import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from vidueu_config import load_config

from run_pipeline_http_dashboard import DEFAULT_PORT as DEFAULT_DASHBOARD_PORT
from run_pipeline_http_dashboard import start_dashboard_server

PipelinePlanner = None
_PIPELINE_PLANNER_IMPORT_ERROR = None

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
SKIP_VIDEO_SCAN_DIRS = {
    "__pycache__",
    "_tmp",
    ".git",
    ".pytest_cache",
}
BENCH_KEYS = (
    "llava_bench",
    "mmt_bench",
    "mme",
    "edit",
    "dialog_visdial_like",
    "dialog_mmdu_like",
    "unify",
    "generation",
)
BENCH_ALIASES = {
    "magicbrush": "edit",
}
VIDEO_RUN_MODE_SKIP_COMPLETED = "skip_completed"
VIDEO_RUN_MODE_FORCE_OVERWRITE = "force_overwrite"
DEFAULT_WORKSPACE_OUTPUT_ROOT = "data/runs"
OUTPUT_INDEX_CACHE_VERSION = 1


def _get_pipeline_planner_class():
    """Import the heavy planner only after cheap startup work is complete."""
    global PipelinePlanner, _PIPELINE_PLANNER_IMPORT_ERROR
    if PipelinePlanner is not None:
        return PipelinePlanner
    try:
        from orchestrator.planner import PipelinePlanner as imported
    except Exception as exc:
        _PIPELINE_PLANNER_IMPORT_ERROR = exc
        raise RuntimeError(f"PipelinePlanner import failed: {exc!r}") from exc
    PipelinePlanner = imported
    _PIPELINE_PLANNER_IMPORT_ERROR = None
    return PipelinePlanner


def _normalize_benchmark_name(value: object) -> str:
    bench = BENCH_ALIASES.get(str(value or "").strip(), str(value or "").strip())
    if bench.startswith("edit."):
        return "edit"
    if bench.startswith("generation."):
        return "generation"
    if bench.startswith("understanding."):
        return bench.split(".", 1)[1]
    if bench in {"geneval", "wise", "generation"}:
        return "generation"
    if bench == "unify" or bench.startswith("unify."):
        return "unify"
    return bench


def _empty_video_stats() -> dict[str, object]:
    return {
        "total_items": 0,
        "qa_items": 0,
        "by_bench": {k: 0 for k in BENCH_KEYS},
    }


def _canonical_video_path(video_path: object) -> str:
    text = str(video_path or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return os.path.abspath(text)


def _runtime_cfg(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    return config.setdefault("runtime", {})


def _paths_cfg(config: dict) -> dict:
    if not isinstance(config, dict):
        return {}
    return config.setdefault("paths", {})


def _resolve_video_run_mode(config: dict) -> str:
    runtime = _runtime_cfg(config)
    raw = str(runtime.get("video_run_mode") or runtime.get("resume_mode") or "").strip().lower()
    aliases = {
        "": VIDEO_RUN_MODE_SKIP_COMPLETED,
        "skip": VIDEO_RUN_MODE_SKIP_COMPLETED,
        "skip_completed": VIDEO_RUN_MODE_SKIP_COMPLETED,
        "resume": VIDEO_RUN_MODE_SKIP_COMPLETED,
        "resume_completed": VIDEO_RUN_MODE_SKIP_COMPLETED,
        "overwrite": VIDEO_RUN_MODE_FORCE_OVERWRITE,
        "force": VIDEO_RUN_MODE_FORCE_OVERWRITE,
        "force_overwrite": VIDEO_RUN_MODE_FORCE_OVERWRITE,
    }
    resolved = aliases.get(raw, VIDEO_RUN_MODE_SKIP_COMPLETED)
    runtime["video_run_mode"] = resolved
    return resolved


def _video_parallelism_cfg(config: dict) -> dict:
    runtime = _runtime_cfg(config)
    raw = runtime.get("video_parallelism")
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _video_parallelism_enabled(config: dict) -> bool:
    cfg = _video_parallelism_cfg(config)
    raw = cfg.get("enabled", False)
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_video_parallelism_workers(config: dict, planner, pending_count: int) -> int:
    if pending_count <= 0:
        return 1
    cfg = _video_parallelism_cfg(config)
    configured = int(cfg.get("max_workers", 0) or 0)
    if configured > 0:
        return max(1, min(configured, pending_count))
    try:
        pool_size = int(planner.model_manager.get_vlm_pool_size() or 1)
    except Exception:
        pool_size = 1
    slots_per_video = max(1, int(cfg.get("vlm_slots_per_video", 1) or 1))
    auto_workers = max(1, pool_size // slots_per_video)
    return max(1, min(auto_workers, pending_count))


def _video_state_dir(config: dict) -> Path:
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    state_dir = output_dir / "_video_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def _video_identity(video_path: str) -> dict[str, object]:
    p = Path(_canonical_video_path(video_path))
    stat = p.stat()
    payload = f"{str(p)}|{int(stat.st_size)}|{int(getattr(stat, 'st_mtime_ns', int(stat.st_mtime * 1e9)))}"
    key = hashlib.sha1(payload.encode("utf-8", errors="replace")).hexdigest()[:16]
    return {
        "video_key": key,
        "video_path": str(p),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _manifest_path_for_identity(config: dict, identity: dict[str, object]) -> Path:
    return _video_state_dir(config) / f"{str(identity.get('video_key') or '')}.json"


def _video_segment_log_path(config: dict, video_path: str) -> Path:
    paths = _paths_cfg(config)
    log_dir = Path(str(paths.get("log_dir") or "logs"))
    return log_dir / f"{Path(str(video_path)).stem}_segments.jsonl"


def _video_has_segment_log(config: dict, video_path: str) -> bool:
    return _video_segment_log_path(config, video_path).exists()


def _load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _path_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except Exception:
        return ""
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    return f"{stat.st_size}:{mtime_ns}"


def _output_index_cache_path(config: dict) -> Path:
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    return output_dir / "_video_state" / "output_index_cache.json"


def _output_index_signature(config: dict) -> dict[str, object]:
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    bench_dir = output_dir / "benchmarks_jsonl"
    return {
        "data_jsonl": _path_signature(output_dir / "data.jsonl"),
        "benchmarks_jsonl": {
            bench: _path_signature(bench_dir / f"{bench}.jsonl")
            for bench in BENCH_KEYS
        },
    }


def _load_output_index_cache(config: dict, signature: dict[str, object]) -> dict[str, dict[str, object]] | None:
    cache_path = _output_index_cache_path(config)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", 0) or 0) != OUTPUT_INDEX_CACHE_VERSION:
        return None
    if payload.get("signature") != signature:
        return None
    index = payload.get("index")
    if not isinstance(index, dict):
        return None
    out: dict[str, dict[str, object]] = {}
    for key, value in index.items():
        if isinstance(value, dict):
            out[str(key)] = dict(value)
    return out


def _write_output_index_cache(config: dict, index: dict[str, dict[str, object]]) -> None:
    cache_path = _output_index_cache_path(config)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": OUTPUT_INDEX_CACHE_VERSION,
            "signature": _output_index_signature(config),
            "index": index,
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _iter_jsonl_dicts(path: Path):
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except Exception:
        return


def _build_output_index(config: dict) -> dict[str, dict[str, object]]:
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    signature = _output_index_signature(config)
    cached = _load_output_index_cache(config, signature)
    if cached is not None:
        return cached

    index: dict[str, dict[str, object]] = {}

    def _entry(video_path: str) -> dict[str, object]:
        return index.setdefault(_canonical_video_path(video_path), _empty_video_stats())

    data_file = output_dir / "data.jsonl"
    for item in _iter_jsonl_dicts(data_file):
        video_path = _canonical_video_path(item.get("video_path"))
        if not video_path:
            continue
        ent = _entry(video_path)
        ent["total_items"] = int(ent.get("total_items", 0) or 0) + 1
        bench = _normalize_benchmark_name(item.get("benchmark", ""))
        if bench != "edit":
            ent["qa_items"] = int(ent.get("qa_items", 0) or 0) + 1

    bench_dir = output_dir / "benchmarks_jsonl"
    for bench in BENCH_KEYS:
        bench_file = bench_dir / f"{bench}.jsonl"
        for item in _iter_jsonl_dicts(bench_file):
            video_path = _canonical_video_path(item.get("video_path"))
            if not video_path:
                continue
            ent = _entry(video_path)
            by_bench = dict(ent.get("by_bench", {}) or {})
            by_bench[bench] = int(by_bench.get(bench, 0) or 0) + 1
            ent["by_bench"] = by_bench

    for ent in index.values():
        by_bench = dict(ent.get("by_bench", {}) or {})
        for bench in BENCH_KEYS:
            by_bench.setdefault(bench, 0)
        ent["by_bench"] = by_bench
    _write_output_index_cache(config, index)
    return index


def _completed_video_stats(
    manifest: dict | None,
    output_index: dict[str, dict[str, object]],
    video_path_hint: object = None,
) -> dict[str, object]:
    video_path = _canonical_video_path((manifest or {}).get("video_path") if isinstance(manifest, dict) else video_path_hint)
    actual = dict(output_index.get(video_path, {}) or {}) if video_path else {}
    if int(actual.get("total_items", 0) or 0) > 0:
        return actual
    return dict(manifest or {})


def _remove_video_rows_from_file(path: Path, video_path: str) -> None:
    video_path = _canonical_video_path(video_path)
    if not path.exists():
        return
    kept: list[str] = []
    changed = False
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.rstrip("\n")
                stripped = text.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except Exception:
                    kept.append(text)
                    continue
                if isinstance(obj, dict) and _canonical_video_path(obj.get("video_path")) == video_path:
                    changed = True
                    continue
                kept.append(text)
    except Exception:
        return
    if not changed:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(row + "\n")


def _purge_video_outputs(config: dict, video_path: str) -> None:
    video_path = _canonical_video_path(video_path)
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    _remove_video_rows_from_file(output_dir / "data.jsonl", video_path)
    bench_dir = output_dir / "benchmarks_jsonl"
    for bench in BENCH_KEYS:
        _remove_video_rows_from_file(bench_dir / f"{bench}.jsonl", video_path)


def _prune_other_manifests_for_video(config: dict, video_path: str, *, keep_key: str) -> None:
    video_path = _canonical_video_path(video_path)
    state_dir = _video_state_dir(config)
    for manifest_path in state_dir.glob("*.json"):
        if manifest_path.stem == keep_key:
            continue
        manifest = _load_manifest(manifest_path)
        if not isinstance(manifest, dict):
            continue
        if _canonical_video_path(manifest.get("video_path")) != video_path:
            continue
        try:
            manifest_path.unlink()
        except Exception:
            pass


def _set_output_index_entry(output_index: dict[str, dict[str, object]], video_path: str, *, total_items: int, qa_items: int, by_bench: dict[str, int]) -> None:
    output_index[_canonical_video_path(video_path)] = {
        "total_items": int(total_items),
        "qa_items": int(qa_items),
        "by_bench": {bench: int(by_bench.get(bench, 0) or 0) for bench in BENCH_KEYS},
    }


def _reset_outputs_for_force_overwrite(config: dict) -> None:
    paths = _paths_cfg(config)
    output_dir = Path(str(paths.get("output_dir") or "dataset/output"))
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)


try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _resolve_config_output_root(config: dict) -> str:
    runtime = _runtime_cfg(config)
    explicit = str(runtime.get("output_root") or "").strip()
    if explicit:
        return explicit

    explicit_work_root = str(runtime.get("work_root") or "").strip()
    if explicit_work_root:
        return explicit_work_root

    paths = _paths_cfg(config)
    output_dir = str(paths.get("output_dir") or "").strip()
    log_dir = str(paths.get("log_dir") or "").strip()
    if not output_dir or not log_dir:
        return ""

    output_path = Path(output_dir)
    log_path = Path(log_dir)
    if output_path.name != "output" or log_path.name != "logs":
        return ""
    if output_path.parent != log_path.parent:
        return ""
    return str(output_path.parent)


def _resolve_effective_output_root(config: dict, cli_output_root: str) -> str:
    explicit = str(cli_output_root or "").strip()
    if explicit:
        return explicit
    from_config = _resolve_config_output_root(config)
    if from_config:
        return from_config
    return DEFAULT_WORKSPACE_OUTPUT_ROOT


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def is_fixed_file(p: Path) -> bool:
    stem_lower = p.stem.lower()
    suffixes_lower = [suffix.lower() for suffix in p.suffixes]

    if stem_lower.endswith("_fixed"):
        return True
    if ".fix" in suffixes_lower or ".fixed" in suffixes_lower:
        return True
    return False


def _video_scan_cfg(config: dict) -> dict:
    runtime = _runtime_cfg(config)
    raw = runtime.get("video_scan") or runtime.get("input_scan") or {}
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _load_video_list_file(path: Path) -> list[str]:
    videos: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Video list file does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            item_path = ""
            if text.startswith("{"):
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = {}
                if isinstance(payload, dict):
                    item_path = str(payload.get("video_path") or payload.get("path") or payload.get("input_path") or "").strip()
            else:
                item_path = text
            if item_path:
                videos.append(item_path)
    return sorted(set(videos))


def collect_videos(input_path: str, scan_cfg: dict | None = None) -> list[str]:
    scan_cfg = dict(scan_cfg or {})
    video_list_file = str(
        scan_cfg.get("video_list_file")
        or scan_cfg.get("manifest_file")
        or scan_cfg.get("video_manifest")
        or ""
    ).strip()
    if video_list_file:
        return _load_video_list_file(Path(video_list_file))

    p = Path(input_path)

    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if p.is_file():
        if not is_video_file(p):
            raise ValueError(f"Input is not a supported video file: {p}")
        if is_fixed_file(p):
            print(f"Skip _fixed file: {p}")
            return []
        return [str(p)]

    if p.is_dir():
        files = []
        skip_dirs = set(SKIP_VIDEO_SCAN_DIRS)
        skip_dirs.update(str(x) for x in list(scan_cfg.get("skip_dirs", []) or []) if str(x).strip())
        max_depth_raw = scan_cfg.get("max_depth", 0)
        try:
            max_depth = int(max_depth_raw or 0)
        except Exception:
            max_depth = 0
        base_depth = len(p.resolve().parts)
        for root, dirs, filenames in os.walk(p, onerror=lambda exc: print(f"Skip unreadable input dir: {exc}")):
            current_depth = max(0, len(Path(root).resolve().parts) - base_depth)
            if max_depth > 0 and current_depth >= max_depth:
                dirs[:] = []
            else:
                dirs[:] = [
                    dirname
                    for dirname in dirs
                    if dirname not in skip_dirs and not dirname.startswith(".")
                ]
            for filename in filenames:
                child = Path(root) / filename
                try:
                    if not is_video_file(child):
                        continue
                    if is_fixed_file(child):
                        continue
                except (FileNotFoundError, PermissionError, OSError) as exc:
                    print(f"Skip inaccessible input file during scan: {child} | {exc}")
                    continue
                files.append(str(child))
        return sorted(set(files))

    raise ValueError(f"Input path is neither a file nor a directory: {input_path}")


def _normalize_input_spec(raw_spec: object) -> dict[str, str]:
    if isinstance(raw_spec, str):
        input_path = raw_spec.strip()
        if not input_path:
            raise ValueError("Empty input path in config runtime.input_paths")
        return {"input_path": input_path, "source_name": ""}

    if isinstance(raw_spec, dict):
        input_path = str(
            raw_spec.get("path")
            or raw_spec.get("input")
            or raw_spec.get("input_path")
            or ""
        ).strip()
        if not input_path:
            raise ValueError(f"Invalid input spec without path: {raw_spec!r}")
        source_name = str(raw_spec.get("source_name") or "").strip()
        return {"input_path": input_path, "source_name": source_name}

    raise ValueError(f"Unsupported input spec: {raw_spec!r}")


def _resolve_input_specs(config: dict, args) -> list[dict[str, str]]:
    cli_input = str(args.input or "").strip()
    cli_source_name = str(args.source_name or "").strip()
    if cli_input:
        return [{"input_path": cli_input, "source_name": cli_source_name}]

    runtime = _runtime_cfg(config)
    raw_specs = runtime.get("input_paths")
    if raw_specs is None:
        raw_specs = runtime.get("input_path")
    if raw_specs is None:
        raise ValueError(
            "No input specified. Provide --input or set runtime.input_path / runtime.input_paths in config."
        )

    if isinstance(raw_specs, list):
        specs = [_normalize_input_spec(item) for item in raw_specs]
    else:
        specs = [_normalize_input_spec(raw_specs)]

    if cli_source_name:
        if len(specs) != 1:
            raise ValueError(
                "--source-name cannot be used with multiple config-defined inputs; "
                "set source_name inside each runtime.input_paths entry instead."
            )
        specs[0]["source_name"] = cli_source_name

    return specs


def _count_by_bench(items) -> dict[str, int]:
    """Count items by known benchmark name."""
    out = {k: 0 for k in BENCH_KEYS}
    if not isinstance(items, list):
        return out

    for item in items:
        if not isinstance(item, dict):
            continue
        bench = _normalize_benchmark_name(item.get("benchmark", ""))
        if bench in out:
            out[bench] += 1
    return out


def _count_items(items) -> tuple[int, int]:
    """
    Returns:
      total_items: total T4 items including edit.
      qa_items: QA-like items excluding edit.
    """
    if not isinstance(items, list):
        return 0, 0

    total = 0
    qa = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        bench = _normalize_benchmark_name(str(item.get("benchmark", "")).strip().lower())
        if bench != "edit":
            qa += 1
    return total, qa


def _format_bench_counts(by_bench: dict[str, int]) -> str:
    return (
        f"by_bench: llava={by_bench.get('llava_bench', 0)}, "
        f"edit={by_bench.get('edit', 0)}, "
        f"mmt={by_bench.get('mmt_bench', 0)}, "
        f"mme={by_bench.get('mme', 0)}, "
        f"visdial={by_bench.get('dialog_visdial_like', 0)}, "
        f"mmdu={by_bench.get('dialog_mmdu_like', 0)}, "
        f"unify={by_bench.get('unify', 0)}, "
        f"generation={by_bench.get('generation', 0)}"
    )


def _derive_source_name(input_path: str, explicit_source_name: str = "") -> str:
    input_p = Path(input_path).resolve()
    if explicit_source_name:
        return explicit_source_name
    return input_p.name if input_p.is_dir() else input_p.parent.name


def _configure_workspace_layout(
    config: dict,
    *,
    input_path: str,
    output_root: str,
    source_name: str = "",
) -> None:
    """
    Override output-related paths so every run writes under:
      {output_root}/{source_name}/...
    """
    if not output_root:
        return

    resolved_source_name = _derive_source_name(input_path, source_name)
    if resolved_source_name in {".", ".."} or "/" in resolved_source_name or "\\" in resolved_source_name:
        raise ValueError("unsafe_source_name")
    output_root_path = Path(output_root).resolve()
    work_root = output_root_path / resolved_source_name
    dataset_output_file = work_root / "output" / "data.jsonl"

    paths = _paths_cfg(config)
    paths["temp_dir"] = str(work_root / "temp")
    paths["output_dir"] = str(work_root / "output")
    paths["semantic_dir"] = str(work_root / "semantic")
    paths["log_dir"] = str(work_root / "logs")
    paths["debug_dir"] = str(work_root / "logs" / "debug")
    paths["dataset_output"] = str(dataset_output_file)

    dataset_cfg = config.setdefault("dataset", {})
    dataset_cfg["output_file"] = str(dataset_output_file)

    t4_dataset_cfg = (config.setdefault("tasks", {})).setdefault("T4_dataset", {})
    t4_dataset_cfg["output_file"] = str(dataset_output_file)

    runtime = config.setdefault("runtime", {})
    runtime["output_root"] = str(output_root_path)
    runtime["source_name"] = str(resolved_source_name)
    runtime["work_root"] = str(work_root)
    runtime["source_root"] = str(work_root)
    runtime.setdefault("video_run_mode", VIDEO_RUN_MODE_SKIP_COMPLETED)

    (work_root / "temp").mkdir(parents=True, exist_ok=True)
    (work_root / "output").mkdir(parents=True, exist_ok=True)
    (work_root / "semantic").mkdir(parents=True, exist_ok=True)
    (work_root / "logs" / "debug").mkdir(parents=True, exist_ok=True)

    print("Workspace enabled:")
    print(f"   output_root = {output_root_path}")
    print(f"   source_name = {resolved_source_name}")
    print(f"   work_root   = {work_root}")


def _maybe_start_dashboard(args):
    if not getattr(args, "dashboard", True):
        print("[run_pipeline] Dashboard disabled by --no-dashboard.")
        return None
    Path(str(args.resolved_output_root)).mkdir(parents=True, exist_ok=True)
    print(
        f"[run_pipeline] Starting dashboard server for output_root={args.resolved_output_root} "
        f"on port={int(args.dashboard_port)} ..."
    )
    try:
        server = start_dashboard_server(args.resolved_output_root, int(args.dashboard_port))
        print("[run_pipeline] Dashboard server started successfully.")
        return server
    except Exception as exc:
        print(
            f"[run_pipeline] Dashboard start skipped for output_root={args.resolved_output_root} "
            f"port={int(args.dashboard_port)}: {exc}"
        )
        return None


def _process_input_spec(base_config: dict, args, input_spec: dict[str, str]) -> dict[str, object]:
    config = copy.deepcopy(base_config)
    input_path = str(input_spec.get("input_path") or "").strip()
    source_name = str(input_spec.get("source_name") or "").strip()

    _configure_workspace_layout(
        config,
        input_path=input_path,
        output_root=args.resolved_output_root,
        source_name=source_name,
    )
    video_run_mode = _resolve_video_run_mode(config)
    if video_run_mode == VIDEO_RUN_MODE_FORCE_OVERWRITE:
        _reset_outputs_for_force_overwrite(config)
        _video_state_dir(config)
    output_index = _build_output_index(config)

    all_videos = collect_videos(input_path, _video_scan_cfg(config))
    if not all_videos:
        print(f"No videos to process for input: {input_path}")
        return {"elapsed": 0.0, "total_items": 0, "qa_items": 0, "by_bench": {k: 0 for k in BENCH_KEYS}}

    video_list = list(all_videos)

    print(f"Input root: {input_path}")
    print(f"Total videos found: {len(all_videos)}")
    print(f"Videos selected for this run: {len(video_list)}")
    print(f"Video run mode: {video_run_mode}")

    planner = None

    grand_total_items = 0
    grand_qa_items = 0
    grand_elapsed = 0.0
    grand_by_bench = {k: 0 for k in BENCH_KEYS}

    def _ensure_planner():
        nonlocal planner
        if planner is None:
            planner_cls = _get_pipeline_planner_class()
            planner = planner_cls(config)
        return planner

    def _finalize_video_result(video_path: str, identity: dict[str, object], manifest_path: Path, items, elapsed: float) -> None:
        nonlocal grand_total_items, grand_qa_items, grand_elapsed
        total_items, qa_items = _count_items(items)
        by_bench = _count_by_bench(items)
        bench_str = _format_bench_counts(by_bench)

        manifest_payload = {
            "version": 1,
            "status": "done",
            "video_key": str(identity["video_key"]),
            "video_path": str(identity["video_path"]),
            "size_bytes": int(identity["size_bytes"]),
            "mtime_ns": int(identity["mtime_ns"]),
            "finished_at": time.time(),
            "total_items": int(total_items),
            "qa_items": int(qa_items),
            "by_bench": {k: int(by_bench.get(k, 0) or 0) for k in BENCH_KEYS},
            "output_dir": str((Path(config.get("paths", {}).get("output_dir", "dataset/output"))).resolve()),
        }
        _write_manifest(manifest_path, manifest_payload)
        _prune_other_manifests_for_video(config, str(identity["video_path"]), keep_key=str(identity["video_key"]))
        _set_output_index_entry(
            output_index,
            str(identity["video_path"]),
            total_items=total_items,
            qa_items=qa_items,
            by_bench=by_bench,
        )
        _write_output_index_cache(config, output_index)

        grand_total_items += total_items
        grand_qa_items += qa_items
        grand_elapsed += elapsed
        for key in BENCH_KEYS:
            grand_by_bench[key] += by_bench.get(key, 0)

        avg_per_qa = (elapsed / qa_items) if qa_items > 0 else None
        avg_per_item = (elapsed / total_items) if total_items > 0 else None

        if avg_per_qa is not None:
            print(
                f"Video time: {elapsed:.2f}s | items(total)={total_items} | "
                f"QA(wo edit)={qa_items} | avg_sec_per_QA={avg_per_qa:.4f} | {bench_str}"
            )
        else:
            print(
                f"Video time: {elapsed:.2f}s | items(total)={total_items} | "
                f"QA(wo edit)={qa_items} | avg_sec_per_QA=N/A (qa_items=0) | {bench_str}"
            )

        if avg_per_item is not None:
            print(f"(extra) avg_sec_per_item(total)={avg_per_item:.4f}")

    def _run_video(video_path: str):
        worker_planner = _ensure_planner()
        acquired_vlm_worker = False
        if parallel_enabled and hasattr(worker_planner, "acquire_video_vlm_worker"):
            worker_planner.acquire_video_vlm_worker(video_path)
            acquired_vlm_worker = True
        try:
            start_time = time.time()
            items = worker_planner.run_pipeline(video_path)
            elapsed = time.time() - start_time
            return items, elapsed
        finally:
            if acquired_vlm_worker and hasattr(worker_planner, "release_video_vlm_worker"):
                worker_planner.release_video_vlm_worker()

    try:
        requested_parallel = _video_parallelism_enabled(config)
        parallel_enabled = requested_parallel and len(video_list) > 1

        if parallel_enabled:
            planner = _ensure_planner()
            planner.enable_concurrent_video_mode(True)
            planner.warm_vlm_pool()
            parallel_workers = _resolve_video_parallelism_workers(config, planner, len(video_list))
            parallel_enabled = parallel_workers > 1
        else:
            parallel_workers = 1

        if parallel_enabled:
            print(f"Video parallelism: enabled | workers={parallel_workers}")
            first_error = None
            with ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="video-pipeline") as executor:
                future_to_video = {}
                for i, video_path in enumerate(video_list):
                    print("=" * 60)
                    print(f"[video {i + 1}/{len(video_list)}] Inspecting: {video_path}")
                    print("=" * 60)

                    if not os.path.exists(video_path):
                        print(f"File disappeared, skip: {video_path}")
                        continue

                    identity = _video_identity(video_path)
                    manifest_path = _manifest_path_for_identity(config, identity)
                    manifest = _load_manifest(manifest_path)

                    if video_run_mode == VIDEO_RUN_MODE_SKIP_COMPLETED and isinstance(manifest, dict) and manifest.get("status") == "done" and _video_has_segment_log(config, video_path):
                        stats = _completed_video_stats(manifest, output_index, video_path)
                        total_items = int(stats.get("total_items", 0) or 0)
                        qa_items = int(stats.get("qa_items", 0) or 0)
                        by_bench = {k: int((stats.get("by_bench", {}) or {}).get(k, 0) or 0) for k in BENCH_KEYS}
                        bench_str = _format_bench_counts(by_bench)
                        grand_total_items += total_items
                        grand_qa_items += qa_items
                        for key in BENCH_KEYS:
                            grand_by_bench[key] += by_bench.get(key, 0)
                        print(
                            f"Skip completed video: {video_path} | "
                            f"segment_log={_video_segment_log_path(config, video_path)} | "
                            f"items(total)={total_items} | QA(wo edit)={qa_items} | {bench_str}"
                        )
                        continue

                    if video_run_mode == VIDEO_RUN_MODE_SKIP_COMPLETED:
                        if manifest is not None:
                            print(f"Detected incomplete/stale outputs for video, rerun: {video_path}")
                        _purge_video_outputs(config, str(identity["video_path"]))
                        _set_output_index_entry(
                            output_index,
                            str(identity["video_path"]),
                            total_items=0,
                            qa_items=0,
                            by_bench={k: 0 for k in BENCH_KEYS},
                        )

                    print("=" * 60)
                    print(f"[video {i + 1}/{len(video_list)}] Processing: {video_path}")
                    print("=" * 60)
                    future = executor.submit(_run_video, video_path)
                    future_to_video[future] = (video_path, identity, manifest_path)

                for future in as_completed(future_to_video):
                    video_path, identity, manifest_path = future_to_video[future]
                    try:
                        items, elapsed = future.result()
                    except Exception as exc:
                        print(f"[run_pipeline] Video failed under concurrent execution: {video_path} | {exc!r}")
                        if first_error is None:
                            first_error = exc
                        continue
                    _finalize_video_result(video_path, identity, manifest_path, items, elapsed)
            if first_error is not None:
                raise first_error
        else:
            if requested_parallel and len(video_list) > 1:
                print("Video parallelism: requested but effective workers=1; falling back to sequential mode.")
            for i, video_path in enumerate(video_list):
                print("=" * 60)
                print(f"[video {i + 1}/{len(video_list)}] Inspecting: {video_path}")
                print("=" * 60)

                if not os.path.exists(video_path):
                    print(f"File disappeared, skip: {video_path}")
                    continue

                identity = _video_identity(video_path)
                manifest_path = _manifest_path_for_identity(config, identity)
                manifest = _load_manifest(manifest_path)

                if video_run_mode == VIDEO_RUN_MODE_SKIP_COMPLETED and isinstance(manifest, dict) and manifest.get("status") == "done" and _video_has_segment_log(config, video_path):
                    stats = _completed_video_stats(manifest, output_index, video_path)
                    total_items = int(stats.get("total_items", 0) or 0)
                    qa_items = int(stats.get("qa_items", 0) or 0)
                    by_bench = {k: int((stats.get("by_bench", {}) or {}).get(k, 0) or 0) for k in BENCH_KEYS}
                    bench_str = _format_bench_counts(by_bench)
                    grand_total_items += total_items
                    grand_qa_items += qa_items
                    for key in BENCH_KEYS:
                        grand_by_bench[key] += by_bench.get(key, 0)
                    print(
                        f"Skip completed video: {video_path} | "
                        f"segment_log={_video_segment_log_path(config, video_path)} | "
                        f"items(total)={total_items} | QA(wo edit)={qa_items} | {bench_str}"
                    )
                    continue

                if video_run_mode == VIDEO_RUN_MODE_SKIP_COMPLETED:
                    if manifest is not None:
                        print(f"Detected incomplete/stale outputs for video, rerun: {video_path}")
                    _purge_video_outputs(config, str(identity["video_path"]))
                    _set_output_index_entry(
                        output_index,
                        str(identity["video_path"]),
                        total_items=0,
                        qa_items=0,
                        by_bench={k: 0 for k in BENCH_KEYS},
                    )

                print("=" * 60)
                print(f"[video {i + 1}/{len(video_list)}] Processing: {video_path}")
                print("=" * 60)
                items, elapsed = _run_video(video_path)
                _finalize_video_result(video_path, identity, manifest_path, items, elapsed)
    finally:
        if planner is not None:
            planner.enable_concurrent_video_mode(False)
            planner.close()

    grand_bench_str = _format_bench_counts(grand_by_bench)
    if grand_qa_items > 0:
        print(
            f"INPUT summary: elapsed={grand_elapsed:.2f}s | total_items={grand_total_items} | QA={grand_qa_items} | "
            f"avg_sec_per_QA={grand_elapsed / grand_qa_items:.4f} | {grand_bench_str}"
        )
    else:
        print(
            f"INPUT summary: elapsed={grand_elapsed:.2f}s | total_items={grand_total_items} | QA={grand_qa_items} | "
            f"avg_sec_per_QA=N/A (qa_items=0) | {grand_bench_str}"
        )

    return {
        "elapsed": grand_elapsed,
        "total_items": grand_total_items,
        "qa_items": grand_qa_items,
        "by_bench": grand_by_bench,
        "debug_dir": config["paths"]["debug_dir"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help="Input video path or input directory. Optional if runtime.input_path/input_paths is set in config.",
    )
    parser.add_argument("--config", type=str, default="configs/paper.yaml", help="Path to config file")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Root output directory for workspace-style run outputs. If omitted, fall back to config/runtime paths.",
    )
    parser.add_argument(
        "--source-name",
        type=str,
        default="",
        help="Optional source folder name; defaults to a name derived from the input path",
    )
    parser.add_argument(
        "--dashboard",
        dest="dashboard",
        action="store_true",
        default=True,
        help="Start the pipeline dashboard HTTP server in the background.",
    )
    parser.add_argument(
        "--no-dashboard",
        dest="dashboard",
        action="store_false",
        help="Do not start the pipeline dashboard HTTP server.",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=DEFAULT_DASHBOARD_PORT,
        help="HTTP port for the background pipeline dashboard server.",
    )
    args = parser.parse_args()

    dashboard_server = None
    try:
        overrides = {}
        if args.input:
            overrides["input"] = str(Path(args.input).expanduser().resolve())
        if args.output_root:
            overrides["output"] = str(Path(args.output_root).expanduser().resolve())
        base_config = load_config(args.config, overrides=overrides)
        if not args.source_name:
            args.source_name = base_config.get("_vidueu", {}).get("source_name", "")
        args.resolved_output_root = _resolve_effective_output_root(base_config, args.output_root)
        input_specs = _resolve_input_specs(base_config, args)
        dashboard_server = _maybe_start_dashboard(args)
        total_elapsed = 0.0
        total_items = 0
        total_qa_items = 0
        total_by_bench = {k: 0 for k in BENCH_KEYS}

        for idx, input_spec in enumerate(input_specs):
            input_path = str(input_spec.get("input_path") or "").strip()
            source_name = str(input_spec.get("source_name") or "").strip()
            print("#" * 72)
            print(
                f"[run_pipeline] Input spec {idx + 1}/{len(input_specs)} | "
                f"input={input_path} | source_name={source_name or '(auto)'}"
            )
            print("#" * 72)
            stats = _process_input_spec(base_config, args, input_spec)
            total_elapsed += float(stats.get("elapsed", 0.0) or 0.0)
            total_items += int(stats.get("total_items", 0) or 0)
            total_qa_items += int(stats.get("qa_items", 0) or 0)
            by_bench = dict(stats.get("by_bench", {}) or {})
            for bench in BENCH_KEYS:
                total_by_bench[bench] += int(by_bench.get(bench, 0) or 0)

        total_bench_str = _format_bench_counts(total_by_bench)
        if total_qa_items > 0:
            print(
                f"ALL INPUTS summary: elapsed={total_elapsed:.2f}s | total_items={total_items} | "
                f"QA={total_qa_items} | avg_sec_per_QA={total_elapsed / total_qa_items:.4f} | {total_bench_str}"
            )
        else:
            print(
                f"ALL INPUTS summary: elapsed={total_elapsed:.2f}s | total_items={total_items} | "
                f"QA={total_qa_items} | avg_sec_per_QA=N/A (qa_items=0) | {total_bench_str}"
            )

        print("Done.")
    finally:
        if dashboard_server is not None:
            dashboard_server.shutdown()
            dashboard_server.server_close()


if __name__ == "__main__":
    main()
