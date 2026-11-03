from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import time
import urllib.parse
from collections import defaultdict
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_HOST = os.getenv("AGENTPACK_PIPELINE_DASHBOARD_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("AGENTPACK_PIPELINE_DASHBOARD_PORT", "8766"))
DASHBOARD_CACHE_VERSION = 1
DASHBOARD_DATA_TTL_S = float(os.getenv("AGENTPACK_PIPELINE_DASHBOARD_TTL_S", "10") or 10)
BENCH_KEYS = (
    "llava_bench",
    "edit",
    "mmt_bench",
    "mme",
    "dialog_visdial_like",
    "dialog_mmdu_like",
    "unify",
    "generation",
)
BENCH_ALIASES = {
    "magicbrush": "edit",
}
STAGE_ORDER = {
    "T1": 10,
    "T1_REJECT": 15,
    "T1_ERROR": 19,
    "T2": 20,
    "T2_REJECT": 25,
    "T2_ERROR": 29,
    "T3": 30,
    "T3_ERROR": 39,
    "T4": 40,
    "T4_ERROR": 49,
}


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _default_pipeline_root() -> str:
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "data" / "runs",
        repo_root / "data" / "run_pipeline_smoke_fix3",
        repo_root / "data" / "run_pipeline_smoke_fix2",
        repo_root / "data" / "run_pipeline_smoke_fix",
        repo_root / "data" / "run_pipeline_smoke",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(repo_root / "data" / "runs")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _is_source_dir(path: Path) -> bool:
    return (path / "logs").is_dir() and (path / "output").is_dir()


def _resolve_dashboard_root(raw: str) -> Path:
    path = Path(str(raw or "").strip() or _default_pipeline_root()).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _discover_sources(root: Path) -> List[Path]:
    if _is_source_dir(root):
        return [root.resolve()]
    found: List[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if _is_source_dir(child):
            found.append(child.resolve())
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and _is_source_dir(grandchild):
                found.append(grandchild.resolve())
    return found


def _discover_workspaces(root: Path) -> List[Path]:
    return _discover_sources(root)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _count_benchmarks(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = {key: 0 for key in BENCH_KEYS}
    for item in items:
        bench = _normalize_benchmark_name(item.get("benchmark", ""))
        if bench in counts:
            counts[bench] += 1
    return counts


def _normalize_benchmark_name(value: Any) -> str:
    bench = str(value or "").strip()
    bench = BENCH_ALIASES.get(bench, bench)
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


def _stage_instance_rows(stage_block: Dict[str, Any], benchmark: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in _iter_stage_instance_items(stage_block):
        if _normalize_benchmark_name(item.get("benchmark", "")) == benchmark:
            rows.append(item)
    return rows


def _path_signature(path: Path) -> str:
    try:
        stat = path.stat()
    except Exception:
        return ""
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    return f"{stat.st_size}:{mtime_ns}"


def _load_dashboard_cache(root: Path) -> Dict[str, Any]:
    cache_path = root / "pipeline_dashboard_cache.json"
    if not cache_path.exists():
        return {"version": DASHBOARD_CACHE_VERSION, "sources": {}}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": DASHBOARD_CACHE_VERSION, "sources": {}}
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) != DASHBOARD_CACHE_VERSION:
        return {"version": DASHBOARD_CACHE_VERSION, "sources": {}}
    payload.setdefault("sources", {})
    return payload


def _write_dashboard_cache(root: Path, cache: Dict[str, Any]) -> None:
    cache_path = root / "pipeline_dashboard_cache.json"
    cache_path.write_text(json.dumps(_jsonable(cache), ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_stage_instance_items(stage_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    instances = (stage_block.get("instances", {}) or {}) if isinstance(stage_block, dict) else {}
    if not isinstance(instances, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for value in instances.values():
        if isinstance(value, list):
            rows.extend([item for item in value if isinstance(item, dict)])
    return rows


def _infer_t4_module(item: Dict[str, Any]) -> str:
    module = str(item.get("module", "") or "").strip()
    if module:
        return module
    bench = _normalize_benchmark_name(item.get("benchmark", ""))
    if bench in {"edit", "unify", "generation"}:
        return bench
    if bench:
        return "understanding"
    return ""


def _count_t4_modules(stage_block: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in _iter_stage_instance_items(stage_block):
        module = _infer_t4_module(item)
        if module:
            counts[module] = counts.get(module, 0) + 1
    return dict(sorted(counts.items()))


def _infer_task_type(item: Dict[str, Any]) -> str:
    task_type = str(item.get("task_type", "") or "").strip()
    if task_type:
        return task_type
    task = str(item.get("task", "") or "").strip()
    if task:
        return task
    bench = str(item.get("benchmark", "") or "").strip()
    if bench:
        return BENCH_ALIASES.get(bench, bench)
    return ""


def _count_task_types(items: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        task_type = _infer_task_type(item)
        if task_type:
            counts[task_type] = counts.get(task_type, 0) + 1
    return dict(sorted(counts.items()))


def _first_text_deep(value: Any, preferred_keys: List[str] | None = None) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        ordered_keys = list(preferred_keys or [])
        seen = set(ordered_keys)
        for key in value.keys():
            if key not in seen:
                ordered_keys.append(key)
        for key in ordered_keys:
            text = _first_text_deep(value.get(key), preferred_keys=None)
            if text:
                return text
    if isinstance(value, list):
        for child in value:
            text = _first_text_deep(child, preferred_keys=preferred_keys)
            if text:
                return text
    return ""


def _collect_paths(value: Any, out: set[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"img", "image", "source_img", "mask_img"} or key.endswith("_path"):
                if isinstance(child, str) and child:
                    out.add(child)
            elif key == "images" and isinstance(child, list):
                for item in child:
                    if isinstance(item, str) and item:
                        out.add(item)
            _collect_paths(child, out)
    elif isinstance(value, list):
        for item in value:
            _collect_paths(item, out)


def _sample_stage_images(stage_payload: Dict[str, Any], limit: int = 6) -> List[str]:
    found: set[str] = set()
    _collect_paths(stage_payload, found)
    return sorted(found)[:limit]


def _trim_text(value: Any, limit: int = 220) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _pick_first_text(item: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_first_image(item: Dict[str, Any]) -> str:
    for key in ("image", "source_img", "img", "mask_img"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    images = item.get("images")
    if isinstance(images, list):
        for value in images:
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _format_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _format_compact(value: Any, limit: int = 280) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _trim_text(value.strip(), limit)
    if isinstance(value, list):
        parts = [_format_scalar(item) for item in value]
        parts = [part for part in parts if part]
        return _trim_text(" | ".join(parts), limit)
    if isinstance(value, dict):
        return _trim_text(json.dumps(value, ensure_ascii=False), limit)
    return _trim_text(str(value), limit)


def _segment_keyframe_map(item: Dict[str, Any]) -> Dict[int, str]:
    meta = item.get("meta")
    if not isinstance(meta, dict):
        return {}
    keyframes = meta.get("segment_keyframes")
    if not isinstance(keyframes, list):
        return {}
    mapping: Dict[int, str] = {}
    for frame in keyframes:
        if not isinstance(frame, dict):
            continue
        try:
            pos = int(frame.get("pos"))
        except Exception:
            continue
        img = str(frame.get("img", "") or "").strip()
        if img:
            mapping[pos] = img
    return mapping


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _labelize_render_key(key: str) -> str:
    text = str(key or "").replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split()) if text else "Render"


def _collect_render_paths(value: Any, out: List[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key or "")
            if key_text in {"image", "img", "source_img", "target_img", "mask_img"} or key_text.endswith("_path"):
                if isinstance(child, str) and child.strip():
                    out.append(child.strip())
                continue
            if key_text == "images" and isinstance(child, list):
                out.extend([str(item or "").strip() for item in child if isinstance(item, str) and item.strip()])
                continue
            _collect_render_paths(child, out)
        return
    if isinstance(value, list):
        for child in value:
            _collect_render_paths(child, out)
        return
    if isinstance(value, str) and value.strip().lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".mp4")):
        out.append(value.strip())


def _t4_unify_dialogue(item: Dict[str, Any]) -> List[Dict[str, str]]:
    task_type = _infer_task_type(item)
    turns_list = item.get("turns")
    if isinstance(turns_list, list):
        dialogue: List[Dict[str, str]] = []
        for idx, turn in enumerate(turns_list, start=1):
            if not isinstance(turn, dict):
                continue
            render = turn.get("render")
            if not isinstance(render, dict):
                continue
            t_name = str(turn.get("t", "") or f"turn{idx}")
            dialogue.append({"role": "turn", "text": f"{idx}. {t_name} ({str(turn.get('task_type', '') or '')})"})
            for section_name in ("query", "answer"):
                section = render.get(section_name)
                if not isinstance(section, dict):
                    continue
                rows = section.get("messages")
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    role = str(row.get("role", "") or section_name).strip()
                    text = row.get("content")
                    if text is None:
                        text = row.get("text")
                    content = _format_compact(text, 360)
                    if content:
                        dialogue.append({"role": role, "text": content})
        if dialogue:
            return dialogue
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    render = item.get("render") if isinstance(item.get("render"), dict) else {}
    dialogue: List[Dict[str, str]] = []

    def say(role: str, text: Any) -> None:
        content = _format_compact(text, 360)
        if content:
            dialogue.append({"role": role, "text": content})

    def say_render_messages(rows: Any) -> None:
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "") or "").strip() or "assistant"
            text = row.get("content")
            if text is None:
                text = row.get("text")
            say(role, text)

    query = render.get("query") if isinstance(render.get("query"), dict) else {}
    answer = render.get("answer") if isinstance(render.get("answer"), dict) else {}
    if query or answer:
        say_render_messages(query.get("messages"))
        say_render_messages(answer.get("messages"))
        if dialogue:
            return dialogue

    if task_type == "unify.seed.l3.gts":
        select = (turns.get("select") or {}) if isinstance(turns.get("select"), dict) else {}
        say("user", select.get("question") or turns.get("question"))
        say("assistant", select.get("answer"))
        say("assistant", select.get("rationale"))
    elif task_type == "unify.weave.uie_select_recall":
        qa_open = (turns.get("qa_open") or {}) if isinstance(turns.get("qa_open"), dict) else {}
        select = (turns.get("select") or {}) if isinstance(turns.get("select"), dict) else {}
        say("user", qa_open.get("question"))
        say("assistant", qa_open.get("answer"))
        say("user", select.get("question"))
        say("assistant", select.get("answer"))
        say("assistant", select.get("rationale"))
    elif task_type == "unify.comm.task4.c1_steps":
        say("user", turns.get("question"))
        for row in turns.get("select") or []:
            if not isinstance(row, dict):
                continue
            say("user", row.get("question"))
            say("assistant", row.get("answer"))
            say("assistant", row.get("rationale"))
    elif task_type == "unify.comm.task4.c2_fill_gap":
        say("user", turns.get("question"))
        select = (turns.get("select") or {}) if isinstance(turns.get("select"), dict) else {}
        say("user", select.get("question"))
        say("assistant", select.get("answer"))
        say("assistant", select.get("rationale"))
    elif task_type == "unify.isg.i2.seq_select":
        for row in turns.get("select") or []:
            if not isinstance(row, dict):
                continue
            say("user", row.get("question"))
            say("assistant", row.get("answer"))
            say("assistant", row.get("rationale"))
    return dialogue


def _t4_example_images(item: Dict[str, Any]) -> List[str]:
    benchmark = str(item.get("benchmark", "") or "").strip()
    task_type = _infer_task_type(item)
    render = item.get("render") if isinstance(item.get("render"), dict) else {}
    images: List[str] = []
    if benchmark == "edit":
        for key in ("source_img", "target_img", "mask_img"):
            value = str(item.get(key, "") or "").strip()
            if value:
                images.append(value)
        return _ordered_unique(images)
    if benchmark in {"geneval", "generation"}:
        value = str(item.get("image", "") or "").strip()
        if value:
            return [value]
        meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        value = str(meta.get("ground_truth_image", "") or "").strip()
        if value:
            return [value]
        raw_images = item.get("images")
        if isinstance(raw_images, list):
            for value in raw_images:
                text = str(value or "").strip()
                if text:
                    return [text]
        return []
    for key in ("source_img", "target_img", "mask_img", "image", "img"):
        value = str(item.get(key, "") or "").strip()
        if value:
            images.append(value)
    raw_images = item.get("images")
    if isinstance(raw_images, list):
        images.extend([str(value or "").strip() for value in raw_images if str(value or "").strip()])
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    for key in ("source_image", "ground_truth_image"):
        value = str(meta.get(key, "") or "").strip()
        if value:
            images.append(value)
    frame_map = _segment_keyframe_map(item)
    positions: List[int] = []
    if benchmark.startswith("unify.") or task_type.startswith("unify."):
        _collect_render_paths(render, images)
        turns_list = item.get("turns")
        if isinstance(turns_list, list):
            for turn in turns_list:
                if isinstance(turn, dict):
                    _collect_render_paths(turn.get("render"), images)
        for key in ("source_frame", "target_frame", "earlier_frame", "middle_frame", "later_frame"):
            if key in render:
                try:
                    positions.append(int(render.get(key)))
                except Exception:
                    pass
        for key in ("frame_positions", "context_positions", "future_positions"):
            values = render.get(key)
            if isinstance(values, list):
                for value in values:
                    try:
                        positions.append(int(value))
                    except Exception:
                        continue
    elif benchmark in {"geneval", "generation"}:
        positions.extend(sorted(frame_map.keys()))
    for pos in positions:
        img = frame_map.get(pos)
        if img:
            images.append(img)
    return _ordered_unique(images)


def _append_render_fields(fields: List[Dict[str, str]], value: Any, prefix: str = "Render", depth: int = 0) -> None:
    if depth > 4:
        text = _format_compact(value)
        if text:
            fields.append({"label": prefix, "value": text})
        return
    if isinstance(value, dict):
        if depth == 0 and value.get("kind"):
            kind = _format_compact(value.get("kind"))
            if kind:
                fields.append({"label": "Render Kind", "value": kind})
        for key, child in value.items():
            key_text = str(key or "")
            if key_text == "kind":
                continue
            if key_text in {
                "source_frame",
                "target_frame",
                "earlier_frame",
                "middle_frame",
                "later_frame",
                "frame_positions",
                "context_positions",
                "future_positions",
                "image",
                "img",
                "source_img",
                "target_img",
                "mask_img",
            } or key_text.endswith("_path"):
                continue
            child_prefix = f"{prefix} {_labelize_render_key(key_text)}" if prefix else _labelize_render_key(key_text)
            _append_render_fields(fields, child, child_prefix, depth + 1)
        return
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            for idx, item in enumerate(value, start=1):
                text = _format_compact(item)
                if text:
                    fields.append({"label": f"{prefix} {idx}", "value": text})
            return
        for idx, item in enumerate(value, start=1):
            _append_render_fields(fields, item, f"{prefix} {idx}", depth + 1)
        return
    text = _format_compact(value)
    if text:
        fields.append({"label": prefix, "value": text})


def _t4_example_fields(item: Dict[str, Any]) -> List[Dict[str, str]]:
    benchmark = str(item.get("benchmark", "") or "").strip()
    task_type = _infer_task_type(item)
    turns = item.get("turns") if isinstance(item.get("turns"), dict) else {}
    render = item.get("render") if isinstance(item.get("render"), dict) else {}
    review = ((item.get("meta") or {}).get("review", {}) or {}) if isinstance(item.get("meta"), dict) else {}
    fields: List[Dict[str, str]] = []

    def add(label: str, value: Any, limit: int = 280) -> None:
        text = _format_compact(value, limit)
        if text:
            fields.append({"label": label, "value": text})

    def add_lines(prefix: str, values: Any, limit: int = 280) -> None:
        if isinstance(values, list):
            added = 0
            for idx, value in enumerate(values, start=1):
                text = _format_compact(value, limit)
                if text:
                    fields.append({"label": f"{prefix} {idx}", "value": text})
                    added += 1
            if added:
                return
        add(prefix, values, limit)

    def add_select_rows(rows: Any, prefix: str = "Select") -> None:
        if not isinstance(rows, list):
            return
        for idx, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            question = _format_compact(row.get("question"))
            answer = _format_compact(row.get("answer"))
            rationale = _format_compact(row.get("rationale"))
            if question:
                fields.append({"label": f"{prefix} {idx} Question", "value": question})
            if answer:
                fields.append({"label": f"{prefix} {idx} Answer", "value": answer})
            if rationale:
                fields.append({"label": f"{prefix} {idx} Rationale", "value": rationale})

    if benchmark == "edit":
        add("Instruction", item.get("instruction") or item.get("edit_instruction"))
        add_lines("Reviewed Edit", review.get("edits"))
        add("Review Reason", review.get("reason"))
    elif benchmark.startswith("unify.") or task_type.startswith("unify."):
        turns_list = item.get("turns")
        if isinstance(turns_list, list):
            for idx, turn in enumerate(turns_list, start=1):
                if not isinstance(turn, dict):
                    continue
                turn_name = str(turn.get("t", "") or f"turn{idx}")
                turn_type = str(turn.get("task_type", "") or "")
                fields.append({"label": f"Turn {idx}", "value": f"{turn_name} ({turn_type})".strip()})
                _append_render_fields(fields, turn.get("render"), f"Turn {idx} Render")
        else:
            _append_render_fields(fields, render, "Render")
        if not fields:
            add("Render Kind", render.get("kind"))
    elif benchmark in {"geneval", "generation"}:
        add("Prompt", item.get("prompt") or item.get("query"))
    else:
        add("Instruction", item.get("instruction") or item.get("edit_instruction"))
        add("Prompt", item.get("prompt") or item.get("query") or item.get("question"))
        add("Answer", item.get("caption") or item.get("answer") or item.get("target"))
    if not fields:
        add("Raw", item)
    return fields


def _build_t1_shot_cards(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    shots = list(out.get("shot_profiles") or out.get("shots") or [])
    gate_a = out.get("gate_a", {}) or {}
    decisions = {
        str(item.get("shot_id", "") or ""): item
        for item in list(gate_a.get("shot_decisions") or [])
        if isinstance(item, dict)
    }
    kept_ids = {str(x) for x in list(gate_a.get("kept_shot_ids") or [])}
    cards: List[Dict[str, Any]] = []
    for idx, shot in enumerate(shots):
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("shot_id", "") or f"shot_{idx}")
        decision = decisions.get(shot_id, {})
        rep_frames = []
        for rep in list(shot.get("rep_frames") or [])[:6]:
            if not isinstance(rep, dict):
                continue
            img = str(rep.get("img_path", "") or rep.get("img", "") or "")
            if not img:
                continue
            rep_frames.append(
                {
                    "img": img,
                    "sec": float(rep.get("sec", 0.0) or 0.0),
                    "kind": str(rep.get("kind", "") or ""),
                }
            )
        cards.append(
            {
                "shot_id": shot_id,
                "shot_role": str(shot.get("shot_role", "") or "generic"),
                "scene_type": str(shot.get("scene_type", "") or ""),
                "summary": str(shot.get("vlm_summary", "") or ""),
                "decision": str(decision.get("decision", "keep" if shot_id in kept_ids else "unknown")),
                "decision_reason": str(decision.get("reason", "") or ""),
                "kept": bool(shot_id in kept_ids or str(decision.get("decision", "") or "") != "drop"),
                "rep_frames": rep_frames,
            }
        )
    return cards


def _build_t2_shot_cards(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    shot_plans = ((out.get("tool_plan", {}) or {}).get("shot_plans", {}) or {}) if isinstance(out, dict) else {}
    gate_b = out.get("gate_b", {}) or {}
    gate_d = out.get("gate_d", {}) or {}
    usable_by_shot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in list(gate_b.get("usable_frames") or []):
        if isinstance(item, dict):
            usable_by_shot[str(item.get("shot_id", "") or "")].append(item)
    selected_by_shot: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in list(gate_d.get("selected_frames") or []):
        if isinstance(item, dict):
            selected_by_shot[str(item.get("shot_id", "") or "")].append(item)
    shot_ids = sorted({str(k) for k in shot_plans.keys()} | set(usable_by_shot.keys()) | set(selected_by_shot.keys()))
    cards: List[Dict[str, Any]] = []
    for shot_id in shot_ids:
        shot_plan = dict(shot_plans.get(shot_id, {}) or {})
        tool_actions = []
        for action in list(shot_plan.get("tool_actions") or []):
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "")
            if tool_name:
                tool_actions.append(
                    {
                        "tool_name": tool_name,
                        "reason": str(action.get("reason", "") or ""),
                    }
                )
        usable_images = []
        for item in list(usable_by_shot.get(shot_id) or [])[:8]:
            img = str(item.get("img", "") or "")
            if img:
                usable_images.append(img)
        selected_images = []
        for item in list(selected_by_shot.get(shot_id) or [])[:6]:
            img = str(item.get("img", "") or "")
            if img:
                selected_images.append(img)
        cards.append(
            {
                "shot_id": shot_id,
                "plan_reason": str(shot_plan.get("plan_reason", "") or ""),
                "tool_actions": tool_actions,
                "usable_images": usable_images,
                "selected_images": selected_images,
                "usable_count": len(list(usable_by_shot.get(shot_id) or [])),
                "selected_count": len(list(selected_by_shot.get(shot_id) or [])),
            }
        )
    return cards


def _build_t3_keyframe_cards(t2_out: Dict[str, Any], t3_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    keyframes = list(t2_out.get("keyframes") or [])
    frames = list(t3_out.get("frames") or [])
    cards: List[Dict[str, Any]] = []
    total = max(len(keyframes), len(frames))
    for idx in range(total):
        keyframe = keyframes[idx] if idx < len(keyframes) and isinstance(keyframes[idx], dict) else {}
        frame = frames[idx] if idx < len(frames) and isinstance(frames[idx], dict) else {}
        img = str(keyframe.get("img", "") or frame.get("img", "") or "")
        if not img:
            continue
        cards.append(
            {
                "index": idx,
                "img": img,
                "sec": float(keyframe.get("sec", frame.get("sec", 0.0)) or 0.0),
                "shot_id": str(keyframe.get("shot_id", "") or ""),
                "caption": str(frame.get("caption", "") or ""),
                "transition_to_next": str(frame.get("transition_to_next", "") or ""),
                "cause": str(frame.get("cause", "") or ""),
            }
        )
    return cards


def _build_t4_examples(stage_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    seen_labels: set[str] = set()
    for item in _iter_stage_instance_items(stage_block):
        if not isinstance(item, dict):
            continue
        bench = _normalize_benchmark_name(item.get("benchmark", ""))
        task_type = _infer_task_type(item)
        label = bench if not task_type or task_type == bench else f"{bench} / {task_type}"
        if label in seen_labels:
            continue
        seen_labels.add(label)
        examples.append(
            {
                "benchmark": bench,
                "task_type": task_type,
                "label": label,
                "image": _pick_first_image(item),
                "images": _t4_example_images(item),
                "dialogue": _t4_unify_dialogue(item) if (bench == "unify" or task_type.startswith("unify.")) else [],
                "fields": _t4_example_fields(item),
                "raw_preview": _trim_text(json.dumps(item, ensure_ascii=False), 320),
            }
        )
    return examples


def _extract_t1_metrics(stage_block: Dict[str, Any]) -> Dict[str, Any]:
    out = (stage_block.get("output", {}) or {}) if isinstance(stage_block, dict) else {}
    runtime = (stage_block.get("runtime", {}) or {}) if isinstance(stage_block, dict) else {}
    shots = list(out.get("shot_profiles") or out.get("shots") or [])
    chapters = list(out.get("chapters") or [])
    rep_frames = []
    for shot in shots:
        rep_frames.extend(list((shot or {}).get("rep_frames") or []))
    return {
        "shots": len(shots),
        "chapters": len(chapters),
        "candidate_frames": len(rep_frames),
        "suitable_for_dataset": bool(out.get("suitable_for_dataset", True)),
        "video_type": str(out.get("video_type", "") or ""),
        "fallback_summary": runtime.get("fallback_summary", {}) or {},
        "sample_images": [str((x or {}).get("img_path", "") or "") for x in rep_frames[:6] if (x or {}).get("img_path")],
        "shot_cards": _build_t1_shot_cards(out),
    }


def _extract_t2_metrics(stage_block: Dict[str, Any]) -> Dict[str, Any]:
    out = (stage_block.get("output", {}) or {}) if isinstance(stage_block, dict) else {}
    gate_b = out.get("gate_b", {}) or {}
    gate_d = out.get("gate_d", {}) or {}
    tool_execution = out.get("tool_execution", {}) or {}
    semantic_by_shot = tool_execution.get("semantic_by_shot", {}) or {}
    return {
        "quality_passed": bool(out.get("quality_passed", False)),
        "candidate_pool": len(list(out.get("candidate_pool") or [])),
        "gateb_passed": len(list(gate_b.get("usable_frames") or [])),
        "gateb_rejected": len(list(gate_b.get("rejected_frames") or [])),
        "final_selected": len(list(gate_d.get("selected_frames") or [])),
        "semantic_shots": len(semantic_by_shot),
        "reason": str(out.get("reason", "") or gate_d.get("final_reason", "") or ""),
        "sample_images": [str((x or {}).get("img", "") or "") for x in list(gate_d.get("selected_frames") or [])[:6] if (x or {}).get("img")],
        "shot_cards": _build_t2_shot_cards(out),
    }


def _extract_t3_metrics(stage_block: Dict[str, Any], t2_out: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = (stage_block.get("output", {}) or {}) if isinstance(stage_block, dict) else {}
    return {
        "dataset_type": str(out.get("dataset_type", "") or "none"),
        "frames": len(list(out.get("frames") or [])),
        "segment_summary": str(out.get("segment_summary", "") or ""),
        "sample_images": [str((x or {}).get("img", "") or "") for x in list(out.get("frames") or [])[:6] if (x or {}).get("img")],
        "keyframe_cards": _build_t3_keyframe_cards(t2_out or {}, out),
    }


def _extract_t4_metrics(stage_block: Dict[str, Any]) -> Dict[str, Any]:
    runtime = (stage_block.get("runtime", {}) or {}) if isinstance(stage_block, dict) else {}
    decision = (stage_block.get("decision", {}) or {}) if isinstance(stage_block, dict) else {}
    stage_items = _iter_stage_instance_items(stage_block)
    counts = _count_benchmarks(stage_items)
    task_type_counts = _count_task_types(stage_items)
    module_counts = _count_t4_modules(stage_block)
    sample_images: List[str] = []
    for bench in BENCH_KEYS:
        for item in _stage_instance_rows(stage_block, bench):
            if isinstance(item, dict):
                for key in ("image", "source_img"):
                    if item.get(key):
                        sample_images.append(str(item.get(key)))
                        break
                images = item.get("images")
                if isinstance(images, list) and images:
                    sample_images.append(str(images[0]))
            if len(sample_images) >= 6:
                break
        if len(sample_images) >= 6:
            break
    selected_benchmarks = [key for key, value in counts.items() if _safe_int(value, 0) > 0]
    raw_edit_pairs = list(decision.get("edit_pairs") or decision.get("magicbrush_pairs") or [])
    return {
        "instances": counts,
        "task_types": task_type_counts,
        "modules": module_counts,
        "total_items": sum(counts.values()),
        "selected_benchmarks": selected_benchmarks,
        "generated_dialog_visdial": _safe_int(runtime.get("generated_dialog_visdial", 0), 0),
        "generated_dialog_mmdu": _safe_int(runtime.get("generated_dialog_mmdu", 0), 0),
        "generated_generation": _safe_int(runtime.get("generated_generation", 0), 0),
        "generation_reviewed_count": _safe_int(runtime.get("generation_reviewed_count", 0), 0),
        "generation_suitable_count": _safe_int(runtime.get("generation_suitable_count", 0), 0),
        "generated_unify": _safe_int(runtime.get("generated_unify", 0), 0),
        "raw_pair_count": _safe_int(runtime.get("raw_pair_count", 0), 0),
        "reviewed_pair_count": _safe_int(runtime.get("reviewed_pair_count", 0), 0),
        "candidate_pair_count": _safe_int(runtime.get("candidate_pair_count", 0), 0),
        "edit_pair_count": len([item for item in raw_edit_pairs if isinstance(item, dict)]),
        "selected_modules": list(runtime.get("selected_modules", []) or list(module_counts.keys())),
        "module_route_reason": str(runtime.get("module_route_reason", "") or ""),
        "sample_images": sample_images[:6],
        "examples": _build_t4_examples(stage_block),
    }


def _sum_counter_maps(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        for name, value in ((row.get(key, {}) or {}).items() if isinstance(row.get(key, {}), dict) else []):
            counts[str(name)] = counts.get(str(name), 0) + _safe_int(value, 0)
    return dict(sorted(counts.items()))


def _build_segment_summary(segment_index: int, segment_path: str, stage_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(stage_rows, key=lambda row: (STAGE_ORDER.get(str(row.get("stage", "")), 999), len(json.dumps(row, ensure_ascii=False))))
    final_stage = str((ordered[-1] or {}).get("stage", "") or "UNKNOWN") if ordered else "UNKNOWN"
    stage_map = {str(row.get("stage", "")): row for row in ordered}
    t2_stage_block = (stage_map.get("T2", {}) or {}).get("T2", {}) or {}
    t1 = _extract_t1_metrics((stage_map.get("T1", {}) or {}).get("T1", {}) or {})
    t2 = _extract_t2_metrics(t2_stage_block)
    t3 = _extract_t3_metrics((stage_map.get("T3", {}) or {}).get("T3", {}) or {}, (t2_stage_block.get("output", {}) or {}))
    t4 = _extract_t4_metrics((stage_map.get("T4", {}) or {}).get("T4", {}) or {})
    error_row = next((row for row in ordered if str(row.get("stage", "")).endswith("_ERROR")), None)
    reject_stage = next((str(row.get("stage", "")) for row in ordered if str(row.get("stage", "")).endswith("_REJECT")), "")
    sample_images = []
    for block in (t4, t3, t2, t1):
        sample_images.extend(list(block.get("sample_images", []) or []))
    deduped = []
    seen = set()
    for path in sample_images:
        if path and path not in seen:
            deduped.append(path)
            seen.add(path)
    return {
        "segment_index": segment_index,
        "segment_path": segment_path,
        "final_stage": final_stage,
        "status": "error" if error_row else ("rejected" if reject_stage else ("done" if final_stage == "T4" else "partial")),
        "is_terminal": bool(error_row or reject_stage or final_stage == "T4"),
        "timeline": [
            {
                "stage": str(row.get("stage", "") or ""),
                "has_error": str(row.get("stage", "")).endswith("_ERROR"),
                "has_reject": str(row.get("stage", "")).endswith("_REJECT"),
            }
            for row in ordered
        ],
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "t4": t4,
        "error": {
            "stage": str(error_row.get("stage", "") or ""),
            "type": str(error_row.get("error_type", "") or ""),
            "message": str(error_row.get("error_message", "") or ""),
            "traceback": str(error_row.get("traceback", "") or ""),
        } if error_row else {},
        "sample_images": deduped[:6],
    }


def _build_video_summary(log_path: Path, cache_entry: Dict[str, Any] | None = None) -> Dict[str, Any]:
    signature = _path_signature(log_path)
    cached_summary = (cache_entry or {}).get("summary", {}) if isinstance(cache_entry, dict) else {}
    if signature and str((cache_entry or {}).get("signature", "") or "") == signature and isinstance(cached_summary, dict):
        return cached_summary
    rows = _load_jsonl(log_path)
    by_segment: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_segment[_safe_int(row.get("segment_index", -1), -1)].append(row)
    segments = [
        _build_segment_summary(seg_idx, str((seg_rows[0] or {}).get("segment_path", "") or ""), seg_rows)
        for seg_idx, seg_rows in sorted(by_segment.items())
        if seg_idx >= 0
    ]
    bench_counts = {key: 0 for key in BENCH_KEYS}
    for seg in segments:
        for key, value in (seg.get("t4", {}).get("instances", {}) or {}).items():
            if key in bench_counts:
                bench_counts[key] += _safe_int(value, 0)
    task_type_counts = _sum_counter_maps([seg.get("t4", {}) or {} for seg in segments], "task_types")
    module_counts = _sum_counter_maps([seg.get("t4", {}) or {} for seg in segments], "modules")
    partial_segments = [seg for seg in segments if not bool(seg.get("is_terminal"))]
    active_segment = partial_segments[-1] if partial_segments else (segments[-1] if segments else None)
    last_row = rows[-1] if rows else {}
    return {
        "video_name": log_path.name.replace("_segments.jsonl", ""),
        "video_log_path": str(log_path),
        "log_mtime_epoch": log_path.stat().st_mtime if log_path.exists() else 0.0,
        "segment_count": len(segments),
        "segments_done": sum(1 for seg in segments if seg.get("status") == "done"),
        "segments_rejected": sum(1 for seg in segments if seg.get("status") == "rejected"),
        "segments_error": sum(1 for seg in segments if seg.get("status") == "error"),
        "has_active_run": bool(partial_segments),
        "active_segment_index": _safe_int((active_segment or {}).get("segment_index", -1), -1),
        "active_stage": str((active_segment or {}).get("final_stage", "") or str(last_row.get("stage", "") or "")),
        "bench_counts": bench_counts,
        "task_type_counts": task_type_counts,
        "module_counts": module_counts,
        "segments": segments,
    }


def _build_source_summary(source_dir: Path, cache_entry: Dict[str, Any] | None = None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    log_dir = source_dir / "logs"
    output_dir = source_dir / "output"
    video_logs = sorted(log_dir.glob("*_segments.jsonl"))
    data_jsonl = output_dir / "data.jsonl"
    benchmark_paths = {key: output_dir / "benchmarks_jsonl" / f"{key}.jsonl" for key in BENCH_KEYS}
    source_signature = {
        "video_logs": {str(path): _path_signature(path) for path in video_logs},
        "data_jsonl": _path_signature(data_jsonl),
        "benchmarks_jsonl": {key: _path_signature(path) for key, path in benchmark_paths.items()},
    }
    cached_signature = (cache_entry or {}).get("signature", {}) if isinstance(cache_entry, dict) else {}
    cached_summary = (cache_entry or {}).get("summary", {}) if isinstance(cache_entry, dict) else {}
    cached_videos = (cache_entry or {}).get("videos", {}) if isinstance(cache_entry, dict) else {}
    if source_signature == cached_signature and isinstance(cached_summary, dict):
        return cached_summary, {"signature": source_signature, "summary": cached_summary, "videos": cached_videos}

    videos = []
    next_video_cache: Dict[str, Any] = {}
    for path in video_logs:
        path_text = str(path)
        video_summary = _build_video_summary(path, cached_videos.get(path_text, {}) if isinstance(cached_videos, dict) else {})
        videos.append(video_summary)
        next_video_cache[path_text] = {
            "signature": source_signature["video_logs"].get(path_text, ""),
            "summary": video_summary,
        }

    all_items = _load_jsonl(data_jsonl)
    benchmark_files = {}
    for key in BENCH_KEYS:
        benchmark_files[key] = len(_load_jsonl(benchmark_paths[key]))
    bench_counts = _count_benchmarks(all_items)
    for key in BENCH_KEYS:
        bench_counts[key] = max(bench_counts.get(key, 0), benchmark_files.get(key, 0))
    task_type_counts = _count_task_types(all_items)
    module_counts = _sum_counter_maps(videos, "module_counts")
    active_videos = [video for video in videos if bool(video.get("has_active_run"))]
    active_video = active_videos[-1] if active_videos else (max(videos, key=lambda x: _safe_float(x.get("log_mtime_epoch", 0.0), 0.0)) if videos else None)
    summary = {
        "source_name": source_dir.name,
        "workspace_name": source_dir.name,
        "source_path": str(source_dir),
        "workspace_path": str(source_dir),
        "source_mtime_epoch": max((_safe_float(video.get("log_mtime_epoch", 0.0), 0.0) for video in videos), default=0.0),
        "video_count": len(videos),
        "segment_count": sum(int(video.get("segment_count", 0)) for video in videos),
        "segments_done": sum(int(video.get("segments_done", 0)) for video in videos),
        "segments_rejected": sum(int(video.get("segments_rejected", 0)) for video in videos),
        "segments_error": sum(int(video.get("segments_error", 0)) for video in videos),
        "dataset_items": len(all_items),
        "has_active_run": any(bool(video.get("has_active_run")) for video in videos),
        "active_video_name": str((active_video or {}).get("video_name", "") or ""),
        "active_stage": str((active_video or {}).get("active_stage", "") or ""),
        "active_segment_index": _safe_int((active_video or {}).get("active_segment_index", -1), -1),
        "bench_counts": bench_counts,
        "task_type_counts": task_type_counts,
        "module_counts": module_counts,
        "videos": videos,
    }
    return summary, {"signature": source_signature, "summary": summary, "videos": next_video_cache}


def _build_dashboard_payload(root: Path) -> Dict[str, Any]:
    cache = _load_dashboard_cache(root)
    cached_sources = cache.get("sources", {}) if isinstance(cache.get("sources", {}), dict) else {}
    sources = []
    next_sources_cache: Dict[str, Any] = {}
    for path in _discover_sources(root):
        source_summary, source_cache = _build_source_summary(path, cached_sources.get(str(path), {}))
        sources.append(source_summary)
        next_sources_cache[str(path)] = source_cache
    aggregate_bench = {key: 0 for key in BENCH_KEYS}
    aggregate_task_types = _sum_counter_maps(sources, "task_type_counts")
    aggregate_modules = _sum_counter_maps(sources, "module_counts")
    for source in sources:
        for key, value in (source.get("bench_counts", {}) or {}).items():
            if key in aggregate_bench:
                aggregate_bench[key] += _safe_int(value, 0)
    active_sources = [source for source in sources if bool(source.get("has_active_run"))]
    active_source = active_sources[-1] if active_sources else (max(sources, key=lambda x: _safe_float(x.get("source_mtime_epoch", 0.0), 0.0)) if sources else None)
    payload = {
        "root_path": str(root),
        "generated_at_epoch": time.time(),
        "loading": False,
        "source_count": len(sources),
        "workspace_count": len(sources),
        "video_count": sum(int(source.get("video_count", 0)) for source in sources),
        "segment_count": sum(int(source.get("segment_count", 0)) for source in sources),
        "segments_done": sum(int(source.get("segments_done", 0)) for source in sources),
        "segments_rejected": sum(int(source.get("segments_rejected", 0)) for source in sources),
        "segments_error": sum(int(source.get("segments_error", 0)) for source in sources),
        "dataset_items": sum(int(source.get("dataset_items", 0)) for source in sources),
        "active_source_name": str((active_source or {}).get("source_name", "") or ""),
        "active_workspace_name": str((active_source or {}).get("source_name", "") or ""),
        "active_video_name": str((active_source or {}).get("active_video_name", "") or ""),
        "active_stage": str((active_source or {}).get("active_stage", "") or ""),
        "active_segment_index": _safe_int((active_source or {}).get("active_segment_index", -1), -1),
        "bench_counts": aggregate_bench,
        "task_type_counts": aggregate_task_types,
        "module_counts": aggregate_modules,
        "sources": sources,
        "workspaces": sources,
    }
    _filter_existing_sample_images_inplace(root, payload)
    _write_dashboard_cache(root, {"version": DASHBOARD_CACHE_VERSION, "sources": next_sources_cache})
    return payload


def _empty_dashboard_payload(root: Path, *, loading: bool = True, message: str = "") -> Dict[str, Any]:
    return {
        "root_path": str(root),
        "generated_at_epoch": time.time(),
        "loading": bool(loading),
        "message": str(message or ""),
        "source_count": 0,
        "workspace_count": 0,
        "video_count": 0,
        "segment_count": 0,
        "segments_done": 0,
        "segments_rejected": 0,
        "segments_error": 0,
        "dataset_items": 0,
        "active_source_name": "",
        "active_workspace_name": "",
        "active_video_name": "",
        "active_stage": "",
        "active_segment_index": -1,
        "bench_counts": {key: 0 for key in BENCH_KEYS},
        "task_type_counts": {},
        "module_counts": {},
        "sources": [],
        "workspaces": [],
    }


def _collect_asset_paths(payload: Dict[str, Any]) -> List[str]:
    assets: set[str] = set()
    _collect_paths(payload, assets)
    return sorted(assets)


def _normalize_path_text(value: str) -> str:
    return str(value or "").replace("\\", "/").strip()


def _resolve_asset_path(root: Path, allowed_assets: set[str], raw_path: str) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None

    raw_norm = _normalize_path_text(raw)
    allowed_raw = {str(item or "").strip() for item in allowed_assets if str(item or "").strip()}
    allowed_norm = {_normalize_path_text(item) for item in allowed_raw}

    resolved_allowed: set[str] = set()
    candidate_paths: List[Path] = []

    for item in allowed_raw:
        item_path = Path(item)
        if item_path.is_absolute():
            candidate_paths.append(item_path)
        else:
            candidate_paths.append(root / item_path)

    raw_candidate = Path(raw)
    if raw_candidate.is_absolute():
        candidate_paths.append(raw_candidate)
    else:
        candidate_paths.append(root / raw_candidate)
        for source_dir in _discover_sources(root):
            candidate_paths.append(source_dir / raw_candidate)
            candidate_paths.append(source_dir.parent / raw_candidate)

    seen_candidates: set[str] = set()
    for candidate in candidate_paths:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        resolved_str = str(resolved)
        if resolved_str in seen_candidates:
            continue
        seen_candidates.add(resolved_str)
        if resolved.exists() and resolved.is_file():
            resolved_allowed.add(resolved_str)

    for resolved_str in sorted(resolved_allowed):
        resolved_norm = _normalize_path_text(resolved_str)
        if raw_norm == resolved_norm or raw_norm.endswith(resolved_norm) or resolved_norm.endswith(raw_norm):
            return Path(resolved_str)

    if raw_norm not in allowed_norm:
        return None

    return None


def _filter_existing_sample_images_inplace(root: Path, node: Any) -> None:
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if key in {"sample_images", "images"} and isinstance(value, list):
                filtered: List[str] = []
                for item in value:
                    raw = str(item or "").strip()
                    if not raw:
                        continue
                    if _resolve_asset_path(root, {raw}, raw) is not None:
                        filtered.append(raw)
                node[key] = filtered
            else:
                _filter_existing_sample_images_inplace(root, value)
        return
    if isinstance(node, list):
        for item in node:
            _filter_existing_sample_images_inplace(root, item)


def _build_dashboard_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Run Pipeline Dashboard</title>
  <style>
    :root { --bg:#f5f1e8; --panel:#fffdf8; --ink:#1e2630; --muted:#677382; --line:#dad1c4; --blue:#2d6dcf; --green:#2f8e59; --red:#c34f4f; --amber:#b88426; }
    * { box-sizing:border-box; } body { margin:0; font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif; color:var(--ink); background:linear-gradient(180deg,#f9f6ef,#efe7d9); }
    .layout { display:grid; grid-template-columns:320px 1fr; min-height:100vh; } .sidebar { position:sticky; top:0; height:100vh; overflow:auto; padding:18px; border-right:1px solid var(--line); background:rgba(255,253,248,.94); }
    .brand { font-size:24px; font-weight:800; } .hint,.muted,.path { color:var(--muted); font-size:12px; line-height:1.5; word-break:break-all; }
    .search { width:100%; padding:10px 12px; border-radius:12px; border:1px solid var(--line); margin:14px 0; }
    .list { display:grid; gap:10px; } .button, .seg-btn { width:100%; text-align:left; padding:12px; border-radius:14px; border:1px solid var(--line); background:#fff; cursor:pointer; }
    .button.active, .seg-btn.active { border-color:var(--blue); box-shadow:0 0 0 2px rgba(45,109,207,.15); background:#f3f8ff; }
    .button.running, .seg-btn.running { border-color:var(--green); box-shadow:0 0 0 2px rgba(47,142,89,.18); background:#eef9f1; }
    .content { padding:24px; overflow:auto; } .panel, .hero, .card { background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 12px 32px rgba(30,38,48,.06); }
    .hero, .panel { padding:16px; margin-bottom:14px; } .grid { display:grid; gap:12px; } .top-grid { grid-template-columns:repeat(6,minmax(0,1fr)); } .mid-grid { grid-template-columns:1.1fr .9fr; } .seg-grid { grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); } .shot-grid,.kframe-grid,.example-grid { display:grid; gap:10px; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); }
    .card { padding:12px; } .label { font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; } .value { margin-top:6px; font-size:22px; font-weight:800; }
    .chip { display:inline-flex; padding:3px 8px; border-radius:999px; margin:0 6px 6px 0; background:#f7f1e6; border:1px solid #e7d3ab; font-size:11px; }
    .state { display:inline-flex; padding:3px 8px; border-radius:999px; font-weight:700; font-size:10px; }
    .state.done { background:#ebf8ef; border:1px solid #bfe3c8; color:#1f6a3f; } .state.rejected { background:#fff7e5; border:1px solid #ecd193; color:#865a0d; } .state.error { background:#fff0f0; border:1px solid #efc0c0; color:#a33b3b; } .state.partial { background:#edf4ff; border:1px solid #c8daf7; color:#244d86; }
    .images { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; } .mini-images { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:6px; } .images img,.mini-images img { width:100%; aspect-ratio:16/9; object-fit:cover; border-radius:8px; background:#eceef2; border:1px solid var(--line); }
    .shot-card,.kframe-card,.example-card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; box-shadow:0 6px 16px rgba(30,38,48,.05); }
    .shot-card.pass { border:2px solid rgba(47,142,89,.55); } .shot-card.drop { border:2px solid rgba(195,79,79,.55); } .shot-card.unknown { border:2px solid rgba(184,132,38,.45); }
    .shot-header { display:flex; justify-content:space-between; gap:8px; align-items:flex-start; margin-bottom:6px; }
    .tool-list { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0; } .tool-pill { display:inline-flex; padding:2px 8px; border-radius:999px; background:#eef4ff; border:1px solid #cbd9f7; font-size:11px; }
    .section-title { font-size:16px; margin:0 0 8px; } .stage-copy { margin:4px 0 0; color:var(--muted); line-height:1.35; font-size:12px; }
    .kv { display:grid; gap:4px; font-size:11px; color:var(--muted); margin:6px 0; } .kv strong { color:var(--ink); font-size:11px; }
    .tight { margin-top:6px; }
    .trace { white-space:pre-wrap; word-break:break-word; background:#221e1b; color:#f5eadb; border-radius:14px; padding:14px; font-size:12px; max-height:220px; overflow:auto; }
    table { width:100%; border-collapse:collapse; } th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; font-size:13px; } th { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
    @media (max-width:1280px) { .top-grid { grid-template-columns:repeat(3,minmax(0,1fr)); } .mid-grid { grid-template-columns:1fr; } }
    @media (max-width:980px) { .layout { grid-template-columns:1fr; } .sidebar { position:static; height:auto; border-right:none; border-bottom:1px solid var(--line); } .shot-grid,.kframe-grid,.example-grid,.images,.mini-images { grid-template-columns:1fr; } }
    @media (max-width:720px) { .content { padding:12px; } .top-grid,.images,.mini-images { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="brand">Pipeline Dashboard</div>
      <div class="hint">Complete run_pipeline view across sources, videos, segments, stage status, errors, and T4 dataset output.</div>
      <div id="refresh-meta" class="hint">Auto refresh: every 5s</div>
      <input id="search" class="search" type="search" placeholder="Filter sources or videos">
      <div id="source-list" class="list"></div>
    </aside>
    <main class="content"><div id="app"></div></main>
  </div>
  <script>
    async function loadData(){ const res = await fetch('./pipeline_dashboard_data.json'); if(!res.ok) throw new Error('Failed to load dashboard data'); return res.json(); }
    function assetUrl(path){ return '/__asset__?path=' + encodeURIComponent(path || ''); }
    function esc(text){ return String(text ?? '').replace(/[&<>\"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[ch])); }
    function num(v){ return Number(v || 0).toLocaleString(); }
    function stateBadge(status){ return `<span class="state ${esc(status || 'partial')}">${esc(status || 'partial')}</span>`; }
    function metric(label, value){ return `<div class="card"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div></div>`; }
    function chips(map){ const entries = Object.entries(map || {}).filter(([,v]) => Number(v || 0) > 0); return entries.length ? entries.map(([k,v]) => `<span class="chip">${esc(k)}: ${num(v)}</span>`).join('') : '<div class="muted">None</div>'; }
    function images(rows, cls='images'){ return (rows || []).length ? `<div class="${cls}">${rows.map((path) => `<img src="${assetUrl(path)}" alt="${esc(path)}">`).join('')}</div>` : '<div class="muted">No images.</div>'; }
    function toolPills(rows){ return (rows || []).length ? `<div class="tool-list">${rows.map((item) => `<span class="tool-pill" title="${esc(item.reason || '')}">${esc(item.tool_name || '-')}</span>`).join('')}</div>` : '<div class="muted">No tools.</div>'; }
    function renderT1Shots(seg){ const rows = seg.t1?.shot_cards || []; return rows.length ? `<div class="shot-grid">${rows.map((shot) => `<div class="shot-card ${shot.kept ? 'pass' : (shot.decision === 'drop' ? 'drop' : 'unknown')}"><div class="shot-header"><div><strong>${esc(shot.shot_id || '-')}</strong><div class="muted">${esc(shot.shot_role || 'generic')}${shot.scene_type ? ` | ${esc(shot.scene_type)}` : ''}</div></div><div>${stateBadge(shot.kept ? 'done' : 'rejected')}</div></div><div class="kv"><div><strong>GateA:</strong> ${esc(shot.decision || '-')}</div><div><strong>Reason:</strong> ${esc(shot.decision_reason || '-')}</div></div>${images((shot.rep_frames || []).map((x) => x.img).filter(Boolean), 'mini-images')}</div>`).join('')}</div>` : '<div class="muted">No T1 shot cards.</div>'; }
    function renderT2Shots(seg){ const rows = seg.t2?.shot_cards || []; return rows.length ? `<div class="shot-grid">${rows.map((shot) => `<div class="shot-card"><div class="shot-header"><div><strong>${esc(shot.shot_id || '-')}</strong><div class="muted">u=${num(shot.usable_count)} | s=${num(shot.selected_count)}</div></div></div>${toolPills(shot.tool_actions || [])}<div class="kv"><div><strong>Plan:</strong> ${esc(shot.plan_reason || '-')}</div></div><div class="tight">${images(shot.usable_images || [], 'mini-images')}</div>${(shot.selected_images || []).length ? `<div class="tight"><div class="muted">Selected</div>${images(shot.selected_images || [], 'mini-images')}</div>` : ''}</div>`).join('')}</div>` : '<div class="muted">No T2 shot evidence.</div>'; }
    function renderT3Keyframes(seg){ const rows = seg.t3?.keyframe_cards || []; return rows.length ? `<div class="kframe-grid">${rows.map((item) => `<div class="kframe-card"><div><strong>K${num((item.index || 0) + 1)}</strong> <span class="muted">${esc(item.shot_id || '-')}${item.sec || item.sec === 0 ? ` | ${Number(item.sec).toFixed(2)}s` : ''}</span></div><div class="tight">${images([item.img], 'mini-images')}</div><div class="kv"><div><strong>Cap:</strong> ${esc(item.caption || '-')}</div><div><strong>Trans:</strong> ${esc(item.transition_to_next || '-')}</div></div></div>`).join('')}</div>` : '<div class="muted">No T3 keyframes.</div>'; }
    function renderDialogue(rows){ return (rows || []).length ? `<div class="kv">${rows.map((turn, idx) => `<div><strong>${esc(turn.role || (idx % 2 === 0 ? 'user' : 'assistant'))}:</strong> ${esc(turn.text || '-')}</div>`).join('')}</div>` : ''; }
    function renderT4Examples(seg){ const rows = seg.t4?.examples || []; return rows.length ? `<div class="example-grid">${rows.map((item) => `<div class="example-card"><div><strong>${esc(item.label || item.benchmark || '-')}</strong></div>${(item.images || []).length ? `<div class="tight">${images(item.images || [], 'mini-images')}</div>` : (item.image ? `<div class="tight">${images([item.image], 'mini-images')}</div>` : '')}${(item.dialogue || []).length ? `<div class="tight"><strong>Dialogue</strong>${renderDialogue(item.dialogue || [])}</div>` : ''}<div class="kv">${(item.fields || []).length ? item.fields.map((field) => `<div><strong>${esc(field.label || '-')}:</strong> ${esc(field.value || '-')}</div>`).join('') : '<div class="muted">No structured fields.</div>'}</div></div>`).join('')}</div>` : '<div class="muted">No T4 examples generated yet.</div>'; }
    function renderSourceList(data, state){ const q = String(state.search || '').trim().toLowerCase(); const root = document.getElementById('source-list'); root.innerHTML = (data.sources || []).map((source, sourceIdx) => ({source, sourceIdx})).filter(({source}) => { const names = [source.source_name, source.source_path].concat((source.videos || []).map((v) => v.video_name)).join(' ').toLowerCase(); return !q || names.includes(q); }).map(({source, sourceIdx}) => { const running = Boolean(source.has_active_run); return `<button class="button ${sourceIdx === state.sourceIndex ? 'active' : ''} ${running ? 'running' : ''}" data-source="${sourceIdx}"><div><strong>${esc(source.source_name || '-')}</strong></div><div class="muted">videos=${num(source.video_count)} | items=${num(source.dataset_items)}</div><div>${stateBadge(running ? 'partial' : (source.segments_error ? 'error' : (source.segments_rejected ? 'rejected' : 'done')))}</div>${running ? `<div class="muted">running ${esc(source.active_video_name || '-')} seg ${esc(source.active_segment_index ?? '-')} @ ${esc(source.active_stage || '-')}</div>` : ''}<div class="path">${esc(source.source_path || '-')}</div></button>`; }).join('') || '<div class="muted">No source matched.</div>'; document.querySelectorAll('[data-source]').forEach((btn) => btn.addEventListener('click', () => { state.sourceIndex = Number(btn.dataset.source || '0'); state.videoIndex = 0; state.segmentIndex = 0; render(data, state); })); }
    function renderSegment(seg, idx, active, running){ return `<button class="seg-btn ${idx === active ? 'active' : ''} ${running ? 'running' : ''}" data-seg="${idx}"><div><strong>Segment ${esc(seg.segment_index)}</strong></div><div class="muted">${esc(seg.final_stage || '-')}</div><div style="margin-top:6px;">${stateBadge(seg.status)}</div></button>`; }
    function keepSelection(data, state){ const sources = data.sources || []; if(!sources.length){ state.sourceIndex = 0; state.videoIndex = 0; state.segmentIndex = 0; return; } if(state.sourceIndex >= sources.length) state.sourceIndex = 0; const source = sources[state.sourceIndex] || null; const videos = source ? (source.videos || []) : []; if(state.videoIndex >= videos.length) state.videoIndex = 0; const video = videos[state.videoIndex] || null; const segments = video ? (video.segments || []) : []; if(state.segmentIndex >= segments.length) state.segmentIndex = 0; }
    function render(data, state){ keepSelection(data, state); renderSourceList(data, state); document.getElementById('search').value = state.search || ''; const source = (data.sources || [])[state.sourceIndex] || null; const videos = source ? (source.videos || []) : []; const video = videos[state.videoIndex] || null; const segments = video ? (video.segments || []) : []; const seg = segments[state.segmentIndex] || null; const stamp = data.generated_at_epoch ? new Date(Number(data.generated_at_epoch) * 1000) : null; const refreshMeta = document.getElementById('refresh-meta'); if(refreshMeta) refreshMeta.textContent = stamp ? `Auto refresh: every 5s | last sync ${stamp.toLocaleTimeString()}` : 'Auto refresh: every 5s'; const app = document.getElementById('app'); if(!source || !video){ app.innerHTML = '<div class="panel"><div class="muted">No source/video data available.</div></div>'; return; }
      const activeLine = data.active_source_name ? `Running now: ${data.active_source_name} / ${data.active_video_name || '-'} / segment ${data.active_segment_index ?? '-'} / ${data.active_stage || '-'}` : 'Running now: idle';
      app.innerHTML = `<section class="hero"><h1>${esc(data.root_path || '-')}</h1><div class="muted">Selected source: ${esc(source.source_name || '-')}</div><div class="hint" style="margin-top:8px;">${esc(activeLine)}</div><div class="grid top-grid" style="margin-top:16px;">${metric('Sources', num(data.source_count))}${metric('Videos', num(data.video_count))}${metric('Segments', num(data.segment_count))}${metric('Done', num(data.segments_done))}${metric('Rejected', num(data.segments_rejected))}${metric('Errors', num(data.segments_error))}</div></section>
      <section class="panel"><h2>Source Summary</h2><div class="path">${esc(source.source_path || '-')}</div><div style="margin-top:12px;"><strong>Benchmarks</strong><div style="margin-top:8px;">${chips(source.bench_counts)}</div></div><div style="margin-top:12px;"><strong>Modules</strong><div style="margin-top:8px;">${chips(source.module_counts)}</div></div><div class="grid top-grid" style="margin-top:14px;">${metric('Source Videos', num(source.video_count))}${metric('Source Segments', num(source.segment_count))}${metric('Dataset Items', num(source.dataset_items))}${metric('Done', num(source.segments_done))}${metric('Rejected', num(source.segments_rejected))}${metric('Errors', num(source.segments_error))}</div></section>
      <section class="panel"><h2>Videos</h2><div class="list">${videos.map((item, idx) => `<button class="button ${idx === state.videoIndex ? 'active' : ''} ${item.has_active_run ? 'running' : ''}" data-video="${idx}"><div><strong>${esc(item.video_name || '-')}</strong></div><div class="muted">segments=${num(item.segment_count)} | done=${num(item.segments_done)} | error=${num(item.segments_error)}</div>${item.has_active_run ? `<div class="muted">running segment ${esc(item.active_segment_index ?? '-')} @ ${esc(item.active_stage || '-')}</div>` : ''}<div><strong>Benchmarks</strong></div><div>${chips(item.bench_counts)}</div><div style="margin-top:8px;"><strong>Modules</strong></div><div>${chips(item.module_counts)}</div></button>`).join('')}</div></section>
      <section class="panel"><h2>Segments</h2><div class="grid seg-grid">${segments.map((item, idx) => renderSegment(item, idx, state.segmentIndex, !item.is_terminal)).join('')}</div></section>
      <section class="panel"><h2>Selected Segment</h2><div class="grid mid-grid"><div><table><tbody>
        <tr><th>Segment</th><td>${esc(seg.segment_index)}</td></tr>
        <tr><th>Path</th><td>${esc(seg.segment_path || '-')}</td></tr>
        <tr><th>Status</th><td>${stateBadge(seg.status)}</td></tr>
        <tr><th>Final Stage</th><td>${esc(seg.final_stage || '-')}</td></tr>
        <tr><th>T1</th><td>shots=${num(seg.t1?.shots)} | chapters=${num(seg.t1?.chapters)} | reps=${num(seg.t1?.candidate_frames)}</td></tr>
        <tr><th>T2</th><td>pass=${String(Boolean(seg.t2?.quality_passed))} | pool=${num(seg.t2?.candidate_pool)} | gateB=${num(seg.t2?.gateb_passed)}/${num((seg.t2?.gateb_passed||0)+(seg.t2?.gateb_rejected||0))} | final=${num(seg.t2?.final_selected)}</td></tr>
        <tr><th>T3</th><td>type=${esc(seg.t3?.dataset_type || 'none')} | frames=${num(seg.t3?.frames)}</td></tr>
        <tr><th>T4</th><td>items=${num(seg.t4?.total_items)} | benchmarks=${esc((seg.t4?.selected_benchmarks || []).join(', ') || '-')}</td></tr>
        <tr><th>T4 Pairs</th><td>raw=${num(seg.t4?.raw_pair_count)} | reviewed=${num(seg.t4?.reviewed_pair_count)} | candidate=${num(seg.t4?.candidate_pair_count)} | edit_pairs=${num(seg.t4?.edit_pair_count)}</td></tr>
        <tr><th>T4 Route</th><td>modules=${esc((seg.t4?.selected_modules || []).join(', ') || '-')} | reason=${esc(seg.t4?.module_route_reason || '-')} | unify=${num(seg.t4?.generated_unify)} | generation=${num(seg.t4?.generated_generation)}</td></tr>
      </tbody></table><div style="margin-top:14px;"><strong>Timeline</strong><div style="margin-top:8px;">${(seg.timeline || []).map((step) => `<span class="chip">${esc(step.stage)}</span>`).join('') || '<div class="muted">No timeline.</div>'}</div></div><div style="margin-top:14px;"><strong>T4 Benchmarks</strong><div style="margin-top:8px;">${chips(seg.t4?.instances || {})}</div></div><div style="margin-top:14px;"><strong>T4 Modules</strong><div style="margin-top:8px;">${chips(seg.t4?.modules || {})}</div></div>${seg.error && seg.error.message ? `<div style="margin-top:14px;"><strong>${esc(seg.error.stage || 'ERROR')}</strong><div class="muted">${esc(seg.error.type || '-')}</div><div style="margin-top:8px;">${esc(seg.error.message || '-')}</div><div class="trace" style="margin-top:10px;">${esc(seg.error.traceback || '-')}</div></div>` : ''}</div><div><strong>Stage Samples</strong><div style="margin-top:10px;">${images(seg.sample_images || [])}</div></div></div></section>
      <section class="panel"><h2 class="section-title">T1 Representative Frames + GateA</h2>${renderT1Shots(seg)}</section>
      <section class="panel"><h2 class="section-title">T2 Tools + Quality-Passed Frames</h2>${renderT2Shots(seg)}</section>
      <section class="panel"><h2 class="section-title">T3 Final Keyframes + Caption / Transition</h2><div class="stage-copy">${esc(seg.t3?.segment_summary || '-')}</div><div style="margin-top:14px;">${renderT3Keyframes(seg)}</div></section>
      <section class="panel"><h2 class="section-title">T4 Example Outputs</h2>${renderT4Examples(seg)}</section>`;
      document.querySelectorAll('[data-video]').forEach((btn) => btn.addEventListener('click', () => { state.videoIndex = Number(btn.dataset.video || '0'); state.segmentIndex = 0; render(data, state); }));
      document.querySelectorAll('[data-seg]').forEach((btn) => btn.addEventListener('click', () => { state.segmentIndex = Number(btn.dataset.seg || '0'); render(data, state); }));
      document.getElementById('search').oninput = (ev) => { state.search = ev.target.value || ''; render(data, state); };
    }
    loadData().then((data) => { const state = { sourceIndex:0, videoIndex:0, segmentIndex:0, search:'' }; render(data, state); window.setInterval(async () => { try { const next = await loadData(); render(next, state); } catch (err) { const refreshMeta = document.getElementById('refresh-meta'); if(refreshMeta) refreshMeta.textContent = `Auto refresh: retry failed (${String(err.message || err)})`; } }, 5000); }).catch((err) => { document.getElementById('app').innerHTML = `<div class="panel"><div class="muted">${esc(err.message || String(err))}</div></div>`; });
  </script>
</body>
</html>
"""


def _write_dashboard_artifacts(root: Path, payload: Dict[str, Any]) -> None:
    (root / "pipeline_dashboard_data.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "pipeline_dashboard.html").write_text(_build_dashboard_html(), encoding="utf-8")


def _new_dashboard_state(root: Path, initial_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = initial_payload if isinstance(initial_payload, dict) else _empty_dashboard_payload(root)
    return {
        "lock": threading.RLock(),
        "payload": payload,
        "last_refresh_ts": 0.0 if payload.get("loading") else time.time(),
        "refreshing": False,
        "last_error": "",
        "allowed_assets": set(_collect_asset_paths(payload)),
    }


def _refresh_dashboard_payload(root: Path, state: Dict[str, Any]) -> None:
    try:
        payload = _build_dashboard_payload(root)
        allowed_assets = set(_collect_asset_paths(payload))
        _write_dashboard_artifacts(root, payload)
        with state["lock"]:
            state["payload"] = payload
            state["allowed_assets"] = allowed_assets
            state["last_refresh_ts"] = time.time()
            state["last_error"] = ""
            state["refreshing"] = False
    except Exception as exc:
        with state["lock"]:
            payload = dict(state.get("payload") or _empty_dashboard_payload(root))
            payload["loading"] = False
            payload["message"] = f"dashboard refresh failed: {exc}"
            state["payload"] = payload
            state["last_error"] = repr(exc)
            state["last_refresh_ts"] = time.time()
            state["refreshing"] = False
        print(f"[{_ts()}] [PipelineDashboard] refresh failed: {exc!r}")


def _start_dashboard_refresh(root: Path, state: Dict[str, Any]) -> bool:
    with state["lock"]:
        if state.get("refreshing"):
            return False
        state["refreshing"] = True
    thread = threading.Thread(
        target=_refresh_dashboard_payload,
        args=(root, state),
        name="run-pipeline-dashboard-refresh",
        daemon=True,
    )
    thread.start()
    return True


def _dashboard_payload_for_request(root: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    with state["lock"]:
        payload = dict(state.get("payload") or _empty_dashboard_payload(root))
        last_refresh_ts = float(state.get("last_refresh_ts", 0.0) or 0.0)
        refreshing = bool(state.get("refreshing"))
    stale = (now - last_refresh_ts) >= DASHBOARD_DATA_TTL_S
    if stale and not refreshing:
        _start_dashboard_refresh(root, state)
        with state["lock"]:
            payload = dict(state.get("payload") or payload)
            refreshing = bool(state.get("refreshing"))
    payload["refreshing"] = refreshing
    return payload


def _make_handler(root: Path, state: Dict[str, Any]):
    class DashboardHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self.path = "/pipeline_dashboard.html"
                return super().do_GET()
            if parsed.path == "/pipeline_dashboard_data.json":
                payload = _dashboard_payload_for_request(root, state)
                data = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if parsed.path == "/__asset__":
                raw_path = str((urllib.parse.parse_qs(parsed.query).get("path") or [""])[0] or "")
                if not raw_path:
                    self.send_error(404, "asset not found")
                    return
                with state["lock"]:
                    allowed_assets = set(state.get("allowed_assets", set()) or set())
                asset_path = _resolve_asset_path(root, allowed_assets, raw_path)
                if asset_path is None:
                    self.send_error(404, "asset missing")
                    return
                mime_type = mimetypes.guess_type(str(asset_path))[0] or "application/octet-stream"
                data = asset_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            return super().do_GET()

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[{_ts()}] [PipelineDashboard] {format % args}")

    return DashboardHandler


def _log_dashboard_endpoints(host: str, port: int, root: Path) -> None:
    bind_host = host or DEFAULT_HOST
    display_host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
    print(f"[{_ts()}] [PipelineDashboard] listening on http://{bind_host}:{port}")
    print(f"[{_ts()}] [PipelineDashboard] open via http://{display_host}:{port}/pipeline_dashboard.html")
    print(f"[{_ts()}] [PipelineDashboard] root={root}")


def _serve(root: Path, host: str, port: int, state: Dict[str, Any]) -> None:
    server = ThreadingHTTPServer((host, port), _make_handler(root, state))
    server.daemon_threads = True
    _log_dashboard_endpoints(host, port, root)
    _start_dashboard_refresh(root, state)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"[{_ts()}] [PipelineDashboard] stopped")
    finally:
        server.server_close()


def serve_dashboard_forever(
    run_root: str | Path,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
) -> None:
    root = _resolve_dashboard_root(str(run_root))
    payload = _empty_dashboard_payload(root)
    _write_dashboard_artifacts(root, payload)
    _serve(root, host, int(port), _new_dashboard_state(root, payload))


def start_dashboard_server(
    run_root: str | Path,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
) -> ThreadingHTTPServer:
    root = _resolve_dashboard_root(str(run_root))
    payload = _empty_dashboard_payload(root)
    _write_dashboard_artifacts(root, payload)
    state = _new_dashboard_state(root, payload)
    server = ThreadingHTTPServer((host, int(port)), _make_handler(root, state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="run-pipeline-dashboard", daemon=True)
    thread.start()
    _log_dashboard_endpoints(host, int(port), root)
    _start_dashboard_refresh(root, state)
    return server


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a run_pipeline workspace dashboard over HTTP.")
    parser.add_argument("--run-root", default=_default_pipeline_root(), help="Workspace root, source root, or output root.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    serve_dashboard_forever(args.run_root, int(args.port), str(args.host))


if __name__ == "__main__":
    main()
