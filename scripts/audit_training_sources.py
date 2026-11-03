#!/usr/bin/env python3
"""Read-only, deduplicated audit of training records and their real media.

Default inputs are the canonical benchmark files immediately under source-root;
merged files, shards, maps and sampled copies are excluded unless --include is
explicit. This checks format/media, never semantic quality or optimizer coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.iteration.bagel_data import _candidate_turns
from orchestrator.iteration.paper_rounds import validate_paper_rounds
from orchestrator.iteration.taxonomy import TASK_TYPES, TASK_CAPABILITY, canonical_task

CANONICAL_FILES = (
    "mmt_bench.jsonl", "llava_bench.jsonl", "mme.jsonl",
    "dialog_mmdu_like.jsonl", "dialog_visdial_like.jsonl",
    "magicbrush.jsonl", "edit.jsonl", "unify.jsonl",
)
ARTIFACTS = ("source_inventory.json", "task_source_matrix.json",
             "dataset_provenance.json", "shortage_report.json")


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def row_digest(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def review_ok(row: dict) -> bool:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    review = row.get("review") if "review" in row else meta.get("review")
    return isinstance(review, dict) and review.get("ok") is True


def source_task(row: dict) -> str | None:
    benchmark = str(row.get("benchmark") or "")
    # MME/VisDial are source material, not the paper's MCQ/reasoning templates.
    return (canonical_task(row.get("task_type")) or canonical_task(benchmark)
            or ("localized_editing" if benchmark == "magicbrush" else None))


def source_features(row: dict) -> dict[str, int]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    pack = row.get("evidencepack") if isinstance(row.get("evidencepack"), dict) else {}
    return {
        "reviewed_edit_pair_rows": int(review_ok(row) and bool(row.get("source_img")) and bool(row.get("target_img"))),
        "rows_with_three_keyframes": int(len(meta.get("segment_keyframes") or []) >= 3),
        "rows_with_explicit_transitions": int(any(k.endswith("_TRANS") for k in pack)),
        "rows_with_legacy_change_evidence": int(any(k.startswith("C") and k[1:].isdigit() for k in pack)),
    }


def task_turns(row: dict, task: str) -> list[dict]:
    if task == "localized_editing" and not review_ok(row):
        raise ValueError("missing_or_rejected_edit_review")
    turns = _candidate_turns(row, task)
    if not turns:
        raise ValueError("empty_episode")
    validate_paper_rounds(turns, task)
    return turns


def required_images(turns: list[dict]) -> list[str]:
    paths = []
    for turn in turns:
        for side_name, role in (("query", "user"), ("answer", "assistant")):
            side = turn["render"][side_name]
            images, messages = side["images"], side["messages"]
            if not isinstance(images, list) or any(not isinstance(p, str) or not p for p in images):
                raise ValueError("invalid_media_paths")
            if not isinstance(messages, list) or not messages:
                raise ValueError("missing_render_messages")
            refs = [m.get("content") for m in messages if m.get("type") == "image"]
            if refs != [f"Image #{i+1}" for i in range(len(images))]:
                raise ValueError("invalid_ordered_image_references")
            if any(m.get("role") != role or not m.get("content") for m in messages):
                raise ValueError("invalid_render_message")
            if side_name == "answer" and not (images or any(m.get("type") == "text" for m in messages)):
                raise ValueError("missing_training_target")
            paths.extend(images)
        if set(turn["render"]["query"]["images"]) & set(turn["render"]["answer"]["images"]):
            raise ValueError("current_target_is_also_current_input")
    return list(dict.fromkeys(paths))


def audit_sources(source_root: Path, output_dir: Path, *, max_examples: int = 3,
                  include: list[str] | None = None, media_roots: list[Path] | None = None) -> dict[str, Any]:
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError("output_must_be_outside_read_only_source")
    if any((output_dir / name).exists() for name in ARTIFACTS):
        raise FileExistsError("use_a_new_audit_output_directory")
    files = sorted({p.resolve() for pattern in include or CANONICAL_FILES
                    for p in source_root.glob(pattern) if p.is_file() and p.suffix == ".jsonl"})
    if not files:
        raise ValueError("no_source_jsonl_files_selected")
    roots = [p.resolve() for p in (media_roots or [source_root])]
    # Memoize path lookup, but do not guess a basename or rename an asset.
    media_cache: dict[str, dict] = {}
    def inspect_media(name: str) -> dict:
        if name not in media_cache:
            p = Path(name)
            matches = {candidate.resolve() for candidate in ([p] if p.is_absolute() else [root / p for root in roots])
                       if candidate.is_file()}
            media_cache[name] = {"source_path": name, "status": "found" if len(matches) == 1 else (
                "ambiguous" if matches else "missing"), "resolved_paths": sorted(str(x) for x in matches)}
        return media_cache[name]

    counts = {task: Counter() for task in TASK_TYPES}
    examples = {task: [] for task in TASK_TYPES}
    reasons = {task: Counter() for task in TASK_TYPES}
    features, file_reports, seen = Counter(), [], set()
    for path in files:
        before = path.stat()
        fingerprint = {"path": str(path), "sha256": sha256(path), "bytes": before.st_size}
        stats, benchmarks, errors = Counter(), Counter(), []
        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("row_is_not_object")
                except (ValueError, TypeError) as exc:
                    stats["parse_errors"] += 1
                    if len(errors) < max_examples:
                        errors.append({"source_line": line_no, "error": str(exc)})
                    continue
                stats["rows"] += 1
                benchmarks[str(row.get("benchmark") or "unknown")] += 1
                signature = row_digest(row)
                if signature in seen:
                    stats["duplicate_content_rows"] += 1
                    continue
                seen.add(signature)
                stats["unique_content_rows"] += 1
                features.update(source_features(row))
                task = source_task(row)
                if not task:
                    stats["unmapped_source_rows"] += 1
                    continue
                counts[task]["declared_unique_rows"] += 1
                pointer = {"source_file": str(path), "source_line": line_no,
                           "source_key": str(row.get("record_key") or row.get("id") or row.get("unify_id")
                                             or row.get("dialog_id") or row.get("img_id") or f"line_{line_no}"),
                           "source_row_sha256": signature, "source_file_sha256": fingerprint["sha256"]}
                try:
                    media = [inspect_media(p) for p in required_images(task_turns(row, task))]
                    if not media:
                        raise ValueError("missing_required_media")
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    reasons[task][f"format:{exc}"] += 1
                    continue
                counts[task]["format_rows"] += 1
                if any(m["status"] != "found" for m in media):
                    counts[task]["media_unresolved_rows"] += 1
                    reasons[task]["missing_or_ambiguous_media"] += 1
                    if len(examples[task]) < max_examples:
                        examples[task].append({**pointer, "status": "media_unresolved", "media": media})
                    continue
                counts[task]["format_and_media_rows"] += 1
                # Decode/hash bounded representative examples. Selected training
                # episodes still go through convert_manifest's full media check.
                if sum(e.get("status") == "example_decoded" for e in examples[task]) < max_examples:
                    try:
                        from PIL import Image
                        decoded = []
                        for item in media:
                            image_path = Path(item["resolved_paths"][0])
                            with Image.open(image_path) as image:
                                image.verify()
                            decoded.append({**item, "sha256": sha256(image_path)})
                    except (OSError, ValueError) as exc:
                        counts[task]["decode_failed_examples"] += 1
                        reasons[task][f"decode:{type(exc).__name__}"] += 1
                    else:
                        counts[task]["decoded_examples"] += 1
                        examples[task].append({**pointer, "status": "example_decoded", "media": decoded})
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise RuntimeError(f"source_changed_during_audit:{path}")
        file_reports.append({**fingerprint, **stats, "benchmarks": dict(benchmarks), "parse_error_examples": errors})
        print(f"audited {path.name}: {stats['rows']} rows, {stats['duplicate_content_rows']} duplicates", file=sys.stderr, flush=True)

    matrix = {"schema_version": 2, "coverage_scope": "format_and_media_only_not_quality_or_optimizer", "tasks": {}}
    shortage = {"schema_version": 2, "target_per_task": 1, "tasks": {}}
    for task in TASK_TYPES:
        c = counts[task]
        ready = c["decoded_examples"] > 0
        status = "supported" if ready else ("incomplete" if c["declared_unique_rows"] else "unsupported")
        matrix["tasks"][task] = {"capability": TASK_CAPABILITY[task], "status": status,
                                  **c, "reasons": dict(reasons[task]), "examples": examples[task]}
        shortage["tasks"][task] = {"status": status, "target": 1, "decoded_examples": c["decoded_examples"],
                                    "deficit": int(not ready), "quality_review": "not_run"}
    inventory = {"schema_version": 2, "source_root": str(source_root), "selected_patterns": list(include or CANONICAL_FILES),
                 "file_count": len(files), "files": file_reports, "unique_content_rows": len(seen),
                 "source_material_features": dict(features), "media_lookup": dict(Counter(m["status"] for m in media_cache.values()))}
    provenance = {"schema_version": 2, "source_root": str(source_root), "media_roots": [str(p) for p in roots],
                  "deduplication": "exact_canonical_json_content_not_semantic_duplicates",
                  "files": [{k: row[k] for k in ("path", "sha256", "bytes")} for row in file_reports],
                  "quality_review": "not_run", "examples": {task: examples[task] for task in TASK_TYPES}}
    result = dict(zip(("source_inventory", "task_source_matrix", "dataset_provenance", "shortage_report"),
                      (inventory, matrix, provenance, shortage)))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in result.items():
        (output_dir / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include", action="append", help="Explicit glob relative to source-root; repeatable")
    parser.add_argument("--media-root", type=Path, action="append", help="Exact media root; no basename guessing")
    parser.add_argument("--max-examples", type=int, default=3)
    args = parser.parse_args()
    if args.max_examples < 1:
        parser.error("--max-examples must be positive")
    report = audit_sources(args.source_root, args.output_dir, max_examples=args.max_examples,
                           include=args.include, media_roots=args.media_root)
    print(json.dumps({k: {key: v for key, v in row.items() if key not in {"examples", "reasons"}}
                      for k, row in report["task_source_matrix"]["tasks"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
