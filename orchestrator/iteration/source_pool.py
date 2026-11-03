"""Admit bounded, explicitly declared historical training sources for smoke."""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .bagel_data import validated_turns
from .jobs import file_fingerprint
from .paper_rounds import validate_paper_rounds
from .state import read_jsonl, write_json, write_jsonl
from .taxonomy import TASK_CAPABILITY, TASK_TYPES, canonical_task


def row_sha256(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def restore_unified(row: dict[str, Any], task: str) -> tuple[dict[str, Any], str]:
    """Recover only preserved legacy fields; never infer a missing target."""
    candidate = copy.deepcopy(row)
    turns = candidate.get("turns")
    if isinstance(turns, list):
        try:
            validate_paper_rounds(turns, task)
            return candidate, "already_complete"
        except ValueError:
            legacy = (candidate.get("meta") or {}).get("legacy_unify")
            if not isinstance(legacy, dict) or not isinstance(legacy.get("turns"), dict):
                raise ValueError("incomplete_rounds_without_preserved_legacy")
            for key in ("turns", "answer", "render", "evidence_refs", "unify_id"):
                if key in legacy:
                    candidate[key] = copy.deepcopy(legacy[key])
    elif not isinstance(turns, dict):
        raise ValueError("missing_original_unified_turns")
    answer = candidate.get("answer") or {}
    positions = (candidate.get("meta") or {}).get("frame_positions") or []
    if task in {"grounded_edit_to_select", "understand_recall_generate_select", "middle_step_gap"}:
        if len(positions) < (3 if task == "middle_step_gap" else 2):
            raise ValueError("missing_original_frame_positions")
        if answer.get("target_frame_pos") != positions[1]:
            raise ValueError("missing_or_conflicting_original_target")
    else:
        key = "target_frame_positions" if task == "stepwise_sequence" else "future_frame_positions"
        targets = answer.get(key)
        if not isinstance(targets, list) or not targets or len(targets) != len(set(targets)):
            raise ValueError("missing_original_target_sequence")
        if targets != positions[1 if task == "stepwise_sequence" else 2:]:
            raise ValueError("conflicting_original_target_sequence")
    from orchestrator.paper_render import render_paper_unified

    rendered = render_paper_unified(candidate)
    validate_paper_rounds(rendered["turns"], task)
    return rendered, "rerendered_preserved_legacy"


def prepare_source_pool(sources: list[dict[str, Any]], output_dir: Path, *,
                        max_per_task: int = 16,
                        forbidden_media: list[Path] | None = None) -> dict[str, Any]:
    """Publish candidates and provenance after decoding every selected image.

    A source must explicitly be a training split. Missing task coverage is a
    reportable shortage, not permission to relabel, repeat, or invent samples.
    """
    if isinstance(max_per_task, bool) or not isinstance(max_per_task, int) or max_per_task < 1:
        raise ValueError("max_per_task_must_be_positive_integer")
    if not sources:
        raise ValueError("training_sources_required")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("use_new_source_pool_output_directory")
    checked_sources = []
    for spec in sources:
        if spec.get("split") != "train":
            raise ValueError("only_explicit_training_sources_allowed")
        root = Path(spec["root"]).resolve()
        media_root = Path(spec["media_root"]).resolve()
        if not root.is_dir() or not media_root.is_dir():
            raise FileNotFoundError("source_or_media_root_missing")
        if root == output_dir or root in output_dir.parents or media_root == output_dir or media_root in output_dir.parents:
            raise ValueError("output_must_be_outside_read_only_source")
        if "videodataset_eval" in str(root).lower():
            raise ValueError("evaluation_sources_forbidden")
        files = []
        for name in spec.get("files", []):
            path = (root / name).resolve()
            if root not in path.parents or path.suffix != ".jsonl" or not path.is_file():
                raise ValueError("source_file_must_be_explicit_jsonl_inside_root")
            files.append(path)
        if not files:
            raise ValueError("explicit_source_files_required")
        checked_sources.append((spec, root, media_root, files))

    forbidden_hashes = {file_fingerprint(Path(p))["sha256"] for p in forbidden_media or []}
    media: dict[str, dict[str, Any]] = {}
    candidates, manifests, file_reports, rejection_examples = [], [], [], []
    counts, transforms, seen = Counter(), Counter(), set()
    for spec, root, media_root, files in checked_sources:
        for path in files:
            before = file_fingerprint(path)
            stats, reasons = Counter(), Counter()
            with path.open(encoding="utf-8") as stream:
                for line_no, text in enumerate(stream, 1):
                    if not text.strip():
                        continue
                    stats["rows_scanned"] += 1
                    try:
                        row = json.loads(text)
                        if not isinstance(row, dict):
                            raise ValueError("source_row_must_be_object")
                        signature = row_sha256(row)
                        if signature in seen:
                            stats["duplicate_content_rows"] += 1
                            continue
                        seen.add(signature)
                        task = (canonical_task(row.get("task_type") or row.get("benchmark"))
                                or ("localized_editing" if row.get("benchmark") == "magicbrush" else None))
                        if not task:
                            raise ValueError("unsupported_source_task")
                        if counts[task] >= max_per_task:
                            stats["above_task_cap"] += 1
                            continue
                        candidate, transform = copy.deepcopy(row), "unchanged"
                        if TASK_CAPABILITY[task] == "unified":
                            candidate, transform = restore_unified(row, task)
                        if task == "localized_editing":
                            review = row.get("review", (row.get("meta") or {}).get("review"))
                            if not isinstance(review, dict) or review.get("ok") is not True:
                                raise ValueError("missing_or_rejected_edit_review")
                        turns = validated_turns(candidate, task, media_root, media, require_paper_rounds=True)
                        used = {p for turn in turns for side in ("query", "answer")
                                for p in turn["render"][side]["images"]}
                        if any(media[p]["sha256"] in forbidden_hashes for p in used):
                            raise ValueError("evaluation_media_overlap")
                    except (ValueError, KeyError, TypeError, IndexError, OSError) as exc:
                        reasons[f"{type(exc).__name__}:{exc}"] += 1
                        if len(rejection_examples) < 40:
                            rejection_examples.append({"source_file": str(path), "source_line": line_no,
                                                       "error": f"{type(exc).__name__}:{exc}"})
                        continue
                    provenance = {
                        "record_key": f"historical-{signature}", "task_type": task,
                        "capability": TASK_CAPABILITY[task], "source_file": str(path), "source_line": line_no,
                        "source_row_sha256": signature, "source_sha256": signature,
                        "source_file_sha256": before["sha256"],
                        "source_pointer": f"{path}#L{line_no}",
                        "asset_root": str(media_root), "media_root": str(media_root),
                        "source_origin": "historical_training_source", "source_split": "train",
                        "source_audit": {"source_name": spec.get("name"), "transform": transform,
                                         "format_media_paper_rounds": "passed",
                                         "current_quality_review": "not_run",
                                         "media_sha256": {p: media[p]["sha256"] for p in sorted(used)}},
                    }
                    # Materialize complete turns with resolved paths; preserve
                    # all original fields and their fingerprint separately.
                    candidate.update(provenance)
                    candidate["turns"] = turns
                    candidate.setdefault("input_row", line_no)
                    candidates.append(candidate)
                    manifests.append({**provenance, "candidate": candidate})
                    counts[task] += 1
                    transforms[transform] += 1
                    stats["selected"] += 1
            if file_fingerprint(path) != before:
                raise RuntimeError(f"source_changed_during_selection:{path}")
            file_reports.append({**before, **stats, "rejections": dict(reasons)})
    missing = [task for task in TASK_TYPES if not counts[task]]
    report = {"schema_version": 1, "status": "shortage" if missing else "ready",
              "coverage_scope": "historical_source_format_media_rounds_not_quality_or_optimizer",
              "max_per_task": max_per_task, "candidate_count": len(candidates),
              "task_counts": {task: counts[task] for task in TASK_TYPES},
              "missing_task_types": missing, "transforms": dict(transforms),
              "source_files": file_reports, "rejection_examples": rejection_examples,
              "forbidden_media_sha256": sorted(forbidden_hashes)}
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidate_pool.jsonl", candidates)
    write_jsonl(output_dir / "train_manifest.jsonl", manifests)
    write_json(output_dir / "source_pool_report.json", report)
    write_json(output_dir / "shortage_report.json", {"target_per_task": 1, "missing_task_types": missing})
    return report


def load_audited_pool(path: Path) -> list[dict[str, Any]]:
    """Recheck the declared source and every media hash at loop admission."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("audited_source_pool_empty")
    files, media, seen = {}, {}, set()
    for row in rows:
        audit = row.get("source_audit") or {}
        if (row.get("source_origin") != "historical_training_source" or row.get("source_split") != "train"
                or audit.get("format_media_paper_rounds") != "passed"):
            raise ValueError("audited_source_pool_requires_training_provenance")
        key = row.get("record_key")
        if not key or key in seen:
            raise ValueError("audited_source_pool_duplicate_record_key")
        seen.add(key)
        source = Path(row["source_file"])
        if str(source) not in files:
            files[str(source)] = file_fingerprint(source)
        if files[str(source)]["sha256"] != row.get("source_file_sha256"):
            raise ValueError("audited_source_file_changed")
        task = canonical_task(row.get("task_type"))
        if task is None or row.get("capability") != TASK_CAPABILITY[task]:
            raise ValueError("audited_source_task_capability_mismatch")
        turns = validated_turns(row, task, Path(row["asset_root"]), media, require_paper_rounds=True)
        used = {p for turn in turns for side in ("query", "answer") for p in turn["render"][side]["images"]}
        if {p: media[p]["sha256"] for p in used} != audit.get("media_sha256"):
            raise ValueError("audited_source_media_changed")
    return rows
