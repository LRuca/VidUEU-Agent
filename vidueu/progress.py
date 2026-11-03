"""Read-only progress summaries; never infer a hang from GPU utilization."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def source_directories(root):
    root = Path(root).resolve()
    if (root / "output" / "data.jsonl").is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                  and (p / "output" / "data.jsonl").is_file()) if root.is_dir() else []


def capability(row):
    module = str(row.get("module") or "")
    if module in {"edit", "unify", "understanding", "generation"}:
        return module
    value = str(row.get("task_type") or row.get("benchmark") or row.get("module") or "")
    if value == "edit" or value == "magicbrush" or value.startswith("edit."):
        return "edit"
    if value == "unify" or value.startswith("unify."):
        return "unify"
    if value in {"generation", "geneval", "wise"} or value.startswith("generation."):
        return "generation"
    if value in {"llava_bench", "mmt_bench", "mme", "dialog_visdial_like", "dialog_mmdu_like"} or value.startswith("understanding."):
        return "understanding"
    raise ValueError("record_task_unknown_or_missing")


def inspect_progress(root):
    sources = []
    for source in source_directories(root):
        data = source / "output" / "data.jsonl"
        counts = Counter()
        invalid = 0
        with data.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    counts[capability(json.loads(line))] += 1
                except (ValueError, AttributeError):
                    invalid += 1
        states = Counter()
        state_dir = source / "output" / "_video_state"
        for path in state_dir.glob("*.json"):
            try:
                states[str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))] += 1
            except (OSError, ValueError):
                states["unreadable"] += 1
        latest = None
        for log in (source / "logs").glob("*_segments.jsonl"):
            modified = log.stat().st_mtime
            if latest is None or modified > latest["modified_at"]:
                latest = {"file": str(log), "modified_at": modified}
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 16384))
                    tail = stream.read().splitlines()
                if tail:
                    try:
                        row = json.loads(tail[-1])
                        latest["stage"] = row.get("stage")
                        latest["segment_index"] = row.get("segment_index")
                    except ValueError:
                        latest["partial_line"] = True
        sources.append({"source": source.name, "counts": dict(counts), "invalid_lines": invalid,
                        "bytes": data.stat().st_size, "modified_at": data.stat().st_mtime,
                        "video_states": dict(states), "last_segment_event": latest})
    return {"sources": sources, "total_items": sum(sum(s["counts"].values()) for s in sources),
            "note": "Counts indicate writes, not quality approval. Compare timestamps over time; no automatic restart."}
