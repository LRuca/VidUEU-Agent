"""Deterministic source-level scheduling for iteration selection."""
from __future__ import annotations

from collections import defaultdict
import posixpath
import random
import re
from typing import Any

from .taxonomy import labels_for_record


def _provenance(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("review_provenance")
    return value if isinstance(value, dict) else {}


def _mapping(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    return value if isinstance(value, dict) else {}


def _canonical_path(value: Any) -> str:
    """Normalize separators without resolving machine-specific absolute paths."""
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return posixpath.normpath(text) if text else ""


def _segment_path(record: dict[str, Any]) -> str:
    meta = _mapping(record, "meta")
    provenance = _provenance(record)
    return str(
        record.get("segment_path")
        or meta.get("segment_path")
        or provenance.get("segment_path")
        or ""
    )


def _video_path(record: dict[str, Any]) -> str:
    meta = _mapping(record, "meta")
    provenance = _provenance(record)
    return str(
        record.get("video_path")
        or meta.get("video_path")
        or provenance.get("video_path")
        or ""
    )


def source_key(record: dict[str, Any]) -> str:
    """Return the single source identity used by schedule and selection audit.

    A segment path is preferred because all records from one fixed 30-second
    segment must share a source bucket.  The fallback order is deliberately
    shared by every caller; content hashes remain separate provenance fields
    and are not silently substituted for a source identity.
    """
    meta = _mapping(record, "meta")
    provenance = _provenance(record)
    segment = _segment_path(record)
    if segment:
        return f"segment:{_canonical_path(segment)}"
    video = _video_path(record)
    if video:
        return f"video:{_canonical_path(video)}"
    source_sha = record.get("source_sha256") or provenance.get("source_sha256")
    if source_sha:
        return f"sha256:{str(source_sha).strip()}"
    pointer = record.get("source_pointer") or meta.get("source_pointer") or provenance.get("source_pointer")
    if pointer:
        return f"pointer:{_canonical_path(pointer)}"
    return f"record:{str(record.get('record_key') or '').strip()}"


def _quality(record: dict[str, Any]) -> float:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    try:
        return max(0.0, float(record.get("quality_score", meta.get("quality_score", 1.0))))
    except (TypeError, ValueError):
        return 1.0


def build_source_schedule(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    selected: list[dict[str, Any]] | None = None,
    seed: int,
) -> list[dict[str, Any]]:
    """Build an auditable source schedule with stable source-level weights."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[source_key(record)].append(record)
    selected_keys = {source_key(row) for row in (selected or [])}
    selected_counts: dict[str, int] = defaultdict(int)
    for row in selected or []:
        selected_counts[source_key(row)] += 1
    total = max(1, len(records))
    available_domain_counts: dict[str, int] = defaultdict(int)
    for row in records:
        domain = labels_for_record(row).get("domain")
        if domain:
            available_domain_counts[str(domain)] += 1
    rows: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        labels = [labels_for_record(row) for row in group]
        # The source is a single video/segment in normal operation.  For
        # legacy pools with mixed rows, use the most common canonical label.
        domain_counts: dict[str, int] = defaultdict(int)
        type_counts: dict[str, int] = defaultdict(int)
        for item in labels:
            if item.get("domain"):
                domain_counts[str(item["domain"])] += 1
            if item.get("video_type"):
                type_counts[str(item["video_type"])] += 1
        # labels_for_record intentionally does not expose video_type; read it
        # only for scheduling weight and never use it to infer a domain.
        for row in group:
            value = str(row.get("video_type") or (row.get("meta") or {}).get("video_type") or "unknown")
            type_counts[value] += 1
        domain = max(domain_counts, key=lambda value: (domain_counts[value], value)) if domain_counts else None
        video_type = max(type_counts, key=lambda value: (type_counts[value], value)) if type_counts else "unknown"
        target_ratio = float((policy.get("domain") or {}).get(domain, 0.0) or 0.0) if domain else 0.0
        available_ratio = available_domain_counts.get(domain, 0) / total if domain else 0.0
        epsilon = 1.0 / total
        base = float((policy.get("video_type_weights") or {}).get(video_type, 1.0) or 1.0)
        quality = sum(_quality(row) for row in group) / max(1, len(group))
        shortage = target_ratio / max(available_ratio, epsilon) if domain else 0.0
        weight = base * max(shortage, 1.0) * max(quality, 0.0)
        rows.append({
            "source_key": key,
            "source_pointer": str(group[0].get("source_pointer") or _video_path(group[0]) or ""),
            "video_path": _video_path(group[0]),
            "segment_path": _segment_path(group[0]),
            "source_sha256": str(group[0].get("source_sha256") or _provenance(group[0]).get("source_sha256") or ""),
            "video_type": video_type if video_type != "unknown" else None,
            "policy_domain": domain,
            "target_ratio": target_ratio,
            "available_count": len(group),
            "available_domain_ratio": available_ratio,
            "base_video_type_weight": base,
            "quality_weight": quality,
            "sampling_weight": weight,
            "selected_count": selected_counts.get(key, 0),
            "selected": key in selected_keys,
            "selection_reason": "domain_shortage" if domain and target_ratio > available_ratio else "largest_deficit",
            "iteration": int(policy.get("iteration", 1)),
            "random_seed": int(seed),
        })
    return rows


def select_source_records(
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    source_cap: int,
    seed: int,
) -> dict[str, Any]:
    """Select source videos for a future generation round.

    A cap of zero means all known sources are retained.  When a cap is set,
    selection is deterministic and gives the largest score to underrepresented
    policy domains, then to the reflected video-type weight and source quality.
    The returned schedule is an audit artifact; source records themselves are
    never rewritten.
    """
    if source_cap < 0:
        raise ValueError("source_cap_must_be_nonnegative")
    schedule = build_source_schedule(records, policy, seed=seed)
    by_key = {row["source_key"]: row for row in schedule}
    source_rows: dict[str, dict[str, Any]] = {}
    for record in records:
        key = source_key(record)
        source_rows.setdefault(key, record)
    keys = sorted(source_rows)
    target = len(keys) if source_cap == 0 else min(int(source_cap), len(keys))
    selected: list[str] = []
    remaining = set(keys)
    while remaining and len(selected) < target:
        def score(key: str) -> tuple[float, float, str]:
            row = by_key[key]
            # A stable pseudo-random tie breaker avoids dependence on input
            # ordering while keeping the schedule reproducible for a seed.
            tie = random.Random(f"{seed}:{key}").random()
            return float(row.get("sampling_weight", 1.0) or 1.0), tie, key

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)

    selected_set = set(selected)
    selected_rows = [source_rows[key] for key in selected]
    for row in schedule:
        row["selected"] = row["source_key"] in selected_set
        row["selection_reason"] = (
            "generation_source_cap"
            if row["selected"] and source_cap > 0
            else ("generation_source_withheld" if source_cap > 0 else row["selection_reason"])
        )
    return {
        "schedule": schedule,
        "selected": selected_rows,
        "selected_source_keys": selected,
        "source_cap": int(source_cap),
        "available_source_count": len(keys),
        "selected_source_count": len(selected_rows),
        "selection_seed": int(seed),
    }


__all__ = ["build_source_schedule", "select_source_records", "source_key"]
