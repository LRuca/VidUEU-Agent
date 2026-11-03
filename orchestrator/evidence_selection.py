"""Select complete evidence records without cutting IDs or structured states."""
from __future__ import annotations

import json
from typing import Any


def complete_evidence_text(units: list[dict[str, Any]], required_ids: set[str],
                           max_chars: int = 2400) -> str:
    rows = {row["evidence_id"]: row for row in units if row.get("evidence_id") and row.get("text")}
    missing = required_ids - rows.keys()
    if missing:
        raise ValueError(f"missing_required_evidence: {sorted(missing)}")
    required = [f"[{key}] {row['text']}" for key, row in rows.items() if key in required_ids]
    # The soft limit applies to auxiliary records. Required observations must
    # survive intact even when they exceed it; otherwise grounding is lost.
    used = len("\n".join(required))
    selected = set(required_ids)
    omitted = []
    for key, row in rows.items():
        if key in selected:
            continue
        length = len(f"[{key}] {row['text']}") + 1
        if used + length <= max_chars:
            selected.add(key)
            used += length
        else:
            omitted.append(key)
    text = "\n".join(f"[{key}] {row['text']}" for key, row in rows.items() if key in selected)
    if omitted:
        text += "\nOMITTED_AUXILIARY_EVIDENCE_IDS: " + json.dumps(omitted)
    return text
