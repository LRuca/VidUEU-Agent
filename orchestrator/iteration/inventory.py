"""Candidate/source inventory helpers for the iteration loop."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .controller import normalize_candidate
from .state import read_jsonl


def load_inventory(path: Path) -> list[dict[str, Any]]:
    """Read and normalize a candidate JSONL without mutating the source."""
    return [normalize_candidate(row, index) for index, row in enumerate(read_jsonl(Path(path)))]


def inventory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(rows),
        "record_keys_unique": len({str(row.get("record_key")) for row in rows}) == len(rows),
        "source_pointers_present": sum(bool(row.get("source_pointer")) for row in rows),
        "source_hashes_present": sum(bool(row.get("source_sha256")) for row in rows),
        "unknown_domains": sum(not row.get("policy_domain") for row in rows),
    }


__all__ = ["load_inventory", "inventory_summary"]
