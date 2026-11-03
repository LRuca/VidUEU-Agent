"""Small manifest contract helpers shared by scripts and the controller."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .state import digest, write_json


def write_iteration_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.setdefault("schema_version", 1)
    result["payload_sha256"] = digest({key: value for key, value in result.items() if key != "payload_sha256"})
    write_json(Path(path), result)
    return result


def validate_manifest_metadata(value: dict[str, Any]) -> None:
    required = {"iteration", "target_train_samples", "actual_samples", "shortage"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError("manifest_missing_fields:" + ",".join(missing))
    for key, minimum in (("iteration", 1), ("target_train_samples", 1), ("actual_samples", 0)):
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise ValueError(f"manifest_invalid_integer:{key}")
    actual, target = value["actual_samples"], value["target_train_samples"]
    if actual > target:
        raise ValueError("manifest_actual_exceeds_target")
    if not isinstance(value["shortage"], bool) or value["shortage"] != (actual < target):
        raise ValueError("manifest_shortage_inconsistent")


__all__ = ["write_iteration_manifest", "validate_manifest_metadata"]
