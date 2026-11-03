"""Content identities for external models and immutable run artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .jobs import file_fingerprint
from .state import digest, read_json, write_json


def content_identity(path: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path.is_file():
        return file_fingerprint(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = []
    for child in sorted(path.rglob("*")):
        relative = child.relative_to(path)
        if (not child.is_file() or child.name in exclude
                or any(part in {"__pycache__", ".cache", ".git"} for part in relative.parts)):
            continue
        item = file_fingerprint(child)
        item["path"] = relative.as_posix()
        files.append(item)
    if not files:
        raise ValueError(f"empty_artifact_directory:{path}")
    return {"path": str(path), "files": files, "sha256": digest(files)}


def publish_immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"immutable_artifact_changed:{path}")
    else:
        write_json(path, value)
