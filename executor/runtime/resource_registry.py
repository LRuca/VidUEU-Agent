from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ResourceRegistry:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else Path(__file__).with_name("resource_registry.json")
        self._resources = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Resource registry must be a JSON object: {self.path}")
        out: Dict[str, Dict[str, Any]] = {}
        for resource_name, spec in raw.items():
            if not isinstance(spec, dict):
                raise TypeError(f"Resource spec for {resource_name} must be an object")
            name = str(resource_name or "").strip()
            if not name:
                raise ValueError("Resource name cannot be empty")
            class_path = str((((spec.get("factory", {}) or {}).get("class_path")) or "")).strip()
            if not class_path:
                raise ValueError(f"Resource {name} is missing factory.class_path")
            out[name] = dict(spec)
        return out

    def enabled_resources(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: dict(spec)
            for name, spec in self._resources.items()
            if bool(spec.get("enabled", True))
        }

    def get(self, name: str) -> Dict[str, Any]:
        return dict(self._resources[str(name)])
