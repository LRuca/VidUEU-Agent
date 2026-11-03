from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ToolRegistry:
    def __init__(self, path: str | Path | None = None, *, enabled=None):
        self.path = Path(path) if path is not None else Path(__file__).with_name("tool_registry.json")
        self._tools = self._load()
        if enabled is not None:
            if set(enabled) - set(self._tools):
                raise ValueError("unknown_enabled_tool")
            for name, spec in self._tools.items():
                spec["enabled"] = name in enabled and bool(spec.get("enabled", True))

    def _load(self) -> Dict[str, Dict[str, Any]]:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError(f"Tool registry must be a JSON object: {self.path}")
        out: Dict[str, Dict[str, Any]] = {}
        for tool_name, spec in raw.items():
            if not isinstance(spec, dict):
                raise TypeError(f"Tool spec for {tool_name} must be an object")
            name = str(tool_name or "").strip()
            if not name:
                raise ValueError("Tool name cannot be empty")
            class_path = str((((spec.get("adapter", {}) or {}).get("class_path")) or "")).strip()
            if not class_path:
                raise ValueError(f"Tool {name} is missing adapter.class_path")
            resources = spec.get("resources", [])
            if not isinstance(resources, list):
                raise TypeError(f"Tool {name} resources must be a list")
            out[name] = dict(spec)
        return out

    def enabled_tools(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: dict(spec)
            for name, spec in self._tools.items()
            if bool(spec.get("enabled", True))
        }

    def get(self, tool_name: str) -> Dict[str, Any]:
        return dict(self._tools[str(tool_name)])

    def semanticizer_enabled_tools(self) -> List[str]:
        return [
            name for name, spec in self.enabled_tools().items()
            if bool(((spec.get("semanticizer", {}) or {}).get("enabled", True)))
        ]

    def scorer_enabled_tools(self) -> List[str]:
        return [
            name for name, spec in self.enabled_tools().items()
            if bool(((spec.get("value_scoring", {}) or {}).get("enabled", True)))
        ]
