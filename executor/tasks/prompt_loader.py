from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from string import Template
from typing import Any


_RESOURCE_ROOT = Path(__file__).resolve().parents[2]
PROMPT_JSON_PATH = _RESOURCE_ROOT / "extracted_system_prompts.json"
if not PROMPT_JSON_PATH.is_file():
    # Wheel installs carry the legacy prompt in the prompts resource package.
    PROMPT_JSON_PATH = PROMPT_JSON_PATH.parent / "prompts" / "legacy.json"
PAPER_PROMPT_PATH = _RESOURCE_ROOT / "prompts" / "paper_v1.json"
PAPER_PROFILES = {"paper_v1", "paper_v2", "paper_v3", "paper_v4", "paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}
_PROFILE = ContextVar("vidueu_prompt_profile", default="legacy")
_SOURCES = ContextVar("vidueu_prompt_sources", default=None)


def configured_prompt_profile(config: dict[str, Any]) -> str:
    t4 = config.get("tasks", {}).get("T4", {})
    return t4.get("prompt_profile", "paper_v2") if t4.get("task_profile", "paper") == "paper" else "legacy"


def current_prompt_profile() -> str:
    return _PROFILE.get()


def structured_evidence_profile() -> bool:
    return current_prompt_profile() in {"paper_v4", "paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}


@contextmanager
def prompt_profile_scope(profile: str):
    if profile not in {"legacy", *PAPER_PROFILES}:
        raise ValueError(f"Unknown prompt profile: {profile}")
    token = _PROFILE.set(profile)
    sources_token = _SOURCES.set({})
    try:
        yield _SOURCES.get()
    finally:
        _SOURCES.reset(sources_token)
        _PROFILE.reset(token)


def loaded_prompt_sources() -> list[dict[str, Any]]:
    return list((_SOURCES.get() or {}).values())


def load_prompt_text(*path: str, language: str = "english", profile: str | None = None) -> str:
    profile = profile or current_prompt_profile()
    prompt_id = "/".join(path)
    if profile not in {"legacy", *PAPER_PROFILES}:
        raise ValueError(f"Unknown prompt profile: {profile}")
    source = None
    inherited_source = None
    if profile in PAPER_PROFILES:
        paper_path = PAPER_PROMPT_PATH.with_name(profile + ".json")
        paper = json.loads(paper_path.read_text(encoding="utf-8"))
        inherited_source = paper.get("legacy_sources", {}).get(path[0])
        source = paper["prompts"].get(prompt_id)
        if source is not None:
            if language != "english":
                raise ValueError("Paper training prompts are English only")
            text = _stringify_prompt_value(source["text"])
            _record_source(prompt_id, text, profile, source, paper_path)
            return text
    obj = _load_prompt_json()
    node: Any = obj
    for key in path:
        if not isinstance(node, dict) or key not in node:
            joined = ".".join(path)
            raise KeyError(f"Prompt path not found: {joined}")
        node = node[key]

    if isinstance(node, dict):
        if language not in node:
            joined = ".".join(path)
            raise KeyError(f"Prompt language '{language}' not found for path: {joined}")
        node = node[language]

    if isinstance(node, (list, str)):
        text = _stringify_prompt_value(node)
        _record_source(prompt_id, text, profile, {"paper": inherited_source} if inherited_source else None, PROMPT_JSON_PATH)
        return text

    joined = ".".join(path)
    raise TypeError(f"Prompt value at path '{joined}' is not a string or list of strings")


def render_prompt_text(*path: str, language: str = "english", profile: str | None = None, **replacements: Any) -> str:
    profile = profile or current_prompt_profile()
    text = load_prompt_text(*path, language=language, profile=profile)
    data = {key: _stringify_prompt_value(value) for key, value in replacements.items()}
    if profile in PAPER_PROFILES:
        return Template(text).substitute(data)
    return Template(text).safe_substitute(data)


def _record_source(prompt_id: str, text: str, profile: str, source: dict | None, file: Path) -> None:
    sources = _SOURCES.get()
    if sources is not None:
        sources[prompt_id] = {"prompt_id": prompt_id, "profile": profile,
            "template_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "source_file": file.name, "paper": (source or {}).get("paper"),
            "adaptations": (source or {}).get("adaptations", "Unchanged legacy template")}


def _load_prompt_json() -> dict[str, Any]:
    try:
        text = PROMPT_JSON_PATH.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Prompt JSON not found: {PROMPT_JSON_PATH}") from exc

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise TypeError(f"Prompt JSON root must be an object: {PROMPT_JSON_PATH}")
    return obj


def _stringify_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(x) for x in value)
    return str(value)
