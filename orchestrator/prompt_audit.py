"""Record final prompts and evidence without changing model interfaces."""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
import time
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from executor.tasks.prompt_loader import configured_prompt_profile, loaded_prompt_sources, prompt_profile_scope

_SESSION = ContextVar("vidueu_prompt_session", default=None)
_WRITE_LOCK = threading.Lock()


def image_identity(image: Any) -> dict:
    filename = str(image if isinstance(image, (str, Path)) else getattr(image, "filename", "") or "")
    if filename and Path(filename).is_file():
        return {"path": filename, "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
    if hasattr(image, "tobytes"):
        data = str(getattr(image, "size", "")).encode() + str(getattr(image, "mode", "")).encode() + image.tobytes()
        return {"path": filename or None, "pixel_sha256": hashlib.sha256(data).hexdigest()}
    return {"path": filename or None}


def paper_prompt_stage(stage: str):
    def decorate(fn):
        signature = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            planner, segment = bound.get("planner"), bound.get("segment")
            config = getattr(planner, "config", {}) or {}
            profile = configured_prompt_profile(config)
            context = {"stage": stage, "prompt_profile": profile}
            if isinstance(bound.get("metadata"), dict):
                context["metadata_id"] = bound["metadata"].get("metadata_id")
            if isinstance(bound.get("item"), dict):
                context.update(metadata_id=bound["item"].get("meta", {}).get("metadata_id"),
                               record_key=bound["item"].get("record_key"))
            if segment is not None:
                source = str(getattr(segment, "video_path", "") or getattr(segment, "segment_path", ""))
                index, start = getattr(segment, "index", 0), float(getattr(segment, "start_sec", 0) or 0)
                context.update(source_file=source, segment_index=index,
                    metadata_id=hashlib.sha256(f"{source}|{index}|{start}".encode()).hexdigest()[:24])
            session = {"context": context, "records": [], "path": config.get("paths", {}).get("prompt_audit_path")}
            token = _SESSION.set(session)
            try:
                with prompt_profile_scope(profile):
                    result = fn(*args, **kwargs)
                    if isinstance(result, dict):
                        result.setdefault("runtime", {})["prompt_audit"] = {
                            "profile": profile, "loaded_templates": loaded_prompt_sources(),
                            "call_ids": [r["call_id"] for r in session["records"]],
                            "final_prompt_hashes": [r["prompt_sha256"] for r in session["records"]]}
                    return result
            finally:
                _SESSION.reset(token)
        return wrapped
    return decorate


def audited_call(call, prompt: str, *, images: list | None = None, context: dict | None = None, client: Any = None) -> str:
    session = _SESSION.get()
    if session is None:
        return call()
    record = {"call_id": uuid.uuid4().hex, "context": {**session["context"], **(context or {})},
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "stage_templates_loaded": loaded_prompt_sources(), "images": [image_identity(i) for i in images or []]}
    started = time.monotonic()
    try:
        raw = call()
        record.update(status="response", raw_response=raw)
        identity = getattr(client, "last_attestation", None) or getattr(getattr(client, "_tls", None), "last_attestation", None)
        if isinstance(identity, dict):
            record["model_identity"] = dict(identity)
        return raw
    except Exception as exc:
        error_text = str(exc)
        api_key = getattr(client, "api_key", None)
        if isinstance(api_key, str) and api_key and api_key != "EMPTY":
            error_text = error_text.replace(api_key, "[REDACTED]")
        record.update(status="call_failed", error_type=type(exc).__name__, error=error_text,
                      semantic_decision=None)
        raise
    finally:
        record["elapsed_s"] = time.monotonic() - started
        session["records"].append(record)
        if session["path"]:
            with _WRITE_LOCK:
                path = Path(session["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def audited_text(llm: Any, prompt: str, **kwargs) -> str:
    return audited_call(lambda: llm.chat(prompt, **kwargs), prompt, client=llm)


def audited_vision(planner: Any, images: list, prompt: str, **kwargs) -> str:
    return audited_call(lambda: planner._run_single_vlm_chat(images, prompt, **kwargs), prompt, images=images, client=getattr(planner, "vlm", None))


class AuditedTextClient:
    def __init__(self, llm: Any):
        self.llm = llm

    def chat(self, prompt: str, **kwargs) -> str:
        return audited_text(self.llm, prompt, **kwargs)
