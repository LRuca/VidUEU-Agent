"""Qwen3.5 metadata audit for canonical iteration-domain labels."""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from .taxonomy import DOMAINS


AUDIT_PROMPT = """You are a metadata auditor for a video instruction-data pipeline.
Inspect the supplied representative frame(s) and segment summary. Return JSON only.
Choose exactly one video_type from: tutorial, lecture, sports, gaming, vlog, drama,
surveillance, nature, or other. Choose exactly one policy_domain from:
Technology&Tutorial, Sports&Action, DailyLife&HumanActivities, Nature&Scenery,
Gaming&ACG, or unknown. Use unknown when the evidence is insufficient.
Do not infer a domain from the existing generic label alone.
Schema: {"video_type":"...","policy_domain":"...","confidence":0.0,
"evidence":["short factual visual reasons"]}
"""


def _parse_object(raw: Any) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        value = json.loads(text)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def audit_domain(planner: Any, images: list[str], summary: str, current_video_type: str) -> dict[str, Any] | None:
    """Ask the configured VLM for a canonical domain, preserving audit provenance."""
    def current_model_identity() -> dict[str, Any]:
        vlm = getattr(planner, "vlm", None)
        identity = dict(getattr(vlm, "last_attestation", {}) or {})
        client = getattr(vlm, "api_client", None)
        if not identity:
            identity = dict(getattr(client, "last_attestation", {}) or {})
        if not identity and client is not None:
            requested = getattr(client, "model", None)
            if requested:
                identity = {"requested_model": requested}
        return identity

    model_identity = current_model_identity()
    if not images:
        return {
            "video_type": current_video_type or "unknown",
            "policy_domain": None,
            "domain_source": "qwen3.5_metadata_audit_unavailable",
            "domain_confidence": 0.0,
            "domain_audit_evidence": [],
            "domain_audit_status": "no_images",
            "domain_audit_prompt_sha256": None,
            "domain_audit_model_identity": model_identity,
        }
    prompt = (
        AUDIT_PROMPT
        + f"\nExisting structure label (may be generic): {current_video_type or 'unknown'}"
        + f"\nSegment summary: {summary or ''}"
    )
    try:
        from orchestrator.prompt_audit import audited_vision

        raw = audited_vision(planner, images, prompt, max_new_tokens=350)
        # The adapter updates its attestation only after a successful API
        # response.  Capture it here rather than the pre-call empty value.
        model_identity = current_model_identity()
    except Exception as exc:
        model_identity = current_model_identity()
        return {
            "video_type": current_video_type or "unknown",
            "policy_domain": None,
            "domain_source": "qwen3.5_metadata_audit_failed",
            "domain_confidence": 0.0,
            "domain_audit_evidence": [],
            "domain_audit_status": "call_failed",
            "domain_audit_error": f"{type(exc).__name__}:{exc}",
            "domain_audit_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "domain_audit_model_identity": model_identity,
        }
    value = _parse_object(raw)
    if not value:
        return {
            "video_type": current_video_type or "unknown",
            "policy_domain": None,
            "domain_source": "qwen3.5_metadata_audit_invalid",
            "domain_confidence": 0.0,
            "domain_audit_evidence": [],
            "domain_audit_status": "invalid_json",
            "domain_audit_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "domain_audit_model_identity": model_identity,
        }
    domain = str(value.get("policy_domain") or "unknown").strip()
    if domain not in (*DOMAINS, "unknown"):
        return {
            "video_type": current_video_type or "unknown",
            "policy_domain": None,
            "domain_source": "qwen3.5_metadata_audit_invalid",
            "domain_confidence": 0.0,
            "domain_audit_evidence": [],
            "domain_audit_status": "invalid_domain",
            "domain_audit_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "domain_audit_model_identity": model_identity,
        }
    video_type = str(value.get("video_type") or "other").strip().lower()
    allowed_types = {"tutorial", "lecture", "sports", "gaming", "vlog", "drama", "surveillance", "nature", "other"}
    if video_type not in allowed_types:
        video_type = "other"
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        evidence = []
    return {
        "video_type": video_type,
        "policy_domain": domain if domain != "unknown" else None,
        "domain_source": "qwen3.5_metadata_audit",
        "domain_confidence": confidence,
        "domain_audit_evidence": [str(item) for item in evidence if str(item).strip()][:4],
        "domain_audit_prompt": prompt,
        "domain_audit_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "domain_audit_status": "resolved" if domain != "unknown" else "unknown",
        "domain_audit_model_identity": model_identity,
    }


__all__ = ["audit_domain"]
