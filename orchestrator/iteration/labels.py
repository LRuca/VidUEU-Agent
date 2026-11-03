"""Canonical iteration labels and provenance propagation.

The generation pipeline creates fresh dictionaries for several T4 task
families.  Keeping label propagation in one small pure helper prevents a
label produced by T1/metadata audit from disappearing in one of the flatten
or export paths.
"""
from __future__ import annotations

import copy
from typing import Any

from .taxonomy import DOMAINS, canonical_domain, domain_from_video_type


# The first four fields are the labels consumed by taxonomy/scheduling.  The
# audit fields travel with them so an ``unknown`` result is distinguishable
# from a missing audit or a parser/provider failure after T4 flattens a row.
_LABEL_KEYS = (
    "video_type",
    "policy_domain",
    "domain_source",
    "domain_confidence",
    "domain_audit_status",
    "domain_audit_evidence",
    "domain_audit_model_identity",
    "domain_audit_prompt_sha256",
    "domain_unknown_reason",
)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def canonical_iteration_labels(t1_output: dict[str, Any] | None) -> dict[str, Any]:
    """Return one canonical label envelope from a T1/global-perception row.

    Explicit metadata is preferred, then a domain inferred from the canonical
    video type.  Invalid or missing domains remain ``None`` and are explicitly
    marked unresolved; no task label is used to infer a video domain.
    """
    source = _mapping(t1_output)
    global_perception = _mapping(source.get("global_perception"))
    metadata = _mapping(source.get("meta"))
    metadata_global_perception = _mapping(metadata.get("global_perception"))
    video_type = (
        source.get("video_type")
        or global_perception.get("video_type")
        or metadata.get("video_type")
        or metadata_global_perception.get("video_type")
        or "unknown"
    )
    video_type = str(video_type).strip().lower() or "unknown"
    if video_type in {"generic", "other", "none", "null"}:
        video_type = "unknown"
    raw_domain = (
        source.get("policy_domain")
        or global_perception.get("policy_domain")
        or global_perception.get("domain")
        or metadata.get("policy_domain")
        or metadata_global_perception.get("policy_domain")
        or metadata_global_perception.get("domain")
    )
    domain = canonical_domain(raw_domain)
    domain_source = (
        source.get("domain_source")
        or global_perception.get("domain_source")
        or metadata.get("domain_source")
        or metadata_global_perception.get("domain_source")
    )
    if domain is None:
        domain = domain_from_video_type(video_type)
        if domain:
            domain_source = "video_type_fallback"
    if domain is None:
        domain_source = str(domain_source or "global_perception_unresolved")
    confidence = (
        source.get("domain_confidence")
        if source.get("domain_confidence") is not None
        else global_perception.get("domain_confidence")
    )
    if confidence is None:
        confidence = metadata.get("domain_confidence")
    if confidence is None:
        confidence = metadata_global_perception.get("domain_confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    audit_status = (
        source.get("domain_audit_status")
        or global_perception.get("domain_audit_status")
        or metadata.get("domain_audit_status")
        or metadata_global_perception.get("domain_audit_status")
    )
    if not audit_status:
        if domain in DOMAINS:
            audit_status = "resolved"
        elif domain_source == "video_type_fallback":
            audit_status = "video_type_fallback"
        else:
            audit_status = "not_audited"
    audit_evidence = (
        source.get("domain_audit_evidence")
        or global_perception.get("domain_audit_evidence")
        or metadata.get("domain_audit_evidence")
        or metadata_global_perception.get("domain_audit_evidence")
        or []
    )
    if not isinstance(audit_evidence, list):
        audit_evidence = [str(audit_evidence)] if audit_evidence else []
    model_identity = (
        source.get("domain_audit_model_identity")
        or global_perception.get("domain_audit_model_identity")
        or metadata.get("domain_audit_model_identity")
        or metadata_global_perception.get("domain_audit_model_identity")
        or {}
    )
    if not isinstance(model_identity, dict):
        model_identity = {"value": str(model_identity)}
    prompt_hash = (
        source.get("domain_audit_prompt_sha256")
        or global_perception.get("domain_audit_prompt_sha256")
        or metadata.get("domain_audit_prompt_sha256")
        or metadata_global_perception.get("domain_audit_prompt_sha256")
    )
    unknown_reason = (
        source.get("domain_unknown_reason")
        or global_perception.get("domain_unknown_reason")
        or metadata.get("domain_unknown_reason")
        or metadata_global_perception.get("domain_unknown_reason")
    )
    if domain not in DOMAINS and not unknown_reason:
        unknown_reason = f"metadata_audit:{audit_status}"
    return {
        "video_type": video_type,
        "policy_domain": domain,
        "domain_source": str(domain_source) if domain_source else None,
        "domain_confidence": confidence,
        "domain_audit_status": str(audit_status),
        "domain_audit_evidence": [str(item) for item in audit_evidence if str(item).strip()][:4],
        "domain_audit_model_identity": model_identity,
        "domain_audit_prompt_sha256": str(prompt_hash) if prompt_hash else None,
        "domain_unknown_reason": str(unknown_reason) if unknown_reason else None,
        "domain_label_status": "resolved" if domain in DOMAINS else "unknown",
    }


def propagate_iteration_labels(
    row: dict[str, Any],
    labels: dict[str, Any],
    *,
    provenance: str = "T1_Global_Perception",
) -> dict[str, Any]:
    """Return a candidate with labels mirrored at top-level and under ``meta``.

    Existing explicit candidate labels are preserved.  The returned object is
    a deep copy so callers can safely use the same source row for multiple
    benchmark exports.
    """
    if not isinstance(row, dict):
        raise ValueError("candidate_row_must_be_object")
    result = copy.deepcopy(row)
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    result["meta"] = meta
    nested = _mapping(result.get("global_perception"))
    nested_meta = _mapping(meta.get("global_perception"))
    for key in _LABEL_KEYS:
        value = result.get(key)

        def _missing(candidate: Any) -> bool:
            if candidate in (None, ""):
                return True
            if key == "video_type" and str(candidate).strip().lower() in {"unknown", "generic"}:
                return True
            if key == "domain_audit_evidence" and not isinstance(candidate, list):
                return True
            if key == "domain_audit_model_identity" and not isinstance(candidate, dict):
                return True
            return False

        if _missing(value):
            value = meta.get(key)
        if _missing(value):
            value = nested.get(key)
        if _missing(value):
            value = nested_meta.get(key)
        if _missing(value):
            value = labels.get(key)
        if value is not None:
            if key == "video_type" and str(value).strip().lower() in {"unknown", "generic"}:
                value = "unknown"
            result[key] = value
            if meta.get(key) in (None, ""):
                meta[key] = value
    # Keep the canonical envelope explicit even when the domain is unresolved;
    # downstream reports must distinguish ``policy_domain: null`` from a
    # field that disappeared during T4 flattening.
    for key in _LABEL_KEYS:
        if key not in result and key in labels:
            result[key] = copy.deepcopy(labels[key])
        if key not in meta and key in result:
            meta[key] = copy.deepcopy(result[key])
    if result.get("policy_domain") not in DOMAINS:
        if not result.get("domain_source"):
            result["domain_source"] = str(labels.get("domain_source") or "global_perception_unresolved")
            meta.setdefault("domain_source", result["domain_source"])
        if result.get("domain_confidence") is None:
            result["domain_confidence"] = labels.get("domain_confidence")
            if result["domain_confidence"] is None:
                result["domain_confidence"] = 0.0
            meta.setdefault("domain_confidence", result["domain_confidence"])
        result.setdefault(
            "domain_unknown_reason",
            str(labels.get("domain_unknown_reason") or "no_canonical_domain_after_audit"),
        )
        meta.setdefault("domain_unknown_reason", result["domain_unknown_reason"])

    # Keep a structured copy for inventory/source scheduling and auditing.
    label_payload = {key: result.get(key) for key in _LABEL_KEYS}
    label_payload["domain_label_status"] = (
        "resolved" if result.get("policy_domain") in DOMAINS else "unknown"
    )
    result["global_perception"] = {
        **_mapping(result.get("global_perception")),
        **label_payload,
    }
    meta["global_perception"] = {
        **_mapping(meta.get("global_perception")),
        **label_payload,
    }
    meta.setdefault("iteration_label_provenance", provenance)
    return result


__all__ = ["canonical_iteration_labels", "propagate_iteration_labels"]
