"""Canonical capability, domain, and task labels for iteration control."""
from __future__ import annotations

from typing import Any

CAPABILITIES = ("understanding", "editing", "unified")
DOMAINS = (
    "Technology&Tutorial",
    "Sports&Action",
    "DailyLife&HumanActivities",
    "Nature&Scenery",
    "Gaming&ACG",
)
TASK_TYPES = (
    "fine_grained_perception",
    "single_image_reasoning",
    "multi_image_logic",
    "localized_editing",
    "grounded_edit_to_select",
    "understand_recall_generate_select",
    "stepwise_sequence",
    "middle_step_gap",
    "instructional_future_sequence",
)

TASK_CAPABILITY = {
    "fine_grained_perception": "understanding",
    "single_image_reasoning": "understanding",
    "multi_image_logic": "understanding",
    "localized_editing": "editing",
    "grounded_edit_to_select": "unified",
    "understand_recall_generate_select": "unified",
    "stepwise_sequence": "unified",
    "middle_step_gap": "unified",
    "instructional_future_sequence": "unified",
}

TASK_ALIASES = {
    "visual_reasoning": "single_image_reasoning",
    "editing_and_infilling": "localized_editing",
    "interleaved_understand_recall_generate_select": "understand_recall_generate_select",
    "stepwise_sequence_generate_select": "stepwise_sequence",
    "middle_step_gap_generate_fill": "middle_step_gap",
    "instructional_sequential_generate_select": "instructional_future_sequence",
    "mmt_bench": "fine_grained_perception",
    "llava_bench": "single_image_reasoning",
    "dialog_mmdu_like": "multi_image_logic",
    "edit": "localized_editing",
    "unify.seed.l3.gts": "grounded_edit_to_select",
    "unify.weave.uie_select_recall": "understand_recall_generate_select",
    "unify.comm.task4.c1_steps": "stepwise_sequence",
    "unify.comm.task4.c2_fill_gap": "middle_step_gap",
    "unify.isg.i2.seq_select": "instructional_future_sequence",
}

VIDEO_TYPE_DOMAIN = {
    "tutorial": "Technology&Tutorial",
    "lecture": "Technology&Tutorial",
    "sports": "Sports&Action",
    "gaming": "Gaming&ACG",
    "vlog": "DailyLife&HumanActivities",
    "drama": "DailyLife&HumanActivities",
    "surveillance": "DailyLife&HumanActivities",
}


def canonical_task(value: Any) -> str | None:
    text = str(value or "").strip()
    if text in TASK_CAPABILITY:
        return text
    if text in TASK_ALIASES:
        return TASK_ALIASES[text]
    if text.startswith("understanding."):
        return TASK_ALIASES.get(text.split(".", 1)[1])
    if text.startswith("unify."):
        return TASK_ALIASES.get(text)
    return None


def capability_for_task(task: Any) -> str | None:
    canonical = canonical_task(task)
    return TASK_CAPABILITY.get(canonical) if canonical else None


def canonical_domain(value: Any) -> str | None:
    text = str(value or "").strip()
    return text if text in DOMAINS else None


def domain_from_video_type(value: Any) -> str | None:
    return VIDEO_TYPE_DOMAIN.get(str(value or "").strip().lower())


def labels_for_record(record: dict[str, Any]) -> dict[str, str | None]:
    """Read labels without using a generated task to infer a domain.

    Global-perception metadata is preferred.  ``video_type`` is retained as a
    declared fallback for legacy candidate pools, and callers can inspect
    :func:`label_details_for_record` when provenance of the domain matters.
    """
    details = label_details_for_record(record)
    return {"task_type": details["task_type"], "capability": details["capability"], "domain": details["domain"]}


def label_details_for_record(record: dict[str, Any]) -> dict[str, str | float | None]:
    """Return canonical labels plus domain provenance/confidence."""
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    global_perception = record.get("global_perception")
    if not isinstance(global_perception, dict):
        global_perception = meta.get("global_perception") if isinstance(meta.get("global_perception"), dict) else {}
    task = canonical_task(record.get("task_type") or record.get("benchmark") or meta.get("task_type"))
    capability = str(record.get("capability") or meta.get("capability") or "").strip().lower() or None
    if capability not in CAPABILITIES:
        capability = capability_for_task(task)
    explicit_domain = record.get("policy_domain") or meta.get("policy_domain")
    global_domain = global_perception.get("policy_domain") or global_perception.get("domain")
    domain = canonical_domain(explicit_domain or global_domain)
    domain_source = "explicit" if domain and explicit_domain else ("global_perception" if domain else None)
    if domain is None:
        domain = domain_from_video_type(record.get("video_type") or meta.get("video_type"))
        if domain:
            domain_source = "video_type_fallback"
    confidence = record.get("domain_confidence", meta.get("domain_confidence", global_perception.get("domain_confidence")))
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "task_type": task,
        "capability": capability,
        "domain": domain,
        "domain_source": domain_source,
        "domain_confidence": confidence_value,
        "label_version": record.get("label_version") or meta.get("label_version") or "domain_v1",
    }
