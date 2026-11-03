"""Apply image-bound observation patches without replacing entity inventories."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from executor.tasks.prompt_loader import render_prompt_text
from orchestrator.visual_state import validate_frame_observation


ENTITY_FIELDS = {"visible_state", "position", "spatial_relations", "role", "uncertainties"}
FRAME_FIELDS = {"caption", "observations", "uncertainties", "unsupported_original_claims"}
FRAME_LIST_FIELDS = ("observations", "uncertainties", "unsupported_original_claims")


def observation_payload(frame: dict, frame_id: str) -> dict:
    observed = frame["visual_state"]
    image_hash = hashlib.sha256(Path(frame["img"]).read_bytes()).hexdigest()
    if observed.get("frame_id") != frame_id or observed.get("image_sha256") != image_hash:
        raise ValueError("patch_original_image_binding_changed")
    validation = frame.get("caption_validation") or {}
    return {"frame_id": frame_id, "image_sha256": image_hash, "caption": frame["caption"],
            "entities": copy.deepcopy(frame["visual_state"]["entities"]),
            **{key: copy.deepcopy(validation.get(key, [])) for key in FRAME_LIST_FIELDS}}


def observation_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def observation_patch_request(frame: dict, frame_id: str, focus_ids: list[str]) -> tuple[str, dict]:
    source = observation_payload(frame, frame_id)
    ids = [row["entity_id"] for row in source["entities"]]
    if not focus_ids or len(set(focus_ids)) != len(focus_ids) or any(key not in ids for key in focus_ids):
        raise ValueError("invalid_patch_focus_entities")
    digest = observation_sha256(source)
    prompt = render_prompt_text("paper", "observation_patch", frame_id=frame_id,
                               source_sha256=digest, focus_ids_json=json.dumps(focus_ids),
                               observation_json=json.dumps(source, ensure_ascii=False, sort_keys=True))
    binding = {"frame_id": frame_id, "source_observation_sha256": digest,
               "focus_entity_ids": focus_ids,
               "image_sha256": hashlib.sha256(Path(frame['img']).read_bytes()).hexdigest(),
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    return prompt, binding


def _updates(target: dict, update: Any, allowed: set[str]) -> None:
    if not isinstance(update, dict):
        raise ValueError("invalid_observation_patch_update")
    before, after = update.get("before"), update.get("after")
    if (not isinstance(before, dict) or not isinstance(after, dict) or not before
            or set(before) != set(after) or not set(after) <= allowed):
        raise ValueError("patch_fields_not_allowed_or_unbound")
    if any(target.get(key) != value for key, value in before.items()):
        raise ValueError("patch_before_value_changed")
    if not isinstance(update.get("reason"), str) or not update["reason"].strip():
        raise ValueError("patch_requires_image_reason")
    target.update(copy.deepcopy(after))


def apply_observation_patch(frame: dict, frame_id: str, focus_ids: list[str], patch: Any) -> dict:
    """Establish patch integrity, not visual truth or metadata admission."""
    source = observation_payload(frame, frame_id)
    if (not isinstance(patch, dict) or patch.get("frame_id") != frame_id
            or patch.get("source_observation_sha256") != observation_sha256(source)):
        raise ValueError("observation_patch_source_binding_changed")
    if set(patch) != {"frame_id", "source_observation_sha256", "status", "entity_checks",
                      "entity_updates", "frame_updates", "unchanged_entity_ids", "uncertainties"}:
        raise ValueError("unknown_or_missing_patch_field")
    if patch.get("status") not in {"patched", "unchanged", "hold"}:
        raise ValueError("invalid_observation_patch_status")
    entities = {row["entity_id"]: row for row in source["entities"]}
    if not focus_ids or len(set(focus_ids)) != len(focus_ids) or any(key not in entities for key in focus_ids):
        raise ValueError("invalid_patch_focus_entities")
    checks = patch.get("entity_checks")
    if (not isinstance(checks, list) or [v.get("entity_id") for v in checks if isinstance(v, dict)] != focus_ids
            or any(v.get("status") not in {"preserve", "patch", "uncertain"}
                   or not isinstance(v.get("reason"), str) or not v["reason"].strip() for v in checks)):
        raise ValueError("patch_requires_ordered_focus_entity_checks")
    changes, frame_changes = patch.get("entity_updates"), patch.get("frame_updates")
    if not isinstance(changes, list) or not isinstance(frame_changes, list):
        raise ValueError("patch_requires_explicit_updates")
    if not isinstance(patch.get("uncertainties"), list) or any(not isinstance(v, str) or not v.strip() for v in patch["uncertainties"]):
        raise ValueError("invalid_patch_uncertainties")
    touched = []
    for update in changes:
        key = update.get("entity_id") if isinstance(update, dict) else None
        if key not in focus_ids or key in touched:
            raise ValueError("patch_unknown_duplicate_or_out_of_scope_entity")
        if update.get("evidence_refs") != [frame_id + "_IMAGE"]:
            raise ValueError("patch_wrong_image_evidence")
        _updates(entities[key], update, ENTITY_FIELDS)
        touched.append(key)
    expected_unchanged = [key for key in entities if key not in touched]
    if patch.get("unchanged_entity_ids") != expected_unchanged:
        raise ValueError("patch_entity_inventory_not_preserved")
    touched_fields = set()
    for update in frame_changes:
        fields = set(update.get("after", {})) if isinstance(update, dict) else set()
        if fields & touched_fields:
            raise ValueError("duplicate_frame_patch_field")
        if not isinstance(update, dict) or update.get("evidence_refs") != [frame_id + "_IMAGE"]:
            raise ValueError("patch_wrong_image_evidence")
        _updates(source, update, FRAME_FIELDS)
        touched_fields |= fields
    if patch["status"] in {"unchanged", "hold"} and (changes or frame_changes):
        raise ValueError("non_patched_status_contains_updates")
    if patch["status"] == "patched" and not (changes or frame_changes):
        raise ValueError("patched_status_without_updates")
    if not isinstance(source["caption"], str) or not source["caption"].strip():
        raise ValueError("invalid_patched_caption")
    for key in FRAME_LIST_FIELDS:
        if not isinstance(source[key], list) or any(not isinstance(v, str) or not v.strip() for v in source[key]):
            raise ValueError("invalid_patched_" + key)
    observed = validate_frame_observation(source, frame_id, frame["img"])
    if patch["status"] != "hold":
        missing = []
        for key in focus_ids:
            entity = entities[key]
            if entity["category"] == "body_part" and not str(entity.get("role", "")).strip():
                missing.append({"entity_id": key, "field": "role", "before": None})
            if entity["category"] != "body_part" and not isinstance(entity.get("spatial_relations"), list):
                missing.append({"entity_id": key, "field": "spatial_relations", "before": None})
        if missing:
            raise ValueError("patch_missing_required_focus_observations: " + json.dumps(missing))
    result = copy.deepcopy(frame)
    result.setdefault("observation_patch_history", []).append({"source": observation_payload(frame, frame_id),
                                                              "patch": copy.deepcopy(patch)})
    result.update(caption=source["caption"], visual_state=observed,
                  caption_validation={**source, "raw_response": json.dumps(patch, ensure_ascii=False),
                                      "patch_visual_acceptance": "pending"})
    return result


def require_task_role_coverage(frame: dict) -> None:
    """Check explicit coverage, while leaving visual truth to the reviewer."""
    entities = (frame.get('visual_state') or {}).get('entities')
    if not isinstance(entities, list) or not entities:
        raise ValueError('task_observation_missing_entity_inventory')
    for entity in entities:
        category = entity.get('category')
        if category == 'body_part' and not str(entity.get('role', '')).strip():
            raise ValueError('task_observation_missing_body_part_role')
        interacting = category in {'container', 'utensil', 'fabric', 'food'} and (
            entity.get('position', {}).get('anchor_entity') is not None)
        if interacting and not isinstance(entity.get('spatial_relations'), list):
            raise ValueError('task_observation_missing_explicit_spatial_relations')


def validate_patched_observation(frame: dict, frame_id: str) -> None:
    """Recheck saved repairs under the current contract before admission/reuse."""
    history = frame.get("observation_patch_history")
    if not history:
        return
    if not isinstance(history, list) or not isinstance(history[-1], dict):
        raise ValueError("invalid_observation_patch_history")
    source, patch = history[-1].get("source"), history[-1].get("patch")
    if not isinstance(source, dict) or not isinstance(patch, dict):
        raise ValueError("invalid_observation_patch_history")
    original = {"img": frame["img"], "caption": source.get("caption"),
                "visual_state": {"frame_id": source.get("frame_id"), "image_sha256": source.get("image_sha256"),
                                 "entities": source.get("entities")},
                "caption_validation": {key: source.get(key, []) for key in FRAME_LIST_FIELDS}}
    checks = patch.get("entity_checks")
    if not isinstance(checks, list) or any(not isinstance(row, dict) for row in checks):
        raise ValueError("invalid_observation_patch_history")
    rebuilt = apply_observation_patch(original, frame_id, [row.get("entity_id") for row in checks], patch)
    if observation_payload(rebuilt, frame_id) != observation_payload(frame, frame_id):
        raise ValueError("saved_observation_patch_content_changed")
