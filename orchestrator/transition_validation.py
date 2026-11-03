"""Validate visible changes without rewriting their semantic content."""
from __future__ import annotations

import re
from typing import Any

_FORMAT = re.compile(r"^Edits\((\d+)\):\s*(.*?)\s*\|\s*Shot:\s*(none|minor|scene_cut)\s*\|\s*View:\s*(none|minor|major)\s*\|\s*Subject:\s*(same|unclear|different)\s*\|\s*Uncertainty:\s*(.+)$", re.I)
_VERB = re.compile(r"^(add|remove|move|rotate|open|close|turn on|turn off|change text|zoom|pan|tilt|cut)\b\s+\S", re.I)
_NONE = re.compile(r"^\(?\s*(none|no changes?|no meaningful (visible )?changes?)\s*\)?\.?$", re.I)
_CHAINED_VERB = re.compile(
    r"\b(?:and|then|plus)\s+(?:then\s+)?(?:add|remove|move|rotate|open|close|turn on|turn off|change text|zoom|pan|tilt|cut)\b",
    re.I,
)


def leading_edit_verb(edit: str) -> str | None:
    match = _VERB.match(edit)
    return match.group(1).lower() if match else None


def infer_transition_cause(check: dict) -> str:
    if check.get("status") == "no_change":
        return "none"
    if check.get("status") != "valid":
        return "unknown"
    verbs = {leading_edit_verb(edit) for edit in check["edits"]}
    if "cut" in verbs or check.get("shot") == "scene_cut":
        return "scene_cut"
    if "move" in verbs:
        return "object_motion"
    if verbs - {"pan", "zoom", "tilt", None}:
        return "edit"
    if verbs & {"pan", "zoom", "tilt"} or check.get("view") in {"minor", "major"}:
        return "camera_motion"
    return "unknown"


def validate_transition(raw: str, *, error: Exception | None = None) -> dict[str, Any]:
    result = {"raw_response": raw, "text": "", "edits": [], "status": "invalid", "normalization_reason": None}
    if error is not None:
        return {**result, "status": "call_failed", "error_type": type(error).__name__, "error": str(error)}
    if not isinstance(raw, str):
        return {**result, "reason": "non_string_transition"}
    match = _FORMAT.fullmatch(raw.strip())
    if not match:
        return {**result, "reason": "invalid_transition_format"}
    count, body, shot, view, subject, uncertainty = match.groups()
    count = int(count)
    shot, view, subject = shot.lower(), view.lower(), subject.lower()
    atoms = [re.sub(r"^\d+\)\s*", "", s.strip()) for s in body.split(";") if s.strip()]
    result.update(shot=shot, view=view, subject=subject, uncertainty=uncertainty)
    if len(atoms) == 1 and _NONE.fullmatch(atoms[0]):
        if shot != "none" or view != "none" or subject == "different":
            return {**result, "reason": "no_change_conflicts_with_labels"}
        text = f"Edits(0): none | Shot: {shot} | View: {view} | Subject: {subject} | Uncertainty: {uncertainty}"
        return {**result, "status": "no_change", "text": text,
                "normalization_reason": "no_change_count_or_spelling" if raw.strip() != text else None}
    if count != len(atoms) or not count:
        return {**result, "reason": "edit_count_mismatch"}
    if any(not _VERB.match(a) or _NONE.fullmatch(a) for a in atoms):
        return {**result, "reason": "invalid_atomic_edit"}
    # Conjoined directions or appearance cues can still describe one edit.
    # This catches explicit chained verbs; visual atomicity remains a review task.
    if any(_CHAINED_VERB.search(a) for a in atoms):
        return {**result, "reason": "non_atomic_edit"}
    if shot == "scene_cut" and not any(a.lower().startswith("cut ") for a in atoms):
        return {**result, "reason": "cut_label_without_cut"}
    return {**result, "status": "valid", "text": raw.strip(), "edits": atoms}
