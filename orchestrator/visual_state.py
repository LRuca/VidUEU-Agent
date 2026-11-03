"""Validate image-bound observations and explicit cross-frame entity comparisons.

These checks establish contracts and consistency, not pixel-level correctness.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import re
from typing import Any

from orchestrator.planner_helpers import parse_llm_json
from orchestrator.transition_validation import validate_transition
from executor.tasks.prompt_loader import current_prompt_profile


REGIONS = {"upper_left", "upper_center", "upper_right", "center_left", "center",
           "center_right", "lower_left", "lower_center", "lower_right", "uncertain"}
RELATIONS = {"inside", "above", "below", "image_left_of", "image_right_of", "on",
             "touching", "overlapping", "near", "none", "uncertain"}
CATEGORIES = {"person", "body_part", "container", "utensil", "fabric", "food",
              "surface", "text", "other"}
SPATIAL_AXES = {"horizontal": {"image_left_of", "image_right_of", "overlapping", "uncertain"},
                "vertical": {"above", "below", "overlapping", "uncertain"},
                "contact": {"touching", "on", "inside", "none", "uncertain"}}


def _strings(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if (not isinstance(value, list) or (nonempty and not value) or
            any(not isinstance(v, str) or not v.strip() for v in value)):
        raise ValueError(f"invalid_{field}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_{field}")
    return value


def _refs(value: Any, expected: set[str]) -> None:
    refs = _strings(value, "image_evidence_refs", nonempty=True)
    if set(refs) != expected or len(refs) != len(expected):
        raise ValueError("wrong_image_evidence_refs")


def validate_frame_observation(value: Any, frame_id: str, image_path: str) -> dict:
    if not isinstance(value, dict) or value.get("frame_id") != frame_id:
        raise ValueError("frame_observation_binding_mismatch")
    entities = value.get("entities")
    if not isinstance(entities, list) or not entities:
        raise ValueError("missing_visible_entities")
    ids = []
    for row in entities:
        if not isinstance(row, dict) or not re.fullmatch(r"E\d+", str(row.get("entity_id", ""))):
            raise ValueError("invalid_entity_id")
        ids.append(row["entity_id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate_entity_id")
    for row in entities:
        _text(row.get("label"), "entity_label")
        _text(row.get("visible_state"), "visible_state")
        _strings(row.get("appearance_cues"), "appearance_cues", nonempty=True)
        _strings(row.get("uncertainties"), "entity_uncertainties")
        _refs(row.get("evidence_refs"), {frame_id + "_IMAGE"})
        if not isinstance(row.get("category"), str) or row["category"] not in CATEGORIES:
            raise ValueError("invalid_entity_category")
        pos = row.get("position")
        if (not isinstance(pos, dict) or not isinstance(pos.get("region"), str) or pos["region"] not in REGIONS or
                not isinstance(pos.get("relation"), str) or pos["relation"] not in RELATIONS):
            raise ValueError("invalid_entity_position")
        if not isinstance(pos.get("certain"), bool):
            raise ValueError("missing_position_certainty")
        anchor = pos.get("anchor_entity")
        if anchor is not None and (not isinstance(anchor, str) or anchor not in ids or anchor == row["entity_id"]):
            raise ValueError("unknown_or_self_anchor")
        if (pos["relation"] not in {"none", "uncertain"}) != (anchor is not None):
            raise ValueError("relation_anchor_mismatch")
        if pos["certain"] and (pos["region"] == "uncertain" or pos["relation"] == "uncertain"):
            raise ValueError("uncertain_position_marked_certain")
        if "reference_frame" in pos and pos["reference_frame"] != "image":
            raise ValueError("position_requires_image_reference_frame")
        if "role" in row:
            _text(row["role"], "entity_role")
        relations = row.get("spatial_relations", [])
        if not isinstance(relations, list):
            raise ValueError("invalid_spatial_relations")
        seen = set()
        for relation in relations:
            if not isinstance(relation, dict):
                raise ValueError("invalid_spatial_relation")
            axis, anchor = relation.get("axis"), relation.get("anchor_entity")
            if not isinstance(axis, str) or axis not in SPATIAL_AXES or relation.get("relation") not in SPATIAL_AXES[axis]:
                raise ValueError("spatial_relation_axis_mismatch")
            if not isinstance(anchor, str) or anchor not in ids or anchor == row["entity_id"]:
                raise ValueError("unknown_or_self_spatial_anchor")
            if relation.get("reference_frame") != "image" or not isinstance(relation.get("certain"), bool):
                raise ValueError("spatial_relation_reference_or_certainty_invalid")
            for field in ("subject_part", "anchor_part", "support"):
                _text(relation.get(field), "spatial_" + field)
            _refs(relation.get("evidence_refs"), {frame_id + "_IMAGE"})
            if relation["relation"] == "uncertain" and relation["certain"]:
                raise ValueError("uncertain_spatial_relation_marked_certain")
            key = (axis, anchor, relation["subject_part"], relation["anchor_part"])
            if key in seen:
                raise ValueError("duplicate_spatial_relation_axis")
            seen.add(key)
    return {"frame_id": frame_id, "entities": copy.deepcopy(entities),
            "image_path": str(image_path), "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
            "validation_status": "contract_valid", "visual_acceptance": "pending"}


def validate_entity_identity(value: Any, source: dict, target: dict) -> dict[str, str]:
    """Validate frame-local one-to-one identities independently of edits."""
    if not isinstance(value, dict) or (value.get("source_frame"), value.get("target_frame")) != (source["frame_id"], target["frame_id"]):
        raise ValueError("pair_frame_direction_mismatch")
    left = {e["entity_id"]: e for e in source["entities"]}
    right = {e["entity_id"]: e for e in target["entities"]}
    matches = value.get("matches")
    if not isinstance(matches, list):
        raise ValueError("missing_entity_matches_or_changes")
    _strings(value.get("uncertainties"), "pair_uncertainties")
    refs = {source["frame_id"] + "_IMAGE", target["frame_id"] + "_IMAGE"}
    matched_left, matched_right, certain_matches = set(), set(), {}
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("invalid_entity_match")
        a, b = match.get("source_entity"), match.get("target_entity")
        if not isinstance(a, str) or not isinstance(b, str) or a not in left or b not in right or a in matched_left or b in matched_right:
            raise ValueError("unknown_or_duplicate_entity_match")
        if not isinstance(match.get("status"), str) or match["status"] not in {"same", "uncertain"}:
            raise ValueError("invalid_identity_status")
        _strings(match.get("identity_cues"), "identity_cues", nonempty=True)
        _refs(match.get("evidence_refs"), refs)
        if match["status"] == "same":
            if left[a]["category"] != right[b]["category"]:
                raise ValueError("matched_entity_category_conflict")
            certain_matches[a] = b
        matched_left.add(a)
        matched_right.add(b)
    for field, all_ids, matched in [("unmatched_source", left, matched_left), ("unmatched_target", right, matched_right)]:
        remaining = _strings(value.get(field), field)
        if set(remaining) != set(all_ids) - matched or len(set(remaining)) != len(remaining):
            raise ValueError("entity_inventory_not_accounted_for")
    return certain_matches


def _validate_comparison(value: Any, source: dict, target: dict) -> tuple[dict, list[dict], list[dict]]:
    certain_matches = validate_entity_identity(value, source, target)
    changes = value.get('changes')
    if not isinstance(changes, list):
        raise ValueError('missing_entity_matches_or_changes')
    left = {e['entity_id']: e for e in source['entities']}
    right = {e['entity_id']: e for e in target['entities']}
    refs = {source['frame_id'] + '_IMAGE', target['frame_id'] + '_IMAGE'}
    differences, diagnostics = [], []
    for a, b in certain_matches.items():
        pa, pb = left[a]["position"], right[b]["position"]
        if not pa["certain"] or not pb["certain"]:
            continue
        fields = []
        if pa["region"] != pb["region"]:
            fields.append("image_region")
        aa, ab = pa["anchor_entity"], pb["anchor_entity"]
        if aa in certain_matches and ab in certain_matches.values():
            if certain_matches[aa] != ab:
                fields.append("anchor_entity")
            elif pa["relation"] != pb["relation"]:
                fields.append("anchor_relation")
        elif aa is None and ab is None and pa["relation"] != pb["relation"]:
            fields.append("anchor_relation")
        ra, rb = left[a].get("spatial_relations", []), right[b].get("spatial_relations", [])
        hard_spatial_fields = set()
        target_relations = {(r["axis"], r["anchor_entity"], r["subject_part"], r["anchor_part"]): r
                            for r in rb if r["certain"]}
        for relation in ra:
            if not relation["certain"] or relation["anchor_entity"] not in certain_matches:
                continue
            key = (relation["axis"], certain_matches[relation["anchor_entity"]],
                   relation["subject_part"], relation["anchor_part"])
            counterpart = target_relations.get(key)
            if counterpart and counterpart["relation"] != relation["relation"]:
                fields.append("spatial_" + relation["axis"])
                values = frozenset((relation["relation"], counterpart["relation"]))
                if (values in {frozenset(("image_left_of", "image_right_of")), frozenset(("above", "below"))} or
                        (relation["axis"] == "contact" and "none" in values)):
                    hard_spatial_fields.add("spatial_" + relation["axis"])
        if fields:
            fields = list(dict.fromkeys(fields))
            diagnostics.append({"source_entity": a, "target_entity": b, "fields": fields})
            if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
                # Grid cells and a changed primary anchor are description choices,
                # not proof of physical motion. Preserve them for visual review.
                opposite = {frozenset(("image_left_of", "image_right_of")), frozenset(("above", "below"))}
                primary_opposite = (aa in certain_matches and certain_matches[aa] == ab and
                                    frozenset((pa["relation"], pb["relation"])) in opposite)
                fields = [field for field in fields if field in hard_spatial_fields or
                          (field == "anchor_relation" and primary_opposite)]
            if fields:
                differences.append({"source_entity": a, "target_entity": b, "fields": fields})
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("invalid_entity_change")
        a, b, verb = change.get("source_entity"), change.get("target_entity"), change.get("verb")
        if (a is not None and not isinstance(a, str)) or (b is not None and not isinstance(b, str)):
            raise ValueError("invalid_change_entity_id")
        _text(change.get("source_state"), "change_source_state")
        _text(change.get("target_state"), "change_target_state")
        edit = _text(change.get("edit"), "change_edit")
        _refs(change.get("evidence_refs"), refs)
        if not isinstance(verb, str) or not edit.lower().startswith(verb + " "):
            raise ValueError("change_verb_text_mismatch")
        if verb in {"zoom", "pan", "tilt", "cut"}:
            if a is not None or b is not None:
                raise ValueError("camera_change_has_entity_target")
        elif verb == "add":
            if a is not None or b not in value["unmatched_target"]:
                raise ValueError("add_targets_existing_or_unknown_entity")
        elif verb == "remove":
            if b is not None or a not in value["unmatched_source"]:
                raise ValueError("remove_targets_persistent_or_unknown_entity")
        elif a not in certain_matches or certain_matches[a] != b:
            raise ValueError("change_identity_not_supported")
        if verb not in {"zoom", "pan", "tilt", "cut"}:
            label = right[b]["label"] if a is None else left[a]["label"]
            if label.casefold() not in edit.casefold():
                raise ValueError("change_edit_does_not_identify_bound_entity")
    if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'}:
        validate_entity_coverage(value, source, target)
    return value, differences, diagnostics


def validate_entity_coverage(value: dict, source: dict, target: dict) -> None:
    """Require a fresh disposition for every entity, including omitted movers."""
    coverage = value.get('entity_coverage')
    if not isinstance(coverage, dict) or set(coverage) != {'source', 'target'}:
        raise ValueError('pair_requires_complete_source_and_target_entity_coverage')
    changes = value['changes']
    matches = value['matches']
    evidence = {source['frame_id'] + '_IMAGE', target['frame_id'] + '_IMAGE'}
    for side, observed in [('source', source), ('target', target)]:
        id_field = side + '_entity'
        entries = coverage[side]
        expected = [row['entity_id'] for row in observed['entities']]
        if not isinstance(entries, list) or [row.get('entity_id') for row in entries if isinstance(row, dict)] != expected:
            raise ValueError('pair_entity_coverage_missing_or_reordered_' + side + ':' + ','.join(expected))
        for row in entries:
            status = row.get('status')
            indexes = row.get('change_indexes')
            if (status not in {'changed', 'stable', 'uncertain'} or not isinstance(indexes, list)
                    or any(isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(changes) for index in indexes)
                    or len(set(indexes)) != len(indexes)):
                raise ValueError('invalid_pair_entity_coverage:' + side + ':' + row['entity_id'])
            _text(row.get('reason'), 'entity_coverage_reason')
            _refs(row.get('evidence_refs'), evidence)
            actual = [i for i, change in enumerate(changes) if change.get(id_field) == row['entity_id']]
            if indexes != actual or (status == 'changed') != bool(actual):
                raise ValueError('pair_entity_coverage_change_binding_mismatch:' + side + ':' + row['entity_id'])
            if status == 'stable' and not any(match.get(id_field) == row['entity_id'] and match.get('status') == 'same' for match in matches):
                raise ValueError('pair_stability_requires_supported_identity:' + side + ':' + row['entity_id'])


def validate_pair_observation(raw: str, source: dict, target: dict, *, error: Exception | None = None) -> dict:
    if error is not None:
        return validate_transition(raw, error=error)
    result = {"raw_response": raw, "text": "", "edits": [], "status": "invalid", "normalization_reason": None}
    try:
        parsed_response = parse_llm_json(raw)
        if current_prompt_profile() in {'paper_v11', 'paper_v13', 'paper_v14', 'paper_v15'} and isinstance(parsed_response, dict):
            parsed_response = copy.deepcopy(parsed_response)
            changes = []
            for field, id_field in [('unmatched_source', 'source_entity'), ('unmatched_target', 'target_entity')]:
                rows = parsed_response.get(field)
                if isinstance(rows, list) and rows and all(isinstance(row, dict)
                        and set(row) == {id_field, 'reason'} and isinstance(row[id_field], str)
                        and isinstance(row['reason'], str) and row['reason'].strip() for row in rows):
                    # Preserve exactly the IDs the model emitted. Inventory,
                    # matching and semantic checks below remain mandatory.
                    parsed_response[field] = [row[id_field] for row in rows]
                    changes.append(field + '_object_envelope_to_ids')
            if changes:
                result['response_normalizations'] = changes
        value, differences, diagnostics = _validate_comparison(parsed_response, source, target)
        parsed = validate_transition(value.get("transition"))
        result.update(raw_transition=value.get("transition"), entity_comparison=value,
                      observed_position_differences=differences)
        if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", 'paper_v13', 'paper_v14', 'paper_v15'}:
            result["position_diagnostics"] = diagnostics
        if parsed["status"] == "invalid":
            raise ValueError(parsed["reason"])
        if parsed["edits"] != [c["edit"] for c in value["changes"]]:
            raise ValueError("transition_changes_disagree")
        if parsed["status"] == "no_change":
            _text(value.get("no_change_reason"), "no_change_reason")
            if differences:
                raise ValueError("no_change_contradicts_observed_positions")
        camera_change = any(c["verb"] in {"zoom", "pan", "tilt", "cut"} for c in value["changes"])
        if not camera_change:
            covered = {(c["source_entity"], c["target_entity"]) for c in value["changes"]}
            uncovered = [d for d in differences if (d["source_entity"], d["target_entity"]) not in covered]
            if uncovered:
                source_entities = {e["entity_id"]: e for e in source["entities"]}
                target_entities = {e["entity_id"]: e for e in target["entities"]}
                result["contract_details"] = [{**d, "source_observation": source_entities[d["source_entity"]],
                                               "target_observation": target_entities[d["target_entity"]]} for d in uncovered]
                raise ValueError("observed_position_change_omitted")
        return {**parsed, **result, "status": parsed["status"], "text": parsed["text"],
                "edits": parsed["edits"], "normalization_reason": parsed["normalization_reason"]}
    except ValueError as exc:
        if current_prompt_profile() in {'paper_v11', 'paper_v13', 'paper_v14', 'paper_v15'} and str(exc) == 'matched_entity_category_conflict':
            parsed = parse_llm_json(raw)
            left = {e['entity_id']: e for e in source['entities']}
            right = {e['entity_id']: e for e in target['entities']}
            result['contract_details'] = [{'source_entity': row.get('source_entity'),
                'target_entity': row.get('target_entity'),
                'source_observation': left[row['source_entity']],
                'target_observation': right[row['target_entity']]}
                for row in parsed.get('matches', []) if row.get('status') == 'same'
                and row.get('source_entity') in left and row.get('target_entity') in right
                and left[row['source_entity']]['category'] != right[row['target_entity']]['category']]
        if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'} and str(exc) in {
                'unknown_or_duplicate_entity_match', 'matched_entity_category_conflict',
                'invalid_unmatched_source', 'invalid_unmatched_target', 'entity_inventory_not_accounted_for'}:
            parsed = parse_llm_json(raw)
            if isinstance(parsed, dict):
                left = {row['entity_id']: row for row in source['entities']}
                right = {row['entity_id']: row for row in target['entities']}
                matches = [row for row in parsed.get('matches', []) if isinstance(row, dict)]
                faults = []
                for index, row in enumerate(matches):
                    a, b = row.get('source_entity'), row.get('target_entity')
                    problems = []
                    if a not in left:
                        problems.append('unknown_source_id')
                    if b not in right:
                        problems.append('unknown_target_id')
                    if sum(other.get('source_entity') == a for other in matches) > 1:
                        problems.append('duplicate_source_id')
                    if sum(other.get('target_entity') == b for other in matches) > 1:
                        problems.append('duplicate_target_id')
                    if row.get('status') == 'same' and a in left and b in right and left[a]['category'] != right[b]['category']:
                        problems.append('same_identity_category_conflict')
                    if problems:
                        faults.append({'match_index': index, 'source_entity': a, 'target_entity': b,
                                       'problems': problems})
                # Report all concrete ID faults together; never repair identities or
                # reinterpret missing inventory entries as physical disappearance.
                result['contract_details'] = {
                    'identity_faults': faults,
                    'source_inventory': [{key: row[key] for key in ['entity_id', 'label', 'category']} for row in source['entities']],
                    'target_inventory': [{key: row[key] for key in ['entity_id', 'label', 'category']} for row in target['entities']],
                    'unmatched_source_provided': parsed.get('unmatched_source'),
                    'unmatched_target_provided': parsed.get('unmatched_target'),
                    'policy': 'Use only each side\'s actual IDs. Unresolved or differently grouped background entities may remain unmatched/uncertain; do not invent IDs, add/remove edits, or supported identity.'}
        return {**result, "reason": str(exc)}
