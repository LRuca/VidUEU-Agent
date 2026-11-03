from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from executor.tasks.prompt_loader import prompt_profile_scope
from orchestrator.metadata_admission import admit_metadata, metadata_for_judging
from orchestrator.visual_state import validate_frame_observation, validate_pair_observation


SUFFIX = " | Shot: none | View: none | Subject: same | Uncertainty: none"


def observation(frame, *, spoon_region="center_left", spoon_anchor="E0", spoon_relation="inside"):
    rows = [("E0", "metal pot", "container", "center_left", None, "none"),
            ("E1", "spoon", "utensil", spoon_region, spoon_anchor, spoon_relation),
            ("E2", "cloth", "fabric", "lower_center", None, "none"),
            ("E3", "food tray", "container", "lower_center", None, "none")]
    return {"frame_id": frame, "entities": [
        {"entity_id": eid, "label": label, "category": category, "appearance_cues": [label + " shape"],
         "visible_state": "visible " + label,
         "position": {"region": region, "anchor_entity": anchor, "relation": relation, "certain": True},
         "uncertainties": [], "evidence_refs": [frame + "_IMAGE"]}
        for eid, label, category, region, anchor, relation in rows]}


def bound(frame, path, **kwargs):
    return validate_frame_observation(observation(frame, **kwargs), frame, str(path))


def comparison(source="F00", target="F01", *, changed=True):
    refs = [source + "_IMAGE", target + "_IMAGE"]
    edit = "move spoon from inside the metal pot to above the food tray"
    return {"source_frame": source, "target_frame": target,
            "matches": [{"source_entity": f"E{i}", "target_entity": f"E{i}", "status": "same",
                         "identity_cues": ["same visible shape"], "evidence_refs": refs} for i in range(4)],
            "unmatched_source": [], "unmatched_target": [],
            "changes": [{"source_entity": "E1", "target_entity": "E1", "verb": "move",
                         "source_state": "inside metal pot", "target_state": "above food tray",
                         "edit": edit, "evidence_refs": refs}] if changed else [],
            "transition": ("Edits(1): 1) " + edit if changed else "Edits(0): none") + SUFFIX,
            "uncertainties": [], "no_change_reason": None if changed else "The same objects have the same visible positions."}


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "bound.png"
    Image.new("RGB", (12, 12), "red").save(path)
    return path


def test_frame_state_binds_pixels_and_does_not_claim_visual_acceptance(image_path):
    raw = observation("F00")
    before = copy.deepcopy(raw)
    result = validate_frame_observation(raw, "F00", str(image_path))
    assert raw == before
    assert result["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert result["visual_acceptance"] == "pending"
    assert {r["label"] for r in result["entities"]} == {"metal pot", "spoon", "cloth", "food tray"}


@pytest.mark.parametrize("mutation", ["duplicate_id", "wrong_frame", "self_anchor", "wrong_image", "uncertain_but_certain"])
def test_invalid_frame_identity_and_position_are_rejected(image_path, mutation):
    value = observation("F00")
    if mutation == "duplicate_id": value["entities"][1]["entity_id"] = "E0"
    elif mutation == "wrong_frame": value["frame_id"] = "F01"
    elif mutation == "self_anchor": value["entities"][1]["position"]["anchor_entity"] = "E1"
    elif mutation == "wrong_image": value["entities"][1]["evidence_refs"] = ["F01_IMAGE"]
    else: value["entities"][1]["position"]["region"] = "uncertain"
    with pytest.raises(ValueError): validate_frame_observation(value, "F00", str(image_path))


def test_persistent_spoon_moves_without_replacing_cloth(image_path):
    a = bound("F00", image_path)
    b = bound("F01", image_path, spoon_region="lower_center", spoon_anchor="E3", spoon_relation="above")
    raw = json.dumps(comparison())
    result = validate_pair_observation(raw, a, b)
    assert result["status"] == "valid" and result["raw_response"] == raw
    assert result["observed_position_differences"] == [{"source_entity": "E1", "target_entity": "E1", "fields": ["image_region", "anchor_entity"]}]
    assert result["edits"] == [comparison()["changes"][0]["edit"]]


def test_no_change_cannot_hide_observed_spoon_displacement(image_path):
    a = bound("F00", image_path)
    b = bound("F01", image_path, spoon_region="lower_center", spoon_anchor="E3", spoon_relation="above")
    result = validate_pair_observation(json.dumps(comparison(changed=False)), a, b)
    assert result["status"] == "invalid" and not result["text"]
    assert result["reason"] == "no_change_contradicts_observed_positions"
    with pytest.raises(ValueError, match="incomplete"):
        admit_metadata(SimpleNamespace(config={}), {"transitions": [{"validation": result}]})


def test_omitted_position_diagnostic_identifies_the_bound_entities(image_path):
    a = bound("F00", image_path)
    b = bound("F01", image_path, spoon_region="lower_center", spoon_anchor="E3", spoon_relation="above")
    value = comparison()
    value["changes"][0].update(source_entity="E0", target_entity="E0", edit="move metal pot leftward")
    value["transition"] = "Edits(1): 1) move metal pot leftward" + SUFFIX
    result = validate_pair_observation(json.dumps(value), a, b)
    assert result["reason"] == "observed_position_change_omitted"
    detail = result["contract_details"][0]
    assert detail["source_entity"] == "E1" and detail["target_entity"] == "E1"
    assert detail["source_observation"] == a["entities"][1]
    assert detail["target_observation"] == b["entities"][1]


def test_genuine_no_change_and_uncertain_identity_are_distinct(image_path):
    a, b = bound("F00", image_path), bound("F01", image_path)
    value = comparison(changed=False)
    assert validate_pair_observation(json.dumps(value), a, b)["status"] == "no_change"
    b["entities"][1]["position"]["region"] = "lower_center"
    value["matches"][1]["status"] = "uncertain"
    value["uncertainties"] = ["Tool identity or source position is unresolved."]
    value["transition"] = value["transition"].replace("Uncertainty: none", "Uncertainty: tool identity unresolved")
    assert validate_pair_observation(json.dumps(value), a, b)["status"] == "no_change"
    value["changes"] = comparison()["changes"]
    value["transition"] = comparison()["transition"]
    assert validate_pair_observation(json.dumps(value), a, b)["reason"] == "change_identity_not_supported"


@pytest.mark.parametrize("mutation,reason", [
    ("swap_direction", "pair_frame_direction_mismatch"),
    ("cloth_to_spoon", "matched_entity_category_conflict"),
    ("remove_persistent", "remove_targets_persistent_or_unknown_entity"),
    ("missing_inventory", "entity_inventory_not_accounted_for"),
    ("duplicate_match", "unknown_or_duplicate_entity_match"),
    ("wrong_refs", "wrong_image_evidence_refs"),
    ("wrong_transition", "transition_changes_disagree"),
])
def test_pair_binding_identity_and_transition_failures(image_path, mutation, reason):
    a, b = bound("F00", image_path), bound("F01", image_path)
    value = comparison()
    if mutation == "swap_direction": value["source_frame"], value["target_frame"] = "F01", "F00"
    elif mutation == "cloth_to_spoon":
        value["matches"][1]["target_entity"], value["matches"][2]["target_entity"] = "E2", "E1"
    elif mutation == "remove_persistent":
        value["changes"][0].update(source_entity="E2", target_entity=None, verb="remove", edit="remove cloth")
    elif mutation == "missing_inventory": value["matches"].pop()
    elif mutation == "duplicate_match": value["matches"].append(copy.deepcopy(value["matches"][0]))
    elif mutation == "wrong_refs": value["changes"][0]["evidence_refs"] = ["F00_IMAGE"]
    else: value["transition"] = "Edits(1): 1) move metal pot" + SUFFIX
    result = validate_pair_observation(json.dumps(value), a, b)
    assert result["status"] == "invalid" and result["reason"] == reason


def test_call_error_retains_failure_separate_from_none(image_path):
    result = validate_pair_observation("", bound("F00", image_path), bound("F01", image_path), error=TimeoutError("temporary"))
    assert result["status"] == "call_failed" and not result["text"]


def test_gate_c_receives_states_matches_and_uncertainty_without_old_claims(image_path):
    a, b = bound("F00", image_path), bound("F01", image_path)
    pair = validate_pair_observation(json.dumps(comparison(changed=False)), a, b)
    metadata = {"metadata_id": "bound", "keyframes": [{"frame_id": "F00", "visual_state": a,
        "caption": "visible pot", "original_caption": "old pouring sentinel"}],
        "transitions": [{"source": "F00", "target": "F01", "validation": pair}]}
    before = copy.deepcopy(metadata)
    projected = metadata_for_judging(metadata)
    assert projected["keyframes"][0]["visual_state"] == a
    assert projected["transitions"][0]["entity_comparison"] == comparison(changed=False)
    assert "old pouring sentinel" not in json.dumps(projected) and metadata == before
    with prompt_profile_scope("paper_v4"), pytest.raises(ValueError, match="missing structured"):
        admit_metadata(SimpleNamespace(config={"tasks": {"T4": {"prompt_profile": "paper_v4"}}}),
                       {"keyframes": [{"frame_id": "F00"}]})


@pytest.mark.parametrize("format_retry", [False, True])
@pytest.mark.parametrize("pair_retry", [False, True])
def test_t3_structured_observations_survive_stage_and_metadata(image_path, tmp_path, format_retry, pair_retry):
    from orchestrator.t3_structured import run_t3_structured
    from orchestrator.metadata_admission import build_metadata
    t2 = {"output": {"quality_passed": True, "keyframes": [
        {"img": str(image_path), "caption": "Unverified pot pouring.", "sec": i, "origin_index": i,
         "shot_id": "s", "chapter_id": "c"} for i in range(2)]}}
    original = copy.deepcopy(t2)
    prompts = []
    pair_prompts = []

    def vision(images, prompt, **kwargs):
        prompts.append(prompt)
        if "ORIGINAL CAPTION" in prompt:
            frame = "F00" if "\nFRAME: F00\n" in prompt else "F01"
            value = observation(frame, **({"spoon_region": "lower_center", "spoon_anchor": "E3", "spoon_relation": "above"} if frame == "F01" else {}))
            value.update(caption="The pot and spoon are visible beside food.", observations=["visible tools"],
                         uncertainties=["liquid source unresolved"], unsupported_original_claims=["pot pours"])
            if format_retry and len(prompts) == 1:
                value["entities"][1]["position"]["relation"] = "held_by"
            return json.dumps(value)
        assert '"frame_id": "F00"' in prompt and '"frame_id": "F01"' in prompt
        pair_prompts.append(prompt)
        value = comparison()
        if pair_retry and len(pair_prompts) == 1:
            value["transition"] = "Edits(1): 1) move metal pot" + SUFFIX
        return json.dumps(value)

    summary_prompts = []
    planner = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "paper", "prompt_profile": "paper_v4"}},
        "paths": {"debug_dir": str(tmp_path / "debug")}}, model_manager=None, _run_single_vlm_chat=vision,
        llm=SimpleNamespace(chat=lambda prompt, **kwargs: summary_prompts.append(prompt) or "A spoon moves relative to the food tray."))
    segment = SimpleNamespace(index=0, video_path="source.mp4", start_sec=0)
    result = run_t3_structured(planner, {"output": {}}, t2, segment)
    frames = result["output"]["frames"]
    assert t2 == original and frames[0]["visual_state"]["frame_id"] == "F00"
    assert len(frames[0]["caption_validation"]["attempts"]) == (2 if format_retry else 1)
    assert frames[0]["transition_validation"]["status"] == "valid"
    assert len(frames[0]["transition_validation"]["attempts"]) == (2 if pair_retry else 1)
    if pair_retry:
        assert "transition_changes_disagree" in pair_prompts[1]
    assert comparison()["changes"][0]["edit"] in summary_prompts[0]
    metadata = build_metadata({}, result["output"]["filtered_t2_output"], result["output"], segment)
    assert metadata_for_judging(metadata)["keyframes"][0]["visual_state"]["image_sha256"] == hashlib.sha256(image_path.read_bytes()).hexdigest()
