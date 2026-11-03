import json

import pytest

from executor.tasks.prompt_loader import PAPER_PROFILES, prompt_profile_scope

from orchestrator.paper_tasks import (
    TASKS,
    RouteEvidenceError,
    route_tasks,
    validate_route_evidence,
)


EVIDENCE_IDS = {
    "S_SEGEMENT_SUM",
    "F00_CAP",
    "F01_CAP",
    "F02_CAP",
    "P00_TRANS",
    "P00_REVIEW",
    "P01_TRANS",
    "P01_REVIEW",
}

EVIDENCE_TEXT = "\n".join(
    [
        "[S_SEGEMENT_SUM] A local object change occurs.",
        "[F00_CAP] Frame F00: the cup is on the left.",
        "[F01_CAP] Frame F01: the cup is on the right.",
        "[F02_CAP] Frame F02: the cup is beside the bowl.",
        "[P00_TRANS] Pair F00->F01: Edits(1): move the cup to the right | Shot: none | View: none | Subject: same.",
        "[P00_REVIEW] Reviewed pair P00: F00->F01 ok=true; edit_count=1; instruction=Move the cup right.",
        "[P01_TRANS] Pair F01->F02: Edits(1): move the cup beside the bowl | Shot: none | View: none | Subject: same.",
        "[P01_REVIEW] Reviewed pair P01: F01->F02 ok=true; edit_count=1; instruction=Move the cup beside the bowl.",
    ]
)


def routing_reply(chosen=None):
    return {"tasks": {task: {"quota": int(task in (chosen or {})), "reason": "visible support",
                              "evidence_refs": (chosen or {}).get(task, [])} for task in TASKS}, "edit_pairs": []}


def route_reply(reply, **kwargs):
    class LLM:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, **unused):
            self.calls.append(prompt)
            return json.dumps(reply)

    llm = LLM()
    result = route_tasks(llm, evidence_text=EVIDENCE_TEXT, evidence_ids=EVIDENCE_IDS,
                         limits={task: 1 for task in TASKS}, eligible={task: True for task in TASKS},
                         edit_pairs=[(0, 1), (1, 2)], **kwargs)
    return result, llm


@pytest.mark.parametrize("task,refs", [
    ("fine_grained_perception", ["F00_CAP"]),
    ("single_image_reasoning", ["F00_CAP"]),
    ("multi_image_logic", ["F00_CAP", "F01_CAP", "P00_TRANS"]),
    ("localized_editing", ["F00_CAP", "F01_CAP", "P00_TRANS", "P00_REVIEW"]),
    ("grounded_edit_to_select", ["F00_CAP", "F01_CAP", "P00_TRANS", "P00_REVIEW"]),
    ("understand_recall_generate_select", ["F00_CAP", "F01_CAP", "P00_TRANS"]),
    ("stepwise_sequence", ["F00_CAP", "F01_CAP", "P00_TRANS"]),
    ("middle_step_gap", ["F00_CAP", "F02_CAP", "P00_TRANS", "P01_TRANS"]),
    ("instructional_future_sequence", ["F00_CAP", "F01_CAP", "P01_TRANS"]),
])
def test_all_nine_task_closures_allow_optional_summary(task, refs):
    result = validate_route_evidence(task, refs, evidence_ids=EVIDENCE_IDS, evidence_text=EVIDENCE_TEXT,
                                     edit_pairs=[(0, 1), (1, 2)])
    assert result["valid"], result


@pytest.mark.parametrize("description", [
    "Edits(0): none", "Edits(1): 1) none", "Edits(1): no changes", "Edits(1): pan camera rightward",
    "Edits(1): zoom into the scene", "Edits(1): tilt camera upward", "malformed transition",
])
def test_temporal_routes_do_not_treat_id_as_semantic_proof(description):
    text = EVIDENCE_TEXT.replace("Edits(1): move the cup to the right", description)
    result = validate_route_evidence("stepwise_sequence", ["F00_CAP", "F01_CAP", "P00_TRANS"],
                                     evidence_ids=EVIDENCE_IDS, evidence_text=text)
    assert not result["valid"]
    assert result["validation_error"] == "transition_is_camera_only_or_no_change"


def test_static_joint_verification_needs_no_transition():
    text = EVIDENCE_TEXT.replace("Edits(1): move the cup to the right", "Edits(0): none")
    assert validate_route_evidence("multi_image_logic", ["F00_CAP", "F01_CAP"],
                                   evidence_ids=EVIDENCE_IDS, evidence_text=text)["valid"]


def test_unrelated_transition_cannot_support_temporal_frames():
    result = validate_route_evidence("stepwise_sequence", ["F00_CAP", "F01_CAP", "P01_TRANS"],
                                     evidence_ids=EVIDENCE_IDS, evidence_text=EVIDENCE_TEXT)
    assert result["validation_error"] == "transition_does_not_match_cited_frames"


def test_gap_requires_a_transition_path_between_before_and_after():
    result = validate_route_evidence("middle_step_gap", ["F00_CAP", "F02_CAP", "P00_TRANS"],
                                     evidence_ids=EVIDENCE_IDS, evidence_text=EVIDENCE_TEXT)
    assert result["validation_error"] == "missing_transition_path_between_gap_endpoints"


def test_two_context_states_are_not_a_future_target():
    refs = ["F00_CAP", "F01_CAP", "F00_STATE", "F01_STATE"]
    assert not validate_route_evidence("instructional_future_sequence", refs,
                                       evidence_ids=EVIDENCE_IDS | set(refs), evidence_text=EVIDENCE_TEXT)["valid"]
    refs.append("F02_STATE")
    assert validate_route_evidence("instructional_future_sequence", refs,
                                    evidence_ids=EVIDENCE_IDS | set(refs), evidence_text=EVIDENCE_TEXT)["valid"]


@pytest.mark.parametrize("timestamps", [(0, 0), (2, 1)])
def test_temporal_route_requires_increasing_source_times(timestamps):
    frames = [{"pos": i, "sec": t, "caption": "cup", "shot_id": "same"} for i, t in enumerate(timestamps)]
    result = validate_route_evidence("stepwise_sequence", ["F00_CAP", "F01_CAP", "P00_TRANS"],
                                     evidence_ids=EVIDENCE_IDS, evidence_text=EVIDENCE_TEXT, frames=frames)
    assert result["validation_error"] == "requires_two_distinct_ordered_frames"


def test_structured_pair_binding_wins_over_numeric_id_and_text():
    # Review numbering and transition numbering are independent.  Neither
    # numeric equality nor a pair mentioned in model prose establishes closure.
    units = [{"evidence_id": f"F{i:02d}_CAP", "text": "cup state"} for i in range(3)] + [
        {"evidence_id": "P00_TRANS", "pair_from_pos": 0, "pair_to_pos": 1,
         "text": "Pair F00->F01: Edits(1): move cup right | Shot: none | View: none | Subject: same"},
        {"evidence_id": "P01_TRANS", "pair_from_pos": 1, "pair_to_pos": 2,
         "text": "Pair F01->F02: Edits(1): move cup right | Shot: none | View: none | Subject: same"},
        {"evidence_id": "P00_REVIEW", "pair_from_pos": 1, "pair_to_pos": 2, "text": "Reviewed pair F00->F01"},
    ]
    refs = ["F01_CAP", "F02_CAP", "P00_REVIEW", "P01_TRANS"]
    assert validate_route_evidence("localized_editing", refs, evidence_ids=EVIDENCE_IDS,
                                   evidence_text=EVIDENCE_TEXT, evidence_units=units, edit_pairs=[(1, 2)])["valid"]
    refs[-1] = "P00_TRANS"
    assert not validate_route_evidence("localized_editing", refs, evidence_ids=EVIDENCE_IDS,
                                       evidence_text=EVIDENCE_TEXT, evidence_units=units, edit_pairs=[(1, 2)])["valid"]


def test_missing_pair_binding_and_unreviewed_pair_are_not_silently_admitted():
    refs = ["F00_CAP", "F01_CAP", "P00_TRANS", "P00_REVIEW"]
    assert not validate_route_evidence("localized_editing", refs, evidence_ids=EVIDENCE_IDS,
                                       evidence_units=[], edit_pairs=[(0, 1)])["valid"]
    assert not validate_route_evidence("localized_editing", refs, evidence_ids=EVIDENCE_IDS,
                                       evidence_text=EVIDENCE_TEXT, edit_pairs=[])["valid"]


def test_selected_reviewed_pair_still_must_match_route_citations():
    reply = routing_reply({"fine_grained_perception": ["F00_CAP"],
                           "localized_editing": ["F00_CAP", "F01_CAP", "P00_TRANS", "P00_REVIEW"]})
    reply["edit_pairs"] = [{"from": 1, "to": 2}]
    result, llm = route_reply(reply)
    assert len(llm.calls) == 1
    assert result["tasks"]["localized_editing"]["quota"] == 0
    assert result["tasks"]["fine_grained_perception"]["quota"] == 1


def test_all_positive_closures_invalid_is_hard_failure_but_all_zero_is_success():
    with pytest.raises(RouteEvidenceError, match="All positive") as captured:
        route_reply(routing_reply({"stepwise_sequence": ["F00_CAP"]}))
    assert len(captured.value.routing_attempts) == 2
    assert captured.value.tasks["stepwise_sequence"]["validation_error"]
    result, llm = route_reply(routing_reply())
    assert result["capabilities"] == []
    assert len(llm.calls) == 1


@pytest.mark.parametrize("refs", [None, "F00_CAP", [None], [True], [1], [{}], [["F00_CAP"]], ["F0_CAP"]])
def test_malformed_and_fuzzy_references_remain_hard_errors(refs):
    with pytest.raises(RouteEvidenceError):
        validate_route_evidence("fine_grained_perception", refs, evidence_ids=EVIDENCE_IDS)


def test_exact_existing_ids_take_priority_over_alias_normalization():
    ids = EVIDENCE_IDS | {"S_SUM", "DERIVED_F00->F01_TRANS"}
    result = validate_route_evidence("fine_grained_perception", [" F00_CAP ", "S_SUM", "DERIVED_F00->F01_TRANS"],
                                     evidence_ids=ids)
    assert result["normalized_refs"] == ["F00_CAP", "S_SUM", "DERIVED_F00->F01_TRANS"]
    assert result["raw_refs"][0] == " F00_CAP "


def test_multi_image_logic_accepts_concrete_frames_without_segment_summary():
    result = validate_route_evidence(
        "multi_image_logic",
        ["F00_CAP", "F01_CAP", "P00_TRANS"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
    )
    assert result["valid"] is True
    assert result["normalized_refs"] == ["F00_CAP", "F01_CAP", "P00_TRANS"]


def test_localized_editing_requires_review_transition_and_matching_endpoints():
    result = validate_route_evidence(
        "localized_editing",
        ["P00_REVIEW", "P00_TRANS", "F00_CAP", "F01_CAP"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
        edit_pairs=[(0, 1)],
    )
    assert result["valid"] is True

    missing_review = validate_route_evidence(
        "localized_editing",
        ["P00_TRANS", "F00_CAP", "F01_CAP"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
        edit_pairs=[(0, 1)],
    )
    assert missing_review["valid"] is False
    assert missing_review["validation_error"] == "missing_reviewed_pair_evidence"


def test_handcraft_reported_pair_uses_f08_f09_without_summary():
    refs = ["P00_TRANS", "P00_REVIEW", "F08_CAP", "F09_CAP"]
    text = EVIDENCE_TEXT.replace("F00", "F08").replace("F01", "F09")
    assert validate_route_evidence("localized_editing", refs, evidence_ids=set(refs),
                                   evidence_text=text, edit_pairs=[(8, 9)])["valid"]


def test_temporal_route_without_transition_is_task_level_rejection():
    result = validate_route_evidence(
        "stepwise_sequence",
        ["F00_CAP", "F01_CAP"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
    )
    assert result["valid"] is False
    assert result["validation_error"] == "missing_semantic_transition_evidence"


def test_tool_only_frame_hint_cannot_establish_a_visual_route():
    result = validate_route_evidence(
        "fine_grained_perception",
        ["F00_TOOL_OCR"],
        evidence_ids=EVIDENCE_IDS | {"F00_TOOL_OCR"},
        evidence_text=EVIDENCE_TEXT,
    )
    assert result["valid"] is False
    assert result["validation_error"] == "missing_frame_evidence"


def test_review_and_transition_from_different_pairs_are_rejected():
    result = validate_route_evidence(
        "localized_editing",
        ["P00_REVIEW", "P01_TRANS", "F00_CAP", "F01_CAP", "F02_CAP"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
        edit_pairs=[(0, 1), (1, 2)],
    )
    assert result["valid"] is False
    assert result["validation_error"] == "review_and_transition_reference_different_pairs"


def test_unknown_evidence_id_remains_hard_error():
    with pytest.raises(RouteEvidenceError, match="Unknown routing evidence"):
        validate_route_evidence(
            "fine_grained_perception",
            ["F99_CAP"],
            evidence_ids=EVIDENCE_IDS,
            evidence_text=EVIDENCE_TEXT,
        )


def test_explicit_legacy_evidence_alias_is_audited_without_fuzzy_mapping():
    result = validate_route_evidence(
        "fine_grained_perception",
        [" f00_image ", "S_SEGMENT_SUM"],
        evidence_ids=EVIDENCE_IDS,
        evidence_text=EVIDENCE_TEXT,
    )
    assert result["normalized_refs"] == ["F00_CAP", "S_SEGEMENT_SUM"]
    assert set(result["aliases_applied"]) == {"f00_image->F00_CAP", "S_SEGMENT_SUM->S_SEGEMENT_SUM"}


def test_task_level_downgrade_preserves_other_valid_routes_and_audit_fields():
    reply = {
        "tasks": {
            task: {
                "quota": 1 if task in {"fine_grained_perception", "localized_editing"} else 0,
                "reason": "supported" if task in {"fine_grained_perception", "localized_editing"} else "not selected",
                "evidence_refs": (
                    ["F00_CAP"] if task == "fine_grained_perception" else
                    ["P00_TRANS", "F00_CAP", "F01_CAP"] if task == "localized_editing" else []
                ),
            }
            for task in TASKS
        },
        "edit_pairs": [],
    }

    class LLM:
        def chat(self, prompt, max_new_tokens=0):
            return json.dumps(reply)

    result = route_tasks(
        LLM(),
        evidence_text=EVIDENCE_TEXT,
        evidence_ids=EVIDENCE_IDS,
        limits={task: 1 for task in TASKS},
        eligible={task: True for task in TASKS},
        edit_pairs=[(0, 1)],
    )
    assert result["tasks"]["fine_grained_perception"]["quota"] == 1
    editing = result["tasks"]["localized_editing"]
    assert editing["quota"] == 0
    assert editing["admission_reason"] == "insufficient_task_specific_evidence"
    assert editing["validation_error"] == "missing_reviewed_pair_evidence"
    assert result["capabilities"] == ["understanding"]
    assert "raw_refs" in editing and "normalized_refs" in editing


def test_explicit_paper_task_alias_is_accepted():
    rows = {
        task: {"quota": 0, "reason": "not selected", "evidence_refs": []}
        for task in TASKS
    }
    rows.pop("single_image_reasoning")
    rows["visual_reasoning"] = {
        "quota": 1,
        "reason": "visible frame supports a function inference",
        "evidence_refs": ["F00_CAP"],
    }
    reply = {"tasks": rows, "edit_pairs": []}

    class LLM:
        def chat(self, prompt, max_new_tokens=0):
            return json.dumps(reply)

    result = route_tasks(
        LLM(), evidence_text=EVIDENCE_TEXT, evidence_ids=EVIDENCE_IDS,
        limits={task: 1 for task in TASKS}, eligible={task: True for task in TASKS}, edit_pairs=[]
    )
    assert result["tasks"]["single_image_reasoning"]["quota"] == 1


@pytest.mark.parametrize("profile", sorted(PAPER_PROFILES))
def test_effective_routing_prompt_declares_task_specific_closure(profile):
    rows = {
        task: {"quota": 0, "reason": "not selected", "evidence_refs": []}
        for task in TASKS
    }
    reply = {"tasks": rows, "edit_pairs": []}

    class LLM:
        def __init__(self):
            self.prompt = ""

        def chat(self, prompt, max_new_tokens=0):
            self.prompt = prompt
            return json.dumps(reply)

    llm = LLM()
    with prompt_profile_scope(profile):
        route_tasks(
            llm, evidence_text=EVIDENCE_TEXT, evidence_ids=EVIDENCE_IDS,
            limits={task: 1 for task in TASKS}, eligible={task: True for task in TASKS}, edit_pairs=[]
        )
    assert "S_SEGEMENT_SUM is optional context" in llm.prompt
    assert "reviewed pair (P*_REVIEW)" in llm.prompt
    assert "unknown IDs and malformed JSON remain hard errors" in llm.prompt
    assert "Positive quota requires an existing S_* ID" not in llm.prompt

