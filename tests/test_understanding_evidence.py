import copy
import json
from types import SimpleNamespace

import pytest

from executor.tasks.prompt_loader import prompt_profile_scope
from orchestrator.paper_validation import validate_single_image_item, review_paper_item, preserve_mmt_verifier_source


def complete_reasoning():
    answer = "The strainer separates solids from liquid. Its mesh retains the pieces while the bowl catches passing liquid."
    return {"image_pos": 0, "reasoning_category": "function", "gpt_answer": answer, "evidence_refs": ["F00_CAP"],
        "reasoning_evidence": {"conclusion": "The strainer separates solids from liquid.",
            "visible_cues": [{"claim": "A mesh strainer holds pieces above a bowl containing liquid.", "evidence_refs": ["F00_CAP"]}],
            "inference_link": "Mesh openings allow liquid to pass while retaining larger solid pieces.",
            "answer_claims": [
        {"claim": "The strainer separates solids from liquid.", "support": "inferred", "evidence_refs": ["F00_CAP"]},
        {"claim": "Its mesh retains the pieces while the bowl catches passing liquid.", "support": "inferred", "evidence_refs": ["F00_CAP"]}]}}


def perception():
    return {"image_pos": 0, "A": "pot", "B": "board", "C": "vase", "D": "cloth", "_answer": "A", "evidence_refs": ["F00_CAP"],
        "perception_evidence": {"premise_cues": [{"claim": "A metal container with a handle is held.", "evidence_refs": ["F00_CAP"]}],
            "option_checks": {k: {"status": "supported" if k == "A" else "excluded", "reason": reason,
                                  "evidence_refs": ["F00_CAP"]} for k, reason in {
                "A": "The held object has a pot body and handle.", "B": "The held object is not a flat wooden surface.",
                "C": "The held object has a cooking handle rather than a vase neck.", "D": "The held object is rigid metal rather than fabric."}.items()}}}


def test_reasoning_and_option_evidence_preserve_author_claims():
    with prompt_profile_scope("paper_v4"):
        for item, bench in [(complete_reasoning(), "llava_bench"), (perception(), "mmt_bench")]:
            before = copy.deepcopy(item)
            validate_single_image_item(item, bench, {"F00_CAP"}, 0)
            assert item == before


@pytest.mark.parametrize("mutation,reason", [
    ("description", "description_repeated_as_inference"),
    ("extra_sentence", "answer_claims_do_not_cover_every_sentence"),
    ("summary", "summary_cannot_establish_single_image_claim"),
    ("other_image", "missing_frame_citation"),
    ("all_visible", "reasoning_has_no_inferred_claim"),
])
def test_incomplete_or_misbound_reasoning_contract_is_rejected(mutation, reason):
    item = complete_reasoning()
    contract = item["reasoning_evidence"]
    if mutation == "description": contract["conclusion"] = contract["visible_cues"][0]["claim"]
    elif mutation == "extra_sentence": item["gpt_answer"] += " It is preparing food for later cooking."
    elif mutation == "summary": contract["answer_claims"][0]["evidence_refs"].append("S_SEGEMENT_SUM")
    elif mutation == "other_image": contract["visible_cues"][0]["evidence_refs"] = ["F01_CAP"]
    else:
        for claim in contract["answer_claims"]: claim["support"] = "visible"
    with prompt_profile_scope("paper_v4"), pytest.raises(ValueError, match=reason):
        validate_single_image_item(item, "llava_bench", {"F00_CAP", "F01_CAP", "S_SEGEMENT_SUM"}, 0)


def test_second_supported_or_uncertain_option_is_not_unique():
    for status in ["supported", "uncertain"]:
        item = perception()
        item["perception_evidence"]["option_checks"]["B"]["status"] = status
        with prompt_profile_scope("paper_v4"), pytest.raises(ValueError, match="ambiguous"):
            validate_single_image_item(item, "mmt_bench", {"F00_CAP"}, 0)


@pytest.mark.parametrize("tool", ["WHISPER", "RAFT", "DINO"])
def test_tool_prefix_does_not_prove_same_image_visible_claim(tool):
    item = complete_reasoning()
    reference = "F00_TOOL_" + tool
    item["reasoning_evidence"]["visible_cues"][0]["evidence_refs"] = [reference]
    with prompt_profile_scope("paper_v4"), pytest.raises(ValueError, match="auxiliary_signal"):
        validate_single_image_item(item, "llava_bench", {"F00_CAP", reference}, 0)


def test_explicit_same_image_citations_preserve_the_model_contract():
    item = perception()
    contract = item["perception_evidence"]
    for row in contract["premise_cues"] + list(contract["option_checks"].values()):
        row["same_image_evidence_refs"] = row.pop("evidence_refs")
    before = copy.deepcopy(item)
    with prompt_profile_scope("paper_v4"):
        validate_single_image_item(item, "mmt_bench", {"F00_CAP"}, 0)
    assert item == before
    contract["premise_cues"][0]["evidence_refs"] = ["F01_CAP"]
    with prompt_profile_scope("paper_v4"), pytest.raises(ValueError, match="conflicting_same_image"):
        validate_single_image_item(item, "mmt_bench", {"F00_CAP", "F01_CAP"}, 0)


def test_verifier_acknowledgment_keeps_author_evidence_and_rejects_rewrites():
    source = perception()
    source["question"] = "What is the held object?"
    fixed = {k: v for k, v in source.items() if k != "perception_evidence"}
    before = copy.deepcopy(source)
    assert preserve_mmt_verifier_source(source, fixed) is source
    assert source == before and "perception_evidence" in source
    for field in ["question", "A", "_answer", "evidence_refs"]:
        changed = copy.deepcopy(fixed)
        changed[field] = "different"
        with pytest.raises(ValueError, match="verifier_changed"):
            preserve_mmt_verifier_source(source, changed)
    changed = copy.deepcopy(fixed)
    changed["perception_evidence"] = {}
    with pytest.raises(ValueError, match="author_evidence_contract"):
        preserve_mmt_verifier_source(source, changed)


def test_independent_reasoning_review_must_judge_substantive_inference():
    reply = {"decision": "accept", "premise_supported": True, "answer_supported": True, "unique_answer": True,
        "options_exclusive": True, "independent_answer": "The chef holds a pot.", "visible_cues": ["held pot"],
        "unsupported_claims": [], "reason": "visible action", "task_type_supported": True,
        "inference_check": {"conclusion": "The chef holds a pot.", "visible_cues": ["held pot"],
            "inference_link": "This directly describes the observed action.", "substantive_inference": False}}
    prompts = []
    planner = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "paper", "prompt_profile": "paper_v4"}}},
        _run_single_vlm_chat=lambda images, prompt, **kwargs: prompts.append(prompt) or json.dumps(reply))
    result = review_paper_item(planner, {"benchmark": "llava_bench", "gpt_answer": "held pot", "meta": {"old": "approval sentinel"}}, [], {})
    assert result["decision"] == "hold" and result["answer_comparison"] == "direct_description_is_not_reasoning"
    assert "approval sentinel" not in prompts[0] and result["training_approved"] is False
