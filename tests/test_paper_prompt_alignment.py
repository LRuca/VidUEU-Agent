from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image

from executor.models.api_budget import ApiBudget, ApiBudgetExhausted
from executor.tasks.prompt_loader import (
    load_prompt_text, render_prompt_text, prompt_profile_scope, loaded_prompt_sources,
)
from orchestrator.metadata_admission import admit_metadata, metadata_for_judging
from orchestrator.paper_validation import validate_single_image_item, review_paper_item
from orchestrator.prompt_audit import paper_prompt_stage, audited_text
from orchestrator.transition_validation import validate_transition, infer_transition_cause
from orchestrator.umm_task_producer import (
    TASK_SEED, TASK_WEAVE, TASK_COMM_C1, TASK_COMM_C2, TASK_ISG,
    build_task_context, build_seed_candidates, build_weave_candidates,
    build_comm_c1_candidates, build_comm_c2_candidates, build_isg_candidates,
    attach_evidence_refs, produce_task_samples,
)


class Reply:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def chat(self, prompt, **kwargs):
        self.calls.append(prompt)
        return json.dumps(self.value)


def test_prompt_profiles_keep_legacy_and_reject_missing_variables():
    old = load_prompt_text("t3_local", "pair_transition", profile="legacy")
    with prompt_profile_scope("paper_v1"):
        new = load_prompt_text("t3_local", "pair_transition")
        assert new != old
        source = loaded_prompt_sources()[0]
        assert source["template_sha256"] == hashlib.sha256(new.encode()).hexdigest()
        assert source["paper"]["pdf_pages"] == [26]
        with pytest.raises(KeyError):
            render_prompt_text("t3_local", "pair_transition", video_type="other")
    assert load_prompt_text("t3_local", "pair_transition") == old


def test_final_prompt_audit_binds_inputs_response_and_failure(tmp_path):
    planner = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "paper"}},
                                     "paths": {"prompt_audit_path": str(tmp_path / "audit.jsonl")}})

    @paper_prompt_stage("test")
    def stage(planner, segment):
        prompt = render_prompt_text("paper", "frame_caption", original_caption="hypothesis", frame_id="F00")
        audited_text(Reply({"caption": "visible"}), prompt)
        return {}

    result = stage(planner, SimpleNamespace(video_path="source.mp4", index=2, start_sec=30))
    row = json.loads((tmp_path / "audit.jsonl").read_text())
    assert row["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
    assert row["context"]["metadata_id"] and row["raw_response"]
    assert result["runtime"]["prompt_audit"]["call_ids"] == [row["call_id"]]


def test_budget_survives_instances_failures_and_immutable_limit(tmp_path):
    path = str(tmp_path / "budget.sqlite")
    first = ApiBudget(path, 2)
    identifier = first.reserve("hash1", "model")
    first.finish(identifier, "http_error", {"http_status": 402})
    second = ApiBudget(path, 2)
    second.reserve("hash2", "model")  # Crash before finish must still consume budget.
    with pytest.raises(ValueError, match="exhausted"):
        ApiBudget(path, 2).reserve("hash3", "model")
    with pytest.raises(ValueError, match="differs"):
        ApiBudget(path, 3)
    assert second.snapshot()["attempts_reserved"] == 2


def test_budget_atomic_concurrent_reservations(tmp_path):
    path = str(tmp_path / "concurrent.sqlite")
    ApiBudget(path, 3)

    def reserve(i):
        try:
            return ApiBudget(path, 3).reserve(str(i), "model")
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=6) as pool:
        identifiers = list(pool.map(reserve, range(9)))
    assert len({v for v in identifiers if v}) == 3
    assert ApiBudget(path, 3).snapshot()["attempts_reserved"] == 3


def test_t3_caption_verification_preserves_originals_and_failed_transition(tmp_path):
    from orchestrator.t3_structured import run_t3_structured
    frames = []
    for i in range(2):
        path = tmp_path / f"f{i}.jpg"
        Image.new("RGB", (16, 16), "red").save(path)
        frames.append({"img": str(path), "caption": "Liquid pours from the pot.", "sec": i,
                       "origin_index": i, "shot_id": "s", "chapter_id": "c"})
    t2 = {"output": {"quality_passed": True, "keyframes": frames}}
    original = copy.deepcopy(t2)

    def vision(images, prompt, **kwargs):
        if "ORIGINAL CAPTION" in prompt:
            return json.dumps({"caption": "A pot is held beside the tray.", "observations": ["held pot"],
                               "uncertainties": ["liquid source"], "unsupported_original_claims": ["pot pours"]})
        return "malformed transition"

    planner = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "paper"}},
                                      "paths": {"debug_dir": str(tmp_path / "debug")}},
                              model_manager=None, _run_single_vlm_chat=vision,
                              llm=SimpleNamespace(chat=lambda *a, **k: "A pot is held beside the tray."))
    result = run_t3_structured(planner, {"output": {}}, t2, SimpleNamespace(index=0, video_path="source.mp4"))
    assert t2 == original
    frame = result["output"]["frames"][0]
    assert frame["original_caption"] == "Liquid pours from the pot."
    assert frame["caption"] == "A pot is held beside the tray."
    assert frame["transition_validation"]["status"] == "invalid"
    assert frame["transition_to_next"] == ""


SUFFIX = " | Shot: none | View: none | Subject: same | Uncertainty: none"


@pytest.mark.parametrize("raw,status", [
    ("Edits(0): none" + SUFFIX, "no_change"),
    ("Edits(1): 1) none" + SUFFIX, "no_change"),
    ("Edits(2): 1) move cup from image-left to image-right" + SUFFIX, "invalid"),
    ("Edits(1): 1) move cup from image-left to image-right" + SUFFIX, "valid"),
    ("Edits(1): 1) pour sauce" + SUFFIX, "invalid"),
    ("Edits(0): none | Shot: scene_cut | View: none | Subject: different | Uncertainty: none", "invalid"),
])
def test_transition_status_and_direction(raw, status):
    result = validate_transition(raw)
    assert result["status"] == status and result["raw_response"] == raw
    if status == "valid":
        assert "from image-left to image-right" in result["text"]
    if status == "no_change":
        assert result["text"].startswith("Edits(0): none") and not result["edits"]
    if status == "invalid":
        assert not result["text"] and not result["edits"]


@pytest.mark.parametrize("edit,status", [
    ("move hand holding spoon down and slightly right", "valid"),
    ("move black and white cup leftward", "valid"),
    ("move small metal pot left and slightly upward", "valid"),
    ("move cup left and rotate it upright", "invalid"),
    ("move cup left then remove it", "invalid"),
])
def test_conjunctions_distinguish_one_direction_from_chained_verbs(edit, status):
    raw = "Edits(1): 1) " + edit + SUFFIX
    result = validate_transition(raw)
    assert result["status"] == status
    if status == "valid":
        assert result["text"] == raw and result["edits"] == [edit]


@pytest.mark.parametrize("edit,view,cause", [
    ("move spoon above the rectangular pan", "none", "object_motion"),
    ("move spoon above the rectangular pan", "minor", "object_motion"),
    ("rotate pot from a diagonal tilt toward upright", "none", "edit"),
    ("pan camera toward the spoon", "minor", "camera_motion"),
])
def test_cause_comes_from_edit_verbs_not_container_nouns(edit, view, cause):
    raw = "Edits(1): 1) " + edit + SUFFIX.replace("View: none", "View: " + view)
    assert infer_transition_cause(validate_transition(raw)) == cause


def test_failed_prompt_audit_redacts_client_key_and_has_no_semantic_verdict(tmp_path):
    client = SimpleNamespace(api_key="fake-secret-for-test")
    planner = SimpleNamespace(config={"paths": {"prompt_audit_path": str(tmp_path / "audit.jsonl")}})

    @paper_prompt_stage("test")
    def stage(planner, segment):
        from orchestrator.prompt_audit import audited_call
        def failed():
            raise TimeoutError("temporary fake-secret-for-test")
        return audited_call(failed, "bound prompt", client=client)

    with pytest.raises(TimeoutError):
        stage(planner, None)
    row = json.loads((tmp_path / "audit.jsonl").read_text())
    assert "fake-secret-for-test" not in row["error"]
    assert "[REDACTED]" in row["error"] and row["semantic_decision"] is None


def test_gate_c_call_failure_is_not_no_change_or_rejection():
    failure = validate_transition("", error=TimeoutError("timeout"))
    assert failure["status"] == "call_failed"
    planner = SimpleNamespace(config={}, _run_single_vlm_chat=lambda *a, **k: pytest.fail("should not call"))
    with pytest.raises(ValueError, match="incomplete"):
        admit_metadata(planner, {"transitions": [{"validation": failure}]})


def test_gate_c_uses_verified_claims_and_retains_original_hypotheses():
    metadata = {"metadata_id": "bound", "video": {"summary": "A pot is held beside the tray."},
                "chapters": [{"summary": "Old pouring claim"}], "keyframes": [{"frame_id": "F00",
                "caption": "A pot is held beside the tray.", "original_caption": "Old pouring claim",
                "quality_profile": {"caption": "Old pouring claim"},
                "caption_validation": {"observations": ["held pot"], "uncertainties": ["liquid source"]}}]}
    before = copy.deepcopy(metadata)
    reviewed = metadata_for_judging(metadata)
    assert "Old pouring claim" not in json.dumps(reviewed)
    assert metadata == before and reviewed["keyframes"][0]["caption"] == metadata["keyframes"][0]["caption"]


@pytest.mark.parametrize("change", [
    {"evidence_refs": ["F01_CAP"]}, {"image_pos": 1}, {"reasoning_category": None},
    {"gpt_answer": "This is one sentence."},
])
def test_single_image_scope_and_reasoning_schema(change):
    item = {"image_pos": 0, "evidence_refs": ["F00_CAP"], "reasoning_category": "function",
            "gpt_answer": "The long handle provides a grip. It separates the hand from the pan body."}
    item.update(change)
    with pytest.raises(ValueError):
        validate_single_image_item(item, "llava_bench", {"F00_CAP", "F01_CAP"}, 0)


def test_contradictory_visual_accept_is_held():
    reply = {"decision": "accept", "premise_supported": False, "answer_supported": True,
             "unique_answer": True, "options_exclusive": True, "visible_cues": ["held pot"],
             "unsupported_claims": ["flow from pot"], "independent_answer": "uncertain", "reason": "uncertain source"}
    planner = SimpleNamespace(config={}, _run_single_vlm_chat=lambda *a, **k: json.dumps(reply))
    result = review_paper_item(planner, {"question": "Pouring from the pot?"}, [], {"image_pos": 0})
    assert result["decision"] == "hold" and result["training_approved"] is False


def test_blind_review_hides_metadata_caption_and_mcq_key():
    calls = []
    response = {"decision": "accept", "premise_supported": True, "answer_supported": True,
                "unique_answer": True, "options_exclusive": True, "independent_answer": "B",
                "visible_cues": ["visible pot"], "unsupported_claims": [], "reason": "option B is supported"}
    planner = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "paper", "prompt_profile": "paper_v3"}}},
        _run_single_vlm_chat=lambda images, prompt, **k: calls.append(prompt) or json.dumps(response))
    item = {"benchmark": "mmt_bench", "question": "Which object?", "A": "bowl", "B": "pot", "C": "pan", "D": "cup",
            "_answer": "A", "caption": "stale-caption-sentinel", "meta": {"gate_c": "prior-approval-sentinel"}}
    result = review_paper_item(planner, item, [], {"single_image": True})
    assert result["decision"] == "hold" and result["answer_comparison"] == "independent_answer_does_not_match_candidate"
    assert all(s not in calls[0] for s in ["stale-caption-sentinel", "prior-approval-sentinel", '"_answer"'])


def test_choice_letter_cannot_pass_an_open_response_review():
    response = {"decision": "accept", "premise_supported": True, "answer_supported": True,
                "unique_answer": True, "options_exclusive": True, "independent_answer": "A",
                "visible_cues": ["held pot"], "unsupported_claims": [], "reason": "generic verdict"}
    planner = SimpleNamespace(config={}, _run_single_vlm_chat=lambda *a, **k: json.dumps(response))
    result = review_paper_item(planner, {"benchmark": "llava_bench", "question": "Why?"}, [], {})
    assert result["decision"] == "hold" and result["answer_comparison"] == "choice_letter_is_not_an_open_response_answer"


def bundle():
    frames = [{"pos": i, "origin_index": i, "img": f"f{i}.jpg", "caption": f"Cup state {i}",
               "shot_id": "s", "chapter_id": "c", "cause": "object_change"} for i in range(4)]
    pairs = [{"pair_from_pos": i, "pair_to_pos": i + 1,
              "transition_text": "Edits(1): 1) move cup from image-left to image-right" + SUFFIX,
              **({"review": {"ok": True, "edit_count": 1, "edits": ["move cup"], "instruction": "Move cup."}} if i == 0 else {})}
             for i in range(3)]
    units = [{"evidence_id": "S_SEGEMENT_SUM", "text": "The cup moves."}]
    units += [{"evidence_id": f"F{i:02d}_CAP", "text": f"Cup state {i}"} for i in range(4)]
    units += [{"evidence_id": f"P{i:02d}_TRANS", "text": row["transition_text"],
               "pair_from_pos": i, "pair_to_pos": i + 1} for i, row in enumerate(pairs)]
    units += [{"evidence_id": "P00_REVIEW", "text": "one local move", "pair_from_pos": 0, "pair_to_pos": 1}]
    return {"t3_out": {"task_profile": "paper", "segment_summary": "The cup moves."},
            "frames_aligned": frames, "pair_records": pairs, "evidence_units": units,
            "reviewed_ok_pairs": [{"from": 0, "to": 1}], "candidate_pairs": [{"from": 0, "to": 1}]}


def task_reply(task, ctx):
    builders = {TASK_SEED: build_seed_candidates, TASK_WEAVE: build_weave_candidates,
                TASK_COMM_C1: lambda c: build_comm_c1_candidates(c, k=3),
                TASK_COMM_C2: build_comm_c2_candidates, TASK_ISG: lambda c: build_isg_candidates(c, k=2)}
    candidate = builders[task](ctx)[0]
    targets = ([candidate.to_pos] if task in {TASK_SEED, TASK_WEAVE} else
               candidate.frame_positions[1:] if task == TASK_COMM_C1 else
               [candidate.frame_positions[1]] if task == TASK_COMM_C2 else candidate.future_positions)
    selects = [{"question": "Which image matches this target?", "answer": f"target:F{p:02d}",
                "rationale": "The cup occupies the observed target position."} for p in targets]
    rows = {
        TASK_SEED: {"plan": {"goal": "Move the cup rightward", "invariants": ["same table"], "edits": ["move cup"]},
                    "gen": {"summary": "A cup at image-right"}, "select": selects[0]},
        TASK_WEAVE: {"understand": {"facts": ["earlier cup left", "later cup right"]},
                     "plan": {"edits": ["move cup"]}, "qa_open": {"question": "Where was the cup in the later image?",
                     "answer": "At image-right", "rubric": ["image-right"]}, "select": selects[0]},
        TASK_COMM_C1: {"question": "Follow the cup positions.", "overall": "The cup moves across the table.",
                       "step_texts": [f"Cup state {p}" for p in targets], "select": selects},
        TASK_COMM_C2: {"question": "What state connects these endpoints?", "explanation": "The cup is at the middle position.",
                       "select": selects[0]},
        TASK_ISG: {"overall": "Continue the cup's observed motion.", "edits": ["move cup"] * len(targets),
                   "because": ["the adjacent position changes"] * len(targets),
                   "so_that": ["cup at the target position"] * len(targets), "check": ["cup position"] * len(targets),
                   "select": selects},
    }
    obj = rows[task]
    obj["evidence_refs"] = attach_evidence_refs(ctx, task, candidate).evidence_ids
    return obj


@pytest.mark.parametrize("task", [TASK_SEED, TASK_WEAVE, TASK_COMM_C1, TASK_COMM_C2, TASK_ISG])
def test_five_strict_producers_preserve_model_text_and_citations(task):
    data = bundle()
    obj = task_reply(task, build_task_context(data))
    out = produce_task_samples(data, llm=Reply(obj), selected_tasks=[task], strict_generation=True)
    assert len(out["samples"][task]) == 1, out["drops"][task]
    row = out["samples"][task][0]
    assert row["evidence_refs"] == obj["evidence_refs"]
    assert row["meta"]["citation_origin"] == "model"
    assert json.loads(row["meta"]["raw_generation_response"]) == obj
    if task == TASK_SEED:
        assert row["turns"]["plan"]["invariants"] == ["same table"]
    if task == TASK_WEAVE:
        assert row["turns"]["qa_open"] == obj["qa_open"]
    if task == TASK_ISG:
        assert row["turns"]["overall"] == obj["overall"]


@pytest.mark.parametrize("problem", ["unknown_ref", "missing_endpoint", "wrong_target", "missing_nested", "wrong_length"])
def test_strict_generation_withholds_without_repairs(problem):
    data = bundle()
    task = TASK_ISG if problem == "wrong_length" else TASK_SEED
    obj = task_reply(task, build_task_context(data))
    if problem == "unknown_ref": obj["evidence_refs"].append("F99_CAP")
    if problem == "missing_endpoint": obj["evidence_refs"].remove("F01_CAP")
    if problem == "wrong_target": obj["select"]["answer"] = "target:F00"
    if problem == "missing_nested": obj["plan"].pop("invariants")
    if problem == "wrong_length": obj["edits"].pop()
    out = produce_task_samples(data, llm=Reply(obj), selected_tasks=[task], strict_generation=True)
    assert not out["samples"][task] and out["drops"][task]


@pytest.mark.parametrize("error", [TimeoutError("transient"), ApiBudgetExhausted("exhausted")])
def test_generation_call_problem_remains_pending_without_semantic_drop(error):
    class PendingLLM:
        def chat(self, prompt, **kwargs):
            raise error
    out = produce_task_samples(bundle(), llm=PendingLLM(), selected_tasks=[TASK_SEED], strict_generation=True)
    assert not out["samples"][TASK_SEED] and not out["drops"][TASK_SEED]
    pending = out["pending"][TASK_SEED][0]
    assert pending["status"] == "pending" and pending["semantic_decision"] is None
    assert pending["error_type"] == type(error).__name__
