import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from orchestrator.metadata_admission import (
    build_metadata,
    admit_metadata,
    admit_metadata_with_recovery,
    filter_signals,
)
from orchestrator.paper_tasks import TASKS, route_tasks
from orchestrator.paper_render import render_paper_unified
from orchestrator.umm_task_producer import (build_task_context, build_comm_c2_candidates, produce_task_samples,
                                          build_weave_candidates, build_comm_c1_candidates, build_isg_candidates)
from executor.tasks.representative_selector import RepresentativeSelector


class ReplyLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, prompt, max_new_tokens=0):
        self.calls.append(prompt)
        return json.dumps(self.reply)


def route_reply(chosen):
    return {"tasks": {task: {"quota": 1 if task in chosen else 0, "reason": "visible support" if task in chosen else "insufficient support",
                            "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"] if task in chosen else []} for task in TASKS}, "edit_pairs": []}


def test_static_metadata_routes_understanding_without_edit_pairs():
    llm = ReplyLLM(route_reply(["fine_grained_perception"]))
    route = route_tasks(llm, evidence_text="flowers", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                        limits={t: 2 for t in TASKS}, eligible={t: t == "fine_grained_perception" for t in TASKS}, edit_pairs=[])
    assert route["capabilities"] == ["understanding"]
    assert route["tasks"]["fine_grained_perception"]["quota"] == 1
    assert len(route["tasks"]) == 9


def test_route_reasks_once_and_preserves_invalid_response():
    valid = route_reply(["fine_grained_perception"])
    invalid = copy.deepcopy(valid)
    invalid["tasks"]["fine_grained_perception"]["evidence_refs"] = ["F99_CAP"]

    class SequenceLLM:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, max_new_tokens=0):
            self.calls.append(prompt)
            return json.dumps(invalid if len(self.calls) == 1 else valid)

    llm = SequenceLLM()
    route = route_tasks(llm, evidence_text="flowers", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                        limits={t: 2 for t in TASKS}, eligible={t: True for t in TASKS}, edit_pairs=[])
    assert len(llm.calls) == 2
    assert "failed validation" in llm.calls[1]
    assert route["attempts"][0]["validation_error"]
    assert route["attempts"][1]["validation_error"] is None
    assert json.loads(route["attempts"][0]["raw"]) == invalid


def test_route_stops_after_two_invalid_responses():
    llm = ReplyLLM({"tasks": {}})
    with pytest.raises(ValueError) as captured:
        route_tasks(llm, evidence_text="flowers", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                    limits={t: 2 for t in TASKS}, eligible={t: True for t in TASKS}, edit_pairs=[])
    assert len(llm.calls) == 2
    assert len(captured.value.routing_attempts) == 2


@pytest.mark.parametrize("defect", ["invalid_json", "missing_task", "invented_reference", "unreviewed_pair"])
def test_bad_route_cannot_admit_default_tasks(defect):
    reply = route_reply(["fine_grained_perception"])
    if defect == "invalid_json": reply = {}
    elif defect == "missing_task": reply["tasks"].pop("localized_editing")
    elif defect == "invented_reference": reply["tasks"]["fine_grained_perception"]["evidence_refs"].append("F99_CAP")
    else: reply["edit_pairs"] = [{"from": 0, "to": 99}]
    with pytest.raises(ValueError):
        route_tasks(ReplyLLM(reply), evidence_text="facts", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                    limits={t: 2 for t in TASKS}, eligible={t: True for t in TASKS}, edit_pairs=[])


def test_missing_segment_summary_is_allowed_for_frame_grounded_route():
    reply = route_reply(["fine_grained_perception"])
    reply["tasks"]["fine_grained_perception"]["evidence_refs"] = ["F00_CAP"]
    route = route_tasks(ReplyLLM(reply), evidence_text="facts", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                        limits={t: 2 for t in TASKS}, eligible={t: True for t in TASKS}, edit_pairs=[])
    item = route["tasks"]["fine_grained_perception"]
    assert item["quota"] == 1
    assert item["admission_reason"] == "model_decision"
    assert item["validation_error"] is None


def test_signal_filter_keeps_exact_ocr_and_does_not_mutate_source():
    ocr = "Step 2:\nAdd 10 mL"
    t2 = {"keyframes": [{"origin_index": 0, "shot_id": "s1", "semantic_profile": {"tool_descs": {"ocr": ocr, "yolo": "unrelated car"}},
                         "tool_signals": {"tools": {"ocr": {"text": ocr}, "yolo": {"text": "car"}}}}]}
    before = copy.deepcopy(t2)
    llm = ReplyLLM({"retained": [{"id": "F00_TOOL_OCR", "reason": "visible instruction"}],
                   "removed": [{"id": "F00_TOOL_YOLO", "reason": "unsupported"}]})
    out, audit = filter_signals(SimpleNamespace(llm=llm), {"shots": []}, t2)
    assert t2 == before
    assert out["keyframes"][0]["semantic_profile"]["tool_descs"] == {"ocr": ocr}
    assert "yolo" not in out["tool_execution"]["semantic_by_shot"]["s1"]
    assert audit["signals"][0]["signal"] == ocr


def test_gate_c_reject_is_distinct_from_transport_or_parsing_error():
    metadata = {"metadata_id": "x", "keyframes": []}
    p = SimpleNamespace(_run_single_vlm_chat=lambda *a, **k: json.dumps({"decision": "reject", "summary_consistent": False,
                                "primary_reasons": ["contradiction", "sparse evidence"]}))
    assert admit_metadata(p, metadata)["decision"] == "reject"
    p._run_single_vlm_chat = lambda *a, **k: "not JSON"
    with pytest.raises(ValueError): admit_metadata(p, metadata)


def test_gate_c_recovery_reasks_truncated_response_and_keeps_acceptance():
    replies = [
        "```json\\n{\\\"decision\\\": \\\"accept\\\",\\\"primary_reasons\\\":[\\\"truncated",
        json.dumps({
            "decision": "accept",
            "primary_reasons": ["Visible content is grounded", "The segment is coherent"],
            "summary_consistent": True,
        }),
    ]

    class Planner:
        def __init__(self):
            self.calls = 0

        def _run_single_vlm_chat(self, *args, **kwargs):
            reply = replies[self.calls]
            self.calls += 1
            return reply

    planner = Planner()
    gate = admit_metadata_with_recovery(planner, {"metadata_id": "recoverable", "keyframes": []})
    assert gate["decision"] == "accept"
    assert gate["recovery"]["retried"] is True
    assert planner.calls == 2


def test_gate_c_recovery_holds_after_two_contract_failures():
    class Planner:
        def __init__(self):
            self.calls = 0

        def _run_single_vlm_chat(self, *args, **kwargs):
            self.calls += 1
            return "```json\\n{\\\"decision\\\": \\\"accept\\\",\\\"primary_reasons\\\":[\\\"cut"

    planner = Planner()
    gate = admit_metadata_with_recovery(planner, {"metadata_id": "held", "keyframes": []})
    assert gate["decision"] == "hold"
    assert gate["status"] == "execution_pending"
    assert gate["gate_c_error"]["attempts"] == 2
    assert planner.calls == 2


@pytest.mark.parametrize("decision", ["accept", "reject"])
def test_gate_c_preserves_all_reasons_without_reasking(decision):
    reasons = [f"Visible evidence {i}" for i in range(5)]
    calls = []

    def reply(*args, **kwargs):
        calls.append(args)
        return json.dumps({"decision": decision, "primary_reasons": reasons,
                           "summary_consistent": True})

    result = admit_metadata(SimpleNamespace(_run_single_vlm_chat=reply),
                            {"metadata_id": "x", "keyframes": []})
    assert result["decision"] == decision
    assert result["primary_reasons"] == reasons
    assert len(calls) == 1


@pytest.mark.parametrize("reasons", [None, [], "evidence", ["ok", " "], [3]])
def test_gate_c_rejects_missing_or_invalid_reasons(reasons):
    reply = {"decision": "accept", "primary_reasons": reasons, "summary_consistent": True}
    planner = SimpleNamespace(_run_single_vlm_chat=lambda *a, **k: json.dumps(reply))
    with pytest.raises(ValueError, match="nonempty list of nonblank reason strings"):
        admit_metadata(planner, {"metadata_id": "x", "keyframes": []})


@pytest.mark.parametrize("summary", [None, "true", 1])
def test_gate_c_requires_explicit_boolean_summary_check(summary):
    reply = {"decision": "accept", "primary_reasons": ["visible content"]}
    if summary is not None:
        reply["summary_consistent"] = summary
    planner = SimpleNamespace(_run_single_vlm_chat=lambda *a, **k: json.dumps(reply))
    with pytest.raises(ValueError, match="boolean summary_consistent"):
        admit_metadata(planner, {"metadata_id": "x", "keyframes": []})


def test_metadata_preserves_frame_identity_and_source_time():
    seg = SimpleNamespace(video_path="source.mp4", segment_path="seg.mp4", index=1, start_sec=30, end_sec=60)
    t2 = {"keyframes": [{"shot_id": "a", "origin_index": 5, "semantic_profile": {"tool_descs": {"ocr": "EXACT"}}}]}
    t3 = {"frames": [{"shot_id": "a", "origin_index": 5, "sec": 2.5, "img": "a.jpg", "caption": "visible label"}]}
    m = build_metadata({}, t2, t3, seg)
    assert m["keyframes"][0]["source_sec"] == 32.5
    assert m["keyframes"][0]["semantic_profile"]["tool_descs"]["ocr"] == "EXACT"
    assert m["video"]["summary_scope"] == "segment"
    assert build_metadata({}, t2, t3, seg)["metadata_id"] == m["metadata_id"]


def unified_item(benchmark, identifier="sample"):
    return {"module": "unify", "benchmark": benchmark, "unify_id": identifier,
            "turns": {"plan": {"goal": "target", "invariants": ["table"], "edits": ["move cup"]},
                      "understand": {"facts": ["earlier cup", "later cup"]},
                      "qa_open": {"question": "Where was the later cup?", "answer": "near the bowl", "rubric": ["bowl"]},
                      "overall": "cup moves", "steps": ["move cup", "put down spoon"],
                      "explanation": "middle joins two states", "edits": ["move cup", "move spoon"],
                      "because": ["visible move", "visible release"], "so_that": ["cup near bowl", "spoon on table"],
                      "check": ["cup position", "spoon position"], "select": {}},
            "answer": {"target_frame_pos": 1, "target_frame_positions": [1, 2], "future_frame_positions": [2, 3]},
            "meta": {"frame_positions": [0, 1, 2, 3], "segment_keyframes": [{"pos": i, "img": f"f{i}.jpg"} for i in range(4)]}}


def test_weave_round_inputs_and_recall_order():
    result = render_paper_unified(unified_item("unify.weave.uie_select_recall"))
    rounds = result["turns"]
    assert [r["t"] for r in rounds] == ["understand", "plan", "qa_open", "gen", "select_step1"]
    assert [r["in"]["frame_positions"] for r in rounds[:4]] == [[0, 1], [0, 1], [0], [0]]
    assert result["context_mode"] == "ordered_dialogue"


@pytest.mark.parametrize("benchmark,expected", [("unify.seed.l3.gts", ["plan", "gen", "select_step1"]),
                                                ("unify.comm.task4.c2_fill_gap", ["explain", "gen", "select_step1"])])
def test_paper_rounds_are_separate(benchmark, expected):
    item = unified_item(benchmark)
    result = render_paper_unified(item)
    assert [r["t"] for r in result["turns"]] == expected
    assert result["turns"][0]["in"]["frame_positions"] == ([0, 1] if "seed" in benchmark else [0, 2])


def test_candidate_shuffle_answer_binding_and_distribution():
    labels = set()
    for i in range(24):
        item = unified_item("unify.seed.l3.gts", str(i))
        a = render_paper_unified(item)
        assert a == render_paper_unified(item)
        choice = a["eval"]["answer"]["step1"]
        labels.add(choice)
        selected = next(c for c in a["eval"]["candidates"]["step1"] if c["id"] == choice)
        assert selected["image"] == "f1.jpg"
    assert len(labels) >= 3


def test_selection_uses_raw_paper_value_weights_and_honors_zero():
    selector = RepresentativeSelector({"tasks": {"T2_representative_selection": {"paper_formula": True, "evidence_bonus_scale": 0,
                "tool_signal_scale": 0, "text_fusion_weight": 0, "enable_text_fusion": False, "time_decay_tau_sec": None}}})
    assert selector.evidence_bonus_scale == 0
    np.testing.assert_allclose(selector._build_frame_weights([{"value_score": x} for x in [.2, .5, .8]]), [.2, .5, .8])


def test_missing_middle_image_fails_before_binding_transition(tmp_path):
    from orchestrator.t3_structured import run_t3_structured
    first = tmp_path / "first.jpg"
    Image.new("RGB", (8, 8)).save(first)
    p = SimpleNamespace(config={"tasks": {"T4": {"task_profile": "legacy"}}}, model_manager=None)
    with pytest.raises(RuntimeError, match="preserve image/caption bindings"):
        run_t3_structured(p, {"output": {}}, {"output": {"quality_passed": True, "keyframes": [{"img": str(first)},
                         {"img": str(tmp_path / "missing.jpg")}]}}, SimpleNamespace(index=0))


def test_temporal_gap_is_eligible_without_reviewed_local_edit():
    frames = [{"pos": i, "shot_id": "s1", "caption": f"cup state {i}", "cause": "object_change"} for i in range(3)]
    pairs = [{"pair_from_pos": i, "pair_to_pos": i+1,
              "transition_text": "Edits(1): move cup to the center | Shot: none | View: none | Subject: same"} for i in range(2)]
    bundle = {"frames_aligned": frames, "pair_records": pairs, "t3_out": {"task_profile": "paper"},
              "evidence_units": [{"evidence_id": f"F{i:02d}_CAP", "text": f"cup state {i}"} for i in range(3)] +
                                [{"evidence_id": f"P{i:02d}_TRANS", "pair_from_pos": i, "pair_to_pos": i+1} for i in range(2)]}
    assert build_comm_c2_candidates(build_task_context(bundle))
    frames[0]["cause"] = "camera_motion"
    assert not build_comm_c2_candidates(build_task_context(bundle))


@pytest.mark.parametrize("edit,eligible", [
    ("move spoon above the rectangular pan", True),
    ("rotate pot from a diagonal tilt toward upright", True),
    ("pan camera toward the tray", False),
])
def test_temporal_eligibility_distinguishes_object_names_from_camera_verbs(edit, eligible):
    frames = [{"pos": i, "shot_id": "s1", "caption": f"tool state {i}", "cause": "object_motion"} for i in range(3)]
    pairs = [{"pair_from_pos": i, "pair_to_pos": i + 1,
              "transition_text": "Edits(1): 1) " + edit + " | Shot: none | View: minor | Subject: same | Uncertainty: none"}
             for i in range(2)]
    bundle = {"frames_aligned": frames, "pair_records": pairs, "t3_out": {"task_profile": "paper"},
              "evidence_units": [{"evidence_id": f"F{i:02d}_CAP", "text": f"tool state {i}"} for i in range(3)] +
                                [{"evidence_id": f"P{i:02d}_TRANS", "pair_from_pos": i, "pair_to_pos": i + 1} for i in range(2)]}
    assert bool(build_comm_c2_candidates(build_task_context(bundle))) is eligible


@pytest.mark.parametrize("description", ["1) none", "no changes", "(none)"])
def test_no_change_placeholder_never_admits_temporal_tasks(description):
    frames = [{"pos": i, "shot_id": "s1", "caption": "same cooking action", "cause": "unknown"} for i in range(3)]
    pairs = [{"pair_from_pos": i, "pair_to_pos": i + 1,
              "transition_text": f"Edits(1): {description} | Shot: none | View: none | Subject: same"} for i in range(2)]
    bundle = {"frames_aligned": frames, "pair_records": pairs, "t3_out": {"task_profile": "paper"},
              "evidence_units": [{"evidence_id": f"F{i:02d}_CAP", "text": "same cooking action"} for i in range(3)] +
                                [{"evidence_id": f"P{i:02d}_TRANS", "pair_from_pos": i, "pair_to_pos": i + 1} for i in range(2)]}
    ctx = build_task_context(bundle)
    assert not build_weave_candidates(ctx)
    assert not build_comm_c1_candidates(ctx)
    assert not build_comm_c2_candidates(ctx)
    assert not build_isg_candidates(ctx)


def test_ineligible_requested_task_is_withheld_and_audited():
    reply = route_reply(["fine_grained_perception", "stepwise_sequence"])
    reply["tasks"]["stepwise_sequence"]["evidence_refs"] = ["F00_CAP"]
    llm = ReplyLLM(reply)
    route = route_tasks(llm, evidence_text="static scene", evidence_ids={"S_SEGEMENT_SUM", "F00_CAP"},
                        limits={t: 2 for t in TASKS}, eligible={t: t == "fine_grained_perception" for t in TASKS}, edit_pairs=[])
    task = route["tasks"]["stepwise_sequence"]
    assert task["requested_quota"] == 1 and task["quota"] == 0
    assert task["admission_reason"] == "ineligible"
    assert route["capabilities"] == ["understanding"]
    assert len(llm.calls) == 1


def test_isg_future_inputs_exclude_current_and_later_targets():
    item = unified_item("unify.isg.i2.seq_select")
    item["answer"].pop("target_frame_positions")
    item["answer"].pop("target_frame_pos")
    item["turns"]["select"] = [{}, {}]
    result = render_paper_unified(item)
    assert [r["t"] for r in result["turns"][:5]] == ["overall", "edit", "reason", "target", "check"]
    gens = [r for r in result["turns"] if r["t"].startswith("gen")]
    assert [r["in"]["frame_positions"] for r in gens] == [[0, 1], [0, 1, 2]]
    assert [r["gt"]["image"] for r in gens] == ["f2.jpg", "f3.jpg"]


def test_t4_automatically_generates_understanding_without_edit(tmp_path):
    from dataset.builder import DatasetBuilder
    from orchestrator.t4_structured import run_t4_structured
    config = {"paths": {"debug_dir": str(tmp_path / "debug")}, "dataset": {"output_file": str(tmp_path / "data.jsonl")},
              "tasks": {"T4": {"task_profile": "paper", "max_llava": 1, "max_mmt": 0, "max_mme": 0,
                        "max_magicbrush": 0, "max_dialog_visdial": 0, "max_dialog_mmdu": 0,
                        "enable_llm_decision": True, "enable_llm_generation": True}}}

    class PipelineLLM:
        def chat(self, prompt, max_new_tokens=0):
            if "NINE paper task" in prompt:
                return json.dumps(route_reply(["single_image_reasoning"]))
            if "Target benchmark: llava_bench" in prompt:
                return json.dumps({"items": [{"image_pos": 0, "category": "complex", "reasoning_category": "function", "question": "Why is the pan handle extended?",
                     "gpt_answer": "The handle provides a grip away from the pan body. Its long shape lets the cook hold and move the pan.",
                     "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"]}]})
            raise AssertionError(prompt[:120])

    p = SimpleNamespace(config=config, llm=PipelineLLM(), model_manager=None, _video_id=lambda x: "source",
                        _relativize_item_paths_inplace=lambda x: x, _global_bench_counts={}, _global_llava_cat_counts={})
    p.dataset_builder = DatasetBuilder(config)
    segment = SimpleNamespace(index=0, video_path="source.mp4", segment_path="seg.mp4")
    result = run_t4_structured(p, {"output": {"global_summary": "A cook holds a pan."}},
                              {"output": {"keyframes": [{"img": "pan.jpg", "sec": 0, "origin_index": 0}]}},
                              {"output": {"segment_summary": "A cook holds a pan by its handle.", "frames": [
                                  {"img": "pan.jpg", "sec": 0, "origin_index": 0, "caption": "A pan has a long handle."}]}}, segment)
    p.dataset_builder.close()
    assert result["decision"]["selected"] == ["understanding"]
    row = result["instances"]["understanding.llava_bench"][0]
    assert row["capability"] == "understanding"
    assert row["task_type"] == "single_image_reasoning"


@pytest.mark.parametrize("all_zero", [False, True])
def test_t4_task_level_route_drop_keeps_generation_and_zero_output_valid(tmp_path, all_zero):
    from dataset.builder import DatasetBuilder
    from orchestrator.t4_structured import run_t4_structured

    output = tmp_path / "data.jsonl"
    config = {"paths": {"debug_dir": str(tmp_path / "debug")}, "dataset": {"output_file": str(output)},
              "tasks": {"T4": {"task_profile": "paper", "max_llava": 1, "max_mmt": 0,
                                "max_dialog_mmdu": 1, "max_magicbrush": 0, "max_unify": 0,
                                "enable_llm_decision": True, "enable_llm_generation": True}}}
    reply = route_reply([] if all_zero else ["single_image_reasoning", "multi_image_logic"])
    # The multi-image route is feasible in principle, but the model cites only
    # one image.  Its rejection must not trigger a router retry or stop QA.
    reply["tasks"]["multi_image_logic"]["evidence_refs"] = [] if all_zero else ["F00_CAP"]

    class LLM:
        def __init__(self):
            self.calls = []

        def chat(self, prompt, **kwargs):
            self.calls.append(prompt)
            if "NINE paper task" in prompt:
                return json.dumps(reply)
            assert "Target benchmark: llava_bench" in prompt
            return json.dumps({"items": [{"image_pos": 0, "category": "complex", "reasoning_category": "function",
                "question": "How does the long handle help the cook?",
                "gpt_answer": "The handle offers a grip away from the pan body. Its length helps the cook hold the pan.",
                "evidence_refs": ["F00_CAP"]}]})

    llm = LLM()
    p = SimpleNamespace(config=config, llm=llm, model_manager=None, _video_id=lambda x: "source",
                        _relativize_item_paths_inplace=lambda x: x, _global_bench_counts={}, _global_llava_cat_counts={})
    p.dataset_builder = DatasetBuilder(config)
    frames = [{"img": f"pan{i}.jpg", "sec": float(i), "origin_index": i,
               "caption": "A pan has a long handle."} for i in range(2)]
    try:
        result = run_t4_structured(p, {"output": {"global_summary": "A cook holds a pan."}},
            {"output": {"keyframes": frames}},
            {"output": {"segment_summary": "A cook holds a pan.", "frames": frames}},
            SimpleNamespace(index=0, video_path="source.mp4", segment_path="seg.mp4"))
    finally:
        p.dataset_builder.close()
    assert sum("NINE paper task" in prompt for prompt in llm.calls) == 1
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(rows) == (0 if all_zero else 1)
    ids = {unit["evidence_id"] for unit in result["evidence"]["evidence_units"]}
    for row in rows:
        assert set(row["evidence_refs"]) <= ids
        assert row["task_type"] == "single_image_reasoning"
    rejected = result["decision"]["tasks"]["multi_image_logic"]
    assert rejected["quota"] == 0
    if not all_zero:
        assert rejected["admission_reason"] == "insufficient_task_specific_evidence"
