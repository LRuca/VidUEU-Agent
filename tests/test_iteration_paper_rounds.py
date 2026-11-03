import copy

import pytest

from orchestrator.iteration.bagel_data import _candidate_turns, convert_manifest
from orchestrator.iteration.paper_rounds import validate_paper_rounds
from orchestrator.iteration.state import write_jsonl
from orchestrator.paper_render import render_paper_unified
from orchestrator.iteration.taxonomy import TASK_ALIASES


def _raw(benchmark):
    return {
        "benchmark": benchmark, "unify_id": "frozen-example", "meta": {
            "frame_positions": [0, 1, 2, 3] if benchmark.endswith(("c1_steps", "seq_select")) else [0, 1, 2],
            "segment_keyframes": [{"pos": i, "img": f"{i}.png"} for i in range(4)],
        },
        "answer": {"target_frame_pos": 1, "target_frame_positions": [1, 2, 3], "future_frame_positions": [2, 3]},
        "turns": {
            "plan": {"goal": "edit", "edits": ["add"], "invariants": ["background"]},
            "understand": {"facts": ["earlier", "later"]},
            "qa_open": {"question": "recall", "answer": "later", "rubric": ["state"]},
            "overall": "sequence", "steps": ["one", "two", "three"], "explanation": "middle fits",
            "edits": ["first", "second"], "because": ["first", "second"],
            "so_that": ["first", "second"], "check": ["first", "second"],
            "select": [],
        },
    }


@pytest.mark.parametrize("benchmark", [
    "unify.seed.l3.gts", "unify.weave.uie_select_recall", "unify.comm.task4.c1_steps",
    "unify.comm.task4.c2_fill_gap", "unify.isg.i2.seq_select",
])
def test_current_paper_renderer_passes_complete_round_contract(benchmark):
    raw = _raw(benchmark)
    if benchmark.endswith(("gts", "recall", "fill_gap")):
        raw["turns"]["select"] = {"rationale": "target matches"}
    if benchmark.endswith("seq_select"):
        raw["answer"].pop("target_frame_positions")
    rendered = render_paper_unified(raw)
    validate_paper_rounds(rendered["turns"], TASK_ALIASES[benchmark])


def _weave():
    raw = _raw("unify.weave.uie_select_recall")
    raw["turns"]["select"] = {"rationale": "target matches"}
    return render_paper_unified(raw)["turns"]


def test_recall_qa_must_precede_generation_and_selection():
    turns = _weave()
    turns.append(turns.pop(2))
    with pytest.raises(ValueError, match="paper_round_order_mismatch"):
        validate_paper_rounds(turns, "understand_recall_generate_select")


def test_valid_letter_for_wrong_real_target_is_rejected():
    turns = _weave()
    selected = turns[-1]
    text = selected["render"]["answer"]["messages"][0]
    text["content"] = "B" if text["content"] == "A" else "A"
    with pytest.raises(ValueError, match="does_not_match_generated_target"):
        validate_paper_rounds(turns, "understand_recall_generate_select")


def test_missing_paper_round_cannot_be_published_for_training(tmp_path):
    turns = _weave()
    turns.pop(0)
    candidate = {"turns": turns}
    original = copy.deepcopy(candidate)
    source, output = tmp_path / "manifest.jsonl", tmp_path / "episodes.jsonl"
    write_jsonl(source, [{"record_key": "legacy", "task_type": "understand_recall_generate_select", "candidate": candidate}])
    with pytest.raises(ValueError, match="paper_round_order_mismatch"):
        convert_manifest(source, output, asset_root=tmp_path, require_nine_tasks=False, require_paper_rounds=True)
    assert not output.exists()
    assert candidate == original
