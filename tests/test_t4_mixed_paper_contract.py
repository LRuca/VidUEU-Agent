import copy
import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from dataset.builder import DatasetBuilder
from orchestrator.paper_tasks import TASKS
from orchestrator import t4_structured


@pytest.mark.parametrize("write_fails", [False, True])
def test_mmt_export_preserves_unified_quotas_and_pending_generation(tmp_path, monkeypatch, write_fails):
    item = {"image_pos": 0, "question": "Which object is on the table?", "A": "Cup", "B": "Box",
            "C": "Plate", "D": "Vase", "_answer": "A", "category": "object", "l2-category": "object_identity",
            "evidence_refs": ["F00_CAP"], "perception_evidence": {
                "premise_cues": [{"claim": "A cup is on the table.", "evidence_refs": ["F00_CAP"]}],
                "option_checks": {k: {"status": "supported" if k == "A" else "excluded", "reason": "visible object shape",
                                      "evidence_refs": ["F00_CAP"]} for k in "ABCD"}}}
    fixed = {k: v for k, v in item.items() if k != "perception_evidence"}
    def chat(prompt, **kwargs):
        if "Target benchmark: mmt_bench" in prompt:
            return json.dumps({"items": [item]})
        if "fixed_item" in prompt:
            return json.dumps({"results": [{"ok": True, "reason": "supported", "fixed_item": fixed}]})
        return json.dumps({"pairs": []})
    chosen = {"fine_grained_perception", "understand_recall_generate_select"}
    routed = {"tasks": {name: {"quota": int(name in chosen)} for name in TASKS},
              "raw": "mock route", "capabilities": ["understanding", "unified"]}
    monkeypatch.setattr(t4_structured, "route_tasks", lambda *a, **k: copy.deepcopy(routed))
    producer_calls = []
    def produce(bundle, **kwargs):
        producer_calls.append(kwargs["selected_tasks"])
        return {"samples": {}, "drops": {}, "pending": {"unify.weave.uie_select_recall": [
            {"status": "pending", "semantic_decision": None, "reason": "api_budget_exhausted"}]}}
    monkeypatch.setattr(t4_structured, "produce_task_samples", produce)
    config = {"paths": {"debug_dir": str(tmp_path / "debug")}, "dataset": {"output_file": str(tmp_path / "data.jsonl")},
              "tasks": {"T4": {"task_profile": "paper", "prompt_profile": "paper_v4", "max_mmt": 1,
                                  "max_llava": 0, "max_dialog_mmdu": 0, "max_unify": 1}}}
    builder = DatasetBuilder(config)
    if write_fails:
        def fail_write(_row):
            raise OSError("simulated disk full")
        monkeypatch.setattr(builder, "add_item", fail_write)
    planner = SimpleNamespace(config=config, llm=SimpleNamespace(chat=chat), model_manager=None,
        dataset_builder=builder, _global_bench_counts={}, _global_mme_cat_counts={}, _global_mmt_cat_counts={},
        _global_llava_cat_counts={}, _video_id=lambda p: Path(p).stem, _relativize_item_paths_inplace=lambda row: row)
    frames = [{"img": str(tmp_path / f"frame{i}.jpg"), "sec": i, "origin_index": i, "shot_id": "s", "chapter_id": "c",
               "caption": "A cup is on the table.", "cause": "object_motion",
               "transition_to_next": "Edits(1): 1) move cup rightward | Shot: none | View: none | Subject: same | Uncertainty: none" if i == 0 else ""}
              for i in range(2)]
    for frame in frames:
        Path(frame["img"]).write_bytes(b"mock image")
    try:
        if write_fails:
            with pytest.raises(RuntimeError, match="dataset_write_failed"):
                t4_structured.run_t4_structured(planner, {"output": {"video_type": "other"}},
                    {"output": {"quality_passed": True, "keyframes": frames}},
                    {"output": {"frames": frames, "segment_summary": "A cup moves on a table.", "metadata_id": "bound"}},
                    SimpleNamespace(index=0, video_path="source.mp4", segment_path="source.mp4"))
            return
        result = t4_structured.run_t4_structured(planner, {"output": {"video_type": "other"}},
            {"output": {"quality_passed": True, "keyframes": frames}},
            {"output": {"frames": frames, "segment_summary": "A cup moves on a table.", "metadata_id": "bound"}},
            SimpleNamespace(index=0, video_path="source.mp4", segment_path="source.mp4"))
    finally:
        builder.close()
    assert producer_calls == [["unify.weave.uie_select_recall"]]
    exported = result["instances"]["understanding.mmt_bench"]
    assert len(exported) == 1 and exported[0]["perception_evidence"] == item["perception_evidence"]
    assert result["runtime"]["execution_pending"] is True
    assert result["runtime"]["pending_generation"]["unify.weave.uie_select_recall"][0]["semantic_decision"] is None
