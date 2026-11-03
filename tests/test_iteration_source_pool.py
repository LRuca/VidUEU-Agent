import copy
import json

import pytest
from PIL import Image

from orchestrator.iteration.bagel_data import convert_manifest
from orchestrator.iteration.source_pool import prepare_source_pool, load_audited_pool
from orchestrator.iteration.state import read_jsonl, write_jsonl
from orchestrator.iteration.taxonomy import TASK_TYPES


def source(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for i, color in enumerate(("red", "blue", "green", "yellow")):
        Image.new("RGB", (32, 32), color).save(root / f"{i}.png")
    return root, {"name": "history", "split": "train", "root": str(root),
                  "media_root": str(root), "files": ["source.jsonl"]}


def raw_unified(benchmark):
    return {"benchmark": benchmark, "unify_id": benchmark, "meta": {
        "frame_positions": [0, 1, 2, 3] if benchmark.endswith(("c1_steps", "seq_select")) else [0, 1, 2],
        "segment_keyframes": [{"pos": i, "img": f"{i}.png"} for i in range(4)],
    }, "answer": ({"future_frame_positions": [2, 3]} if benchmark.endswith("seq_select") else
                  {"target_frame_positions": [1, 2, 3]} if benchmark.endswith("c1_steps") else
                  {"target_frame_pos": 1}),
        "turns": {"plan": {"goal": "change"}, "understand": {"facts": ["state"]},
                  "qa_open": {"question": "recall", "answer": "later"},
                  "overall": "sequence", "steps": ["one", "two", "three"],
                  "explanation": "middle", "edits": ["change"], "because": ["state"],
                  "so_that": ["target"], "check": ["verify"],
                  "select": [] if benchmark.endswith(("c1_steps", "seq_select")) else {"rationale": "matches"}}}


def legacy(raw):
    return {"benchmark": raw["benchmark"], "turns": [{"t": "compressed"}],
            "meta": {**raw["meta"], "legacy_unify": {k: copy.deepcopy(raw[k])
                                                    for k in ("turns", "answer", "unify_id")}}}


def nine_rows():
    rows = [
        {"benchmark": "mmt_bench", "image": "0.png", "question": "color?", "_answer": "A",
         "A": "red", "B": "blue", "C": "green", "D": "yellow"},
        {"benchmark": "llava_bench", "image": "0.png", "question": "Describe", "gpt_answer": "red"},
        {"benchmark": "dialog_mmdu_like", "images": ["0.png", "1.png"], "turns": [
            {"role": "user", "text": "Compare <ImageHere> <ImageHere>"},
            {"role": "assistant", "text": "different colors"}]},
        {"benchmark": "magicbrush", "source_img": "0.png", "target_img": "1.png",
         "instruction": "Make blue", "meta": {"review": {"ok": True}}},
    ]
    for benchmark in ("unify.seed.l3.gts", "unify.weave.uie_select_recall", "unify.comm.task4.c1_steps",
                      "unify.comm.task4.c2_fill_gap", "unify.isg.i2.seq_select"):
        rows.append(legacy(raw_unified(benchmark)))
    return rows


def test_nine_tasks_preserved_legacy_real_media_and_original_source_lines(tmp_path):
    root, spec = source(tmp_path)
    rows = nine_rows()
    write_jsonl(root / "source.jsonl", rows + [rows[0]])
    original = (root / "source.jsonl").read_bytes()
    output = tmp_path / "pool"
    report = prepare_source_pool([spec], output)
    assert report["missing_task_types"] == []
    assert report["candidate_count"] == 9
    assert len(load_audited_pool(output / "candidate_pool.jsonl")) == 9
    assert report["transforms"]["rerendered_preserved_legacy"] == 5
    assert report["source_files"][0]["duplicate_content_rows"] == 1
    assert (root / "source.jsonl").read_bytes() == original
    assert all(r["source_split"] == "train" for r in read_jsonl(output / "candidate_pool.jsonl"))
    episodes = output / "episodes.jsonl"
    converted = convert_manifest(output / "train_manifest.jsonl", episodes,
                                 asset_root=root, require_paper_rounds=True)
    assert converted["missing_task_types"] == []
    episode = read_jsonl(episodes)[-1]
    assert episode["provenance"]["source_line"] == 9
    assert episode["provenance"]["source_file"] == str(root / "source.jsonl")
    assert episode["provenance"]["source_origin"] == "historical_training_source"
    assert all(p.startswith(str(root)) for e in read_jsonl(episodes)
               for t in e["turns"] for p in t["render"]["query"]["images"])


@pytest.mark.parametrize("defect", ["missing_target", "conflicting_target", "missing_legacy", "missing_media", "unreviewed_edit", "same_content_candidates"])
def test_bad_source_records_report_shortage_without_inventing_targets(tmp_path, defect):
    root, spec = source(tmp_path)
    row = nine_rows()[4]
    if defect == "missing_target":
        row["meta"]["legacy_unify"].pop("answer")
    elif defect == "conflicting_target":
        row["meta"]["legacy_unify"]["answer"]["target_frame_pos"] = 2
    elif defect == "missing_legacy":
        row["meta"].pop("legacy_unify")
    elif defect == "missing_media":
        (root / "1.png").unlink()
    elif defect == "unreviewed_edit":
        row = nine_rows()[3]
        row["meta"]["review"]["ok"] = False
    else:
        (root / "2.png").write_bytes((root / "0.png").read_bytes())
    write_jsonl(root / "source.jsonl", [row])
    report = prepare_source_pool([spec], tmp_path / "pool")
    assert report["candidate_count"] == 0
    assert set(report["missing_task_types"]) == set(TASK_TYPES)
    assert report["rejection_examples"]
    assert read_jsonl(tmp_path / "pool" / "candidate_pool.jsonl") == []


def test_declared_eval_source_and_eval_media_rejected(tmp_path):
    root, spec = source(tmp_path)
    write_jsonl(root / "source.jsonl", [nine_rows()[0]])
    with pytest.raises(ValueError, match="only_explicit_training_sources"):
        prepare_source_pool([{**spec, "split": "eval"}], tmp_path / "bad")
    report = prepare_source_pool([spec], tmp_path / "pool", forbidden_media=[root / "0.png"])
    assert report["candidate_count"] == 0
    assert "evaluation_media_overlap" in report["rejection_examples"][0]["error"]


def test_repeated_build_never_overwrites_source_pool_or_source(tmp_path):
    root, spec = source(tmp_path)
    write_jsonl(root / "source.jsonl", [nine_rows()[0]])
    with pytest.raises(ValueError, match="outside_read_only_source"):
        prepare_source_pool([spec], root / "derived")
    output = tmp_path / "pool"
    prepare_source_pool([spec], output)
    original = (output / "candidate_pool.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_source_pool([spec], output)
    assert (output / "candidate_pool.jsonl").read_bytes() == original


def test_source_media_mutation_is_detected_at_loop_admission(tmp_path):
    root, spec = source(tmp_path)
    write_jsonl(root / "source.jsonl", [nine_rows()[0]])
    output = tmp_path / "pool"
    prepare_source_pool([spec], output)
    Image.new("RGB", (32, 32), "purple").save(root / "0.png")
    with pytest.raises(ValueError, match="media_changed"):
        load_audited_pool(output / "candidate_pool.jsonl")
