import copy
from pathlib import Path

import pytest
from PIL import Image

from orchestrator.iteration.bagel_data import convert_manifest
from orchestrator.iteration.state import read_jsonl, write_jsonl
from orchestrator.iteration.taxonomy import TASK_CAPABILITY, TASK_TYPES


def assets(tmp_path):
    for name, color in (("a.png", "red"), ("b.png", "blue"), ("c.png", "green")):
        Image.new("RGB", (32, 32), color).save(tmp_path / name)


def render_turn(mode, question="Generate the changed image", targets=True):
    return {"t": "gen" if targets else "plan", "mode": mode, "render": {
        "query": {"messages": [
            {"role": "user", "type": "text", "content": question},
            {"role": "user", "type": "image", "content": "Image #1"}], "images": ["a.png"]},
        "answer": {"messages": [
            {"role": "assistant", "type": "image" if targets else "text",
             "content": "Image #1" if targets else "Move the cup"}],
            "images": ["b.png"] if targets else []}}}


def pool():
    common = {"input_row": 7}
    raw = [
        {"benchmark": "mmt_bench", "image": "a.png", "question": "Which color?",
         "A": "red", "B": "blue", "C": "green", "D": "black", "_answer": "A"},
        {"benchmark": "llava_bench", "image": "a.png", "question": "Describe.", "gpt_answer": "Red square."},
        {"benchmark": "dialog_mmdu_like", "images": [{"image": "a.png"}, {"image": "b.png"}],
         "turns": [{"role": "user", "text": "Image1: <ImageHere> Describe."},
                   {"role": "assistant", "text": "Red."},
                   {"role": "user", "text": "Image2: <ImageHere> Compare to Image1."},
                   {"role": "assistant", "text": "Blue replaces red."}]},
        {"benchmark": "edit", "source_img": "a.png", "target_img": "b.png", "instruction": "Make it blue."},
    ]
    for _ in TASK_TYPES[4:]:
        raw.append({"turns": [render_turn("vlm_understand", "Plan", False), render_turn("edit")]})
    return [{"record_key": f"record-{index}", "task_type": task, "capability": TASK_CAPABILITY[task],
             "source_pointer": "video:segment0", "source_sha256": "input-digest", "iteration": 1,
             "candidate": {**common, **candidate}}
            for index, (task, candidate) in enumerate(zip(TASK_TYPES, raw))]


def test_nine_tasks_convert_with_media_and_provenance_without_mutation(tmp_path):
    assets(tmp_path)
    source, dest = tmp_path / "manifest.jsonl", tmp_path / "episodes.jsonl"
    rows = pool()
    original = copy.deepcopy(rows)
    write_jsonl(source, rows)
    report = convert_manifest(source, dest, asset_root=tmp_path)
    assert rows == original
    assert report["episode_count"] == 9
    assert report["turn_count"] == 15
    assert report["capability_counts"] == {"understanding": 3, "editing": 1, "unified": 5}
    assert len(report["media"]) == 2
    episodes = read_jsonl(dest)
    dialog = episodes[2]
    assert dialog["turns"][0]["render"]["query"]["images"] == [str(tmp_path / "a.png")]
    assert dialog["turns"][1]["render"]["query"]["images"] == [str(tmp_path / "b.png")]
    assert all(e["provenance"]["input_row"] == 7 for e in episodes)
    assert episodes[3]["turns"][0]["task_type"] == "edit"
    assert report["coverage_scope"] == "format_and_media_only_not_training_consumption"


def test_dialog_bare_imagehere_placeholders_bind_in_order(tmp_path):
    assets(tmp_path)
    candidate = {
        "benchmark": "dialog_mmdu_like",
        "images": [{"image": "a.png"}, {"image": "b.png"}, {"image": "c.png"}],
        "turns": [
            {"role": "user", "text": "Compare these: <ImageHere> <ImageHere> <ImageHere>"},
            {"role": "assistant", "text": "The third image differs."},
        ],
    }
    source, output = tmp_path / "manifest.jsonl", tmp_path / "episodes.jsonl"
    write_jsonl(source, [{"record_key": "bare-dialog", "task_type": "multi_image_logic",
                         "candidate": candidate}])
    convert_manifest(source, output, asset_root=tmp_path, require_nine_tasks=False)
    turn = read_jsonl(output)[0]["turns"][0]
    assert turn["render"]["query"]["images"] == [
        str(tmp_path / "a.png"), str(tmp_path / "b.png"), str(tmp_path / "c.png")
    ]
    assert [m["content"] for m in turn["render"]["query"]["messages"] if m["type"] == "image"] == [
        "Image #1", "Image #2", "Image #3"
    ]


@pytest.mark.parametrize("defect", ["missing_image", "duplicate_key", "wrong_ref", "leak", "bad_mode", "bad_role", "missing_task"])
def test_bad_data_is_rejected_before_publishing(tmp_path, defect):
    assets(tmp_path)
    rows = pool()
    if defect == "missing_image":
        rows[0]["candidate"]["image"] = "missing.png"
    elif defect == "duplicate_key":
        rows[1]["record_key"] = rows[0]["record_key"]
    elif defect == "missing_task":
        rows.pop()
    else:
        turn = rows[-1]["candidate"]["turns"][-1]
        if defect == "wrong_ref":
            turn["render"]["answer"]["messages"][0]["content"] = "Image #2"
        elif defect == "leak":
            turn["render"]["answer"]["images"] = ["a.png"]
        elif defect == "bad_mode":
            turn["mode"] = "unknown"
        elif defect == "bad_role":
            turn["render"]["query"]["messages"][0]["role"] = "assistant"
    source, output = tmp_path / "manifest.jsonl", tmp_path / "episodes.jsonl"
    write_jsonl(source, rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        convert_manifest(source, output, asset_root=tmp_path)
    assert not output.exists()


def test_raw_grounded_unified_task_uses_existing_paper_renderer(tmp_path):
    assets(tmp_path)
    candidate = {"module": "unify", "benchmark": "unify.seed.l3.gts", "unify_id": "test",
                 "turns": {"plan": {"goal": "change color", "invariants": ["square"], "edits": ["blue"]},
                           "select": {"rationale": "blue target"}},
                 "answer": {"target_frame_pos": 1}, "render": {"source_frame": 0, "target_frame": 1},
                 "meta": {"frame_positions": [0, 1], "segment_keyframes": [
                     {"pos": 0, "img": "a.png"}, {"pos": 1, "img": "b.png"}]}}
    source, output = tmp_path / "manifest.jsonl", tmp_path / "episodes.jsonl"
    write_jsonl(source, [{"record_key": "gts", "task_type": "grounded_edit_to_select", "candidate": candidate}])
    report = convert_manifest(source, output, asset_root=tmp_path, require_nine_tasks=False)
    assert report["turn_count"] == 3
    turns = read_jsonl(output)[0]["turns"]
    assert turns[0]["render"]["query"]["images"] == [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    assert turns[1]["render"]["query"]["images"] == [str(tmp_path / "a.png")]
    assert turns[1]["render"]["answer"]["images"] == [str(tmp_path / "b.png")]
