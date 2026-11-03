import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.audit_training_sources import audit_sources, task_turns


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _image(path: Path, color="red"):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _edit():
    return {"benchmark": "magicbrush", "img_id": "edit-1", "source_img": "a.jpg",
            "target_img": "b.jpg", "instruction": "add a cup", "meta": {"review": {"ok": True}}}


def test_audit_checks_nested_edit_review_and_real_media(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    _image(root / "a.jpg")
    _image(root / "b.jpg", "blue")
    original = (root / "magicbrush.jsonl").read_bytes()
    result = audit_sources(root, tmp_path / "audit")
    tasks = result["task_source_matrix"]["tasks"]
    assert tasks["localized_editing"]["status"] == "supported"
    assert tasks["localized_editing"]["decoded_examples"] == 1
    assert tasks["grounded_edit_to_select"]["status"] == "unsupported"
    assert result["source_inventory"]["source_material_features"]["reviewed_edit_pair_rows"] == 1
    example = tasks["localized_editing"]["examples"][0]
    assert example["source_line"] == 1
    assert len(example["source_row_sha256"]) == 64
    assert len(example["media"][0]["sha256"]) == 64
    assert (root / "magicbrush.jsonl").read_bytes() == original


@pytest.mark.parametrize("review", [None, {"ok": False}, {"ok": "true"}])
def test_audit_never_accepts_missing_or_rejected_review(tmp_path, review):
    row = _edit()
    row["meta"]["review"] = review
    _write(tmp_path / "train" / "magicbrush.jsonl", [row])
    task = audit_sources(tmp_path / "train", tmp_path / "audit")["task_source_matrix"]["tasks"]["localized_editing"]
    assert task["status"] == "incomplete"
    assert task["reasons"]["format:missing_or_rejected_edit_review"] == 1


def test_audit_missing_media_is_incomplete_not_supported(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    task = audit_sources(root, tmp_path / "audit")["task_source_matrix"]["tasks"]["localized_editing"]
    assert task["status"] == "incomplete"
    assert task["media_unresolved_rows"] == 1


def test_audit_corrupt_image_cannot_pass(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    _image(root / "a.jpg")
    (root / "b.jpg").write_bytes(b"corrupt")
    task = audit_sources(root, tmp_path / "audit")["task_source_matrix"]["tasks"]["localized_editing"]
    assert task["status"] == "incomplete"
    assert task["decode_failed_examples"] == 1


def test_audit_deduplicates_shards_and_excludes_them_by_default(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    _write(root / "shard_0.jsonl", [_edit()])
    report = audit_sources(root, tmp_path / "default")["source_inventory"]
    assert report["file_count"] == 1
    report = audit_sources(root, tmp_path / "recursive", include=["*.jsonl"])["source_inventory"]
    assert report["unique_content_rows"] == 1
    assert sum(f.get("duplicate_content_rows", 0) for f in report["files"]) == 1


def test_audit_records_parse_errors_and_supports_gpt_answer(tmp_path):
    root = tmp_path / "train"
    _image(root / "a.jpg")
    _write(root / "llava_bench.jsonl", [{"benchmark": "llava_bench", "image": "a.jpg", "question": "q", "gpt_answer": "answer"}])
    with (root / "llava_bench.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    result = audit_sources(root, tmp_path / "audit")
    assert result["source_inventory"]["files"][0]["parse_errors"] == 1
    assert result["task_source_matrix"]["tasks"]["single_image_reasoning"]["status"] == "supported"


def test_legacy_unified_loader_readable_does_not_mean_paper_complete():
    side = {"messages": [{"role": "assistant", "type": "text", "content": "steps"}], "images": []}
    row = {"turns": [{"t": "steps", "task_type": "vlm_understand", "render": {"query": copy.deepcopy(side), "answer": side}}]}
    with pytest.raises(ValueError, match="paper_rounds_missing_generation_steps"):
        task_turns(row, "stepwise_sequence")


def test_media_roots_do_not_guess_between_distinct_files(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    for media in (tmp_path / "media1", tmp_path / "media2"):
        _image(media / "a.jpg")
        _image(media / "b.jpg")
    task = audit_sources(root, tmp_path / "audit", media_roots=[tmp_path / "media1", tmp_path / "media2"])["task_source_matrix"]["tasks"]["localized_editing"]
    assert task["status"] == "incomplete"
    assert task["examples"][0]["media"][0]["status"] == "ambiguous"


def test_audit_refuses_source_mutation_or_old_report_overwrite(tmp_path):
    root = tmp_path / "train"
    _write(root / "magicbrush.jsonl", [_edit()])
    with pytest.raises(ValueError, match="outside_read_only_source"):
        audit_sources(root, root / "audit")
    audit_sources(root, tmp_path / "audit")
    with pytest.raises(FileExistsError):
        audit_sources(root, tmp_path / "audit")
