import json
from pathlib import Path

import pytest
from PIL import Image

from vidueu.paper_export import paper_schema_export, verify_paper_package


def _package(tmp_path: Path) -> Path:
    source = tmp_path / "portable"
    (source / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(source / "images" / "seed.png")
    group = "metadata-group-1"
    metadata = {
        "metadata_group_key": group,
        "segment": {
            "video_id": r"C:\\private\\videos\\clip.mp4",
            "start": 0.0,
            "end": 1.0,
            "summary": "A red square.",
            "type": "other",
        },
        "chapters": [{"chapter_id": "C00", "time_span": [0, 1], "summary": "Square.", "role": "main"}],
        "shots": [{"shot_id": "S00", "chapter_id": "C00", "time_span": [0, 1],
                   "summary": "Square.", "role": "main", "scene_type": "demo",
                   "quality_hint": "clear", "info_hint": "color"}],
        "keyframes": [{
            "frame_id": "F00", "shot_id": "S00", "timestamp": 0.0, "caption": "A red square.",
            "quality": 1.0, "value_score": 1.0, "tool_semantics": {},
            "evidence_id": "F00_CAP", "image_path": "images/seed.png",
        }, {"frame_id": "F01", "shot_id": "S00", "timestamp": 1.0, "caption": "A red square.",
            "quality": 1.0, "value_score": 1.0, "tool_semantics": {},
            "evidence_id": "F01_CAP", "image_path": "images/seed.png"}],
        "transitions": [],
        "canonical_evidence": {"F00_CAP": "A red square."},
        "provenance": {"source_image": r"/home/private/frame.jpg", "source_file": r"C:\\private\\metadata.json"},
        "edit_bindings": [],
    }
    edit = {
        "record_key": "edit-1", "benchmark": "edit", "instruction": "Change red to blue.",
        "from_pos": 0, "to_pos": 1, "ok": True, "edit_count": 1,
        "edits": ["change color"], "reason": "small change",
        "source_image": "images/seed.png", "target_image": "images/seed.png",
        "metadata_ref": group, "evidence_refs": ["F00_CAP"],
    }
    understanding = {
        "record_key": "und-1", "benchmark": "llava_bench", "image": "images/seed.png",
        "question": "What color is the square?", "answer": "Red.",
        "image_pos": 0, "category": "attribute",
        "metadata_refs": [group], "evidence_refs": ["F00_CAP"],
    }
    unify = {
        "record_key": "unify-1", "benchmark": "unify.isg", "evidence_refs": ["F00_CAP"],
        "metadata_ref": group,
        "Round_Plan": {"question": {"text": "What changes?", "Frames": ["F00", "F01"]},
                       "answer": {"edits": ["change red to blue"]}},
        "Round_Gen": {"question": {"text": "Generate the target.", "Frames": ["F00"]},
                      "answer": {"image": "images/seed.png"}},
        "Round_Select": {"question": {"text": "Which target?", "Frames": ["F01"]},
                         "answer": "target:F01", "rationale": "Matches the target."},
        "meta": {"video_path": "/home/private/clip.mp4"},
    }
    (source / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    (source / "edit.jsonl").write_text(json.dumps(edit) + "\n", encoding="utf-8")
    (source / "understanding.jsonl").write_text(json.dumps(understanding) + "\n", encoding="utf-8")
    (source / "unify.jsonl").write_text(json.dumps(unify) + "\n", encoding="utf-8")
    return source


def test_paper_export_small_sample_aligns_and_redacts(tmp_path):
    source = _package(tmp_path)
    destination = tmp_path / "paper"

    report = paper_schema_export(source, destination, sample_limit=1)

    assert report["complete"] is True
    assert report["paper_schema_version"] == "vidueu-paper-hf-v2"
    assert report["counts"] == {"metadata": 1, "understanding": 1, "edit": 1, "unify": 1}
    assert report["reference_closure"]["ok"] is True
    assert verify_paper_package(destination)["ok"] is True
    assert (destination / "images" / "seed.png").exists() is False
    assert len(list((destination / "images").iterdir())) == 1

    serialized = "".join(path.read_text(encoding="utf-8") for path in destination.glob("*.jsonl"))
    assert "C:\\\\private" not in serialized
    assert "/home/private" not in serialized
    assert "source_video://" not in serialized
    assert "source_image://" not in serialized
    assert "source://" not in serialized
    assert "images/" in serialized

    metadata = json.loads((destination / "metadata.jsonl").read_text(encoding="utf-8"))
    assert metadata["segment"]["video_id"].startswith("video_")
    assert set(metadata) == {"metadata_group_key", "segment", "chapters", "shots", "keyframes", "transitions", "evidence_registry"}
    assert metadata["keyframes"][0]["image_path"].startswith("images/")

    edit = json.loads((destination / "edit.jsonl").read_text(encoding="utf-8"))
    assert set(edit) == {"record_key", "benchmark", "instruction", "source_image", "target_image",
                         "evidence_refs", "metadata_ref", "from_pos", "to_pos", "ok", "edit_count", "edits", "reason"}
    unify = json.loads((destination / "unify.jsonl").read_text(encoding="utf-8"))
    assert set(unify) == {"record_key", "benchmark", "Round_Plan", "Round_Gen", "Round_Select",
                          "evidence_refs", "metadata_ref"}


def test_paper_export_rejects_broken_evidence_closure(tmp_path):
    source = _package(tmp_path)
    row = json.loads((source / "edit.jsonl").read_text(encoding="utf-8"))
    row["evidence_refs"] = ["F99_CAP"]
    (source / "edit.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        paper_schema_export(source, tmp_path / "paper")
    except ValueError as exc:
        assert "reference_closure_failed" in str(exc)
    else:
        raise AssertionError("broken evidence closure was accepted")


def test_paper_export_rejects_runtime_unified_row_without_paper_rounds(tmp_path):
    source = _package(tmp_path)
    row = json.loads((source / "unify.jsonl").read_text(encoding="utf-8"))
    row.pop("Round_Plan")
    row.pop("Round_Gen")
    row.pop("Round_Select")
    row["render"] = {"images": ["images/seed.png"]}
    (source / "unify.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        paper_schema_export(source, tmp_path / "paper")
    except ValueError as exc:
        assert "unify_round_schema_missing" in str(exc)
    else:
        raise AssertionError("runtime unified row was silently projected")


def test_repaired_inputs_take_priority_and_preserve_source_bytes(tmp_path):
    source = _package(tmp_path)
    (source / "metadata.jsonl").rename(source / "metadata_repaired_v1.jsonl")
    (source / "edit.jsonl").rename(source / "edit_repaired_v1.jsonl")
    (source / "understanding.jsonl").rename(source / "und.jsonl")
    (source / "metadata.jsonl").write_text('{"obsolete": true}\n')
    (source / "edit.jsonl").write_text('{"obsolete": true}\n')
    before = {path.name: path.read_bytes() for path in source.glob("*.jsonl")}
    report = paper_schema_export(source, tmp_path / "release")
    assert report["inputs"]["metadata.jsonl"]["file"] == "metadata_repaired_v1.jsonl"
    assert report["inputs"]["edit.jsonl"]["file"] == "edit_repaired_v1.jsonl"
    assert report["inputs"]["understanding.jsonl"]["file"] == "und.jsonl"
    assert {path.name: path.read_bytes() for path in source.glob("*.jsonl")} == before


def test_known_gaps_are_retained_and_block_publish_acceptance(tmp_path):
    source = _package(tmp_path)
    path = source / "edit.jsonl"
    row = json.loads(path.read_text())
    row["evidence_refs"].append("P99_REVIEW")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="reference_closure_failed"):
        paper_schema_export(source, tmp_path / "strict")
    assert not (tmp_path / "strict").exists()
    report = paper_schema_export(source, tmp_path / "candidate", allow_unresolved=True)
    assert report["complete"] and not report["publish_ready"]
    output = json.loads((tmp_path / "candidate" / "edit.jsonl").read_text())
    assert output["evidence_refs"] == row["evidence_refs"]
    assert verify_paper_package(tmp_path / "candidate")["publish_ready"] is False


def test_unified_registry_option_labels_and_answers_survive_projection(tmp_path):
    source = _package(tmp_path)
    path = source / "unify.jsonl"
    row = json.loads(path.read_text())
    row.pop("record_key")
    row["id"] = "real-unify-identity"
    row["frame_registry"] = {"Frame[F00]": {"position": 0, "image": "images/seed.png", "verification": "old"}}
    row["Round_Select"]["options"] = [{"id": "B", "frame": "Frame[F00]", "image": "images/seed.png", "original_image": "/home/private/image.jpg"}]
    row["Round_Select"]["answer"] = "target:Frame[F00]"
    path.write_text(json.dumps(row) + "\n")
    report = paper_schema_export(source, tmp_path / "out")
    output = json.loads((tmp_path / "out" / "unify.jsonl").read_text())
    assert report["publish_ready"]
    assert output["id"] == output["record_key"] == "real-unify-identity"
    assert output["Round_Select"]["answer"] == row["Round_Select"]["answer"]
    assert output["Round_Select"]["options"][0]["id"] == "B"
    assert output["Round_Select"]["options"][0]["frame"] == "Frame[F00]"
    assert set(output["frame_registry"]["Frame[F00]"]) == {"position", "image"}
    assert "original_image" not in output["Round_Select"]["options"][0]


def test_sample_keeps_metadata_dependencies_and_rejects_escape(tmp_path):
    source = _package(tmp_path)
    # Validating a package must not trust a path outside its root.
    from vidueu.paper_export import _safe_file
    with pytest.raises(ValueError, match="unsafe_relative_path"):
        _safe_file(source, "../outside.png")
    with pytest.raises(ValueError, match="invalid_sample_limit"):
        paper_schema_export(source, tmp_path / "out", sample_limit=0)
    report = paper_schema_export(source, tmp_path / "dry", sample_limit=1, dry_run=True)
    assert report["counts"]["metadata"] == 1
    assert not (tmp_path / "dry").exists()


def test_web_urls_are_not_machine_paths_and_grouping_keys_do_not_collide(tmp_path):
    from vidueu.paper_export import LOCAL_PATH
    assert not LOCAL_PATH.search("The screen shows http://tablet-news.com/ and https://example.org.")
    assert LOCAL_PATH.search(r"C:\private\image.jpg")
    assert LOCAL_PATH.search("/home/private/image.jpg")
    source = _package(tmp_path)
    path = source / "edit.jsonl"
    first = json.loads(path.read_text())
    second = dict(first)
    first["sample_id"], second["sample_id"] = "distinct-edit-1", "distinct-edit-2"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    report = paper_schema_export(source, tmp_path / "out")
    output = [json.loads(line) for line in (tmp_path / "out" / "edit.jsonl").read_text().splitlines()]
    assert report["publish_ready"]
    assert [row["record_key"] for row in output] == ["distinct-edit-1", "distinct-edit-2"]


def test_edit_scalar_metadata_ref_uses_declared_canonical_group(tmp_path):
    source = _package(tmp_path)
    path = source / "edit.jsonl"
    row = json.loads(path.read_text())
    row["metadata_ref"] = "legacy-record-grouping-key"
    row["canonical_metadata_ref"] = {"metadata_group_key": "metadata-group-1", "source_frame_id": "F00", "target_frame_id": "F01"}
    path.write_text(json.dumps(row) + "\n")
    report = paper_schema_export(source, tmp_path / "out")
    result = json.loads((tmp_path / "out" / "edit.jsonl").read_text())
    assert report["publish_ready"]
    assert result["metadata_ref"] == "metadata-group-1"
    lineage = [json.loads(line) for line in (tmp_path / "out" / "record_lineage.jsonl").read_text().splitlines()]
    assert next(x for x in lineage if x["file"] == "edit.jsonl")["source_metadata_ref"] == "legacy-record-grouping-key"


def test_quoted_paths_in_visible_supervision_are_preserved_for_review(tmp_path):
    source = _package(tmp_path)
    path = source / "understanding.jsonl"
    row = json.loads(path.read_text())
    row["answer"] = r"The notebook displays C:\Users\richard\demo.py."
    path.write_text(json.dumps(row) + "\n")
    report = paper_schema_export(source, tmp_path / "out")
    assert report["path_like_supervision_texts_for_review"] == 1
    result = json.loads((tmp_path / "out" / "understanding.jsonl").read_text())
    assert result["answer"] == row["answer"]
    assert (tmp_path / "out" / "path_text_review.jsonl").read_text().strip()
