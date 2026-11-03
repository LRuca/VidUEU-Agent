"""Paper-schema projection; complete materialization and release acceptance are distinct."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

PAPER_SCHEMA_VERSION = "vidueu-paper-hf-v2"
TASK_INPUT_FILES = {
    "understanding.jsonl": ("understanding.jsonl", "und.jsonl"),
    "edit.jsonl": ("edit_repaired_v1.jsonl", "edit.jsonl"),
    "unify.jsonl": ("unify.jsonl",),
}
METADATA_FIELDS = {
    "segment": ("video_id", "start", "end", "summary", "type"),
    "chapters": ("chapter_id", "time_span", "summary", "role"),
    "shots": ("shot_id", "chapter_id", "time_span", "summary", "role", "scene_type", "quality_hint", "info_hint"),
    "keyframes": ("frame_id", "shot_id", "timestamp", "caption", "quality", "value_score", "tool_semantics", "evidence_id", "image_path"),
    "transitions": ("source", "target", "atomic_changes", "same_shot", "viewpoint_change", "subject_consistent", "reliability"),
}
LINK_FIELDS = ("metadata_group_key", "binding_identity_key", "source_frame_id", "target_frame_id", "from_pos", "to_pos")
ROUND_NAMES = {
    "Round_Plan", "Round_Understand", "Round_QA", "Round_Overall", "Round_Steps", "Round_Explain",
    "Round_Edit", "Round_Reason", "Round_Target", "Round_Check", "Round_Gen", "Round_Select",
}
DROP_FIELDS = {"audit", "schema_audit", "raw_record", "raw_metadata", "provenance", "_provenance",
               "_export_provenance", "schema_provenance", "scores_6d", "original_image",
               "original_answer", "original_choice", "source_path"}
IMAGE_FIELDS = {"image", "image_path", "source_image", "target_image", "mask_image", "images"}
LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/](?!/)|\\\\[^\\]+\\)|(?:^|[\s\"'])/(?:home|root|mnt|Users|tmp|data)/")
FRAME_TOKEN = re.compile(r"^Frame\[(F\d+)\]$")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pick(row, fields):
    return {key: copy.deepcopy(row[key]) for key in fields if key in row}


def _rows(path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"paper_export_row_must_be_object:{path.name}:{number}")
                yield number, row


def _strings(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _strings(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _strings(child, path + "/" + str(index))
    elif isinstance(value, str):
        yield path, value


def _evidence_refs(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_refs" and isinstance(child, list):
                yield from child
            else:
                yield from _evidence_refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _evidence_refs(child)


def _clean_payload(value):
    """Remove implementation fields, preserving arbitrary scientific answer keys."""
    if isinstance(value, dict):
        return {key: _clean_payload(child) for key, child in value.items()
                if key not in DROP_FIELDS and not (key.endswith(("_path", "_file", "_root")) and key != "image_path")}
    if isinstance(value, list):
        return [_clean_payload(child) for child in value]
    return value


def _video_id(value):
    if isinstance(value, str) and (LOCAL_PATH.search(value) or value.startswith(("/", "\\"))):
        return "video_" + hashlib.sha256(value.replace("\\", "/").encode()).hexdigest()[:24]
    return value


def _private_prefixes(source):
    # Source workspace and account directories are operational locators. Paths
    # quoted in a programming-video answer can instead be visible supervision.
    prefixes = [str(source).replace("\\", "/").lower() + "/"]
    parts = source.parts
    for marker in ("home", "Users"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                prefixes.append(str(Path(*parts[:index + 2])).replace("\\", "/").lower() + "/")
    return prefixes


def _project_metadata(original):
    row = copy.deepcopy(original.get("metadata", original))
    if "metadata_group_key" not in row and row.get("metadata_id"):
        row["metadata_group_key"] = row["metadata_id"]
    if "segment" not in row and isinstance(row.get("video"), dict):
        row["segment"] = row["video"]
    result = _pick(row, ("metadata_group_key",))
    for container, fields in METADATA_FIELDS.items():
        value = row.get(container)
        if container == "segment":
            result[container] = _pick(value, fields) if isinstance(value, dict) else {}
            if "video_id" in result[container]:
                result[container]["video_id"] = _video_id(result[container]["video_id"])
        elif isinstance(value, list):
            if any(not isinstance(item, dict) for item in value):
                raise ValueError("paper_export_metadata_object_invalid:" + container)
            result[container] = [_clean_payload(_pick(item, fields)) for item in value]
        else:
            result[container] = []
    registry = row.get("canonical_evidence", row.get("evidence_registry"))
    if isinstance(registry, dict) and registry:
        result["evidence_registry"] = _clean_payload(registry)
    return result


def _project_task(row, filename, number):
    result = _pick(row, ("record_key", "sample_id", "id", "benchmark", "capability", "task_type"))
    # Producer record_key groups several edits; the saved sample identity is unique.
    if row.get("sample_id") or row.get("id"):
        result["record_key"] = row.get("sample_id") or row["id"]
    if not result.get("record_key"):
        identity = row.get("sample_id") or row.get("id")
        if not identity:
            payload = json.dumps(row, sort_keys=True, ensure_ascii=False).encode()
            identity = hashlib.sha256(filename.encode() + b":" + str(number).encode() + b":" + payload).hexdigest()
        result["record_key"] = identity
    if filename == "understanding.jsonl":
        result.update(_pick(row, ("image", "images", "image_pos", "image_pos_list", "category", "question",
                                  "options", "answer", "turns", "evidence_refs")))
        if "turns" in result:
            result["turns"] = [_pick(turn, ("role", "text", "evidence_refs")) for turn in result["turns"]]
        if isinstance(result.get("images"), list):
            result["images"] = [_pick(image, ("pos", "image")) if isinstance(image, dict) else image
                                for image in result["images"]]
        if row.get("evidencepack"):
            result["evidence_registry"] = _clean_payload(copy.deepcopy(row["evidencepack"]))
    elif filename == "edit.jsonl":
        result.update(_pick(row, ("from_pos", "to_pos", "ok", "edit_count", "edits", "instruction", "reason",
                                  "source_image", "target_image", "mask_image", "evidence_refs")))
        for old, new in (("source_img", "source_image"), ("target_img", "target_image"), ("mask_img", "mask_image")):
            if new not in result and old in row:
                result[new] = row[old]
    else:
        rounds = [key for key in row if key.startswith("Round_")]
        if not rounds:
            raise ValueError("paper_export_unify_round_schema_missing")
        if set(rounds) - ROUND_NAMES:
            raise ValueError("paper_export_unknown_round")
        for name in rounds:
            result[name] = _clean_payload(copy.deepcopy(row[name]))
        result.update(_pick(row, ("round_order", "evidence_refs")))
        if isinstance(row.get("frame_registry"), dict):
            result["frame_registry"] = {key: _pick(frame, ("position", "image"))
                                        for key, frame in row["frame_registry"].items()}
        if row.get("evidence_registry"):
            result["evidence_registry"] = _clean_payload(copy.deepcopy(row["evidence_registry"]))
    for key in ("metadata_refs", "canonical_metadata_ref"):
        if key in row:
            refs = row[key]
            if isinstance(refs, list):
                result[key] = [_pick(ref, LINK_FIELDS) if isinstance(ref, dict) else ref for ref in refs]
            elif isinstance(refs, dict):
                result[key] = _pick(refs, LINK_FIELDS)
    if isinstance(row.get("metadata_ref"), str):
        result["metadata_ref"] = row["metadata_ref"]
    if isinstance(result.get("canonical_metadata_ref"), dict) and result["canonical_metadata_ref"].get("metadata_group_key"):
        result["metadata_ref"] = result["canonical_metadata_ref"]["metadata_group_key"]
    if "metadata_refs" not in result:
        group = (row.get("meta") or {}).get("metadata_group_key")
        if group:
            result["metadata_refs"] = [group]
    return result


def _validate_closure(tasks, metadata):
    issues, stats, groups = [], Counter(), {}

    def issue(file, line, kind, detail):
        issues.append({"file": file, "line": line, "kind": kind, "detail": detail})

    for number, row in enumerate(metadata, 1):
        key = row.get("metadata_group_key")
        if not isinstance(key, str) or not key:
            issue("metadata.jsonl", number, "metadata_id_missing", "metadata_group_key")
        elif key in groups:
            issue("metadata.jsonl", number, "duplicate_metadata_id", key)
        groups[key] = row
        frame_ids = [f.get("frame_id") for f in row["keyframes"]]
        shot_ids = {s.get("shot_id") for s in row["shots"]}
        chapter_ids = {c.get("chapter_id") for c in row["chapters"]}
        if len(set(frame_ids)) != len(frame_ids):
            issue("metadata.jsonl", number, "duplicate_frame_id", key)
        for container, fields in METADATA_FIELDS.items():
            values = [row[container]] if container == "segment" else row[container]
            for index, item in enumerate(values):
                for field in fields:
                    if field not in item:
                        issue("metadata.jsonl", number, "missing_schema_field", f"{container}[{index}].{field}")
                    elif item[field] is None:
                        stats["nullable:" + container + "." + field] += 1
        for frame in row["keyframes"]:
            if not frame.get("image_path"):
                issue("metadata.jsonl", number, "unavailable_frame_image", frame.get("frame_id"))
            if frame.get("shot_id") is not None and frame["shot_id"] not in shot_ids:
                issue("metadata.jsonl", number, "frame_shot_unresolved", frame.get("frame_id"))
        for shot in row["shots"]:
            if shot.get("chapter_id") is not None and shot["chapter_id"] not in chapter_ids:
                issue("metadata.jsonl", number, "shot_chapter_unresolved", shot.get("shot_id"))
        for tr in row["transitions"]:
            if tr.get("source") not in frame_ids or tr.get("target") not in frame_ids:
                issue("metadata.jsonl", number, "transition_frame_unresolved", [tr.get("source"), tr.get("target")])
    for filename, rows in tasks.items():
        identities = set()
        for number, row in enumerate(rows, 1):
            identity = row.get("record_key")
            if not isinstance(identity, str) or identity in identities:
                issue(filename, number, "duplicate_or_invalid_record_key", identity)
            identities.add(identity)
            linked = []
            refs = row.get("metadata_refs", [])
            refs = refs if isinstance(refs, list) else []
            canonical = row.get("canonical_metadata_ref")
            if isinstance(canonical, dict):
                refs = [*refs, canonical]
            if not refs and row.get("metadata_ref"):
                refs = [row["metadata_ref"]]
            for ref in refs:
                key = ref.get("metadata_group_key") if isinstance(ref, dict) else ref
                if key not in groups:
                    issue(filename, number, "metadata_ref_unresolved", key)
                else:
                    linked.append(groups[key])
                    if isinstance(ref, dict):
                        frames = {f["frame_id"]: f for f in groups[key]["keyframes"]}
                        for field in ("source_frame_id", "target_frame_id"):
                            if ref.get(field) is not None and ref[field] not in frames:
                                issue(filename, number, "binding_frame_unresolved", ref[field])
            if not linked:
                stats[filename + ":without_explicit_metadata_link"] += 1
            evidence = set((row.get("evidence_registry") or {}).keys())
            for group in linked:
                evidence.update((group.get("evidence_registry") or {}).keys())
                evidence.update(f.get("evidence_id") for f in group["keyframes"] if f.get("evidence_id"))
            unresolved = sorted(set(str(ref) for ref in _evidence_refs(row)) - evidence)
            if unresolved:
                issue(filename, number, "evidence_refs_unresolved", unresolved)
            if filename == "edit.jsonl":
                for field in ("from_pos", "to_pos", "ok", "edit_count", "edits", "instruction", "reason", "source_image", "target_image"):
                    if field not in row:
                        issue(filename, number, "missing_schema_field", field)
                if isinstance(canonical, dict) and canonical.get("metadata_group_key") in groups:
                    frames = {f["frame_id"]: f for f in groups[canonical["metadata_group_key"]]["keyframes"]}
                    for side in ("source", "target"):
                        frame = frames.get(canonical.get(side + "_frame_id"))
                        if frame and frame.get("image_path") != row.get(side + "_image"):
                            issue(filename, number, "binding_image_mismatch", side)
            elif filename == "understanding.jsonl":
                required = ("image_pos_list", "turns", "images") if "turns" in row else ("image_pos", "category", "question", "answer", "image")
                for field in required:
                    if field not in row:
                        issue(filename, number, "missing_schema_field", field)
            else:
                registry = row.get("frame_registry") or {}
                for path, text in _strings({k: v for k, v in row.items() if k.startswith("Round_")}):
                    token = text[len("target:"):] if text.startswith("target:") else text
                    if FRAME_TOKEN.fullmatch(token) and token not in registry:
                        issue(filename, number, "unified_frame_unresolved", token)
                for key, frame in registry.items():
                    if not isinstance(frame.get("image"), str):
                        issue(filename, number, "registry_image_missing", key)
    return issues, stats


def _safe_file(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("paper_export_unsafe_relative_path")
    real = root.joinpath(*path.parts).resolve(strict=True)
    if not real.is_relative_to(root) or not real.is_file():
        raise ValueError("paper_export_image_outside_package")
    return real


def _signature(path):
    s = path.stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


class Assets:
    def __init__(self, source):
        self.source, self.entries, self.aliases = source, {}, {}

    def rewrite(self, value, field=""):
        if isinstance(value, dict):
            return {key: self.rewrite(child, key) for key, child in value.items()}
        if isinstance(value, list):
            return [self.rewrite(child, field) for child in value]
        if isinstance(value, str) and field in IMAGE_FIELDS and value:
            if value not in self.aliases:
                real = _safe_file(self.source, value)
                # Existing 64-hex names are verified against actual bytes during materialization.
                sha = PurePosixPath(value).stem
                if not re.fullmatch(r"[0-9a-f]{64}", sha):
                    sha = digest(real)
                relative = "images/" + sha + real.suffix.lower()
                self.entries.setdefault(relative, {"source": real, "stat": _signature(real)})
                self.aliases[value] = relative
            return self.aliases[value]
        return value


def _json(path, payload):
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def _write_rows(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def paper_schema_export(source, destination, *, sample_limit=None, dry_run=False, allow_unresolved=False, workers=4, progress=None):
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if destination.exists() or destination.is_symlink():
        raise ValueError("paper_export_destination_exists")
    if destination.resolve() != destination or destination.is_relative_to(source):
        raise ValueError("paper_export_destination_unsafe")
    if sample_limit is not None and (type(sample_limit) is not int or sample_limit <= 0):
        raise ValueError("paper_export_invalid_sample_limit")

    def log(event, **detail):
        if progress:
            progress({"event": event, **detail})

    inputs = {"metadata.jsonl": ("metadata_repaired_v1.jsonl", "metadata.jsonl"), **TASK_INPUT_FILES}
    paths = {}
    for output, candidates in inputs.items():
        chosen = next((source / name for name in candidates if (source / name).is_file()), None)
        if chosen is None:
            raise ValueError("paper_export_input_missing:" + output)
        paths[output] = chosen
    input_hashes = {key: {"file": path.name, "sha256": digest(path), "stat": _signature(path)} for key, path in paths.items()}
    log("inputs_hashed", inputs={key: item["file"] for key, item in input_hashes.items()})
    metadata = [_project_metadata(row) for _, row in _rows(paths["metadata.jsonl"])]
    log("metadata_projected", records=len(metadata))
    tasks, lineage = {}, []
    for filename in TASK_INPUT_FILES:
        projected = []
        for number, row in _rows(paths[filename]):
            if sample_limit is not None and len(projected) >= sample_limit:
                break
            target = _project_task(row, filename, number)
            projected.append(target)
            lineage.append({"file": filename, "record_key": target["record_key"], "source_record_key": row.get("record_key"), "source_metadata_ref": row.get("metadata_ref"), "source_file": paths[filename].name,
                            "source_line": number, "source_record_sha256": hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()})
        tasks[filename] = projected
        log("tasks_projected", file=filename, records=len(projected))
    # Rehearsals retain all metadata rather than slicing away dependencies.
    issues, stats = _validate_closure(tasks, metadata)
    path_review = []
    private_prefixes = _private_prefixes(source)
    for filename, rows in [("metadata.jsonl", metadata), *tasks.items()]:
        for number, row in enumerate(rows, 1):
            for path, text in _strings(row):
                normalized = text.replace("\\", "/").lower()
                if any(prefix in normalized for prefix in private_prefixes):
                    issues.append({"file": filename, "line": number, "kind": "author_machine_path_in_retained_field", "detail": path})
                elif LOCAL_PATH.search(text):
                    path_review.append({"file": filename, "line": number, "field": path,
                                        "reason": "Path-like original supervision; contextual review required."})
    assets = Assets(source)
    metadata = [assets.rewrite(row) for row in metadata]
    tasks = {name: [assets.rewrite(row) for row in rows] for name, rows in tasks.items()}
    counts = {"metadata": len(metadata), **{Path(name).stem: len(rows) for name, rows in tasks.items()}}
    kinds = dict(Counter(item["kind"] for item in issues))
    report = {"paper_schema_version": PAPER_SCHEMA_VERSION, "counts": counts, "sample_limit": sample_limit,
              "inputs": input_hashes, "image_count": len(assets.entries),
              "image_bytes": sum(item["stat"][2] for item in assets.entries.values()),
              "reference_closure": {"ok": not issues, "issue_count": len(issues), "issue_kinds": kinds, "stats": dict(stats)},
              "publish_ready": not issues, "semantic_visual_certification": False,
              "path_like_supervision_texts_for_review": len(path_review),
              "technical_extensions": ["record identifiers", "metadata_refs", "frame_registry", "evidence_registry", "image_path"],
              "policy": "Retain every selected instruction/metadata record; never guess evidence links or silently drop exceptions."}
    log("preflight_complete", counts=counts, images=len(assets.entries), issues=kinds)
    if issues and not allow_unresolved:
        raise ValueError("paper_export_reference_closure_failed:" + json.dumps(kinds))
    if dry_run:
        return report
    if shutil.disk_usage(destination.parent).free < report["image_bytes"] + 5 * 1024**3:
        raise ValueError("paper_export_insufficient_disk_space")
    for name, path in paths.items():
        if _signature(path) != input_hashes[name]["stat"] or digest(path) != input_hashes[name]["sha256"]:
            raise ValueError("paper_export_input_changed")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()
    for filename, rows in [("metadata.jsonl", metadata), *tasks.items()]:
        _write_rows(destination / filename, rows)
    _write_rows(destination / "reference_exceptions.jsonl", issues)
    _write_rows(destination / "path_text_review.jsonl", path_review)
    _write_rows(destination / "record_lineage.jsonl", lineage)

    def materialize(pair):
        relative, item = pair
        src = item["source"]
        if _signature(src) != item["stat"]:
            raise ValueError("paper_export_image_changed")
        with Image.open(src) as im:
            im.verify()
        with Image.open(src) as im:
            im.load()
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp", "GIF": ".gif"}.get(im.format)
            dims = [im.width, im.height]
        sha = digest(src)
        if PurePosixPath(relative).stem != sha or PurePosixPath(relative).suffix.lower() != extension:
            raise ValueError("paper_export_image_not_content_addressed:" + relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as read, target.open("xb") as write:
            shutil.copyfileobj(read, write, 1024 * 1024)
        if digest(target) != sha or _signature(src) != item["stat"]:
            raise ValueError("paper_export_image_copy_mismatch")
        if target.stat().st_ino == src.stat().st_ino and target.stat().st_dev == src.stat().st_dev:
            raise ValueError("paper_export_image_not_independent")
        return {"path": relative, "sha256": sha, "bytes": item["stat"][2], "dimensions": dims}

    image_manifest = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for entry in executor.map(materialize, sorted(assets.entries.items())):
            image_manifest.append(entry)
            if len(image_manifest) % 5000 == 0:
                log("images_copied", done=len(image_manifest), total=len(assets.entries))
    _write_rows(destination / "image_manifest.jsonl", image_manifest)
    for name, path in paths.items():
        if digest(path) != input_hashes[name]["sha256"] or _signature(path) != input_hashes[name]["stat"]:
            raise ValueError("paper_export_input_changed")
    report["input_hashes_unchanged"] = True
    report["images_independent_copies"] = True
    report["all_images_SHA_and_full_decode_verified"] = True
    report["data_files"] = [{"path": path.name, "sha256": digest(path)} for path in sorted(destination.iterdir()) if path.is_file()]
    report["inputs"] = {key: {k: v for k, v in item.items() if k != "stat"} for key, item in input_hashes.items()}
    report["complete"], report["ok"] = True, True
    _json(destination / "schema_report.json", report)
    _json(destination / "paper_manifest.json", report)
    log("complete", publish_ready=report["publish_ready"], counts=counts)
    return report


def verify_paper_package(root, *, rehash_images=True):
    root = Path(root).resolve(strict=True)
    report = json.loads((root / "paper_manifest.json").read_text(encoding="utf-8"))
    if not report.get("complete") or report.get("paper_schema_version") != PAPER_SCHEMA_VERSION:
        raise ValueError("paper_export_report_incomplete")
    for entry in report["data_files"]:
        if digest(_safe_file(root, entry["path"])) != entry["sha256"]:
            raise ValueError("paper_export_data_mismatch")
    images = {}
    for _, entry in _rows(root / "image_manifest.jsonl"):
        path = _safe_file(root, entry["path"])
        if rehash_images and digest(path) != entry["sha256"]:
            raise ValueError("paper_export_image_mismatch")
        images[entry["path"]] = entry
    counts = {}
    for filename in ["metadata.jsonl", *TASK_INPUT_FILES]:
        count = 0
        for _, row in _rows(root / filename):
            count += 1
            for _, text in _strings(row):
                if text.startswith("images/") and text not in images:
                    raise ValueError("paper_export_image_ref_unresolved")
        counts[Path(filename).stem] = count
    if counts != report["counts"] or len(images) != report["image_count"]:
        raise ValueError("paper_export_count_mismatch")
    return {"ok": True, "publish_ready": report["publish_ready"], "counts": counts, "image_count": len(images),
            "issue_count": report["reference_closure"]["issue_count"]}
