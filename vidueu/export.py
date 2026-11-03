"""Copy a completed workspace into a self-contained, verifiable image package."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image

from vidueu.progress import capability, source_directories

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
IMAGE_FIELDS = {"img", "image", "images", "image_path", "image_paths", "source_img", "target_img", "mask_img",
                "source_image", "target_image", "ground_truth_image", "reference_image", "reference_images",
                "input_image", "output_image", "frame_path", "frame_paths", "keyframes", "segment_keyframes",
                "frames", "path", "file", "filename", "url"}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inside(path, roots):
    return any(path == root or root in path.parents for root in roots)


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width < 1 or image.height < 1:
                raise ValueError("empty image")
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError("image_corrupt") from exc


def rewrite_images(value, resolve, field=""):
    if isinstance(value, dict):
        typed_image = value.get("type") in {"image", "image_url"}
        return {key: child if key == "_provenance" else rewrite_images(child, resolve,
                    "image" if typed_image and key in {"content", "value"} else key)
                for key, child in value.items()}
    if isinstance(value, list):
        return [rewrite_images(child, resolve, field) for child in value]
    image_field = (field in IMAGE_FIELDS or field.startswith("image_") or
                   field.endswith(("_image", "_images", "_image_path", "_image_paths", "_img", "_imgs")))
    if field.endswith(("_id", "_ids")):
        image_field = False
    if isinstance(value, str) and value and image_field:
        suffix = Path(urlsplit(value).path).suffix.lower()
        # Frame IDs and descriptive strings are not image files. Explicit image
        # references must be local decodable files, never fetched during export.
        if suffix in IMAGE_SUFFIXES or value.startswith("data:image/") or field in {"source_img", "target_img", "mask_img", "image_path"}:
            return resolve(value)
    return value


def export_run(root, destination, *, asset_roots=(), dry_run=False):
    root = Path(root).expanduser().resolve(strict=True)
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if destination.exists() or destination.is_symlink():
        raise ValueError("export_destination_exists")
    if destination.resolve() != destination:
        raise ValueError("export_destination_symlink")
    if inside(destination, [root]):
        raise ValueError("export_destination_inside_source")
    roots = [root, *(Path(p).expanduser().resolve(strict=True) for p in asset_roots)]
    sources = source_directories(root)
    if not sources:
        raise ValueError("dataset_source_missing")
    images, snapshots, rows = {}, {}, []
    explicit_keys = {}
    counts = Counter()

    def source_file(path):
        real = path.resolve(strict=True)
        if not inside(real, [root]):
            raise ValueError("export_source_outside_root")
        content = real.read_bytes()
        snapshots[real] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8-sig")

    for source in sources:
        def resolve_image(value):
            path = Path(value).expanduser()
            if not path.is_absolute() and urlsplit(value).scheme:
                raise ValueError("export_image_must_be_local")
            options = [path] if path.is_absolute() else [source / path, *(r / path for r in roots)]
            found = {p.resolve() for p in options if p.is_file()}
            if not found:
                raise ValueError("image_missing")
            if len(found) != 1:
                raise ValueError("ambiguous_image_reference")
            real = found.pop()
            if not inside(real, roots):
                raise ValueError("image_outside_asset_roots")
            sha = digest(real)
            valid_image(real)
            if digest(real) != sha:
                raise ValueError("export_source_changed")
            # Canonical extension comes from decoded content, not its filename.
            with Image.open(real) as image:
                extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp",
                             "TIFF": ".tiff", "GIF": ".gif"}.get(image.format)
            if not extension:
                raise ValueError("unsupported_image_format")
            relative = "images/" + sha + extension
            images.setdefault(relative, {"sha256": sha, "bytes": real.stat().st_size, "sources": set()})["sources"].add(real)
            snapshots[real] = sha
            return relative

        data = source / "output" / "data.jsonl"
        inputs = [(data, None)]
        metadata = sorted((source / "logs" / "debug").glob("*_METADATA.json"))
        if not metadata:
            raise ValueError("metadata_snapshot_missing")
        inputs += [(path, "metadata") for path in metadata]
        for path, kind in inputs:
            text = source_file(path)
            records = [text] if kind else text.splitlines()
            for number, line in enumerate(records, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("dataset_row_must_be_object")
                group = kind or capability(row)
                if kind and not isinstance(row.get("metadata"), dict):
                    raise ValueError("metadata_snapshot_invalid")
                original_sha = hashlib.sha256(line.encode()).hexdigest()
                key = row.get("record_key")
                if key is not None:
                    identity = (source.name, str(key))
                    if identity in explicit_keys:
                        raise ValueError("duplicate_record_key")
                    explicit_keys[identity] = original_sha
                try:
                    packaged = rewrite_images(copy.deepcopy(row), resolve_image)
                except ValueError as exc:
                    exc.source_location = {"source_file": str(path), "source_line": number}
                    raise
                provenance = {"source_file": str(path), "source_line": number, "source_sha256": snapshots[path.resolve()],
                              "original_row_sha256": original_sha, "source_name": source.name}
                # Preserve any producer-provided provenance; export adds its own
                # separate namespace and does not replace source identifiers.
                packaged["_export_provenance"] = provenance
                if key is None:
                    packaged["record_key"] = hashlib.sha256(f"{path.relative_to(root)}:{number}:{original_sha}".encode()).hexdigest()
                rows.append((group, packaged))
                counts[group] += 1

    plan = {"schema_version": 1, "source_root": str(root), "destination": str(destination),
            "dry_run": dry_run, "counts": dict(counts), "image_count": len(images),
            "image_bytes": sum(item["bytes"] for item in images.values()),
            "source_files": [{"path": str(path), "sha256": sha} for path, sha in sorted(snapshots.items())],
            "image_manifest": [{"path": relative, **{k: v for k, v in item.items() if k != "sources"}}
                               for relative, item in sorted(images.items())]}

    def unchanged():
        for path, sha in snapshots.items():
            if not path.is_file() or digest(path) != sha:
                raise ValueError("export_source_changed")

    unchanged()
    if dry_run:
        return plan
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "images").mkdir()
    for relative, entry in images.items():
        shutil.copyfile(sorted(entry["sources"])[0], destination / relative)
        if digest(destination / relative) != entry["sha256"]:
            raise ValueError("export_copy_mismatch")
    for group in counts:
        with (destination / (group + ".jsonl")).open("x", encoding="utf-8") as stream:
            for row_group, row in rows:
                if row_group == group:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    unchanged()
    plan["data_files"] = [{"path": group + ".jsonl", "sha256": digest(destination / (group + ".jsonl"))} for group in counts]
    plan["complete"] = True
    with (destination / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2)
    verify_package(destination)
    return plan


def verify_package(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("export_package_incomplete")
    for item in manifest["image_manifest"] + manifest["data_files"]:
        path = root / item["path"]
        if not inside(path.resolve(), [root]) or not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError("export_package_mismatch")
    declared = {item["path"] for item in manifest["image_manifest"]}

    def check(value):
        if value not in declared:
            raise ValueError("export_undeclared_image")
        valid_image(root / value)
        return value

    counts = Counter()
    for item in manifest["data_files"]:
        with (root / item["path"]).open(encoding="utf-8") as stream:
            for line in stream:
                rewrite_images(json.loads(line), check)
                counts[Path(item["path"]).stem] += 1
    if dict(counts) != manifest["counts"]:
        raise ValueError("export_count_mismatch")
    return {"ok": True, "counts": dict(counts), "image_count": len(declared)}
