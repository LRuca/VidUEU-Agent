"""Offline CPU contract validation with explicit simulated tool/model results."""
from __future__ import annotations

import builtins
import io
import json
import os
import shutil
import socket
import subprocess
import sys
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image, ImageDraw

from dataset.builder import DatasetBuilder
from vidueu.config import template
from vidueu.export import digest, export_run, verify_package
from vidueu.paper_export import paper_schema_export, verify_paper_package
from vidueu_config import load_config


def offline_guards():
    stack = ExitStack()
    importer = builtins.__import__

    def cpu_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "tensorflow", "cupy", "vllm", "paddle", "openai", "transformers"}:
            raise RuntimeError("smoke_model_import_forbidden")
        return importer(name, *args, **kwargs)

    def no_network(*args, **kwargs):
        raise RuntimeError("smoke_network_forbidden")

    stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
    stack.enter_context(patch("builtins.__import__", cpu_import))
    stack.enter_context(patch.object(socket, "socket", no_network))
    stack.enter_context(patch.object(socket, "create_connection", no_network))
    return stack


class FixturePlanner:
    """Test double at the model boundary; the entrypoint and writer stay real."""

    calls = 0

    def __init__(self, config):
        self.config = config
        self.model_manager = SimpleNamespace(get_vlm_pool_size=lambda: 1)

    def close(self):
        pass

    def enable_concurrent_video_mode(self, enabled):
        if enabled:
            raise RuntimeError("smoke_requires_serial_execution")

    def warm_vlm_pool(self):
        pass

    def run_pipeline(self, video_path):
        type(self).calls += 1
        paths = self.config["paths"]
        images = []
        for index, color in enumerate(("red", "blue")):
            path = Path(paths["temp_dir"]) / f"frame_{index:03d}.png"
            image = Image.new("RGB", (128, 128), "white")
            ImageDraw.Draw(image).rectangle((32, 32, 96, 96), fill=color)
            image.save(path)
            images.append(str(path))
        metadata_group_key = "smoke-metadata-group"
        meta = {
            "metadata_group_key": metadata_group_key,
            "segment": {"video_id": video_path, "start": 0.0, "end": 1.0,
                         "summary": "A square changes from red to blue.", "type": "other"},
            "chapters": [{"chapter_id": "C00", "time_span": [0.0, 1.0], "summary": "A square changes color.", "role": "main"}],
            "shots": [{"shot_id": "S00", "chapter_id": "C00", "time_span": [0.0, 1.0],
                       "summary": "A square changes color.", "role": "main", "scene_type": "demo",
                       "quality_hint": "clear", "info_hint": "color"}],
            "keyframes": [
                {"frame_id": "F00", "shot_id": "S00", "timestamp": 0.0,
                 "caption": "A red square.", "quality": 1.0, "value_score": 1.0,
                 "tool_semantics": {}, "evidence_id": "F00_CAP", "image_path": images[0]},
                {"frame_id": "F01", "shot_id": "S00", "timestamp": 1.0,
                 "caption": "A blue square.", "quality": 1.0, "value_score": 1.0,
                 "tool_semantics": {}, "evidence_id": "F01_CAP", "image_path": images[1]},
            ],
            "transitions": [{"source": "F00", "target": "F01",
                             "atomic_changes": ["square color changes from red to blue"],
                             "same_shot": True, "viewpoint_change": False,
                             "subject_consistent": True, "reliability": 1.0}],
            "video_path": video_path, "segment_index": 0, "start_s": 0.0, "end_s": 1.0,
            "simulated": True, "frame_paths": images, "evidence_refs": ["F00_CAP", "F01_CAP"],
        }
        rows = [
            {"benchmark": "llava_bench", "image": images[0], "question": "What color is the square?", "answer": "Red.",
             "image_pos": 0, "category": "attribute",
             "metadata_refs": [metadata_group_key]},
            {"benchmark": "edit", "source_img": images[0], "target_img": images[1], "mask_img": None,
             "from_pos": 0, "to_pos": 1, "ok": True, "edit_count": 1, "edits": ["Change red to blue."],
             "reason": "Only the square color changes.",
             "instruction": "Change the red square to blue. Do not change anything else.",
             "metadata_ref": metadata_group_key},
            {"benchmark": "unify.isg", "Round_Plan": {"question": {"text": "What changes?", "Frames": ["F00", "F01"]},
             "answer": {"edits": ["change red square to blue"]}},
             "Round_Gen": {"question": {"text": "Generate the target.", "Frames": ["F00"]},
             "answer": {"image": images[1]}},
             "Round_Select": {"question": {"text": "Which target?", "Frames": ["F01"]},
             "answer": "target:F01", "rationale": "The square is blue."},
             "metadata_ref": metadata_group_key},
        ]
        rows = [{**row, "record_key": f"smoke-{i}", "meta": meta, "video_path": video_path,
                 "evidence_refs": meta["evidence_refs"], "simulated": True} for i, row in enumerate(rows)]
        builder = DatasetBuilder(self.config)
        try:
            for row in rows:
                builder.add_item(row)
        finally:
            builder.close()
        debug = Path(paths["debug_dir"])
        with (debug / "fixture_seg000_METADATA.json").open("x", encoding="utf-8") as stream:
            json.dump({"metadata": meta, "gate_c": {"accepted": True, "simulated": True}}, stream)
        log = Path(paths["log_dir"]) / (Path(video_path).stem + "_segments.jsonl")
        with log.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": "T4", "video_path": video_path, "segment_index": 0, "simulated": True}) + "\n")
        return rows


def run_smoke(output):
    root = Path(os.path.abspath(Path(output).expanduser()))
    if root.resolve() != root:
        raise ValueError("smoke_output_symlink")
    root.mkdir(parents=True, exist_ok=False)
    with offline_guards():
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError("missing_system_command")
        videos = root / "videos"
        videos.mkdir()
        poster = videos / "poster.png"
        Image.new("RGB", (128, 128), "white").save(poster)
        video = videos / "fixture.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-n", "-loop", "1", "-i", str(poster),
                        "-t", "1", "-c:v", "libx264", "-threads", "1", "-pix_fmt", "yuv420p", str(video)],
                       check=True, timeout=30)
        video_sha = digest(video)
        config_path = root / "smoke.yaml"
        config = template()
        config.update(input=str(videos), output=str(root / "runs"), source_name="fixture", tools=[])
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        from scripts.doctor import inspect
        report = inspect(load_config(config_path), lightweight=True, media=True)
        if not report["ok"]:
            raise RuntimeError("smoke_preflight_failed")
        import run_pipeline
        argv = ["run_pipeline", "--config", str(config_path), "--no-dashboard"]
        FixturePlanner.calls = 0
        with patch.object(run_pipeline, "PipelinePlanner", FixturePlanner), patch.object(sys, "argv", argv):
            with (root / "pipeline.log").open("x", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
                run_pipeline.main()
                data = root / "runs" / "fixture" / "output" / "data.jsonl"
                before_resume = digest(data)
                run_pipeline.main()
        if FixturePlanner.calls != 1 or digest(data) != before_resume or digest(video) != video_sha:
            raise RuntimeError("smoke_resume_or_source_changed")
        plan = export_run(root / "runs", root / "package", dry_run=True)
        if (root / "package").exists():
            raise RuntimeError("smoke_dry_run_wrote_output")
        export_run(root / "runs", root / "package")
        shutil.copytree(root / "package", root / "relocated_package")
        portable = verify_package(root / "relocated_package")
        if portable["counts"] != {"understanding": 1, "edit": 1, "unify": 1, "metadata": 1} or portable["image_count"] != 2:
            raise RuntimeError("smoke_export_contract_failed")
        paper_plan = paper_schema_export(root / "package", root / "paper_package", sample_limit=1)
        paper_package = verify_paper_package(root / "paper_package")
        if paper_package["counts"] != {"understanding": 1, "edit": 1, "unify": 1, "metadata": 1} or paper_package["image_count"] != 2:
            raise RuntimeError("smoke_paper_export_contract_failed")
        result = {"ok": True, "simulated": True, "cuda_visible_devices": "", "model_calls": 0, "paid_api_calls": 0,
                  "real_optimizer_steps": 0, "network": "blocked", "model_imports": "blocked",
                  "planner_fixture_calls": FixturePlanner.calls, "resume_idempotent": True,
                  "source_unchanged": True, "doctor": report, "portable_export": portable,
                  "export_dry_run_counts": plan["counts"], "paper_export": paper_package,
                  "paper_export_counts": paper_plan["counts"], "output": str(root),
                  "limits": "Contract smoke only; no real tool inference, semantic quality, GPU training or benchmark scores."}
        with (root / "smoke_report.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        return result
