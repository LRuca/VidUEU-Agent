from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import yaml

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

import run_pipeline


class FakePlanner:
    instances: list["FakePlanner"] = []
    _write_lock = threading.Lock()

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_manager = types.SimpleNamespace(get_vlm_pool_size=lambda: 8)
        self.calls: list[str] = []
        self.concurrent_mode_enabled = False
        self.closed = False
        self.acquired_videos: list[str] = []
        self.released_video_workers = 0
        FakePlanner.instances.append(self)

    def enable_concurrent_video_mode(self, enabled: bool) -> None:
        self.concurrent_mode_enabled = bool(enabled)

    def warm_vlm_pool(self) -> None:
        return None

    def acquire_video_vlm_worker(self, video_path: str = "") -> None:
        self.acquired_videos.append(str(video_path))

    def release_video_vlm_worker(self) -> None:
        self.released_video_workers += 1

    def close(self) -> None:
        self.closed = True

    def run_pipeline(self, video_path: str):
        self.calls.append(video_path)
        time.sleep(0.05)
        items = [
            {"benchmark": "llava_bench", "video_path": str(Path(video_path).resolve())},
            {"benchmark": "magicbrush", "video_path": str(Path(video_path).resolve())},
            {"benchmark": "mme", "video_path": str(Path(video_path).resolve())},
            {"benchmark": "unify", "video_path": str(Path(video_path).resolve())},
        ]

        output_dir = Path(self.config["paths"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        data_fp = output_dir / "data.jsonl"
        bench_dir = output_dir / "benchmarks_jsonl"
        bench_dir.mkdir(parents=True, exist_ok=True)
        log_dir = Path(self.config["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        segment_log = log_dir / f"{Path(video_path).stem}_segments.jsonl"
        with self._write_lock:
            with data_fp.open("a", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            for item in items:
                bench = run_pipeline._normalize_benchmark_name(item["benchmark"])
                with (bench_dir / f"{bench}.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({**item, "benchmark": bench}, ensure_ascii=False) + "\n")
            with segment_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"stage": "T4", "segment_index": 0, "video_path": str(Path(video_path).resolve())}, ensure_ascii=False) + "\n")

        return items


class FakeDashboardServer:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.server_close_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def server_close(self) -> None:
        self.server_close_called = True


class RunPipelineMockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.gettempdir()) / "agentpack_test_tmp" / f"run_pipeline_mock_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        FakePlanner.instances.clear()

    def test_collect_videos_filters_fixed_files(self) -> None:
        video_dir = self.root / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        keep = video_dir / "keep.mp4"
        skip = video_dir / "skip_fixed.mp4"
        skip_fix = video_dir / "skip.fix.mp4"
        skip_dot_fixed = video_dir / "skip.fixed.mp4"
        keep.write_bytes(b"video")
        skip.write_bytes(b"video")
        skip_fix.write_bytes(b"video")
        skip_dot_fixed.write_bytes(b"video")

        videos = run_pipeline.collect_videos(str(video_dir))

        self.assertEqual(videos, [str(keep)])

    def test_collect_videos_recurses_into_subdirectories(self) -> None:
        video_dir = self.root / "videos_recursive"
        nested = video_dir / "sub" / "deep"
        nested.mkdir(parents=True, exist_ok=True)
        keep_root = video_dir / "root.mp4"
        keep_nested = nested / "nested.mkv"
        skip_nested = nested / "nested_fixed.mov"
        skip_nested_fix = nested / "nested.fix.mov"
        keep_root.write_bytes(b"video")
        keep_nested.write_bytes(b"video")
        skip_nested.write_bytes(b"video")
        skip_nested_fix.write_bytes(b"video")

        videos = run_pipeline.collect_videos(str(video_dir))

        self.assertEqual(videos, sorted([str(keep_root), str(keep_nested)]))

    def test_collect_videos_honors_scan_max_depth(self) -> None:
        video_dir = self.root / "videos_depth"
        nested = video_dir / "sub" / "deep"
        nested.mkdir(parents=True, exist_ok=True)
        keep_root = video_dir / "root.mp4"
        skip_nested = nested / "nested.mp4"
        keep_root.write_bytes(b"video")
        skip_nested.write_bytes(b"video")

        videos = run_pipeline.collect_videos(str(video_dir), {"max_depth": 1})

        self.assertEqual(videos, [str(keep_root)])

    def test_import_does_not_load_pipeline_planner(self) -> None:
        self.assertIsNone(run_pipeline.PipelinePlanner)
        self.assertNotIn("orchestrator.planner", sys.modules)

    def test_existing_segment_log_marks_video_as_completed(self) -> None:
        video = self.root / "existing_items.mp4"
        video.write_bytes(b"video")
        video_path = str(video.resolve())
        config = {"paths": {"log_dir": str(self.root / "logs")}}
        segment_log = run_pipeline._video_segment_log_path(config, video_path)
        segment_log.parent.mkdir(parents=True, exist_ok=True)
        self.assertFalse(run_pipeline._video_has_segment_log(config, video_path))
        segment_log.write_text("{}\n", encoding="utf-8")
        output_index = {
            video_path: {
                "total_items": 2,
                "qa_items": 1,
                "by_bench": {bench: 0 for bench in run_pipeline.BENCH_KEYS},
            }
        }
        manifest = {
            "status": "done",
            "video_path": video_path,
            "total_items": 0,
            "qa_items": 0,
            "by_bench": {bench: 0 for bench in run_pipeline.BENCH_KEYS},
        }

        self.assertTrue(run_pipeline._video_has_segment_log(config, video_path))
        self.assertEqual(
            run_pipeline._completed_video_stats(manifest, output_index, video_path)["total_items"],
            2,
        )

    def test_main_runs_end_to_end_with_mock_planner_and_workspace_outputs(self) -> None:
        video_dir = self.root / "videos"
        output_root = self.root / "runs"
        config_path = self.root / "config.yaml"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / "sample.mp4"
        fixed_path = video_dir / "sample_fixed.mp4"
        video_path.write_bytes(b"video")
        fixed_path.write_bytes(b"video")
        config_path.write_text(yaml.safe_dump({}), encoding="utf-8")

        stdout = io.StringIO()
        argv = [
            "run_pipeline.py",
            "--input",
            str(video_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--source-name",
            "mock_source",
        ]
        fake_dashboard_server = FakeDashboardServer()

        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch(
            "run_pipeline.start_dashboard_server", return_value=fake_dashboard_server
        ) as mock_start_dashboard, patch.object(sys, "argv", argv):
            with redirect_stdout(stdout):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 1)
        planner = FakePlanner.instances[0]
        self.assertEqual(planner.calls, [str(video_path)])
        self.assertTrue(planner.closed)
        mock_start_dashboard.assert_called_once_with(str(output_root), run_pipeline.DEFAULT_DASHBOARD_PORT)
        self.assertTrue(fake_dashboard_server.shutdown_called)
        self.assertTrue(fake_dashboard_server.server_close_called)

        expected_work_root = output_root.resolve() / "mock_source"
        expected_dataset_output = expected_work_root / "output" / "data.jsonl"
        self.assertEqual(planner.config["paths"]["temp_dir"], str(expected_work_root / "temp"))
        self.assertEqual(planner.config["paths"]["output_dir"], str(expected_work_root / "output"))
        self.assertEqual(planner.config["paths"]["semantic_dir"], str(expected_work_root / "semantic"))
        self.assertEqual(planner.config["paths"]["log_dir"], str(expected_work_root / "logs"))
        self.assertEqual(planner.config["paths"]["debug_dir"], str(expected_work_root / "logs" / "debug"))
        self.assertEqual(planner.config["paths"]["dataset_output"], str(expected_dataset_output))
        self.assertEqual(planner.config["dataset"]["output_file"], str(expected_dataset_output))
        self.assertEqual(planner.config["tasks"]["T4_dataset"]["output_file"], str(expected_dataset_output))
        self.assertEqual(planner.config["runtime"]["source_name"], "mock_source")
        self.assertEqual(planner.config["runtime"]["work_root"], str(expected_work_root))

        output = stdout.getvalue()
        self.assertIn("Workspace enabled:", output)
        self.assertIn("Total videos found: 1", output)
        self.assertIn("Videos selected for this run: 1", output)
        self.assertIn("by_bench: llava=1, edit=1, mmt=0, mme=1, visdial=0, mmdu=0, unify=1", output)
        self.assertIn("Done.", output)
        self.assertTrue((expected_work_root / "temp").exists())
        self.assertTrue((expected_work_root / "output").exists())
        self.assertTrue((expected_work_root / "semantic").exists())
        self.assertTrue((expected_work_root / "logs" / "debug").exists())

    def test_default_mode_skips_when_segment_log_exists_and_reruns_when_log_missing(self) -> None:
        video_dir = self.root / "videos"
        output_root = self.root / "runs"
        config_path = self.root / "config.yaml"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / "sample.mp4"
        video_path.write_bytes(b"video")
        config_path.write_text(yaml.safe_dump({}), encoding="utf-8")

        argv = [
            "run_pipeline.py",
            "--input",
            str(video_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--source-name",
            "mock_source",
            "--no-dashboard",
        ]

        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 1)
        self.assertEqual(FakePlanner.instances[0].calls, [str(video_path)])
        self.assertTrue(FakePlanner.instances[0].closed)

        stdout_skip = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout_skip):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 1)
        self.assertIn("Skip completed video:", stdout_skip.getvalue())

        data_fp = output_root.resolve() / "mock_source" / "output" / "data.jsonl"
        lines = data_fp.read_text(encoding="utf-8").splitlines()
        data_fp.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""), encoding="utf-8")

        stdout_partial_skip = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout_partial_skip):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 1)
        self.assertIn("Skip completed video:", stdout_partial_skip.getvalue())
        self.assertIn("segment_log=", stdout_partial_skip.getvalue())
        self.assertIn("items(total)=3", stdout_partial_skip.getvalue())

        data_fp.write_text("", encoding="utf-8")
        log_fp = output_root.resolve() / "mock_source" / "logs" / f"{video_path.stem}_segments.jsonl"
        log_fp.unlink()

        stdout_rerun = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout_rerun):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 2)
        self.assertEqual(FakePlanner.instances[1].calls, [str(video_path)])
        self.assertIn("Detected incomplete/stale outputs for video, rerun:", stdout_rerun.getvalue())

    def test_force_overwrite_mode_reruns_and_overwrites_current_output_dir(self) -> None:
        video_dir = self.root / "videos_force"
        output_root = self.root / "runs_force"
        config_path = self.root / "config_force.yaml"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_path = video_dir / "sample.mp4"
        video_path.write_bytes(b"video")
        config_path.write_text(
            yaml.safe_dump({"runtime": {"video_run_mode": "force_overwrite"}}),
            encoding="utf-8",
        )

        argv = [
            "run_pipeline.py",
            "--input",
            str(video_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--source-name",
            "mock_source",
            "--no-dashboard",
        ]

        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            run_pipeline.main()

        data_fp = output_root.resolve() / "mock_source" / "output" / "data.jsonl"
        first_lines = data_fp.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(first_lines), 4)

        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            run_pipeline.main()

        second_lines = data_fp.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(second_lines), 4)

    def test_config_input_paths_runs_multiple_inputs_in_order(self) -> None:
        video_dir_a = self.root / "videos_a"
        video_dir_b = self.root / "videos_b"
        output_root = self.root / "runs_multi"
        config_path = self.root / "config_multi.yaml"
        video_dir_a.mkdir(parents=True, exist_ok=True)
        video_dir_b.mkdir(parents=True, exist_ok=True)

        video_a = video_dir_a / "a.mp4"
        video_b = video_dir_b / "b.mp4"
        video_a.write_bytes(b"video")
        video_b.write_bytes(b"video")
        config_path.write_text(
            yaml.safe_dump(
                {
                    "runtime": {
                        "input_paths": [
                            str(video_dir_a),
                            {"path": str(video_dir_b), "source_name": "custom_b"},
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        argv = [
            "run_pipeline.py",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--no-dashboard",
        ]

        stdout = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 2)
        self.assertEqual(FakePlanner.instances[0].calls, [str(video_a)])
        self.assertEqual(FakePlanner.instances[1].calls, [str(video_b)])
        self.assertTrue((output_root.resolve() / "videos_a" / "output" / "data.jsonl").exists())
        self.assertTrue((output_root.resolve() / "custom_b" / "output" / "data.jsonl").exists())
        output = stdout.getvalue()
        self.assertIn(f"Input root: {video_dir_a}", output)
        self.assertIn(f"Input root: {video_dir_b}", output)
        self.assertIn("ALL INPUTS summary:", output)

    def test_video_parallelism_uses_single_shared_planner(self) -> None:
        video_dir = self.root / "videos_parallel"
        output_root = self.root / "runs_parallel"
        config_path = self.root / "config_parallel.yaml"
        video_dir.mkdir(parents=True, exist_ok=True)

        video_a = video_dir / "a.mp4"
        video_b = video_dir / "b.mp4"
        video_a.write_bytes(b"video")
        video_b.write_bytes(b"video")
        config_path.write_text(
            yaml.safe_dump(
                {
                    "runtime": {
                        "video_parallelism": {
                            "enabled": True,
                            "max_workers": 2,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        argv = [
            "run_pipeline.py",
            "--input",
            str(video_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--source-name",
            "parallel_source",
            "--no-dashboard",
        ]

        stdout = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout):
                run_pipeline.main()

        self.assertEqual(len(FakePlanner.instances), 1)
        planner = FakePlanner.instances[0]
        self.assertCountEqual(planner.calls, [str(video_a), str(video_b)])
        self.assertCountEqual(planner.acquired_videos, [str(video_a), str(video_b)])
        self.assertEqual(planner.released_video_workers, 2)
        self.assertTrue(planner.closed)
        self.assertIn("Video parallelism: enabled | workers=2", stdout.getvalue())

    def test_auto_video_parallelism_uses_vlm_pool_size(self) -> None:
        video_dir = self.root / "videos_parallel_auto"
        output_root = self.root / "runs_parallel_auto"
        config_path = self.root / "config_parallel_auto.yaml"
        video_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(5):
            (video_dir / f"{idx}.mp4").write_bytes(b"video")
        config_path.write_text(
            yaml.safe_dump(
                {
                    "runtime": {
                        "video_parallelism": {"enabled": True, "max_workers": 0},
                        "single_gpu_scheduler": {"devices": ["cuda:0", "cuda:1", "cuda:2"]},
                    }
                }
            ),
            encoding="utf-8",
        )

        argv = [
            "run_pipeline.py",
            "--input",
            str(video_dir),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--source-name",
            "parallel_auto_source",
            "--no-dashboard",
        ]

        stdout = io.StringIO()
        with patch("run_pipeline.PipelinePlanner", FakePlanner), patch.object(sys, "argv", argv):
            with redirect_stdout(stdout):
                run_pipeline.main()

        self.assertIn("Video parallelism: enabled | workers=5", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
