from __future__ import annotations

import os
import shutil
import unittest
from types import SimpleNamespace
from pathlib import Path
import uuid
from contextlib import contextmanager

from executor.tools.asr_tool import ASRToolAdapter
from executor.tools.base import ToolContext
from executor.tools.ocr_tool import OCRToolAdapter
from executor.tools.optical_flow_tool import OpticalFlowToolAdapter
from executor.tools.yolo_tool import YoloToolAdapter


class FakeModelManager:
    def __init__(self, resources: dict) -> None:
        self.resources = resources

    def get_resource(self, name: str):
        return self.resources[name]

    def run_with_resource(self, name: str, fn, *args, **kwargs):
        return fn(self.resources[name], *args, **kwargs)

    @contextmanager
    def use_resource(self, name: str):
        yield self.resources[name]


class FakeOCRResource:
    def __init__(self, raw: dict) -> None:
        self.raw = raw

    def run(self, frame_paths: list[str]) -> dict:
        return {path: self.raw.get(path, self.raw.get(os.path.basename(path), [])) for path in frame_paths}


class FakeYOLOResource:
    def __init__(self, raw: dict) -> None:
        self.raw = raw

    def run(self, frame_paths: list[str]) -> dict:
        return {path: self.raw.get(path, []) for path in frame_paths}


class FakeRAFTResource:
    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.calls: list[list[str]] = []

    def compute_flow_for_frame_paths(self, pair_paths: list[str], iters: int = 20, resize_short_side: int = 384) -> list[float]:
        _ = (iters, resize_short_side)
        self.calls.append(list(pair_paths))
        score = self.scores[len(self.calls) - 1] if len(self.calls) - 1 < len(self.scores) else 0.0
        return [score]


class ToolAdapterRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = str(Path("logs") / "test_tmp" / f"tool_regressions_{uuid.uuid4().hex[:8]}")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def test_ocr_adapter_uses_basename_fallback_and_deduplicates_lines(self) -> None:
        adapter = OCRToolAdapter()
        img_path = os.path.join(self.temp_dir, "frame_a.jpg")
        ctx = ToolContext(
            shot=None,
            candidates=[{"shot_id": "shot_001", "frame_index": 4, "img": img_path}],
            pre_gate_candidates=[],
            video_path="demo.mp4",
            t2_workdir=self.temp_dir,
            model_manager=FakeModelManager(
                {
                    "ocr": FakeOCRResource(
                        {
                            "frame_a.jpg": [
                                {"text": " Upload File "},
                                {"text": "upload   file"},
                                {"text": "Step 1"},
                            ]
                        }
                    )
                }
            ),
            config={},
            tool_spec={},
        )

        result = adapter.run_shot(ctx)
        hint = result["semantic"]["frame_hints"][0]

        self.assertEqual(hint["text"], "Visible text present: Upload File")
        self.assertEqual(hint["attributes"]["text_lines"], ["Upload File", "Step 1"])
        self.assertEqual(hint["attributes"]["line_count"], 2)
        self.assertTrue(hint["attributes"]["has_text"])
        self.assertIn("Upload File | Step 1", hint["evidence"]["merged_preview"])

    def test_asr_adapter_short_shot_uses_zero_signal_fallback_without_audio_extraction(self) -> None:
        adapter = ASRToolAdapter()
        shot = SimpleNamespace(shot_id="shot_001", start_sec=0.0, end_sec=0.02)
        ctx = ToolContext(
            shot=shot,
            candidates=[{"shot_id": "shot_001", "frame_index": 1, "img": "f.jpg", "sec": 0.01}],
            pre_gate_candidates=[],
            video_path="demo.mp4",
            t2_workdir=self.temp_dir,
            model_manager=FakeModelManager({}),
            config={},
            tool_spec={},
        )

        result = adapter.run_shot(ctx)
        hint = result["semantic"]["frame_hints"][0]

        self.assertEqual(result["raw"], {"text": "", "segments": []})
        self.assertEqual(result["semantic"]["summary"], "")
        self.assertEqual(hint["score"], 0.0)
        self.assertFalse(hint["attributes"]["has_speech"])
        self.assertEqual(hint["evidence"]["window_seconds"], 1.5)

    def test_yolo_adapter_sorts_by_score_and_deduplicates_labels(self) -> None:
        adapter = YoloToolAdapter()
        img_path = os.path.join(self.temp_dir, "frame_b.jpg")
        ctx = ToolContext(
            shot=None,
            candidates=[{"shot_id": "shot_002", "frame_index": 2, "img": img_path}],
            pre_gate_candidates=[],
            video_path="demo.mp4",
            t2_workdir=self.temp_dir,
            model_manager=FakeModelManager(
                {
                    "yolo": FakeYOLOResource(
                        {
                            img_path: [
                                {"label": "button", "score": 0.2},
                                {"label": "dialog", "score": 0.9},
                                {"label": "button", "score": 0.8},
                            ]
                        }
                    )
                }
            ),
            config={},
            tool_spec={},
        )

        result = adapter.run_shot(ctx)
        hint = result["semantic"]["frame_hints"][0]

        self.assertEqual(hint["attributes"]["labels"], ["dialog", "button"])
        self.assertEqual(hint["text"], "Visible entities: dialog, button")
        self.assertEqual(hint["attributes"]["top_scores"], [0.9, 0.8, 0.2])
        self.assertEqual(len(hint["evidence"]["detections"]), 3)

    def test_optical_flow_adapter_only_scores_pairs_touching_usable_candidates(self) -> None:
        adapter = OpticalFlowToolAdapter()
        raft = FakeRAFTResource(scores=[0.1, 0.5])
        pre_gate_candidates = [
            {"shot_id": "shot_003", "frame_index": 0, "img": "f0.jpg", "sec": 0.0},
            {"shot_id": "shot_003", "frame_index": 1, "img": "f1.jpg", "sec": 1.0},
            {"shot_id": "shot_003", "frame_index": 2, "img": "f2.jpg", "sec": 2.0},
        ]
        usable_candidates = [
            {"shot_id": "shot_003", "frame_index": 1, "img": "f1.jpg", "sec": 1.0},
            {"shot_id": "shot_003", "frame_index": 2, "img": "f2.jpg", "sec": 2.0},
        ]
        ctx = ToolContext(
            shot=None,
            candidates=usable_candidates,
            pre_gate_candidates=pre_gate_candidates,
            video_path="demo.mp4",
            t2_workdir=self.temp_dir,
            model_manager=FakeModelManager({"raft": raft}),
            config={},
            tool_spec={},
        )

        result = adapter.run_shot(ctx)

        self.assertEqual(raft.calls, [["f0.jpg", "f1.jpg"], ["f1.jpg", "f2.jpg"]])
        self.assertEqual(result["raw"]["raw_scores"], [0.1, 0.5])
        frame1, frame2 = result["semantic"]["frame_hints"]
        self.assertEqual(frame1["attributes"]["flow_levels"], ["high", "low"])
        self.assertEqual(frame1["score"], 1.0)
        self.assertEqual(frame2["attributes"]["flow_levels"], ["high"])
        self.assertEqual(frame2["text"], "Motion level nearby: high")


if __name__ == "__main__":
    unittest.main()
