from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager

from executor.tools.asr_tool import ASRToolAdapter
from executor.tools.base import ToolContext
from executor.tasks.whisper_asr import WhisperASR


class FakeModelManager:
    def __init__(self) -> None:
        self.requested = False

    def get_resource(self, name: str):
        self.requested = True
        raise AssertionError(f"ASR resource should not be requested during fallback test: {name}")

    def run_with_resource(self, name: str, fn, *args, **kwargs):
        self.requested = True
        raise AssertionError(f"ASR resource should not be requested during fallback test: {name}")

    @contextmanager
    def use_resource(self, name: str):
        self.requested = True
        raise AssertionError(f"ASR resource should not be requested during fallback test: {name}")
        yield


class FakeASRModelManager(FakeModelManager):
    def __init__(self, response: tuple[str, list[dict]]) -> None:
        super().__init__()
        self.response = response

    def run_with_resource(self, name: str, fn, *args, **kwargs):
        self.requested = True
        if name != "asr":
            raise AssertionError(f"Unexpected ASR resource request: {name}")
        return self.response


class ASRToolAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path("logs") / "test_tmp" / "asr_tool"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    def _make_ctx(self) -> tuple[ASRToolAdapter, ToolContext, FakeModelManager]:
        adapter = ASRToolAdapter()
        model_manager = FakeModelManager()
        shot = SimpleNamespace(shot_id="shot_000", start_sec=0.0, end_sec=3.5)
        ctx = ToolContext(
            shot=shot,
            candidates=[
                {
                    "shot_id": "shot_000",
                    "frame_index": 0,
                    "sec": 0.8,
                    "img": str(self.temp_dir / "frame_0000.jpg"),
                }
            ],
            pre_gate_candidates=[],
            video_path=str(self.temp_dir / "sample.mp4"),
            t2_workdir=str(self.temp_dir),
            model_manager=model_manager,
            config={},
            tool_spec={},
        )
        return adapter, ctx, model_manager

    def test_run_shot_skips_cleanly_when_video_has_no_audio_stream(self) -> None:
        adapter, ctx, model_manager = self._make_ctx()

        with patch.object(adapter, "_probe_has_audio_stream", return_value=False):
            result = adapter.run_shot(ctx)

        self.assertFalse(model_manager.requested)
        self.assertEqual(result["raw"]["text"], "")
        self.assertEqual(result["diagnostics"]["reason"], "no_audio_stream")
        self.assertEqual(len(result["semantic"]["frame_hints"]), 1)
        self.assertFalse(result["semantic"]["frame_hints"][0]["attributes"]["has_speech"])

    def test_run_shot_degrades_gracefully_when_ffmpeg_extract_fails(self) -> None:
        adapter, ctx, model_manager = self._make_ctx()

        with patch.object(adapter, "_probe_has_audio_stream", return_value=True), patch.object(
            adapter,
            "_extract_audio_clip",
            return_value=(False, "mock ffmpeg failure"),
        ):
            result = adapter.run_shot(ctx)

        self.assertFalse(model_manager.requested)
        self.assertEqual(result["diagnostics"]["reason"], "audio_extract_failed")
        self.assertIn("mock ffmpeg failure", result["diagnostics"]["extract_error"])
        self.assertEqual(result["raw"]["segments"], [])

    def test_run_shot_normalizes_partial_whisper_timestamps(self) -> None:
        adapter, ctx, _ = self._make_ctx()
        audio_path = Path(ctx.t2_workdir) / "tool_audio" / "shot_000.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake wav data")
        ctx.model_manager = FakeASRModelManager(
            (
                "hello world trailing",
                [
                    {"text": "hello world", "timestamp": [0.2, None]},
                    {"text": "trailing", "timestamp": [None, 1.4]},
                    {"text": "skip me", "timestamp": [None, None]},
                ],
            )
        )

        result = adapter.run_shot(ctx)

        self.assertEqual(
            result["raw"]["segments"],
            [
                {"text": "hello world", "timestamp": [0.2, 0.2]},
                {"text": "trailing", "timestamp": [1.4, 1.4]},
            ],
        )
        self.assertTrue(result["semantic"]["frame_hints"][0]["attributes"]["has_speech"])

    def test_whisper_init_failure_raises_for_resource_manager(self) -> None:
        with patch("executor.tasks.whisper_asr.pipeline", side_effect=RuntimeError("mock init failure")):
            with self.assertRaisesRegex(RuntimeError, "mock init failure"):
                WhisperASR(
                    {
                        "models": {"whisper_model": str(self.temp_dir / "missing-whisper")},
                        "tasks": {"T2_asr": {"enabled": True, "device": "cpu"}},
                    }
                )


if __name__ == "__main__":
    unittest.main()
