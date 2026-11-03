from __future__ import annotations

import unittest

from executor.tasks.structure_profiler import StructureProfiler
from orchestrator.structures import ChapterRecord, ShotRecord


class StructureProfilerRegressionTest(unittest.TestCase):
    def test_profile_video_and_chapters_uses_explicit_video_id_and_preserves_tutorial_objective(self) -> None:
        profiler = StructureProfiler(config={})
        shots = [
            ShotRecord(
                shot_id="shot_001",
                start_sec=0.0,
                end_sec=2.0,
                duration=2.0,
                shot_role="screen_demo",
                scene_type="desktop_ui",
                vlm_summary="The presenter opens a file upload dialog in a desktop app.",
                candidate_sampling_hint="high",
                informativeness_hint="high",
            ),
            ShotRecord(
                shot_id="shot_002",
                start_sec=2.0,
                end_sec=4.0,
                duration=2.0,
                shot_role="screen_demo",
                scene_type="desktop_ui",
                vlm_summary="A progress view shows the upload button and recent files list.",
                candidate_sampling_hint="high",
                informativeness_hint="high",
            ),
        ]
        chapters = [
            ChapterRecord(
                chapter_id="chapter_001",
                shot_ids=["shot_001", "shot_002"],
                start_sec=0.0,
                end_sec=4.0,
                duration=4.0,
                chapter_role="screen_demo",
            )
        ]

        result = profiler.profile_video_and_chapters(
            shots=shots,
            chapters=chapters,
            model_manager=None,
            video_id="explicit_video_id",
        )

        video_profile = result["video_profile"]
        chapter = result["chapters"][0]
        self.assertEqual(video_profile.video_id, "explicit_video_id")
        self.assertEqual(video_profile.structure_type, "tutorial")
        self.assertEqual(video_profile.selection_objective, "clarity_and_step_coverage")
        self.assertEqual(video_profile.global_sampling_hint, "high")
        self.assertIn("upload", video_profile.video_summary.lower())
        self.assertEqual(chapter.chapter_role, "screen_demo")
        self.assertEqual(chapter.importance_hint, "high")
        self.assertIn("layout", chapter.t2_candidate_tool_hints)
        self.assertIn("coverage", chapter.t2_candidate_tool_hints)


if __name__ == "__main__":
    unittest.main()
