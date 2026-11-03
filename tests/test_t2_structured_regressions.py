from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
import shutil
import uuid

from orchestrator.t2_structured import run_t2_structured


class FakeGroupExecutor:
    def __init__(self, execute_result: dict | None = None) -> None:
        self.execute_result = execute_result or {}

    def group_by_shot(self, frames):
        out = {}
        for frame in frames or []:
            out.setdefault(str(frame.get("shot_id", "")), []).append(frame)
        return out

    def execute(self, **kwargs):
        return self.execute_result


class FakeGateB:
    def __init__(self, usable_frames: list[dict]) -> None:
        self.usable_frames = usable_frames

    def run(self, candidate_frames, model_manager):
        _ = (candidate_frames, model_manager)
        return {
            "usable_frames": list(self.usable_frames),
            "rejected_frames": [],
            "all_frame_quality_profiles": [],
            "fallback_flags": {"model_stage_fallback_used": False},
        }


class FakeSemanticizer:
    def build(self, **kwargs):
        return list(kwargs["enriched_frames"])


class FakeValueScorer:
    def analyze(self, **kwargs):
        return list(kwargs["semanticized_frames"])


class FakeQuotaPlanner:
    def plan(self, **kwargs):
        _ = kwargs
        return {"global_plan": {"total_budget": 1}, "shot_plans": {"shot_001": {"final_quota": 1}}}


class FakeRepresentativeSelector:
    def __init__(self, selected_frames: list[dict]) -> None:
        self.selected_frames = selected_frames

    def select(self, **kwargs):
        _ = kwargs
        return {
            "selected_frames": list(self.selected_frames),
            "fallback_flags": {"embedding_fallback_used": False},
        }


class T2StructuredRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = str(Path("logs") / "test_tmp" / f"t2_regressions_{uuid.uuid4().hex[:8]}")
        Path(self.temp_root).mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.temp_root, ignore_errors=True))

    def test_missing_t1_structure_returns_explicit_fallback(self) -> None:
        planner = SimpleNamespace()
        segment = SimpleNamespace(index=0, video_path="video.mp4", segment_path="segment.mp4")

        result = run_t2_structured(planner, {"output": {}}, segment)

        self.assertFalse(result["output"]["quality_passed"])
        self.assertEqual(result["output"]["reason"], "missing_t1_structure")
        self.assertTrue(result["output"]["fallback_flags"]["missing_t1_structure"])

    def test_gate_a_rejects_all_shots_returns_no_kept_shots_fallback(self) -> None:
        planner = SimpleNamespace()
        segment = SimpleNamespace(index=0, video_path="video.mp4", segment_path="segment.mp4")
        t1_stage = {
            "output": {
                "shot_profiles": [
                    {
                        "shot_id": "shot_001",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "duration": 1.0,
                    }
                ],
                "chapters": [
                    {
                        "chapter_id": "chapter_001",
                        "shot_ids": ["shot_001"],
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "duration": 1.0,
                    }
                ],
                "video_profile": {
                    "video_id": "vid",
                    "selection_objective": "balanced_chapter_coverage",
                },
                "gate_a": {"kept_shot_ids": []},
            }
        }

        result = run_t2_structured(planner, t1_stage, segment)

        self.assertFalse(result["output"]["quality_passed"])
        self.assertEqual(result["output"]["reason"], "gate_a_rejected_all_shots")
        self.assertEqual(result["output"]["gate_d"]["final_reason"], "no_kept_shots")
        self.assertTrue(result["output"]["fallback_flags"]["gate_a_rejected_all_shots"])

    def test_happy_path_threads_tool_and_selection_outputs_into_final_t2_payload(self) -> None:
        candidate_frames = [
            {
                "shot_id": "shot_001",
                "chapter_id": "",
                "frame_index": 5,
                "sec": 1.2,
                "img": "frame5.jpg",
                "candidate_type": "raw_pool",
                "tool_signals": {"semantic_summary": "upload button visible"},
                "value_profile": {"reason": "high utility"},
                "value_score": 0.91,
            }
        ]
        execute_result = {
            "tool_outputs_by_shot": {},
            "raw_by_shot": {},
            "semantic_by_shot": {},
            "enriched_candidates_by_shot": {"shot_001": list(candidate_frames)},
        }
        planner = SimpleNamespace(
            temp_root=self.temp_root,
            config={},
            model_manager=object(),
            llm=None,
            modules=SimpleNamespace(
                tool_planner=SimpleNamespace(
                    plan=lambda **kwargs: {
                        "chapter_plans": {
                            "chapter_001": {
                                "tool_actions": [{"tool_name": "omniparser", "reason": "recover UI structure"}]
                            }
                        },
                        "shot_plans": {
                            "shot_001": {
                                "tool_actions": [{"tool_name": "ocr", "reason": "read text cues"}]
                            }
                        },
                    }
                ),
                tool_executor=FakeGroupExecutor(execute_result=execute_result),
                gate_b_quality=FakeGateB(usable_frames=list(candidate_frames)),
                frame_signal_semanticizer=FakeSemanticizer(),
                llm_value_scorer=FakeValueScorer(),
                quota_planner=FakeQuotaPlanner(),
                representative_selector=FakeRepresentativeSelector(selected_frames=list(candidate_frames)),
            ),
            _video_id=lambda _path: "video_001",
        )
        segment = SimpleNamespace(index=1, video_path="video.mp4", segment_path="segment.mp4")
        t1_stage = {
            "output": {
                "shot_profiles": [
                    {
                        "shot_id": "shot_001",
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                        "duration": 2.0,
                        "shot_role": "screen_demo",
                        "scene_type": "desktop_ui",
                        "metadata": {},
                    }
                ],
                "chapters": [
                    {
                        "chapter_id": "chapter_001",
                        "shot_ids": ["shot_001"],
                        "start_sec": 0.0,
                        "end_sec": 2.0,
                        "duration": 2.0,
                        "chapter_role": "screen_demo",
                    }
                ],
                "video_profile": {
                    "video_id": "vid",
                    "selection_objective": "clarity_and_step_coverage",
                },
                "gate_a": {"kept_shot_ids": ["shot_001"]},
                "shot_boundaries": [{"shot_id": "shot_001"}],
                "transition_indices": [5],
                "aux": {"flow_pairs": [{"from": 4, "to": 5}]},
            }
        }

        with patch("orchestrator.t2_structured.prepare_t2_frame_pool", return_value=list(candidate_frames)):
            result = run_t2_structured(planner, t1_stage, segment)

        output = result["output"]
        self.assertTrue(output["quality_passed"])
        self.assertEqual(output["tools"], ["omniparser", "ocr"])
        self.assertEqual(output["tool_reasons"]["omniparser"], "recover UI structure")
        self.assertEqual(output["keyframes"][0]["frame_index"], 5)
        self.assertEqual(output["keyframes"][0]["chapter_id"], "chapter_001")
        self.assertEqual(output["gate_d"]["selected_frame_indices"], [5])
        self.assertEqual(output["gate_d"]["coverage_summary"]["candidate_count"], 1)
        self.assertEqual(output["fallback_flags"]["gate_b_model_stage_fallback_used"], False)
        self.assertEqual(result["runtime"]["fallback_summary"]["gate_b"]["model_stage_fallback_used"], False)
        self.assertIn("seg_001", result["runtime"]["t2_workdir"])


if __name__ == "__main__":
    unittest.main()
