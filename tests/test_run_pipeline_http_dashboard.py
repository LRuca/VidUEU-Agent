from __future__ import annotations

import json
import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import run_pipeline_http_dashboard as dashboard


class RunPipelineHttpDashboardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("logs") / "test_tmp" / f"pipeline_dashboard_{uuid.uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def test_discovers_nested_workspaces(self) -> None:
        workspace = self.root / "source_a" / "sheet0"
        (workspace / "logs").mkdir(parents=True, exist_ok=True)
        (workspace / "output").mkdir(parents=True, exist_ok=True)

        found = dashboard._discover_workspaces(self.root)

        self.assertEqual(found, [workspace.resolve()])

    def test_start_dashboard_server_creates_missing_root_and_listens_before_payload_refresh(self) -> None:
        missing_root = self.root / "new_output_root"
        with patch.object(dashboard, "_build_dashboard_payload", side_effect=lambda root: dashboard._empty_dashboard_payload(root, loading=False)):
            server = dashboard.start_dashboard_server(missing_root, port=0, host="127.0.0.1")
            try:
                self.assertTrue(missing_root.exists())
                self.assertTrue((missing_root / "pipeline_dashboard.html").exists())
                self.assertTrue((missing_root / "pipeline_dashboard_data.json").exists())
            finally:
                server.shutdown()
                server.server_close()

    def test_dashboard_payload_request_uses_ttl_cache(self) -> None:
        payload = dashboard._empty_dashboard_payload(self.root, loading=False)
        state = dashboard._new_dashboard_state(self.root, payload)
        state["last_refresh_ts"] = time.time()

        with patch.object(dashboard, "_start_dashboard_refresh") as mock_refresh:
            first = dashboard._dashboard_payload_for_request(self.root, state)
            second = dashboard._dashboard_payload_for_request(self.root, state)

        self.assertFalse(first["loading"])
        self.assertFalse(second["loading"])
        mock_refresh.assert_not_called()

    def test_builds_payload_from_stage_logs(self) -> None:
        workspace = self.root / "source_a" / "sheet0"
        logs = workspace / "logs"
        output = workspace / "output"
        logs.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        (workspace / "frames").mkdir(parents=True, exist_ok=True)
        (workspace / "frames" / "src.jpg").write_bytes(b"jpg")
        (workspace / "frames" / "tgt.jpg").write_bytes(b"jpg")
        (workspace / "gen_0.jpg").write_bytes(b"jpg")

        stage_rows = [
            {
                "segment_index": 0,
                "segment_path": "seg0.mp4",
                "stage": "T1",
                "T1": {
                    "output": {
                        "shot_profiles": [
                            {
                                "rep_frames": [{"img_path": "frame0.jpg"}],
                            }
                        ],
                        "chapters": [{"chapter_id": "chapter_0"}],
                        "suitable_for_dataset": True,
                        "video_type": "tutorial",
                    },
                    "runtime": {"fallback_summary": {"shot_detector_fallback_used": True}},
                },
            },
            {
                "segment_index": 0,
                "segment_path": "seg0.mp4",
                "stage": "T2",
                "T1": {},
                "T2": {
                    "output": {
                        "quality_passed": True,
                        "candidate_pool": [{"img": "cand0.jpg"}],
                        "gate_b": {"usable_frames": [{"img": "cand0.jpg"}], "rejected_frames": []},
                        "gate_d": {"selected_frames": [{"img": "sel0.jpg"}]},
                    }
                },
            },
            {
                "segment_index": 0,
                "segment_path": "seg0.mp4",
                "stage": "T3",
                "T3": {"output": {"dataset_type": "qa", "frames": [{"img": "t3_0.jpg"}], "segment_summary": "ok"}},
            },
            {
                "segment_index": 0,
                "segment_path": "seg0.mp4",
                "stage": "T4",
                "T4": {
                    "decision": {"selected": ["edit", "unify", "generation"], "magicbrush_pairs": [{"from": 0, "to": 1}]},
                    "instances": {
                        "magicbrush": [{"source_img": "sheet0/frames/src.jpg", "target_img": "sheet0/frames/tgt.jpg", "benchmark": "magicbrush", "module": "edit"}],
                        "mme": [{"image": "t4_0.jpg", "benchmark": "mme", "module": "understanding"}],
                        "generation": [{"image": "gen_0.jpg", "benchmark": "generation", "query": "a complete generation prompt"}],
                        "unify": [
                            {
                                "benchmark": "unify",
                                "module": "unify",
                                "task_type": "unify.seed.l3.gts",
                                "turns": {"question": "which grounded target frame matches the edited scene?"},
                                "answer": {"answer": "target:F01"},
                                "render": {
                                    "kind": "edit",
                                    "source_frame": 0,
                                    "target_frame": 1,
                                    "query": {
                                        "messages": [
                                            {"role": "user", "type": "text", "content": "Describe the grounded target frame."},
                                            {"role": "assistant", "type": "text", "content": "The scene stays grounded."},
                                            {"role": "user", "type": "text", "content": "Which grounded target frame matches the edited scene?"},
                                        ]
                                    },
                                    "answer": {
                                        "messages": [
                                            {"role": "assistant", "type": "text", "content": "target:F01"},
                                            {"role": "assistant", "type": "text", "content": "Because it preserves the grounded update."},
                                        ]
                                    },
                                },
                            },
                            {
                                "benchmark": "unify",
                                "module": "unify",
                                "task_type": "unify.isg.i2.seq_select",
                                "turns": {
                                    "overall": "given context frames, pick the immediate next future step.",
                                    "edits": ["raise hands", "frame face"],
                                    "check": ["hands up", "hands near face"],
                                    "select": [
                                        {
                                            "question": "Which frame is the next future step 1?",
                                            "answer": "target:F02",
                                            "rationale": "It shows hands raised.",
                                        },
                                        {
                                            "question": "Which frame is the next future step 2?",
                                            "answer": "target:F03",
                                            "rationale": "It shows face framing.",
                                        },
                                    ],
                                },
                                "answer": {"future_frame_positions": [2, 3]},
                                "render": {
                                    "kind": "interleaved_mm",
                                    "context_positions": [0, 1],
                                    "future_positions": [2, 3],
                                    "query": {
                                        "messages": [
                                            {"role": "assistant", "type": "text", "content": "Given context frames, find the immediate next future step."},
                                            {"role": "user", "type": "text", "content": "Which frame is the next future step 1?"},
                                            {"role": "user", "type": "text", "content": "Which frame is the next future step 2?"},
                                        ]
                                    },
                                    "answer": {
                                        "messages": [
                                            {"role": "assistant", "type": "text", "content": "target:F02"},
                                            {"role": "assistant", "type": "text", "content": "It shows hands raised."},
                                            {"role": "assistant", "type": "text", "content": "target:F03"},
                                            {"role": "assistant", "type": "text", "content": "It shows face framing."},
                                        ]
                                    },
                                },
                            },
                        ],
                    },
                    "runtime": {
                        "generated_dialog_visdial": 0,
                        "generated_dialog_mmdu": 0,
                        "generated_generation": 1,
                        "generation_reviewed_count": 3,
                        "generation_suitable_count": 1,
                        "generated_unify": 1,
                        "raw_pair_count": 3,
                        "reviewed_pair_count": 2,
                        "candidate_pair_count": 1,
                        "selected_modules": ["edit", "unify", "generation"],
                        "module_route_reason": "edit_pairs_available",
                    },
                },
            },
            {
                "segment_index": 1,
                "segment_path": "seg1.mp4",
                "stage": "T1",
                "T1": {"output": {"shot_profiles": [], "chapters": [], "suitable_for_dataset": True}},
            },
            {
                "segment_index": 1,
                "segment_path": "seg1.mp4",
                "stage": "T2_ERROR",
                "T1": {},
                "error_type": "ModuleNotFoundError",
                "error_message": "No module named 'transformers'",
                "traceback": "traceback text",
            },
        ]
        self._write_jsonl(logs / "demo_segments.jsonl", stage_rows)
        self._write_jsonl(
            output / "data.jsonl",
            [
                {"benchmark": "magicbrush"},
                {"benchmark": "mme"},
                {"benchmark": "generation"},
                {"benchmark": "unify", "task_type": "unify.seed.l3.gts"},
                {"benchmark": "unify", "task_type": "unify.isg.i2.seq_select"},
            ],
        )

        payload = dashboard._build_dashboard_payload(self.root)

        self.assertEqual(payload["workspace_count"], 1)
        self.assertEqual(payload["video_count"], 1)
        self.assertEqual(payload["segment_count"], 2)
        self.assertEqual(payload["segments_done"], 1)
        self.assertEqual(payload["segments_error"], 1)
        self.assertEqual(payload["bench_counts"]["edit"], 1)
        self.assertEqual(payload["bench_counts"]["mme"], 1)
        self.assertEqual(payload["bench_counts"]["generation"], 1)
        self.assertEqual(payload["bench_counts"]["unify"], 2)
        self.assertEqual(payload["module_counts"]["edit"], 1)
        self.assertEqual(payload["module_counts"]["understanding"], 1)
        self.assertEqual(payload["module_counts"]["generation"], 1)
        self.assertEqual(payload["module_counts"]["unify"], 2)
        self.assertEqual(payload["task_type_counts"]["edit"], 1)
        self.assertEqual(payload["task_type_counts"]["mme"], 1)
        self.assertEqual(payload["task_type_counts"]["generation"], 1)
        self.assertEqual(payload["task_type_counts"]["unify.seed.l3.gts"], 1)
        self.assertEqual(payload["task_type_counts"]["unify.isg.i2.seq_select"], 1)
        self.assertEqual(payload["active_workspace_name"], "sheet0")
        self.assertEqual(payload["active_video_name"], "demo")
        self.assertEqual(payload["active_segment_index"], 1)
        self.assertEqual(payload["active_stage"], "T2_ERROR")
        workspace_payload = payload["workspaces"][0]
        self.assertEqual(workspace_payload["dataset_items"], 5)
        self.assertFalse(workspace_payload["has_active_run"])
        video_payload = workspace_payload["videos"][0]
        self.assertEqual(video_payload["segments_done"], 1)
        self.assertEqual(video_payload["segments_error"], 1)
        self.assertFalse(video_payload["has_active_run"])
        segment_done = video_payload["segments"][0]
        self.assertEqual(segment_done["status"], "done")
        self.assertEqual(segment_done["t4"]["total_items"], 5)
        self.assertEqual(segment_done["t4"]["instances"]["edit"], 1)
        self.assertEqual(segment_done["t4"]["instances"]["generation"], 1)
        self.assertEqual(segment_done["t4"]["instances"]["unify"], 2)
        self.assertEqual(segment_done["t4"]["selected_benchmarks"], ["edit", "mme", "unify", "generation"])
        self.assertEqual(segment_done["t4"]["modules"]["edit"], 1)
        self.assertEqual(segment_done["t4"]["modules"]["understanding"], 1)
        self.assertEqual(segment_done["t4"]["modules"]["generation"], 1)
        self.assertEqual(segment_done["t4"]["modules"]["unify"], 2)
        self.assertEqual(segment_done["t4"]["task_types"]["unify.seed.l3.gts"], 1)
        self.assertEqual(segment_done["t4"]["task_types"]["unify.isg.i2.seq_select"], 1)
        self.assertEqual(segment_done["t4"]["candidate_pair_count"], 1)
        self.assertEqual(segment_done["t4"]["selected_modules"], ["edit", "unify", "generation"])
        self.assertEqual(segment_done["t4"]["generated_generation"], 1)
        generation_example = next(item for item in segment_done["t4"]["examples"] if item["benchmark"] == "generation")
        generation_fields = {field["label"]: field["value"] for field in generation_example["fields"]}
        self.assertEqual(generation_fields["Prompt"], "a complete generation prompt")
        self.assertEqual(list(generation_fields.keys()), ["Prompt"])
        self.assertEqual(generation_example["images"], ["gen_0.jpg"])
        edit_example = next(item for item in segment_done["t4"]["examples"] if item["benchmark"] == "edit")
        self.assertEqual(edit_example["images"], ["sheet0/frames/src.jpg", "sheet0/frames/tgt.jpg"])
        unify_example = next(item for item in segment_done["t4"]["examples"] if item.get("task_type") == "unify.seed.l3.gts")
        self.assertEqual(unify_example["label"], "unify / unify.seed.l3.gts")
        unify_fields = {field["label"]: field["value"] for field in unify_example["fields"]}
        self.assertEqual(unify_fields["Render Kind"], "edit")
        self.assertEqual(unify_fields["Render Query Messages 1 Content"], "Describe the grounded target frame.")
        self.assertEqual(unify_fields["Render Answer Messages 1 Content"], "target:F01")
        self.assertEqual(unify_example["images"], [])
        self.assertEqual(unify_example["dialogue"][0]["role"], "user")
        self.assertEqual(unify_example["dialogue"][0]["text"], "Describe the grounded target frame.")
        self.assertEqual(unify_example["dialogue"][1]["role"], "assistant")
        self.assertIn("grounded target frame", unify_example["dialogue"][2]["text"])
        future_example = next(item for item in segment_done["t4"]["examples"] if item.get("task_type") == "unify.isg.i2.seq_select")
        future_fields = {field["label"]: field["value"] for field in future_example["fields"]}
        self.assertEqual(future_fields["Render Kind"], "interleaved_mm")
        self.assertEqual(future_fields["Render Answer Messages 1 Content"], "target:F02")
        self.assertEqual(future_example["images"], [])
        self.assertEqual(future_example["dialogue"][0]["text"], "Given context frames, find the immediate next future step.")
        self.assertEqual(future_example["dialogue"][1]["text"], "Which frame is the next future step 1?")
        self.assertEqual(future_example["dialogue"][3]["text"], "target:F02")
        self.assertEqual(future_example["dialogue"][5]["text"], "target:F03")
        segment_error = video_payload["segments"][1]
        self.assertEqual(segment_error["status"], "error")
        self.assertIn("transformers", segment_error["error"]["message"])

    def test_t4_examples_read_aligned_unify_turn_render(self) -> None:
        examples = dashboard._build_t4_examples(
            {
                "instances": {
                    "unify.seed.l3.gts": [
                        {
                            "benchmark": "unify.seed.l3.gts",
                            "module": "unify",
                            "task": "seed.l3.gts",
                            "turns": [
                                {
                                    "t": "plan",
                                    "task_type": "vlm_understand",
                                    "render": {
                                        "query": {
                                            "messages": [
                                                {"role": "user", "type": "text", "content": "Write a plan."},
                                                {"role": "user", "type": "image", "content": "Image #1"},
                                            ],
                                            "images": ["frames/source.jpg"],
                                        },
                                        "answer": {
                                            "messages": [
                                                {"role": "assistant", "type": "text", "content": '{"edits":["move object"]}'},
                                            ],
                                            "images": [],
                                        },
                                    },
                                },
                                {
                                    "t": "select",
                                    "task_type": "vlm_understand",
                                    "render": {
                                        "query": {
                                            "messages": [
                                                {"role": "user", "type": "text", "content": "Select the real image."},
                                                {"role": "user", "type": "image", "content": "Image #1"},
                                            ],
                                            "images": ["frames/target.jpg"],
                                        },
                                        "answer": {
                                            "messages": [{"role": "assistant", "type": "text", "content": "A"}],
                                            "images": [],
                                        },
                                    },
                                },
                            ],
                            "meta": {},
                        }
                    ]
                }
            }
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["benchmark"], "unify")
        self.assertEqual(example["task_type"], "seed.l3.gts")
        self.assertEqual(example["images"], ["frames/source.jpg", "frames/target.jpg"])
        self.assertEqual(example["dialogue"][0]["role"], "turn")
        self.assertEqual(example["dialogue"][0]["text"], "1. plan (vlm_understand)")
        self.assertEqual(example["dialogue"][1]["text"], "Write a plan.")
        self.assertEqual(example["dialogue"][3]["text"], '{"edits":["move object"]}')
        fields = {field["label"]: field["value"] for field in example["fields"]}
        self.assertEqual(fields["Turn 1"], "plan (vlm_understand)")
        self.assertEqual(fields["Turn 2"], "select (vlm_understand)")
        self.assertEqual(fields["Turn 2 Render Query Messages 1 Content"], "Select the real image.")
        self.assertEqual(fields["Turn 2 Render Answer Messages 1 Content"], "A")

    def test_generation_example_uses_only_prompt_image(self) -> None:
        examples = dashboard._build_t4_examples(
            {
                "instances": {
                    "geneval": [
                        {
                            "benchmark": "geneval",
                            "module": "generation",
                            "image": "frames/prompt_image.jpg",
                            "images": ["frames/extra_candidate.jpg"],
                            "prompt": "a single generated frame prompt",
                            "meta": {"ground_truth_image": "frames/ground_truth.jpg"},
                        }
                    ]
                }
            }
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["label"], "generation / geneval")
        self.assertEqual(examples[0]["images"], ["frames/prompt_image.jpg"])


if __name__ == "__main__":
    unittest.main()
