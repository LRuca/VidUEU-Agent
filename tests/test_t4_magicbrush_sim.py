from __future__ import annotations

import json
import os
import sys
import types
import unittest
import uuid
import shutil
from pathlib import Path
from types import SimpleNamespace

if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from dataset.builder import DatasetBuilder
from orchestrator.t4_structured import run_t4_structured


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, prompt: str, max_new_tokens: int = 0) -> str:
        text = str(prompt)
        self.calls.append(text[:160])
        if "validating MULTIPLE consecutive keyframe pairs for MagicBrush-style image editing" in text:
            return json.dumps(
                {
                    "pairs": [
                        {
                            "from": 0,
                            "to": 1,
                            "ok": True,
                            "edit_count": 1,
                            "edits": ["remove the sale sticker on the bottle label"],
                            "instruction": "Remove the sale sticker on the bottle label. Do not change anything else.",
                            "reason": "single local edit",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "You are a quota allocator for dataset instance generation." in text:
            return json.dumps(
                {
                    "quotas": {
                        "llava_bench": 0,
                        "edit": 1,
                        "mmt_bench": 0,
                        "mme": 1,
                        "dialog_visdial_like": 1,
                        "dialog_mmdu_like": 0,
                        "unify": 1,
                    },
                    "edit_pairs": [{"from": 0, "to": 1}],
                    "rationales": [
                        {
                            "type": "edit",
                            "evidence_refs": ["S_SEGEMENT_SUM", "P00_REVIEW"],
                            "reason": "reviewed pair supports a small local edit",
                        },
                        {
                            "type": "mme",
                            "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"],
                            "reason": "single-frame fact is easy to verify",
                        },
                        {
                            "type": "dialog_visdial_like",
                            "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"],
                            "reason": "stable frame can support a short grounded dialog",
                        },
                        {
                            "type": "unify",
                            "evidence_refs": ["S_SEGEMENT_SUM", "P00_REVIEW"],
                            "reason": "reviewed edit pair can anchor unify generation",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "TASK TYPE:" in text:
            return "{}"
        if "Target benchmark: edit" in text:
            return json.dumps(
                {
                    "items": [
                        {
                            "from_pos": 0,
                            "to_pos": 1,
                            "instruction": "Remove the sale sticker on the bottle label. Do not change anything else.",
                            "evidence_refs": ["P00_REVIEW", "S_SEGEMENT_SUM", "F00_CAP", "F01_CAP"],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Target benchmark: mme" in text:
            return json.dumps(
                {
                    "items": [
                        {
                            "image_pos": 0,
                            "category": "presence",
                            "question": "Is there a bottle on the table in the image?",
                            "answer": "Yes",
                            "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"],
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Generate 1 dialog(s) of type: dialog_visdial_like" in text:
            return json.dumps(
                {
                    "dialogs": [
                        {
                            "turns": [
                                {
                                    "role": "user",
                                    "text": "Goal: Verify the bottle label details in the image. Plan: Step 1: locate the bottle. Step 2: inspect the label. Step 1: Where is the bottle placed in the image?",
                                    "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"],
                                },
                                {
                                    "role": "assistant",
                                    "text": "Answer: The bottle stands on the table near the center-right area. Evidence: The bottle is upright on the tabletop and sits beside a bowl that anchors its position.",
                                    "evidence_refs": ["S_SEGEMENT_SUM", "F00_CAP"],
                                },
                                {
                                    "role": "user",
                                    "text": "Step 2: On that same bottle, is there a sticker attached to the label area?",
                                    "evidence_refs": ["F00_CAP"],
                                },
                                {
                                    "role": "assistant",
                                    "text": "Answer: Yes, a sticker is attached to the label area. Evidence: A bright sale sticker is visible on the bottle label and contrasts with the rest of the bottle surface.",
                                    "evidence_refs": ["F00_CAP"],
                                },
                            ]
                        }
                    ]
                },
                ensure_ascii=False,
            )
        raise RuntimeError(f"Unexpected prompt:\n{text[:300]}")


class FakeVLM:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, images, prompt: str, max_new_tokens: int = 0) -> str:
        text = str(prompt or "")
        self.calls.append(text[:160])
        if "suitable for a text-to-image generation benchmark" not in text:
            raise RuntimeError(f"Unexpected VLM prompt:\n{text[:300]}")
        if "bright sale sticker" in text:
            return json.dumps(
                {
                    "suitable": True,
                    "subject_clarity": "high",
                    "composition_clean": True,
                    "text_overlay_heavy": False,
                    "background_clean": True,
                    "generation_reason": "The bottle is clearly framed and the tabletop scene is easy to describe. The visible objects are stable and promptable.",
                    "benchmark_style": "geneval",
                    "generation_prompt": "A realistic product-style tabletop photo of a single bottle standing on a table beside a bowl, with a bright sale sticker attached to the bottle label, centered in a stable indoor composition, natural lighting, clear details, documentary photography style.",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "suitable": False,
                "subject_clarity": "medium",
                "composition_clean": False,
                "text_overlay_heavy": False,
                "background_clean": False,
                "generation_reason": "The frame is weaker as a standalone generation target because the subject is less distinctive after the sticker is removed.",
            },
            ensure_ascii=False,
        )


class FakeModelManager:
    def __init__(self, llm: FakeLLM, vlm: FakeVLM) -> None:
        self._llm = llm
        self._vlm = vlm

    def get_resource(self, name: str):
        if name == "llm":
            return self._llm
        if name == "vlm":
            return self._vlm
        raise KeyError(name)

    def run_vlm_jobs(self, jobs, worker_fn, max_workers=None):
        return [worker_fn(self._vlm, job) for job in list(jobs or [])]


class FakePlanner:
    def __init__(self, config: dict, *, use_fake_vlm: bool = False) -> None:
        config.setdefault("tasks", {}).setdefault("T4", {}).setdefault("task_profile", "legacy")
        self.config = config
        self.llm = FakeLLM()
        self.vlm = FakeVLM() if use_fake_vlm else None
        self.model_manager = FakeModelManager(self.llm, self.vlm) if use_fake_vlm and self.vlm is not None else None
        self.dataset_builder = DatasetBuilder(config)
        self._global_bench_counts = {}
        self._global_mme_cat_counts = {}
        self._global_mmt_cat_counts = {}
        self._global_llava_cat_counts = {}

    def _video_id(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _relativize_item_paths_inplace(self, item: dict) -> dict:
        return item


class T4MagicBrushSimulationTest(unittest.TestCase):
    def test_magicbrush_branch_runs_end_to_end_for_pair_starting_at_zero(self) -> None:
        root = Path("logs") / "test_tmp" / f"t4_magicbrush_{uuid.uuid4().hex[:8]}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        try:
            data_dir = root / "data"
            output_dir = root / "dataset" / "output"
            debug_dir = root / "logs" / "debug"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)

            frame0 = data_dir / "sim_f0.jpg"
            frame1 = data_dir / "sim_f1.jpg"
            frame0.write_bytes(b"sim")
            frame1.write_bytes(b"sim")

            output_file = output_dir / "t4_magicbrush_sim.jsonl"
            config = {
                "paths": {"debug_dir": str(debug_dir)},
                "dataset": {"output_file": str(output_file)},
                "tasks": {
                    "T4": {
                        "enable_llm_decision": True,
                        "enable_llm_generation": True,
                        "enable_llm_edit_instruction": True,
                        "enable_mmt_verification": False,
                        "max_llava": 0,
                        "max_edit": 1,
                        "max_mmt": 0,
                        "max_mme": 1,
                        "max_dialog_visdial": 1,
                        "max_dialog_mmdu": 0,
                        "max_unify": 1,
                        "dialog_visdial_rounds": 2,
                        "global_mix": {
                            "warmup_items": 999,
                            "single_cap": 1.0,
                            "edit_cap": 1.0,
                            "dialog_min": 0.0,
                            "dialog_max": 1.0,
                        },
                    }
                },
            }

            planner = FakePlanner(config)
            segment = SimpleNamespace(index=0, segment_path=str(data_dir / "sim_segment.mp4"))
            video_path = str(data_dir / "sim_video.mp4")

            t1_stage = {
                "output": {
                    "video_type": "product_demo",
                    "global_summary": "A short product demo shows a bottle on a table while the camera stays stable.",
                    "modality_hint": "visual",
                    "motion_hint": "low_motion",
                    "shot_profiles": [
                        {"shot_id": "shot_001", "frame_indices": [0, 1], "metadata": {"chapter_id": "chapter_001"}}
                    ],
                    "chapters": [{"chapter_id": "chapter_001", "shot_ids": ["shot_001"]}],
                }
            }
            t2_stage = {
                "output": {
                    "shot_boundaries": [],
                    "transition_indices": [],
                    "tools": ["ocr"],
                    "tool_reasons": {"ocr": "label text may help describe the product scene"},
                    "keyframes": [
                        {"index": 0, "frame_index": 0, "img": str(frame0), "sec": 0.0, "score": 0.92},
                        {"index": 1, "frame_index": 1, "img": str(frame1), "sec": 1.0, "score": 0.90},
                    ],
                    "tool_execution": {
                        "semantic_by_shot": {
                            "shot_001": {
                                "ocr": {"summary": "A bottle label and a sale sticker are visible on the table scene."}
                            }
                        }
                    },
                    "semantic_profiles": [
                        {
                            "shot_id": "shot_001",
                            "origin_index": 0,
                            "semantic_profile": {
                                "semantic_text": "Bottle with sticker on a table.",
                                "tool_descs": {"ocr": "The sticker is attached to the bottle label."},
                            },
                        },
                        {
                            "shot_id": "shot_001",
                            "origin_index": 1,
                            "semantic_profile": {
                                "semantic_text": "Bottle without sticker on a table.",
                                "tool_descs": {"ocr": "The label area looks cleaner after the sticker is removed."},
                            },
                        },
                    ],
                }
            }
            t3_stage = {
                "output": {
                    "segment_summary": "The segment focuses on a bottle placed on a table, with a small sticker change across two keyframes.",
                    "summary_hierarchy": {
                        "shot_summaries": [
                            {
                                "shot_id": "shot_001",
                                "chapter_id": "chapter_001",
                                "summary": "A stable tabletop shot centers on a bottle and nearby objects.",
                            }
                        ],
                        "chapter_summaries": [
                            {
                                "chapter_id": "chapter_001",
                                "summary": "A short product-view chapter with one consistent composition.",
                            }
                        ],
                    },
                    "frames": [
                        {
                            "img": str(frame0),
                            "sec": 0.0,
                            "caption": "A bottle with a bright sale sticker stands on a table beside a bowl.",
                            "transition_to_next": "Edits(1): remove the sale sticker on the bottle label | Shot: none | View: none | Subject: same | Uncertainty: low",
                            "cause": "object_change",
                            "shot_id": "shot_001",
                            "chapter_id": "chapter_001",
                        },
                        {
                            "img": str(frame1),
                            "sec": 1.0,
                            "caption": "The same bottle remains on the table, now without the sale sticker on the label.",
                            "cause": "object_change",
                            "shot_id": "shot_001",
                            "chapter_id": "chapter_001",
                        },
                    ],
                }
            }

            result = run_t4_structured(planner, t1_stage, t2_stage, t3_stage, segment)
            planner.dataset_builder.close()

            self.assertEqual(result["runtime"]["raw_pair_count"], 1)
            self.assertEqual(result["runtime"]["reviewed_pair_count"], 1)
            self.assertEqual(result["runtime"]["candidate_pair_count"], 1)
            self.assertEqual(result["decision"]["selected"], ["edit", "unify"])
            self.assertEqual(len(result["instances"]["edit.magicbrush"]), 1)
            self.assertEqual(len(result["instances"]["understanding.mme"]), 0)
            self.assertEqual(len(result["instances"]["understanding.dialog_visdial_like"]), 0)
            self.assertEqual(len(result["instances"]["unify.seed.l3.gts"]), 1)
            self.assertEqual(len(result["instances"]["unify.weave.uie_select_recall"]), 1)

            magic_item = result["instances"]["edit.magicbrush"][0]
            self.assertEqual(magic_item["module"], "edit")
            self.assertEqual(magic_item["meta"]["from_keyframe_pos"], 0)
            self.assertEqual(magic_item["meta"]["to_keyframe_pos"], 1)
            self.assertIn("P00_REVIEW", magic_item["evidence_refs"])
            unify_items = []
            for key in ("unify.seed.l3.gts", "unify.weave.uie_select_recall"):
                unify_items.extend(result["instances"][key])
            unify_task_types = [item["benchmark"] for item in unify_items]
            self.assertEqual(unify_task_types, ["unify.seed.l3.gts", "unify.weave.uie_select_recall"])
            self.assertEqual(unify_items[0]["module"], "unify")

            with output_file.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(rows), 3)
            self.assertEqual(
                [row["benchmark"] for row in rows],
                ["edit", "unify.seed.l3.gts", "unify.weave.uie_select_recall"],
            )
        finally:
            planner = locals().get("planner")
            if planner is not None:
                try:
                    planner.dataset_builder.close()
                except Exception:
                    pass

    def test_generation_route_writes_ground_truth_prompt_pair(self) -> None:
        root = Path("logs") / "test_tmp" / f"t4_generation_{uuid.uuid4().hex[:8]}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))

        try:
            data_dir = root / "data"
            output_dir = root / "dataset" / "output"
            debug_dir = root / "logs" / "debug"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)

            frame0 = data_dir / "sim_f0.jpg"
            frame1 = data_dir / "sim_f1.jpg"
            frame0.write_bytes(b"sim")
            frame1.write_bytes(b"sim")

            output_file = output_dir / "t4_generation_sim.jsonl"
            config = {
                "paths": {"debug_dir": str(debug_dir)},
                "dataset": {"output_file": str(output_file)},
                "tasks": {
                    "T4": {
                        "enable_llm_decision": True,
                        "enable_llm_generation": True,
                        "enable_llm_edit_instruction": True,
                        "enable_mmt_verification": False,
                        "max_llava": 0,
                        "max_edit": 1,
                        "max_mmt": 0,
                        "max_mme": 0,
                        "max_dialog_visdial": 0,
                        "max_dialog_mmdu": 0,
                        "max_unify": 1,
                        "max_generation": 1,
                        "enable_generation_routing": True,
                        "generation_vlm_max_new_tokens": 256,
                        "generation_ground_truth_dirname": "generation_ground_truth",
                    }
                },
            }

            planner = FakePlanner(config, use_fake_vlm=True)
            segment = SimpleNamespace(index=0, segment_path=str(data_dir / "sim_segment.mp4"))

            t1_stage = {
                "output": {
                    "video_type": "product_demo",
                    "global_summary": "A short product demo shows a bottle on a table while the camera stays stable.",
                    "modality_hint": "visual",
                    "motion_hint": "low_motion",
                    "shot_profiles": [
                        {"shot_id": "shot_001", "frame_indices": [0, 1], "metadata": {"chapter_id": "chapter_001"}}
                    ],
                    "chapters": [{"chapter_id": "chapter_001", "shot_ids": ["shot_001"]}],
                }
            }
            t2_stage = {
                "output": {
                    "shot_boundaries": [],
                    "transition_indices": [],
                    "tools": [],
                    "tool_reasons": {},
                    "keyframes": [
                        {"index": 0, "frame_index": 0, "img": str(frame0), "sec": 0.0, "score": 0.92},
                        {"index": 1, "frame_index": 1, "img": str(frame1), "sec": 1.0, "score": 0.90},
                    ],
                    "tool_execution": {"semantic_by_shot": {}},
                    "semantic_profiles": [],
                }
            }
            t3_stage = {
                "output": {
                    "segment_summary": "The segment focuses on a bottle placed on a table, with a small sticker change across two keyframes.",
                    "summary_hierarchy": {"shot_summaries": [], "chapter_summaries": []},
                    "frames": [
                        {
                            "img": str(frame0),
                            "sec": 0.0,
                            "caption": "A bottle with a bright sale sticker stands on a table beside a bowl.",
                            "transition_to_next": "Edits(1): remove the sale sticker on the bottle label | Shot: none | View: none | Subject: same | Uncertainty: low",
                            "cause": "object_change",
                            "shot_id": "shot_001",
                            "chapter_id": "chapter_001",
                        },
                        {
                            "img": str(frame1),
                            "sec": 1.0,
                            "caption": "The same bottle remains on the table, now without the sale sticker on the label.",
                            "cause": "object_change",
                            "shot_id": "shot_001",
                            "chapter_id": "chapter_001",
                        },
                    ],
                }
            }

            result = run_t4_structured(planner, t1_stage, t2_stage, t3_stage, segment)
            planner.dataset_builder.close()

            self.assertEqual(result["runtime"]["generated_generation"], 1)
            self.assertEqual(result["runtime"]["generation_reviewed_count"], 2)
            self.assertEqual(result["runtime"]["generation_suitable_count"], 1)
            self.assertIn("generation", result["decision"]["selected"])
            generation_items = []
            for key, value in result["instances"].items():
                if key.startswith("generation."):
                    generation_items.extend(value)
            self.assertEqual(len(generation_items), 1)

            gen_item = generation_items[0]
            self.assertEqual(gen_item["module"], "generation")
            self.assertEqual(gen_item["benchmark"], "geneval")
            self.assertTrue(gen_item["query"])
            self.assertEqual(gen_item["prompt"], gen_item["query"])
            self.assertTrue(Path(gen_item["image"]).exists())
            self.assertEqual(Path(gen_item["image"]).parent.name, "generation_ground_truth")
            self.assertEqual(gen_item["meta"]["source_image"], str(frame0))
            self.assertEqual(gen_item["meta"]["ground_truth_image"], gen_item["image"])
            self.assertEqual(gen_item["meta"]["generation_review"]["subject_clarity"], "high")

            with output_file.open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len([row for row in rows if row.get("benchmark") == "geneval"]), 1)
        finally:
            planner = locals().get("planner")
            if planner is not None:
                try:
                    planner.dataset_builder.close()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
