from __future__ import annotations

import json
import unittest
from contextlib import contextmanager

from executor.tasks.frame_signal_semanticizer import FrameSignalSemanticizer
from executor.tools.base import ToolContext
from executor.tools.omniparser_tool import OmniParserToolAdapter
from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile
from orchestrator.t2_structured import _summarize_tools


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str, max_new_tokens: int = 0) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "frames": [
                    {
                        "frame_id": "shot_001:7",
                        "tool_descs": {
                            "ocr": "Readable text confirms the upload button is available."
                        },
                        "semantic_text": "The UI shows an available upload action.",
                    }
                ]
            }
        )


class FakeOmniParserResource:
    def __init__(self, raw: dict) -> None:
        self.raw = raw

    def run(self, frame_paths: list[str]) -> dict:
        return {path: self.raw[path] for path in frame_paths}


class FakeModelManager:
    def __init__(self, raw: dict) -> None:
        self.resource = FakeOmniParserResource(raw)

    def get_resource(self, name: str):
        if name != "omniparser":
            raise KeyError(name)
        return self.resource

    def run_with_resource(self, name: str, fn, *args, **kwargs):
        if name != "omniparser":
            raise KeyError(name)
        return fn(self.resource, *args, **kwargs)

    @contextmanager
    def use_resource(self, name: str):
        if name != "omniparser":
            raise KeyError(name)
        yield self.resource


class SemanticizerAndOmniParserRegressionTest(unittest.TestCase):
    def test_semanticizer_treats_raw_hint_as_real_evidence_and_passes_structured_specs(self) -> None:
        semanticizer = FrameSignalSemanticizer(config={})
        llm = FakeLLM()
        shot = ShotRecord(shot_id="shot_001", start_sec=0.0, end_sec=2.0, duration=2.0, shot_role="screen_demo")
        chapter = ChapterRecord(
            chapter_id="chapter_001",
            shot_ids=["shot_001"],
            start_sec=0.0,
            end_sec=2.0,
            duration=2.0,
            chapter_role="screen_demo",
        )
        video_profile = VideoProfile(video_id="demo_video", selection_objective="clarity_and_step_coverage")
        frames = [
            {
                "shot_id": "shot_001",
                "chapter_id": "chapter_001",
                "frame_index": 7,
                "sec": 1.2,
                "candidate_type": "representative",
                "tool_signals": {
                    "tools": {
                        "ocr": {
                            "text": "",
                            "score": 0.0,
                            "attributes": {},
                            "evidence": {},
                            "raw_hint": {"text_lines": ["Upload complete"]},
                        }
                    }
                },
            }
        ]

        out = semanticizer.build(
            enriched_frames=frames,
            shots=[shot],
            chapters=[chapter],
            video_profile=video_profile,
            tool_plan={"shot_plans": {"shot_001": {"tool_actions": []}}},
            llm=llm,
        )

        self.assertEqual(len(llm.prompts), 1)
        prompt = llm.prompts[0]
        self.assertIn('"raw_hint": {', prompt)
        self.assertIn('"tool_semanticization_specs"', prompt)
        self.assertIn("Tool-specific semanticization rules:", prompt)
        self.assertEqual(out[0]["semantic_profile"]["semantic_text"], "The UI shows an available upload action.")
        self.assertEqual(
            out[0]["semantic_profile"]["tool_descs"]["ocr"],
            "Readable text confirms the upload button is available.",
        )

    def test_omniparser_adapter_builds_ui_focused_hints_and_t2_summary_prioritizes_it(self) -> None:
        img_path = "C:/tmp/frame_001.png"
        raw = {
            img_path: {
                "summary": "A desktop app shows an upload button and a sidebar.",
                "label_coordinates": {"0": [0.1, 0.1, 0.3, 0.2]},
                "parsed_content_list": [{"content": "Upload"}],
                "elements": [
                    {"element_id": "0", "type": "button", "content": "Upload", "interactable": True},
                    {"element_id": "1", "type": "text", "content": "Recent Files", "interactable": False},
                ],
            }
        }
        adapter = OmniParserToolAdapter()
        ctx = ToolContext(
            shot=None,
            candidates=[{"shot_id": "shot_002", "frame_index": 3, "img": img_path}],
            pre_gate_candidates=[],
            video_path="demo.mp4",
            t2_workdir="logs/test_tmp",
            model_manager=FakeModelManager(raw),
            config={},
            tool_spec={},
        )

        result = adapter.run_shot(ctx)

        self.assertEqual(result["raw"][img_path]["summary"], raw[img_path]["summary"])
        hint = result["semantic"]["frame_hints"][0]
        self.assertEqual(hint["frame_key"], "shot_002:3")
        self.assertEqual(hint["text"], "Interactive UI elements visible: Upload")
        self.assertGreater(hint["score"], 0.0)
        self.assertEqual(hint["attributes"]["interactive_count"], 1)
        self.assertEqual(hint["attributes"]["text_lines"], ["Recent Files"])
        self.assertEqual(len(hint["evidence"]["elements"]), 2)

        tools, reasons = _summarize_tools(
            {
                "chapter_plans": {
                    "chapter_001": {
                        "tool_actions": [
                            {"tool_name": "ocr", "reason": "read labels"},
                            {"tool_name": "omniparser", "reason": "recover actionable UI structure"},
                        ]
                    }
                },
                "shot_plans": {
                    "shot_002": {
                        "tool_actions": [
                            {"tool_name": "yolo", "reason": "ground visible objects"},
                        ]
                    }
                },
            }
        )
        self.assertEqual(tools[:3], ["omniparser", "ocr", "yolo"])
        self.assertEqual(reasons["omniparser"], "recover actionable UI structure")


if __name__ == "__main__":
    unittest.main()
