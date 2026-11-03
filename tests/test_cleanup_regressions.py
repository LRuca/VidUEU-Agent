from __future__ import annotations

import unittest

from executor.tasks.frame_signal_semanticizer import FrameSignalSemanticizer
from executor.tasks.tool_executor import ToolExecutor


class CleanupRegressionTest(unittest.TestCase):
    def test_fallback_semantic_profile_uses_real_tool_descriptions(self) -> None:
        semanticizer = FrameSignalSemanticizer(config={})

        frame = {
            "tool_signals": {
                "tools": {
                    "ocr": {
                        "text": "OCR: sale sticker visible on bottle label",
                        "score": 0.9,
                        "attributes": {},
                        "evidence": {},
                        "raw_hint": {},
                    },
                    "yolo": {
                        "text": "",
                        "score": 0.75,
                        "attributes": {},
                        "evidence": {},
                        "raw_hint": {},
                    },
                }
            }
        }

        profile = semanticizer._fallback_semantic_profile(frame)

        self.assertEqual(profile["tool_descs"]["ocr"], "OCR: sale sticker visible on bottle label")
        self.assertEqual(profile["tool_descs"]["yolo"], "yolo: score=0.750")
        self.assertEqual(
            profile["semantic_text"],
            "OCR: sale sticker visible on bottle label | yolo: score=0.750",
        )

    def test_tool_executor_does_not_duplicate_raw_payload_on_each_candidate(self) -> None:
        executor = ToolExecutor(config={})
        executor.registry.get = lambda _tool_name: {"semanticizer": {"default_weight": 1.0}}

        enriched = executor._attach_semantic_signals(
            candidates=[{"shot_id": "shot_001", "frame_index": 7}],
            tool_actions=[{"tool_name": "ocr"}],
            selected_tools=["ocr"],
            shot_semantic={
                "ocr": {
                    "frame_hints": [
                        {
                            "frame_key": "shot_001:7",
                            "text": "sale sticker visible",
                            "score": 0.8,
                            "attributes": {"density": "medium"},
                            "evidence": {"lines": 1},
                        }
                    ]
                }
            },
            shot_raw={"ocr": {"debug_blob": "unused"}},
        )

        tool_signals = enriched[0]["tool_signals"]
        self.assertNotIn("raw", tool_signals)
        self.assertEqual(tool_signals["semantic_summary"], "sale sticker visible")


if __name__ == "__main__":
    unittest.main()
