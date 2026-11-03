from __future__ import annotations

import copy
import unittest

from align_unify_render_format import align_item


class AlignUnifyRenderFormatTest(unittest.TestCase):
    def test_preserves_legacy_unify_payload_in_meta(self) -> None:
        item = {
            "module": "unify",
            "benchmark": "unify.seed.l3.gts",
            "unify_id": "sample_0",
            "turns": {
                "plan": {"goal": "make the later state visible", "invariants": ["same scene"], "edits": ["move cup"]},
                "select": {"question": "Which image is real?", "answer": "A"},
            },
            "answer": {"target_frame_pos": 1},
            "render": {"source_frame": 0, "target_frame": 1},
            "evidence_refs": ["F00_CAP", "F01_CAP"],
            "meta": {
                "frame_positions": [0, 1],
                "segment_keyframes": [
                    {"pos": 0, "img": "frame_0000.jpg", "caption": "source"},
                    {"pos": 1, "img": "frame_0001.jpg", "caption": "target"},
                ],
            },
        }
        original = copy.deepcopy(item)

        aligned = align_item(item)

        self.assertIsInstance(aligned.get("turns"), list)
        legacy = aligned["meta"]["legacy_unify"]
        self.assertEqual(legacy["unify_id"], original["unify_id"])
        self.assertEqual(legacy["turns"], original["turns"])
        self.assertEqual(legacy["answer"], original["answer"])
        self.assertEqual(legacy["render"], original["render"])
        self.assertEqual(legacy["evidence_refs"], original["evidence_refs"])
        self.assertEqual(aligned["meta"]["frame_positions"], original["meta"]["frame_positions"])


if __name__ == "__main__":
    unittest.main()
