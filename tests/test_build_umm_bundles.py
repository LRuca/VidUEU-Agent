from __future__ import annotations

import json
import unittest
from pathlib import Path
import shutil
import uuid

from build_umm_bundles import build_bundle_from_stage_set, build_manifest


class BuildUMMBundlesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path("logs") / "test_tmp" / f"umm_bundle_test_{uuid.uuid4().hex[:8]}"
        self.tmpdir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.tmpdir, ignore_errors=True))

    def _write(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path

    def test_build_bundle_from_stage_set(self) -> None:
        debug_dir = self.tmpdir / "logs" / "debug"
        t1 = self._write(debug_dir / "demo_seg000_T1.json", {"output": {"video_type": "cooking"}})
        t2 = self._write(debug_dir / "demo_seg000_T2.json", {"output": {"tools": ["ocr"]}})
        t3 = self._write(debug_dir / "demo_seg000_T3.json", {"output": {"segment_summary": "prep summary"}})
        t4 = self._write(
            debug_dir / "demo_seg000_T4.json",
            {
                "evidence": {
                    "video_id": "demo",
                    "frames": [{"pos": 0, "img": "f0.jpg", "caption": "cap0"}],
                    "evidence_units": [{"evidence_id": "S_SEGEMENT_SUM", "text": "summary"}],
                    "aligned_pairs": [{"pair_from_pos": 0, "pair_to_pos": 1, "transition_text": "change"}],
                    "candidate_pairs": [{"from": 0, "to": 1}],
                    "reviewed_ok_pairs": [{"from": 0, "to": 1}],
                },
                "decision": {"selected": ["edit"]},
                "runtime": {"segment_index": 0, "video_path": "videos/demo.mp4", "segment_path": "segments/demo_seg000.mp4"},
            },
        )

        bundle = build_bundle_from_stage_set({"T1": t1, "T2": t2, "T3": t3, "T4": t4}, root=self.tmpdir)
        self.assertEqual(bundle["segment"]["video_id"], "demo")
        self.assertEqual(bundle["segment"]["index"], 0)
        self.assertEqual(bundle["t1_out"]["video_type"], "cooking")
        self.assertEqual(bundle["frames_aligned"][0]["caption"], "cap0")
        self.assertEqual(bundle["candidate_pairs"][0]["from"], 0)
        self.assertIn("T4", bundle["segment"]["debug_paths"])

    def test_build_manifest_groups_segments_by_video(self) -> None:
        bundles = [
            {"segment": {"index": 0, "video_id": "demo", "video_path": "videos/demo.mp4", "segment_path": "segments/demo_seg000.mp4", "debug_paths": {}}},
            {"segment": {"index": 1, "video_id": "demo", "video_path": "videos/demo.mp4", "segment_path": "segments/demo_seg001.mp4", "debug_paths": {}}},
        ]
        manifest = build_manifest(bundles, input_root=self.tmpdir)
        self.assertEqual(manifest["run_meta"]["video_count"], 1)
        self.assertEqual(manifest["run_meta"]["segment_count"], 2)
        self.assertEqual(len(manifest["videos"][0]["segments"]), 2)


if __name__ == "__main__":
    unittest.main()
