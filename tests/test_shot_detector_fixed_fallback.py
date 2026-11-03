from __future__ import annotations

import unittest

from executor.tasks.shot_detector import ShotDetector


class ShotDetectorFixedFallbackTest(unittest.TestCase):
    def test_falls_back_to_fixed_windows_when_pyscenedetect_is_unavailable(self) -> None:
        detector = ShotDetector(config={"tasks": {"T1_shot_detection": {"fixed_fallback_sec": 10}}})
        detector._detect_with_pyscenedetect = lambda _video_path: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("PySceneDetect unavailable")
        )
        detector._probe_duration = lambda _video_path: 25.0  # type: ignore[method-assign]

        result = detector.detect(
            video_path=__file__,
            prefer_backend="pyscenedetect",
            allow_fallback_backend=True,
            video_type_hint="",
        )

        self.assertEqual(result["backend"], "fixed")
        self.assertEqual(len(result["shots"]), 3)
        self.assertEqual([round(x["duration"], 2) for x in result["shots"]], [10.0, 10.0, 5.0])
        self.assertEqual(result["meta"]["fixed_window_sec"], 10.0)


if __name__ == "__main__":
    unittest.main()
