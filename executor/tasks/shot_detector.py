from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np


class ShotDetector:
    """
    Minimal shot detector with backend preference:
    1. PySceneDetect AdaptiveDetector
    2. OpenCV content-diff fallback

    TODO:
    - add TransNetV2 backend when local dependency path is stable.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        t1_cfg = ((self.config.get("tasks", {}) or {}).get("T1_shot_detection", {}) or {})
        self.target_sample_fps = float(t1_cfg.get("sample_fps", 2.0) or 2.0)
        self.min_shot_sec = float(t1_cfg.get("min_shot_sec", 1.0) or 1.0)
        self.content_threshold = float(t1_cfg.get("content_threshold", 24.0) or 24.0)
        self.fixed_fallback_sec = float(t1_cfg.get("fixed_fallback_sec", 10.0) or 10.0)
        self.detector_name = str(t1_cfg.get("detector", "content") or "content").strip().lower()
        self.allow_opencv_fallback = bool(t1_cfg.get("allow_opencv_fallback", False))

    def detect(
        self,
        video_path: str,
        prefer_backend: str = "pyscenedetect",
        allow_fallback_backend: bool = True,
        video_type_hint: str = "",
    ) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            return {"shots": [], "backend": "none", "meta": {"reason": "missing_video"}}

        backends = [prefer_backend]
        if self.allow_opencv_fallback and prefer_backend != "opencv":
            backends.append("opencv")

        last_error = None
        for backend in backends:
            try:
                if backend == "pyscenedetect":
                    result = self._detect_with_pyscenedetect(video_path)
                elif backend == "opencv":
                    result = self._detect_with_opencv(video_path)
                else:
                    continue
                shots = result.get("shots", []) or []
                meta = result.get("meta", {}) or {}
                if backend == "pyscenedetect" and self._need_transnet_retry(shots, meta, video_type_hint):
                    meta["transnet_retry_recommended"] = True
                if shots:
                    return {"shots": shots, "backend": backend, "meta": meta}
            except Exception as e:
                last_error = repr(e)
                if not allow_fallback_backend:
                    break
        fixed_result = self._detect_with_fixed_windows(video_path, reason=last_error or "no_shots")
        if fixed_result.get("shots"):
            return {"shots": fixed_result["shots"], "backend": "fixed", "meta": fixed_result.get("meta", {}) or {}}
        return {"shots": [], "backend": "none", "meta": {"reason": last_error or "no_shots"}}

    def _detect_with_pyscenedetect(self, video_path: str) -> Dict[str, Any]:
        try:
            from scenedetect import AdaptiveDetector, ContentDetector, SceneManager, open_video
        except Exception as e:
            raise RuntimeError(f"PySceneDetect unavailable: {e}")

        video = open_video(video_path)
        fps = float(video.frame_rate or 25.0)
        duration = self._probe_duration(video_path)
        min_scene_len = max(1, int(round(fps * self.min_shot_sec)))

        content_manager = SceneManager()
        content_manager.add_detector(
            ContentDetector(
                threshold=float(self.content_threshold),
                min_scene_len=min_scene_len,
            )
        )
        content_manager.detect_scenes(video=video)
        content_scenes = content_manager.get_scene_list()

        adaptive_scenes = []
        if self.detector_name == "adaptive":
            try:
                video.reset()
            except Exception:
                pass
            adaptive_manager = SceneManager()
            adaptive_manager.add_detector(AdaptiveDetector(min_scene_len=min_scene_len))
            adaptive_manager.detect_scenes(video=video)
            adaptive_scenes = adaptive_manager.get_scene_list()

        if self.detector_name == "adaptive":
            scenes = adaptive_scenes
            detector_used = "adaptive"
        else:
            scenes = content_scenes
            detector_used = "content"

        shots = []
        for idx, (start, end) in enumerate(scenes):
            s = float(start.get_seconds())
            e = float(end.get_seconds())
            if e - s < self.min_shot_sec and shots:
                shots[-1]["end_sec"] = e
                shots[-1]["duration"] = e - float(shots[-1]["start_sec"])
                continue
            shots.append(
                {
                    "shot_id": f"shot_{idx:03d}",
                    "start_sec": s,
                    "end_sec": e,
                    "duration": max(0.0, e - s),
                    "boundary_confidence_in": 0.6 if idx > 0 else 0.0,
                    "boundary_confidence_out": 0.6 if idx < len(scenes) - 1 else 0.0,
                }
            )
        if not shots and duration > 0:
            shots = [self._single_shot(duration)]
        return {
            "shots": shots,
            "meta": {
                "fps": fps,
                "scene_count": len(shots),
                "scene_detector": detector_used,
                "content_scene_count": len(content_scenes),
                "adaptive_scene_count": len(adaptive_scenes),
                "content_threshold": float(self.content_threshold),
                "min_scene_len_frames": int(min_scene_len),
            },
        }

    def _detect_with_opencv(self, video_path: str) -> Dict[str, Any]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("cv2.VideoCapture open failed")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else self._probe_duration(video_path)
        sample_step = max(1, int(round(fps / max(self.target_sample_fps, 0.2))))

        prev_small = None
        boundaries = [0.0]
        scores: List[float] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step != 0:
                frame_idx += 1
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
            if prev_small is not None:
                score = float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))))
                scores.append(score)
                sec = frame_idx / max(fps, 1e-6)
                if score >= self.content_threshold and (sec - boundaries[-1]) >= self.min_shot_sec:
                    boundaries.append(sec)
            prev_small = small
            frame_idx += 1
        cap.release()

        boundaries = sorted(set([max(0.0, b) for b in boundaries if b < duration]))
        if not boundaries:
            boundaries = [0.0]
        if duration <= 0:
            duration = boundaries[-1] if boundaries else 0.0
        boundaries.append(duration)

        shots = []
        for idx in range(len(boundaries) - 1):
            s, e = boundaries[idx], boundaries[idx + 1]
            if e - s < self.min_shot_sec and shots:
                shots[-1]["end_sec"] = e
                shots[-1]["duration"] = e - float(shots[-1]["start_sec"])
                continue
            shots.append(
                {
                    "shot_id": f"shot_{idx:03d}",
                    "start_sec": s,
                    "end_sec": e,
                    "duration": max(0.0, e - s),
                    "boundary_confidence_in": 0.45 if idx > 0 else 0.0,
                    "boundary_confidence_out": 0.45 if idx < len(boundaries) - 2 else 0.0,
                }
            )
        if not shots and duration > 0:
            shots = [self._single_shot(duration)]
        score_arr = np.asarray(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32)
        return {
            "shots": shots,
            "meta": {
                "fps": fps,
                "scene_count": len(shots),
                "mean_content_score": float(score_arr.mean()) if score_arr.size else 0.0,
                "max_content_score": float(score_arr.max()) if score_arr.size else 0.0,
            },
        }

    def _need_transnet_retry(self, shots: List[Dict[str, Any]], meta: Dict[str, Any], video_type_hint: str) -> bool:
        if not shots:
            return True
        durations = [float(x.get("duration", 0.0) or 0.0) for x in shots]
        if not durations:
            return True
        mean_dur = float(np.mean(durations))
        hint = (video_type_hint or "").lower()
        if "game" in hint or "fast" in hint:
            return mean_dur < 1.2 or len(shots) > 120
        return mean_dur < 0.8 or len(shots) > 180

    def _probe_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or frame_count <= 0:
            return 0.0
        return frame_count / fps

    def _single_shot(self, duration: float) -> Dict[str, Any]:
        return {
            "shot_id": "shot_000",
            "start_sec": 0.0,
            "end_sec": duration,
            "duration": duration,
            "boundary_confidence_in": 0.0,
            "boundary_confidence_out": 0.0,
        }

    def _detect_with_fixed_windows(self, video_path: str, reason: str = "") -> Dict[str, Any]:
        duration = float(self._probe_duration(video_path) or 0.0)
        if duration <= 0:
            return {"shots": [], "meta": {"reason": reason or "invalid_duration"}}

        chunk = max(self.fixed_fallback_sec, self.min_shot_sec, 0.5)
        boundaries = [0.0]
        cursor = chunk
        while cursor < duration:
            boundaries.append(float(cursor))
            cursor += chunk
        boundaries.append(float(duration))

        shots = []
        for idx in range(len(boundaries) - 1):
            start_sec = float(boundaries[idx])
            end_sec = float(boundaries[idx + 1])
            shots.append(
                {
                    "shot_id": f"shot_{idx:03d}",
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration": max(0.0, end_sec - start_sec),
                    "boundary_confidence_in": 0.2 if idx > 0 else 0.0,
                    "boundary_confidence_out": 0.2 if idx < len(boundaries) - 2 else 0.0,
                }
            )

        return {
            "shots": shots,
            "meta": {
                "reason": reason or "fixed_window_fallback",
                "scene_count": len(shots),
                "fixed_window_sec": float(chunk),
                "duration": duration,
            },
        }
