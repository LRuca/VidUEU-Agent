from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np

from executor.tasks.quality_iqa import compute_vlm_quality_scores


class GateBQuality:
    """
    Strict two-stage quality gate.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_gate_b", {}) or {})
        self.fast_reject_threshold = float(cfg.get("fast_reject_threshold", 0.22) or 0.22)
        self.hard_floor = float(cfg.get("hard_floor", 0.35) or 0.35)
        self.enable_model_stage = bool(cfg.get("enable_model_stage", True))
        self.vlm_enable = bool(cfg.get("vlm_enable", True))
        self.vlm_max_new_tokens = int(cfg.get("vlm_max_new_tokens", 768) or 768)

    def run(self, candidate_frames: List[Dict[str, Any]], model_manager: Any) -> Dict[str, Any]:
        quality_profiles = []
        fast_pass = []
        rejected = []

        for cand in candidate_frames:
            profile = self._fast_profile(cand)
            quality_profiles.append(profile)
            if profile["fast_quality_score"] >= self.fast_reject_threshold:
                fast_pass.append(profile)
            else:
                profile["hard_pass"] = False
                profile["reason"] = profile["fast_reject_reason"]
                rejected.append(profile)

        if self.enable_model_stage:
            vlm_meta = self._apply_model_scores(fast_pass, model_manager)
        else:
            for profile in fast_pass:
                profile["model_quality_score"] = profile["fast_quality_score"]
                profile["model_backend"] = "fast_profile_only"
            vlm_meta = {
                "vlm_attempted": False,
                "vlm_fallback_used": True,
                "vlm_available_count": 0,
                "vlm_total_count": len(fast_pass),
                "vlm_backend": "disabled",
            }

        usable = []
        for profile in fast_pass:
            if profile.get("vlm_available", False):
                profile["hard_pass"] = bool(profile.get("vlm_all_pass", False))
            else:
                profile["hard_pass"] = bool(profile["model_quality_score"] >= self.hard_floor)

            if profile["hard_pass"]:
                profile["reason"] = ""
            else:
                if profile.get("vlm_attempted") and profile.get("vlm_available", False):
                    failed = list(profile.get("vlm_failed_dimensions", []) or [])
                    profile["reason"] = (
                        "vlm_quality_failed:" + ",".join(failed)
                        if failed
                        else "vlm_quality_failed"
                    )
                elif profile.get("vlm_attempted"):
                    profile["reason"] = "vlm_unavailable_fallback_below_quality_hard_floor"
                else:
                    profile["reason"] = "below_quality_hard_floor"

            if profile["hard_pass"]:
                usable.append(profile)
            else:
                rejected.append(profile)

        return {
            "usable_frames": [self._to_frame_payload(x) for x in usable],
            "rejected_frames": [self._to_frame_payload(x) for x in rejected],
            "all_frame_quality_profiles": quality_profiles,
            "fallback_flags": {
                "model_stage_fallback_used": bool(vlm_meta.get("vlm_fallback_used", not self.enable_model_stage)),
                "model_stage_backend": str(vlm_meta.get("vlm_backend", "fast_profile_only")),
                "vlm_attempted": bool(vlm_meta.get("vlm_attempted", False)),
                "vlm_available_count": int(vlm_meta.get("vlm_available_count", 0)),
                "vlm_total_count": int(vlm_meta.get("vlm_total_count", len(fast_pass))),
            },
        }

    def _fast_profile(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        img_path = str(cand.get("img", "") or "")
        blur_score = 0.0
        brightness_score = 0.0
        contrast_score = 0.0
        entropy_score = 0.0
        artifact_proxy = 1.0
        fast_quality_score = 0.0
        reason = "image_load_failed"

        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                small = cv2.resize(img, (192, 192), interpolation=cv2.INTER_AREA)
                lap = float(cv2.Laplacian(small, cv2.CV_64F).var())
                blur_score = max(0.0, min(1.0, lap / 320.0))
                mean_val = float(small.mean())
                std_val = float(small.std())
                brightness_score = 1.0 - min(1.0, abs(mean_val - 128.0) / 128.0)
                contrast_score = max(0.0, min(1.0, std_val / 64.0))
                hist = cv2.calcHist([small], [0], None, [256], [0, 256]).flatten()
                hist = hist / max(1.0, float(hist.sum()))
                entropy = float(-(hist * np.log2(hist + 1e-8)).sum())
                entropy_score = max(0.0, min(1.0, entropy / 8.0))
                artifact_proxy = max(0.0, min(1.0, 1.0 - abs(std_val - 42.0) / 42.0))
                fast_quality_score = max(
                    0.0,
                    min(
                        1.0,
                        0.35 * blur_score
                        + 0.20 * brightness_score
                        + 0.20 * contrast_score
                        + 0.15 * entropy_score
                        + 0.10 * artifact_proxy,
                    ),
                )
                reason = ""

        fast_reject_reason = reason
        if not fast_reject_reason:
            if entropy_score < 0.10 and contrast_score < 0.08:
                fast_reject_reason = "low_information"
            elif blur_score < 0.12 and entropy_score >= 0.10:
                fast_reject_reason = "severe_blur"
            elif brightness_score < 0.10:
                fast_reject_reason = "extreme_exposure"
            elif fast_quality_score < self.fast_reject_threshold:
                fast_reject_reason = "fast_quality_reject"
            else:
                fast_reject_reason = ""

        return {
            "frame_index": int(cand.get("frame_index", -1)),
            "shot_id": str(cand.get("shot_id", "")),
            "chapter_id": str(cand.get("chapter_id", "")),
            "img": img_path,
            "sec": float(cand.get("sec", 0.0) or 0.0),
            "candidate_type": str(cand.get("candidate_type", "")),
            "base_scores": dict(cand.get("base_scores", {}) or {}),
            "tool_name": str(cand.get("tool_name", cand.get("candidate_type", "")) or ""),
            "tool_priority_level": str(cand.get("tool_priority_level", "medium") or "medium"),
            "tool_priority_weight": int(cand.get("tool_priority_weight", 2) or 2),
            "tool_signals": dict(cand.get("tool_signals", {}) or {}),
            "fast_quality_score": float(fast_quality_score),
            "model_quality_score": 0.0,
            "blur_score": float(blur_score),
            "brightness_score": float(brightness_score),
            "contrast_score": float(contrast_score),
            "entropy_score": float(entropy_score),
            "artifact_proxy": float(artifact_proxy),
            "fast_reject_reason": fast_reject_reason,
            "vlm_available": False,
            "vlm_attempted": False,
            "vlm_all_pass": False,
            "vlm_passed_count": 0,
            "vlm_failed_count": 0,
            "vlm_passed_dimensions": [],
            "vlm_failed_dimensions": [],
            "vlm_result": None,
            "vlm_caption": "",
            "vlm_raw_response": "",
            "vlm_error": "",
            "model_backend": "fast_profile",
            "hard_pass": False,
            "reason": "",
        }

    def _apply_model_scores(self, profiles: List[Dict[str, Any]], model_manager: Any) -> Dict[str, Any]:
        if not profiles:
            return {
                "vlm_attempted": False,
                "vlm_fallback_used": False,
                "vlm_available_count": 0,
                "vlm_total_count": 0,
                "vlm_backend": "none",
            }

        vlm_attempted = False
        vlm_available_count = 0
        results: List[Dict[str, Any]] = []

        if self.vlm_enable:
            vlm_attempted = True
            try:
                results = compute_vlm_quality_scores(
                    [str(profile.get("img", "") or "") for profile in profiles],
                    model_manager=model_manager,
                    max_new_tokens=self.vlm_max_new_tokens,
                )
            except Exception:
                results = [{"available": False, "error": "vlm_stage_failed"} for _ in profiles]

        for idx, profile in enumerate(profiles):
            result = results[idx] if idx < len(results) else {"available": False, "error": "vlm_result_missing"}
            vlm_available = bool(result.get("available", False))
            if vlm_available:
                vlm_available_count += 1

            profile["vlm_available"] = vlm_available
            profile["vlm_attempted"] = vlm_attempted
            profile["vlm_all_pass"] = bool(result.get("all_pass", False))
            profile["vlm_passed_count"] = int(result.get("passed_count", 0) or 0)
            profile["vlm_failed_count"] = int(result.get("failed_count", 0) or 0)
            profile["vlm_passed_dimensions"] = list(result.get("passed_dimensions", []) or [])
            profile["vlm_failed_dimensions"] = list(result.get("failed_dimensions", []) or [])
            profile["vlm_result"] = result.get("parsed_result")
            profile["vlm_caption"] = str(result.get("caption", "") or "")
            profile["vlm_raw_response"] = str(result.get("raw_response", "") or "")
            profile["vlm_error"] = str(result.get("error", "") or "")
            profile["model_quality_score"] = self._model_score(profile)
            profile["model_backend"] = "fast_profile_plus_vlm" if vlm_available else "fast_profile_fallback"

        return {
            "vlm_attempted": vlm_attempted,
            "vlm_fallback_used": bool(vlm_attempted and vlm_available_count < len(profiles)),
            "vlm_available_count": vlm_available_count,
            "vlm_total_count": len(profiles),
            "vlm_backend": "fast_profile_plus_vlm" if vlm_available_count > 0 else ("fast_profile_fallback" if vlm_attempted else "disabled"),
        }

    def _model_score(self, profile: Dict[str, Any]) -> float:
        if profile.get("vlm_available", False):
            return 1.0 if profile.get("vlm_all_pass", False) else 0.0
        return float(profile.get("fast_quality_score", 0.0))

    def _to_frame_payload(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_index": profile["frame_index"],
            "shot_id": profile["shot_id"],
            "chapter_id": profile["chapter_id"],
            "img": profile["img"],
            "sec": profile["sec"],
            "candidate_type": profile["candidate_type"],
            "base_scores": profile["base_scores"],
            "tool_name": profile.get("tool_name", profile["candidate_type"]),
            "tool_priority_level": profile.get("tool_priority_level", "medium"),
            "tool_priority_weight": profile.get("tool_priority_weight", 2),
            "tool_signals": profile.get("tool_signals", {}),
            "caption": str(profile.get("vlm_caption", "") or ""),
            "quality_profile": {
                "available": profile["vlm_available"],
                "attempted": profile["vlm_attempted"],
                "parsed_result": profile["vlm_result"],
                "caption": str(profile.get("vlm_caption", "") or ""),
                "all_pass": profile["vlm_all_pass"],
                "passed_dimensions": profile["vlm_passed_dimensions"],
                "failed_dimensions": profile["vlm_failed_dimensions"],
                "passed_count": profile["vlm_passed_count"],
                "failed_count": profile["vlm_failed_count"],
                "raw_response": profile["vlm_raw_response"],
                "error": profile["vlm_error"],
                "backend": profile["model_backend"],
                "final_pass": profile["hard_pass"],
                "final_reason": profile["reason"] or profile["fast_reject_reason"],
                "fast_quality_score": profile["fast_quality_score"],
                "model_quality_score": profile["model_quality_score"],
                "blur_score": profile["blur_score"],
                "brightness_score": profile["brightness_score"],
                "contrast_score": profile["contrast_score"],
                "entropy_score": profile["entropy_score"],
                "artifact_proxy": profile["artifact_proxy"],
                "vlm_available": profile["vlm_available"],
                "vlm_attempted": profile["vlm_attempted"],
                "vlm_all_pass": profile["vlm_all_pass"],
                "vlm_passed_count": profile["vlm_passed_count"],
                "vlm_failed_count": profile["vlm_failed_count"],
                "vlm_passed_dimensions": profile["vlm_passed_dimensions"],
                "vlm_failed_dimensions": profile["vlm_failed_dimensions"],
                "vlm_result": profile["vlm_result"],
                "vlm_caption": str(profile.get("vlm_caption", "") or ""),
                "vlm_raw_response": profile["vlm_raw_response"],
                "vlm_error": profile["vlm_error"],
                "model_backend": profile["model_backend"],
                "hard_pass": profile["hard_pass"],
                "reason": profile["reason"] or profile["fast_reject_reason"],
            },
        }
