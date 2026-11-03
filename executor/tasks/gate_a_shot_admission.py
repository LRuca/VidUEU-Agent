from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np

from orchestrator.planner_helpers import parse_llm_json
from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile


class GateAShotAdmission:
    """
    Shot-level admission gate for entering T2.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T1_gate_a", {}) or {})
        self.keep_threshold = int(cfg.get("keep_threshold", 70) or 70)
        self.borderline_threshold = int(cfg.get("borderline_threshold", 50) or 50)

    def run(
        self,
        video_profile: VideoProfile,
        chapters: List[ChapterRecord],
        shots: List[ShotRecord],
        model_manager: Any,
    ) -> Dict[str, Any]:
        chapter_map = {}
        for chapter in chapters:
            for shot_id in chapter.shot_ids:
                chapter_map[shot_id] = chapter

        decisions = []
        kept = []
        dropped = []
        for shot in shots:
            chapter = chapter_map.get(shot.shot_id)
            decision = self._judge_one(video_profile, chapter, shot, model_manager)
            decisions.append(decision)
            if decision.get("decision") == "drop":
                dropped.append(shot.shot_id)
            else:
                kept.append(shot.shot_id)

        return {
            "segment_summary": {
                "total_shots": len(shots),
                "kept_shots": len(kept),
                "dropped_shots": len(dropped),
            },
            "shot_decisions": decisions,
            "kept_shot_ids": kept,
            "dropped_shot_ids": dropped,
        }

    def _judge_one(
        self,
        video_profile: VideoProfile,
        chapter: ChapterRecord | None,
        shot: ShotRecord,
        model_manager: Any,
    ) -> Dict[str, Any]:
        stats = self._visual_stats(shot)
        hard_drop_reason = self._hard_drop_reason(shot, stats)
        if hard_drop_reason:
            return {
                "shot_id": shot.shot_id,
                "usable_score": int(stats["usable_score"]),
                "quality_score": int(stats["quality_score"]),
                "informativeness_score": int(stats["informativeness_score"]),
                "decision": "drop",
                "reason": hard_drop_reason,
                "evidence": [
                    f"entropy_mean={stats['entropy_mean']:.3f}",
                    f"contrast_mean={stats['contrast_mean']:.3f}",
                    f"color_variance_mean={stats['color_variance_mean']:.3f}",
                    f"shot_role={shot.shot_role}",
                    f"vlm_summary={shot.vlm_summary[:120]}",
                ],
            }

        prompt = (
            "You are a strict shot-admission judge for a video frame selection pipeline.\n"
            "Judge whether the current shot is worth sending to downstream frame sampling.\n"
            "Keep shots only if they are visually usable and likely to yield informative frames.\n"
            "Reject shots that are mainly low-information filler, weak transitions, visually poor, "
            "or not worth spending frame-sampling budget on.\n"
            "Return strict JSON with keys usable_score, quality_score, informativeness_score, decision, reason, evidence.\n"
            f"Video summary: {video_profile.video_summary}\n"
            f"Video type: {video_profile.video_type}\n"
            f"Chapter role: {getattr(chapter, 'chapter_role', '')}\n"
            f"Chapter summary: {getattr(chapter, 'chapter_summary', '')}\n"
            f"Shot summary: {shot.vlm_summary}\n"
            f"Shot role: {shot.shot_role}\n"
            f"Scene type: {shot.scene_type}\n"
            f"Visual quality hint: {shot.visual_quality_hint}\n"
            f"Informativeness hint: {shot.informativeness_hint}\n"
            f"Sampling hint: {shot.candidate_sampling_hint}\n"
            f"Representative frame count: {len(shot.rep_frames or [])}\n"
        )
        decision = {}
        llm_fallback_used = False
        try:
            llm = model_manager.get_resource("llm")
            if getattr(llm, "mode", "local") != "api":
                raw = llm.chat(prompt, max_new_tokens=220)
                obj = parse_llm_json(str(raw or ""))
                if isinstance(obj, dict):
                    decision = obj
                else:
                    llm_fallback_used = True
            else:
                llm_fallback_used = True
        except Exception:
            decision = {}
            llm_fallback_used = True

        usable_score = self._safe_int(decision.get("usable_score"), self._fallback_usable_score(shot))
        quality_score = self._safe_int(decision.get("quality_score"), self._hint_to_score(shot.visual_quality_hint))
        info_score = self._safe_int(decision.get("informativeness_score"), self._hint_to_score(shot.informativeness_hint))
        verdict = str(decision.get("decision", "") or "").strip().lower()
        if verdict not in {"keep", "borderline", "drop"}:
            if usable_score >= self.keep_threshold:
                verdict = "keep"
            elif usable_score >= self.borderline_threshold:
                verdict = "borderline"
            else:
                verdict = "drop"

        if shot.shot_role == "transition_bridge" and usable_score < 80:
            verdict = "drop"
        if shot.visual_quality_hint == "low" and quality_score < 55:
            verdict = "drop"

        return {
            "shot_id": shot.shot_id,
            "usable_score": usable_score,
            "quality_score": quality_score,
            "informativeness_score": info_score,
            "decision": verdict,
            "reason": str(decision.get("reason", "") or self._fallback_reason(shot, verdict)),
            "evidence": list(decision.get("evidence", []) or []),
            "fallback_flags": {
                "llm_fallback_used": bool(llm_fallback_used),
            },
        }

    def _safe_int(self, value: Any, default: int) -> int:
        try:
            return max(0, min(100, int(value)))
        except Exception:
            return default

    def _hint_to_score(self, hint: str) -> int:
        return {"low": 35, "medium": 65, "high": 85}.get((hint or "").lower(), 60)

    def _fallback_usable_score(self, shot: ShotRecord) -> int:
        score = 0.55 * self._hint_to_score(shot.visual_quality_hint) + 0.45 * self._hint_to_score(shot.informativeness_hint)
        if shot.shot_role in {"screen_demo", "slide", "title_card"}:
            score += 8
        if shot.shot_role == "transition_bridge":
            score -= 18
        return int(max(0, min(100, round(score))))

    def _fallback_reason(self, shot: ShotRecord, verdict: str) -> str:
        if verdict == "drop":
            if shot.shot_role == "transition_bridge":
                return "short transition shot with weak standalone value"
            if shot.visual_quality_hint == "low":
                return "visual quality is too weak for downstream sampling"
            return "shot is not informative enough for sampling budget"
        return "shot is visually usable and informative enough for downstream sampling"

    def _visual_stats(self, shot: ShotRecord) -> Dict[str, float]:
        entropies: List[float] = []
        contrasts: List[float] = []
        color_vars: List[float] = []
        for rep in shot.rep_frames or []:
            img_path = str(getattr(rep, "img_path", "") or "")
            if not img_path or not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (192, 192), interpolation=cv2.INTER_AREA)
            hist = cv2.calcHist([small], [0], None, [256], [0, 256]).flatten()
            hist = hist / max(1.0, float(hist.sum()))
            entropy = float(-(hist * np.log2(hist + 1e-8)).sum()) / 8.0
            entropies.append(max(0.0, min(1.0, entropy)))
            contrasts.append(max(0.0, min(1.0, float(small.std()) / 64.0)))
            color_vars.append(max(0.0, min(1.0, float(img.astype(np.float32).var()) / 4096.0)))

        entropy_mean = float(np.mean(entropies)) if entropies else 0.0
        contrast_mean = float(np.mean(contrasts)) if contrasts else 0.0
        color_variance_mean = float(np.mean(color_vars)) if color_vars else 0.0
        quality_score = self._hint_to_score(shot.visual_quality_hint)
        informativeness_score = int(
            max(
                0,
                min(
                    100,
                    round(
                        100.0
                        * (
                            0.45 * entropy_mean
                            + 0.25 * contrast_mean
                            + 0.30 * color_variance_mean
                        )
                    ),
                ),
            )
        )
        usable_score = int(round(0.45 * quality_score + 0.55 * informativeness_score))
        return {
            "entropy_mean": entropy_mean,
            "contrast_mean": contrast_mean,
            "color_variance_mean": color_variance_mean,
            "informativeness_score": informativeness_score,
            "quality_score": quality_score,
            "usable_score": usable_score,
        }

    def _hard_drop_reason(self, shot: ShotRecord, stats: Dict[str, float]) -> str:
        if (
            stats["entropy_mean"] < 0.12
            and stats["contrast_mean"] < 0.10
            and stats["color_variance_mean"] < 0.08
        ):
            return "visual information is too weak; likely pure-color or near-empty content"
        if stats["informativeness_score"] < 22:
            return "shot information density is too low for frame sampling"
        if shot.shot_role == "transition_bridge" and stats["informativeness_score"] < 45:
            return "transition shot with insufficient standalone information"
        return ""
