from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile


class ValueAnalyzer:
    """
    Collect per-frame value signals after quality screening and aggregate them
    using ToolPlan discrete score priorities.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def analyze(
        self,
        usable_frames: List[Dict[str, Any]],
        shots: List[ShotRecord],
        chapters: List[ChapterRecord],
        video_profile: VideoProfile,
        tool_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        _ = video_profile
        shot_map = {x.shot_id: x for x in shots}
        chapter_map = {x.chapter_id: x for x in chapters}
        shot_plans = (tool_plan.get("shot_plans", {}) or {}) if isinstance(tool_plan, dict) else {}

        out = []
        for frame in usable_frames:
            shot = shot_map.get(frame.get("shot_id"))
            chapter = chapter_map.get(frame.get("chapter_id"))
            plan = shot_plans.get(frame.get("shot_id"), {})
            value_profile = self._score_one(frame, shot, chapter, plan)
            out.append({**frame, "value_profile": value_profile, "value_score": value_profile["value_score"]})
        return out

    def _score_one(
        self,
        frame: Dict[str, Any],
        shot: ShotRecord | None,
        chapter: ChapterRecord | None,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        base = dict(frame.get("base_scores", {}) or {})
        candidate_type = str(frame.get("candidate_type", "") or "")
        score_levels = dict(plan.get("score_priority_levels", {}) or {})
        signal_scores = {
            "representativeness": float(base.get("representativeness", base.get("utility", 0.0))),
            "informativeness": {"low": 0.35, "medium": 0.60, "high": 0.85}.get(getattr(shot, "informativeness_hint", "medium"), 0.60),
            "change_gain": float(base.get("change_gain", 0.0)),
            "coverage_gain": float(base.get("coverage_gain", 0.25 if candidate_type in {"coverage_peak", "uniform_support"} else 0.0)),
            "layout_signal": float(base.get("layout_signal", 0.0)),
            "tool_signal": 0.0,
            "evidence_bonus": 0.0,
        }
        tool_signals = dict(frame.get("tool_signals", {}) or {})
        tool_signal_scores = dict(tool_signals.get("signal_scores", {}) or {})
        semantic_summary = str(tool_signals.get("semantic_summary", "") or "")
        if tool_signal_scores:
            signal_scores["tool_signal"] = max(float(x or 0.0) for x in tool_signal_scores.values())
        evidence_bonus = 0.0
        if getattr(shot, "shot_role", "") in {"screen_demo", "slide", "title_card"} and candidate_type == "layout_peak":
            evidence_bonus += 0.12
        if candidate_type in {"change_peak", "action_peak"}:
            evidence_bonus += 0.10
        if getattr(chapter, "importance_hint", "") == "high":
            evidence_bonus += 0.08
        if tool_signal_scores.get("ocr", 0.0) >= 0.45:
            evidence_bonus += 0.08
        if tool_signal_scores.get("asr", 0.0) >= 0.45:
            evidence_bonus += 0.06
        if tool_signal_scores.get("yolo", 0.0) >= 0.45:
            evidence_bonus += 0.06
        if tool_signal_scores.get("optical_flow", 0.0) >= 0.60:
            evidence_bonus += 0.08
        signal_scores["evidence_bonus"] = evidence_bonus
        if str(frame.get("tool_priority_level", "medium")) == "high":
            signal_scores["representativeness"] = min(1.0, signal_scores["representativeness"] + 0.05)

        weighted_total = 0.0
        total_weight = 0.0
        signal_weights = {}
        for name, value in signal_scores.items():
            weight = self._priority_weight(score_levels.get(name, "medium"))
            signal_weights[name] = weight
            weighted_total += weight * float(value)
            total_weight += weight
        value_score = max(0.0, min(1.0, weighted_total / max(total_weight, 1e-6)))
        return {
            "signal_scores": signal_scores,
            "signal_weights": signal_weights,
            "representativeness": signal_scores["representativeness"],
            "informativeness": signal_scores["informativeness"],
            "change_gain": signal_scores["change_gain"],
            "coverage_gain": signal_scores["coverage_gain"],
            "layout_signal": signal_scores["layout_signal"],
            "tool_signal": signal_scores["tool_signal"],
            "evidence_bonus": signal_scores["evidence_bonus"],
            "value_score": value_score,
            "tool_signal_scores": tool_signal_scores,
            "semantic_summary": semantic_summary,
            "value_reason": f"candidate_type={candidate_type}; tools={','.join(self._tool_names(tool_signals))}",
        }

    def _priority_weight(self, level: Any) -> float:
        return {"low": 1.0, "medium": 2.0, "high": 3.0}.get(str(level or "medium").lower(), 2.0)

    def _tool_names(self, tool_signals: Dict[str, Any]) -> List[str]:
        actions = list(tool_signals.get("tool_actions", []) or [])
        out: List[str] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            tool_name = str(action.get("tool_name", "") or "")
            if tool_name and tool_name not in out:
                out.append(tool_name)
        return out
