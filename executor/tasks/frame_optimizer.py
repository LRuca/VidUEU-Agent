from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile


class FrameOptimizer:
    """
    Greedy optimizer with chapter coverage first, then marginal gain.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def select(
        self,
        shots: List[ShotRecord],
        chapters: List[ChapterRecord],
        video_profile: VideoProfile,
        candidates_by_shot: Dict[str, List[Dict[str, Any]]],
        quota_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        flat_candidates = self._flatten_candidates(candidates_by_shot)
        if not flat_candidates:
            return {
                "selected_frames": [],
                "chapter_coverage": {},
                "global_coverage": {},
                "optimization_meta": {"reason": "no_candidates"},
            }

        chapter_state = self._init_coverage_state(chapters, quota_plan)
        selected: List[Dict[str, Any]] = []
        total_budget = int((quota_plan.get("global_plan", {}) or {}).get("total_budget", len(flat_candidates)))

        for chapter in chapters:
            plan = (quota_plan.get("chapter_plans", {}) or {}).get(chapter.chapter_id, {})
            min_quota = int(plan.get("min_quota", 0) or 0)
            if min_quota <= 0:
                continue
            cands = [x for x in flat_candidates if x.get("chapter_id") == chapter.chapter_id and x.get("quality_gate_pass")]
            cands.sort(key=lambda x: self._score_candidate(x, video_profile), reverse=True)
            for cand in cands[:min_quota]:
                if len(selected) >= total_budget:
                    break
                if not self._already_selected_frame(cand, selected):
                    selected.append(cand)
                    chapter_state[chapter.chapter_id]["count"] += 1

        while len(selected) < total_budget:
            best = None
            best_gain = None
            for cand in flat_candidates:
                if self._already_selected_frame(cand, selected):
                    continue
                gain = self._compute_marginal_gain(cand, selected, chapter_state, quota_plan, video_profile)
                if best is None or gain > best_gain:
                    best = cand
                    best_gain = gain
            if best is None:
                break
            selected.append(best)
            chapter_id = best.get("chapter_id")
            if chapter_id in chapter_state:
                chapter_state[chapter_id]["count"] += 1

        return {
            "selected_frames": selected,
            "chapter_coverage": chapter_state,
            "global_coverage": {"selected_count": len(selected), "budget": total_budget},
            "optimization_meta": {"candidate_count": len(flat_candidates)},
        }

    def _flatten_candidates(self, candidates_by_shot: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        out = []
        for shot_id, items in (candidates_by_shot or {}).items():
            for item in items or []:
                if "shot_id" not in item:
                    item["shot_id"] = shot_id
                out.append(item)
        return out

    def _score_candidate(self, candidate: Dict[str, Any], video_profile: VideoProfile) -> float:
        base = candidate.get("base_scores", {}) or {}
        quality = float(base.get("quality", 0.0))
        utility = float(base.get("utility", 0.0))
        text = float(base.get("text", 0.0))
        change = float(base.get("change", 0.0))
        score = 0.45 * quality + 0.35 * utility + 0.15 * text + 0.15 * change
        if candidate.get("candidate_type") == "rep_peak":
            score += 0.05
        return score

    def _compute_marginal_gain(
        self,
        candidate: Dict[str, Any],
        selected: List[Dict[str, Any]],
        chapter_state: Dict[str, Dict[str, Any]],
        quota_plan: Dict[str, Any],
        video_profile: VideoProfile,
    ) -> float:
        score = self._score_candidate(candidate, video_profile)
        chapter_id = candidate.get("chapter_id")
        cplan = (quota_plan.get("chapter_plans", {}) or {}).get(chapter_id, {})
        count = int((chapter_state.get(chapter_id, {}) or {}).get("count", 0))
        target = int(cplan.get("target_quota", 1) or 1)
        if count < target:
            score += 0.35
        redundancy_penalty = 0.0
        for other in selected:
            if other.get("shot_id") == candidate.get("shot_id"):
                redundancy_penalty += 0.25
            elif other.get("chapter_id") == candidate.get("chapter_id"):
                redundancy_penalty += 0.08
        score -= redundancy_penalty
        return score

    def _init_coverage_state(self, chapters: List[ChapterRecord], quota_plan: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for chapter in chapters:
            out[chapter.chapter_id] = {
                "count": 0,
                "min_quota": int(((quota_plan.get("chapter_plans", {}) or {}).get(chapter.chapter_id, {}) or {}).get("min_quota", 0)),
                "target_quota": int(((quota_plan.get("chapter_plans", {}) or {}).get(chapter.chapter_id, {}) or {}).get("target_quota", 0)),
            }
        return out

    def _already_selected_frame(self, candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> bool:
        idx = int(candidate.get("frame_index", -1))
        return any(int(x.get("frame_index", -2)) == idx for x in selected)
