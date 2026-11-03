from __future__ import annotations

from typing import Any, Dict, List

from orchestrator.structures import ChapterRecord, ShotRecord, VideoProfile


class RecallReviewer:
    """
    Lightweight coverage repair loop.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def review_and_refine(
        self,
        selected_frames: List[Dict[str, Any]],
        shots: List[ShotRecord],
        chapters: List[ChapterRecord],
        video_profile: VideoProfile,
        candidates_by_shot: Dict[str, List[Dict[str, Any]]],
        quota_plan: Dict[str, Any],
        model_manager: Any,
    ) -> Dict[str, Any]:
        gaps = self._find_coverage_gaps(selected_frames, chapters, quota_plan)
        text_gaps = self._find_text_miss_gaps(selected_frames, candidates_by_shot)
        change_gaps = self._find_change_miss_gaps(selected_frames, candidates_by_shot)
        refinement_plan = self._suggest_append_or_replace(gaps, text_gaps, change_gaps, candidates_by_shot)
        final_selected = self._apply_refinement(selected_frames, refinement_plan, quota_plan)
        return {
            "selected_frames": final_selected,
            "recall_meta": {
                "coverage_gaps": gaps,
                "text_gaps": text_gaps,
                "change_gaps": change_gaps,
                "refinement_plan": refinement_plan,
            },
        }

    def _find_coverage_gaps(
        self,
        selected_frames: List[Dict[str, Any]],
        chapters: List[ChapterRecord],
        quota_plan: Dict[str, Any],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {x.chapter_id: 0 for x in chapters}
        for item in selected_frames:
            cid = item.get("chapter_id")
            if cid in counts:
                counts[cid] += 1
        gaps: Dict[str, int] = {}
        for chapter in chapters:
            min_quota = int(((quota_plan.get("chapter_plans", {}) or {}).get(chapter.chapter_id, {}) or {}).get("min_quota", 0))
            if counts.get(chapter.chapter_id, 0) < min_quota:
                gaps[chapter.chapter_id] = min_quota - counts.get(chapter.chapter_id, 0)
        return gaps

    def _find_text_miss_gaps(
        self, selected_frames: List[Dict[str, Any]], candidates_by_shot: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        selected_idx = {int(x.get("frame_index", -1)) for x in selected_frames}
        gaps = []
        for _, cands in (candidates_by_shot or {}).items():
            text_cands = [x for x in cands if x.get("candidate_type") == "text_peak"]
            if text_cands and all(int(x.get("frame_index", -1)) not in selected_idx for x in text_cands):
                gaps.append(max(text_cands, key=lambda x: float((x.get("base_scores", {}) or {}).get("text", 0.0))))
        return gaps

    def _find_change_miss_gaps(
        self, selected_frames: List[Dict[str, Any]], candidates_by_shot: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        selected_idx = {int(x.get("frame_index", -1)) for x in selected_frames}
        gaps = []
        for _, cands in (candidates_by_shot or {}).items():
            change_cands = [x for x in cands if x.get("candidate_type") == "change_peak"]
            if change_cands and all(int(x.get("frame_index", -1)) not in selected_idx for x in change_cands):
                gaps.append(max(change_cands, key=lambda x: float((x.get("base_scores", {}) or {}).get("change", 0.0))))
        return gaps

    def _suggest_append_or_replace(
        self,
        coverage_gaps: Dict[str, int],
        text_gaps: List[Dict[str, Any]],
        change_gaps: List[Dict[str, Any]],
        candidates_by_shot: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        append: List[Dict[str, Any]] = []
        replace: List[Dict[str, Any]] = []
        for chapter_id in coverage_gaps:
            for _, cands in (candidates_by_shot or {}).items():
                for cand in cands:
                    if cand.get("chapter_id") == chapter_id:
                        append.append(cand)
                        break
                if append:
                    break
        append.extend(text_gaps[:1])
        append.extend(change_gaps[:1])
        return {"append": append, "replace": replace}

    def _apply_refinement(
        self,
        selected_frames: List[Dict[str, Any]],
        refinement_plan: Dict[str, List[Dict[str, Any]]],
        quota_plan: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        out = list(selected_frames)
        selected_idx = {int(x.get("frame_index", -1)) for x in out}
        total_budget = int((quota_plan.get("global_plan", {}) or {}).get("total_budget", len(out)))
        for cand in refinement_plan.get("append", []) or []:
            idx = int(cand.get("frame_index", -1))
            if idx in selected_idx:
                continue
            if len(out) >= total_budget:
                break
            out.append(cand)
            selected_idx.add(idx)
        return out
