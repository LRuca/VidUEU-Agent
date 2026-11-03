from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from executor.tasks.prompt_loader import load_prompt_text


class QuotaPlanner:
    """
    Hierarchical quota planner.

    Canonical output:
    - global_plan: {global_budget, reason}
    - chapter_plans[*]: {chapter_budget, reason}
    - shot_plans[*]: {target_quota, reason}

    Compatibility aliases are also emitted:
    - global_plan.total_budget
    - shot_plans[*].final_quota
    - shot_plans[*].raw_quota
    - shot_plans[*].importance_hint
    - shot_plans[*].coverage_need
    """

    SYSTEM_PROMPT = ""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.system_prompt = load_prompt_text("quota_plan")
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_quota_planning", {}) or {})
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1400) or 1400)
        self.max_shot_quota = int(cfg.get("max_shot_quota", 4) or 4)

    def plan(
        self,
        video_profile: Any,
        chapters: List[Any],
        shots: List[Any],
        value_frames: List[Dict[str, Any]],
        llm: Any = None,
    ) -> Dict[str, Any]:
        shot_stats = self._build_shot_stats(shots, value_frames)
        chapter_stats = self._build_chapter_stats(chapters, shot_stats)
        video_duration_sec = self._estimate_video_duration_sec(video_profile, chapters, shot_stats)
        quota_constraint = self._build_global_quota_constraint(video_duration_sec, shot_stats)

        payload = {
            "selection_objective": str(
                getattr(video_profile, "selection_objective", "") or "balanced_chapter_coverage"
            ),
            "video_context": {
                "video_type": str(getattr(video_profile, "video_type", "") or "generic"),
                "video_summary": str(getattr(video_profile, "video_summary", "") or ""),
                "total_duration_sec": float(round(video_duration_sec, 3)),
                "allowed_total_quota_range": str(quota_constraint["range_text"]),
                "allowed_total_quota_min": int(quota_constraint["min_quota"]),
                "allowed_total_quota_max": int(quota_constraint["max_quota"]),
            },
            "segment_summary": {
                "num_chapters": int(len(chapter_stats)),
                "num_shots": int(len(shot_stats)),
                "total_duration_sec": float(round(video_duration_sec, 3)),
            },
            "chapters": [self._chapter_prompt_view(x) for x in chapter_stats],
            "shots": [self._shot_prompt_view(x) for x in shot_stats],
        }

        raw = self._call_llm(payload, llm=llm)
        if raw is None:
            raw = self._fallback_plan(chapter_stats, shot_stats, quota_constraint)
        return self._validate_and_reconcile(
            raw,
            chapters,
            shots,
            shot_stats,
            chapter_stats,
            quota_constraint,
        )

    def _build_shot_stats(self, shots: List[Any], value_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidate_counts: Dict[str, int] = {}
        for row in value_frames or []:
            shot_id = str(row.get("shot_id", "") or "")
            if shot_id:
                candidate_counts[shot_id] = candidate_counts.get(shot_id, 0) + 1

        out: List[Dict[str, Any]] = []
        for shot_index, shot in enumerate(shots):
            shot_id = str(getattr(shot, "shot_id", "") or "")
            shot_start_sec, shot_end_sec = self._extract_time_range_from_shot(shot, [])
            duration_sec = max(
                0.0,
                self._first_valid_float(
                    [getattr(shot, "duration", None), (getattr(shot, "metadata", {}) or {}).get("duration")],
                    default=(shot_end_sec - shot_start_sec),
                )
                or 0.0,
            )
            shot_summary = str(getattr(shot, "vlm_summary", "") or "")
            process_like_hint = self._infer_process_like_hint(shot, shot_summary)

            out.append(
                {
                    "shot_id": shot_id,
                    "shot_index": int(shot_index),
                    "chapter_id": str(getattr(shot, "metadata", {}).get("chapter_id", "") or ""),
                    "shot_role": str(getattr(shot, "shot_role", "") or "generic"),
                    "scene_type": str(getattr(shot, "scene_type", "") or "generic"),
                    "shot_start_sec": float(shot_start_sec),
                    "shot_end_sec": float(shot_end_sec),
                    "duration_sec": float(duration_sec),
                    "shot_summary": shot_summary,
                    "summary_word_count": int(self._count_summary_words(shot_summary)),
                    "stage_keyword_hits": int(self._count_stage_keywords(shot_summary)),
                    "process_like_hint": bool(process_like_hint),
                    "candidate_capacity": int(max(0, int(candidate_counts.get(shot_id, 0) or 0))),
                }
            )
        return out

    def _build_chapter_stats(self, chapters: List[Any], shot_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_chapter: Dict[str, List[Dict[str, Any]]] = {}
        for row in shot_stats:
            by_chapter.setdefault(str(row.get("chapter_id", "") or ""), []).append(row)

        out: List[Dict[str, Any]] = []
        for chapter in chapters:
            chapter_id = str(getattr(chapter, "chapter_id", "") or "")
            rows = by_chapter.get(chapter_id, [])
            chapter_start = self._first_valid_float(
                [getattr(chapter, "start_sec", None), (getattr(chapter, "metadata", {}) or {}).get("start_sec")],
                default=None,
            )
            chapter_end = self._first_valid_float(
                [getattr(chapter, "end_sec", None), (getattr(chapter, "metadata", {}) or {}).get("end_sec")],
                default=None,
            )
            if chapter_start is None or chapter_end is None or chapter_end < chapter_start:
                starts = [self._safe_float(x.get("shot_start_sec", 0.0)) for x in rows]
                ends = [self._safe_float(x.get("shot_end_sec", 0.0)) for x in rows]
                chapter_start = min(starts) if starts else 0.0
                chapter_end = max(ends) if ends else chapter_start
            duration_sec = max(
                0.0,
                self._first_valid_float(
                    [getattr(chapter, "duration", None), (getattr(chapter, "metadata", {}) or {}).get("duration")],
                    default=(chapter_end - chapter_start),
                )
                or 0.0,
            )
            chapter_summary = str(getattr(chapter, "chapter_summary", "") or "")

            out.append(
                {
                    "chapter_id": chapter_id,
                    "chapter_role": str(getattr(chapter, "chapter_role", "") or "generic"),
                    "chapter_summary": chapter_summary,
                    "num_shots": int(len(rows)),
                    "chapter_start_sec": float(chapter_start),
                    "chapter_end_sec": float(chapter_end),
                    "duration_sec": float(duration_sec),
                    "summary_word_count": int(self._count_summary_words(chapter_summary)),
                    "process_like_shots": int(sum(1 for x in rows if bool(x.get("process_like_hint", False)))),
                    "shot_ids": [str(x.get("shot_id", "") or "") for x in rows],
                }
            )
        return out

    def _shot_prompt_view(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "shot_id": str(stats.get("shot_id", "") or ""),
            "chapter_id": str(stats.get("chapter_id", "") or ""),
            "shot_role": str(stats.get("shot_role", "") or "generic"),
            "scene_type": str(stats.get("scene_type", "") or "generic"),
            "duration_sec": float(stats.get("duration_sec", 0.0) or 0.0),
            "shot_summary": str(stats.get("shot_summary", "") or ""),
        }

    def _chapter_prompt_view(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chapter_id": str(stats.get("chapter_id", "") or ""),
            "chapter_role": str(stats.get("chapter_role", "") or "generic"),
            "duration_sec": float(stats.get("duration_sec", 0.0) or 0.0),
            "num_shots": int(stats.get("num_shots", 0) or 0),
            "chapter_summary": str(stats.get("chapter_summary", "") or ""),
        }

    def _estimate_video_duration_sec(
        self,
        video_profile: Any,
        chapters: List[Any],
        shot_stats: List[Dict[str, Any]],
    ) -> float:
        metadata = getattr(video_profile, "metadata", {}) or {}
        direct_candidates = [
            metadata.get("duration_sec"),
            metadata.get("video_duration_sec"),
            metadata.get("duration"),
        ]
        direct_duration = self._first_valid_float(direct_candidates, default=None)
        if direct_duration is not None and direct_duration > 0:
            return float(direct_duration)

        chapter_starts = []
        chapter_ends = []
        for chapter in chapters or []:
            start_sec = self._first_valid_float(
                [getattr(chapter, "start_sec", None), (getattr(chapter, "metadata", {}) or {}).get("start_sec")],
                default=None,
            )
            end_sec = self._first_valid_float(
                [getattr(chapter, "end_sec", None), (getattr(chapter, "metadata", {}) or {}).get("end_sec")],
                default=None,
            )
            if start_sec is not None and end_sec is not None and end_sec >= start_sec:
                chapter_starts.append(float(start_sec))
                chapter_ends.append(float(end_sec))
        if chapter_starts and chapter_ends:
            return float(max(chapter_ends) - min(chapter_starts))

        shot_starts = [self._safe_float(x.get("shot_start_sec", 0.0)) for x in shot_stats]
        shot_ends = [self._safe_float(x.get("shot_end_sec", 0.0)) for x in shot_stats]
        if shot_starts and shot_ends:
            return float(max(shot_ends) - min(shot_starts))

        return float(sum(self._safe_float(x.get("duration_sec", 0.0)) for x in shot_stats))

    def _build_global_quota_constraint(
        self,
        video_duration_sec: float,
        shot_stats: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        feasible_total = int(sum(self._max_quota_for_shot(x) for x in shot_stats))
        if feasible_total <= 0:
            min_quota = 0
            max_quota = 0
        else:
            raw_min = max(0.0, float(video_duration_sec) * 0.4)
            raw_max = max(0.0, float(video_duration_sec) * 0.6)
            min_quota = max(1, self._round_half_up(raw_min))
            max_quota = max(min_quota, self._round_half_up(raw_max))
            if max_quota == min_quota and feasible_total >= (min_quota + 1):
                max_quota = min_quota + 1
            min_quota = min(min_quota, feasible_total)
            max_quota = min(max_quota, feasible_total)
            if max_quota < min_quota:
                min_quota = max_quota
        return {
            "video_duration_sec": float(max(0.0, video_duration_sec)),
            "min_quota": int(min_quota),
            "max_quota": int(max_quota),
            "feasible_total": int(feasible_total),
            "range_text": f"{int(min_quota)}-{int(max_quota)} frames",
        }

    def _round_half_up(self, value: float) -> int:
        value = float(value or 0.0)
        return int(math.floor(value + 0.5))

    def _call_llm(self, payload: Dict[str, Any], llm: Any) -> Dict[str, Any] | None:
        if llm is None:
            return None
        prompt = (
            self.system_prompt.strip()
            + "\n\n"
            + "Return JSON with exactly this structure:\n"
            + "{\n"
            + '  "global_plan": {"global_budget": 0, "reason": "..."},\n'
            + '  "chapter_plans": {"chapter_id": {"chapter_budget": 0, "reason": "..."}},\n'
            + '  "shot_plans": {"shot_id": {"target_quota": 0, "reason": "..."}}\n'
            + "}\n\n"
            + "Input JSON:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        try:
            raw = llm.chat(prompt, max_new_tokens=self.max_new_tokens)
        except Exception:
            return None
        return self._parse_json_obj(raw)

    def _fallback_plan(
        self,
        chapter_stats: List[Dict[str, Any]],
        shot_stats: List[Dict[str, Any]],
        quota_constraint: Dict[str, Any],
    ) -> Dict[str, Any]:
        global_budget = self._fallback_global_budget(shot_stats, quota_constraint)
        shot_quotas = self._allocate_budget_by_priority(shot_stats, global_budget)

        shot_plans: Dict[str, Dict[str, Any]] = {}
        for row in shot_stats:
            shot_id = str(row.get("shot_id", "") or "")
            quota = int(shot_quotas.get(shot_id, 0))
            shot_plans[shot_id] = {
                "target_quota": quota,
                "reason": self._build_fallback_reason(row, quota),
            }

        chapter_plans: Dict[str, Dict[str, Any]] = {}
        for row in chapter_stats:
            chapter_id = str(row.get("chapter_id", "") or "")
            chapter_budget = sum(
                int(shot_plans[sid]["target_quota"])
                for sid in row.get("shot_ids", []) or []
                if sid in shot_plans
            )
            chapter_plans[chapter_id] = {
                "chapter_budget": int(chapter_budget),
                "reason": self._build_chapter_reason(row, chapter_budget),
            }

        return {
            "global_plan": {
                "global_budget": int(global_budget),
                "reason": self._build_global_reason(quota_constraint, int(global_budget)),
            },
            "chapter_plans": chapter_plans,
            "shot_plans": shot_plans,
        }

    def _validate_and_reconcile(
        self,
        raw: Dict[str, Any],
        chapters: List[Any],
        shots: List[Any],
        shot_stats: List[Dict[str, Any]],
        chapter_stats: List[Dict[str, Any]],
        quota_constraint: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            raw = {}

        shot_stats_by_id = {str(x.get("shot_id", "")): x for x in shot_stats}
        chapter_stats_by_id = {str(x.get("chapter_id", "")): x for x in chapter_stats}
        chapter_ids = [str(getattr(x, "chapter_id", "") or "") for x in chapters]
        shot_ids = [str(getattr(x, "shot_id", "") or "") for x in shots]
        shot_to_chapter = {
            str(getattr(x, "shot_id", "") or ""): str(getattr(x, "metadata", {}).get("chapter_id", "") or "")
            for x in shots
        }

        shot_plans_raw = dict(raw.get("shot_plans", {}) or {})
        parsed_shot_plans: Dict[str, Dict[str, Any]] = {}
        for shot_id in shot_ids:
            row = shot_plans_raw.get(shot_id, {}) or {}
            stats = shot_stats_by_id.get(shot_id, {})
            quota = min(
                self._safe_nonneg_int(
                    row.get("target_quota", row.get("final_quota", row.get("raw_quota", 0)))
                ),
                self._max_quota_for_shot(stats),
            )
            parsed_shot_plans[shot_id] = {
                "target_quota": int(quota),
                "_requested_quota": int(quota),
                "reason": str(row.get("reason", "") or self._build_fallback_reason(stats, quota)),
            }

        requested_global_budget = self._resolve_requested_global_budget(
            raw.get("global_plan", {}) or {},
            parsed_shot_plans,
            quota_constraint,
        )
        parsed_shot_plans = self._reconcile_shot_quotas_to_target(
            parsed_shot_plans,
            shot_stats,
            requested_global_budget,
        )

        shot_plans: Dict[str, Dict[str, Any]] = {}
        for shot_id in shot_ids:
            row = parsed_shot_plans.get(shot_id, {}) or {}
            stats = shot_stats_by_id.get(shot_id, {})
            quota = self._safe_nonneg_int(
                row.get("target_quota", row.get("final_quota", row.get("raw_quota", 0)))
            )
            requested_quota = self._safe_nonneg_int(row.get("_requested_quota", quota))
            reason = str(row.get("reason", "") or "")
            if not reason or requested_quota != quota:
                reason = self._build_fallback_reason(stats, quota)
            importance_hint = str(row.get("importance_hint", "") or self._derive_importance_hint(stats, quota))
            coverage_need = str(row.get("coverage_need", "") or self._derive_coverage_need(stats, quota))

            shot_plans[shot_id] = {
                "target_quota": int(quota),
                "final_quota": int(quota),
                "raw_quota": int(quota),
                "importance_hint": importance_hint,
                "coverage_need": coverage_need,
                "reason": reason,
            }

        chapter_plans_raw = dict(raw.get("chapter_plans", {}) or {})
        chapter_plans: Dict[str, Dict[str, Any]] = {}
        for chapter_id in chapter_ids:
            budget = sum(
                int(v["target_quota"])
                for sid, v in shot_plans.items()
                if shot_to_chapter.get(sid, "") == chapter_id
            )
            row = chapter_plans_raw.get(chapter_id, {}) or {}
            stats = chapter_stats_by_id.get(chapter_id, {})
            reason = str(row.get("reason", "") or "")
            if not reason or self._safe_nonneg_int(row.get("chapter_budget", 0)) != budget:
                reason = self._build_chapter_reason(stats, budget)
            chapter_plans[chapter_id] = {
                "chapter_budget": int(budget),
                "reason": reason,
            }

        global_plan_raw = dict(raw.get("global_plan", {}) or {})
        global_budget = requested_global_budget
        global_reason = str(global_plan_raw.get("reason", "") or "")
        if not global_reason or self._safe_nonneg_int(global_plan_raw.get("global_budget", global_plan_raw.get("total_budget", 0))) != global_budget:
            global_reason = self._build_global_reason(quota_constraint, global_budget)
        global_plan = {
            "global_budget": int(global_budget),
            "total_budget": int(global_budget),
            "reason": global_reason,
        }

        return {
            "global_plan": global_plan,
            "chapter_plans": chapter_plans,
            "shot_plans": shot_plans,
        }

    def _build_fallback_reason(self, stats: Dict[str, Any], quota: int) -> str:
        if not stats:
            return "fallback quota planning using shot summary and duration"
        parts = [
            f"duration={self._safe_float(stats.get('duration_sec', 0.0)):.1f}s",
            f"summary_words={int(stats.get('summary_word_count', 0) or 0)}",
            f"stage_keywords={int(stats.get('stage_keyword_hits', 0) or 0)}",
        ]
        if bool(stats.get("process_like_hint", False)):
            parts.append("process_like=True")
        if not str(stats.get("shot_summary", "") or "").strip():
            parts.append("summary_missing=True")
        return f"fallback quota={int(quota)} based on " + ", ".join(parts)

    def _build_chapter_reason(self, stats: Dict[str, Any], budget: int) -> str:
        if not stats:
            return "reconciled from shot quotas"
        return (
            f"chapter_budget={int(budget)} derived from shot quotas; "
            f"num_shots={int(stats.get('num_shots', 0) or 0)}, "
            f"duration={self._safe_float(stats.get('duration_sec', 0.0)):.1f}s, "
            f"summary_words={int(stats.get('summary_word_count', 0) or 0)}, "
            f"process_like_shots={int(stats.get('process_like_shots', 0) or 0)}"
        )

    def _derive_importance_hint(self, stats: Dict[str, Any], quota: int) -> str:
        priority = self._shot_priority_score(stats)
        if quota >= 3 or priority >= 1.55:
            return "high"
        if quota >= 2 or priority >= 0.95:
            return "medium"
        return "low"

    def _derive_coverage_need(self, stats: Dict[str, Any], quota: int) -> str:
        if quota >= 3:
            return "high"
        if quota <= 0:
            return "low"
        stage_keyword_hits = int(stats.get("stage_keyword_hits", 0) or 0)
        duration_sec = self._safe_float(stats.get("duration_sec", 0.0))
        if stage_keyword_hits >= 2 or (bool(stats.get("process_like_hint", False)) and duration_sec >= 10.0):
            return "high"
        if stage_keyword_hits >= 1 or duration_sec >= 8.0:
            return "medium"
        return "low"

    def _build_global_reason(self, quota_constraint: Dict[str, Any], global_budget: int) -> str:
        return (
            f"global_budget={int(global_budget)} chosen within allowed range "
            f"{quota_constraint.get('range_text', '0-0 frames')} using video/chapter/shot summaries and durations"
        )

    def _resolve_requested_global_budget(
        self,
        global_plan_raw: Dict[str, Any],
        shot_plans: Dict[str, Dict[str, Any]],
        quota_constraint: Dict[str, Any],
    ) -> int:
        requested = self._safe_nonneg_int(
            global_plan_raw.get(
                "global_budget",
                global_plan_raw.get("total_budget", sum(int(v.get("target_quota", 0) or 0) for v in shot_plans.values())),
            )
        )
        min_quota = int(quota_constraint.get("min_quota", 0) or 0)
        max_quota = int(quota_constraint.get("max_quota", 0) or 0)
        if max_quota < min_quota:
            max_quota = min_quota
        if requested <= 0 and max_quota > 0:
            requested = min_quota
        requested = max(min_quota, min(max_quota, requested))
        feasible_total = int(quota_constraint.get("feasible_total", 0) or 0)
        if feasible_total >= 0:
            requested = min(requested, feasible_total)
        return int(requested)

    def _reconcile_shot_quotas_to_target(
        self,
        shot_plans: Dict[str, Dict[str, Any]],
        shot_stats: List[Dict[str, Any]],
        target_total: int,
    ) -> Dict[str, Dict[str, Any]]:
        out = {
            str(k): {
                **dict(v or {}),
                "target_quota": int(max(0, self._safe_nonneg_int((v or {}).get("target_quota", 0)))),
            }
            for k, v in (shot_plans or {}).items()
        }
        for stats in shot_stats:
            shot_id = str(stats.get("shot_id", "") or "")
            out.setdefault(shot_id, {"target_quota": 0, "reason": self._build_fallback_reason(stats, 0)})
            out[shot_id]["target_quota"] = min(out[shot_id]["target_quota"], self._max_quota_for_shot(stats))

        current_total = sum(int(v.get("target_quota", 0) or 0) for v in out.values())
        while current_total < target_total:
            shot_id = self._pick_shot_for_increment(out, shot_stats)
            if not shot_id:
                break
            out[shot_id]["target_quota"] = int(out[shot_id].get("target_quota", 0) or 0) + 1
            current_total += 1

        while current_total > target_total:
            shot_id = self._pick_shot_for_decrement(out, shot_stats)
            if not shot_id:
                break
            out[shot_id]["target_quota"] = max(0, int(out[shot_id].get("target_quota", 0) or 0) - 1)
            current_total -= 1

        return out

    def _allocate_budget_by_priority(
        self,
        shot_stats: List[Dict[str, Any]],
        total_budget: int,
    ) -> Dict[str, int]:
        plans = {str(row.get("shot_id", "") or ""): 0 for row in shot_stats}
        remaining = int(max(0, total_budget))
        while remaining > 0:
            shot_id = self._pick_shot_for_increment(
                {sid: {"target_quota": quota} for sid, quota in plans.items()},
                shot_stats,
            )
            if not shot_id:
                break
            plans[shot_id] = int(plans.get(shot_id, 0)) + 1
            remaining -= 1
        return plans

    def _pick_shot_for_increment(
        self,
        shot_plans: Dict[str, Dict[str, Any]],
        shot_stats: List[Dict[str, Any]],
    ) -> str:
        best_shot_id = ""
        best_key: Tuple[float, float, float, float] | None = None
        for stats in shot_stats:
            shot_id = str(stats.get("shot_id", "") or "")
            current_quota = int((shot_plans.get(shot_id, {}) or {}).get("target_quota", 0) or 0)
            max_quota = self._max_quota_for_shot(stats)
            if current_quota >= max_quota:
                continue
            priority = self._shot_priority_score(stats)
            marginal_gain = priority / (1.0 + 0.85 * current_quota)
            if current_quota == 0:
                marginal_gain += 0.18
            key = (
                float(marginal_gain),
                float(priority),
                self._safe_float(stats.get("duration_sec", 0.0)),
                -self._safe_float(stats.get("shot_start_sec", 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_shot_id = shot_id
        return best_shot_id

    def _pick_shot_for_decrement(
        self,
        shot_plans: Dict[str, Dict[str, Any]],
        shot_stats: List[Dict[str, Any]],
    ) -> str:
        best_shot_id = ""
        best_key: Tuple[float, float, float, float] | None = None
        for stats in shot_stats:
            shot_id = str(stats.get("shot_id", "") or "")
            current_quota = int((shot_plans.get(shot_id, {}) or {}).get("target_quota", 0) or 0)
            if current_quota <= 0:
                continue
            priority = self._shot_priority_score(stats)
            penalty = priority / (1.0 + 0.60 * max(0, current_quota - 1))
            if current_quota == 1:
                penalty += 0.18
            key = (
                -float(penalty),
                -float(priority),
                -self._safe_float(stats.get("duration_sec", 0.0)),
                self._safe_float(stats.get("shot_start_sec", 0.0)),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_shot_id = shot_id
        return best_shot_id

    def _fallback_global_budget(
        self,
        shot_stats: List[Dict[str, Any]],
        quota_constraint: Dict[str, Any],
    ) -> int:
        min_quota = int(quota_constraint.get("min_quota", 0) or 0)
        max_quota = int(quota_constraint.get("max_quota", 0) or 0)
        if max_quota <= 0:
            return 0
        base_target = int(round(float(quota_constraint.get("video_duration_sec", 0.0) or 0.0) * 0.5))
        if base_target <= 0:
            base_target = min_quota
        process_like_count = sum(1 for row in shot_stats if bool(row.get("process_like_hint", False)))
        if process_like_count >= 2 and base_target < max_quota:
            base_target += 1
        return int(max(min_quota, min(max_quota, base_target)))

    def _shot_priority_score(self, stats: Dict[str, Any]) -> float:
        duration_sec = self._safe_float(stats.get("duration_sec", 0.0))
        summary_words = int(stats.get("summary_word_count", 0) or 0)
        stage_keyword_hits = int(stats.get("stage_keyword_hits", 0) or 0)
        score = 0.15
        if str(stats.get("shot_summary", "") or "").strip():
            score += 0.28
        score += min(0.40, float(summary_words) / 40.0)
        score += min(0.42, duration_sec / 12.0)
        score += min(0.36, 0.18 * float(stage_keyword_hits))
        if bool(stats.get("process_like_hint", False)):
            score += 0.24
        if str(stats.get("shot_role", "") or "").strip() and str(stats.get("shot_role", "")).strip().lower() != "generic":
            score += 0.12
        if str(stats.get("scene_type", "") or "").strip() and str(stats.get("scene_type", "")).strip().lower() != "generic":
            score += 0.08
        return float(score)

    def _max_quota_for_shot(self, stats: Dict[str, Any]) -> int:
        return int(max(0, int(stats.get("candidate_capacity", 0) or 0)))

    def _count_summary_words(self, text: str) -> int:
        return int(len(re.findall(r"[A-Za-z0-9]+", str(text or ""))))

    def _count_stage_keywords(self, text: str) -> int:
        lowered = str(text or "").lower()
        keywords = [
            "step",
            "stage",
            "process",
            "then",
            "next",
            "after",
            "before",
            "result",
            "transition",
            "change",
            "prepare",
            "assembly",
            "tutorial",
            "instruction",
            "demonstration",
        ]
        return int(sum(1 for token in keywords if token in lowered))

    def _extract_semantic_text(self, row: Dict[str, Any]) -> str:
        vp = row.get("value_profile", {}) or {}
        sp = row.get("semantic_profile", {}) or {}
        ts = row.get("tool_signals", {}) or {}
        for candidate in [
            vp.get("semantic_text", ""),
            sp.get("semantic_text", ""),
            ts.get("semantic_summary", ""),
        ]:
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    def _pick_semantic_examples(self, rows_by_time: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
        candidates = []
        seen = set()
        for row in rows_by_time:
            text = self._extract_semantic_text(row)
            norm = self._normalize_text(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            candidates.append(
                {
                    "sec": round(self._safe_float(row.get("sec", 0.0)), 2),
                    "text": text[:220],
                    "value_score": round(self._safe_float(row.get("value_score", 0.0)), 4),
                }
            )

        if len(candidates) <= max_n:
            return candidates

        chosen_idx = sorted(set(
            int(round(i * (len(candidates) - 1) / float(max_n - 1)))
            for i in range(max_n)
        ))
        return [candidates[i] for i in chosen_idx[:max_n]]

    def _collect_object_labels(self, rows: List[Dict[str, Any]]) -> List[str]:
        labels: List[str] = []
        for row in rows:
            ts = row.get("tool_signals", {}) or {}
            tools = ts.get("tools", {}) or {}
            yolo = tools.get("yolo", {}) or {}

            for label in list((yolo.get("attributes", {}) or {}).get("labels", []) or []):
                lab = str(label or "").strip().lower()
                if lab:
                    labels.append(lab)

            for det in list((yolo.get("evidence", {}) or {}).get("detections", []) or []):
                lab = str((det or {}).get("label", "") or "").strip().lower()
                if lab:
                    labels.append(lab)

        return [x for x, _ in Counter(labels).most_common()]

    def _collect_motion_peaks(self, rows: List[Dict[str, Any]]) -> Tuple[List[float], int]:
        peaks: List[float] = []
        seen = set()
        high_count = 0
        for row in rows:
            ts = row.get("tool_signals", {}) or {}
            tools = ts.get("tools", {}) or {}
            flow = tools.get("optical_flow", {}) or {}
            pair_hints = list((flow.get("attributes", {}) or {}).get("pair_hints", []) or [])
            for hint in pair_hints:
                key = (
                    self._safe_int(hint.get("from_frame_index", -1)),
                    self._safe_int(hint.get("to_frame_index", -1)),
                )
                if key in seen:
                    continue
                seen.add(key)
                level = str(hint.get("flow_level", "") or "").strip().lower()
                sec = self._safe_float(hint.get("to_sec", hint.get("from_sec", 0.0)))
                if level == "high":
                    high_count += 1
                    peaks.append(sec)
        peaks = sorted(peaks)
        return peaks, high_count

    def _extract_time_range_from_shot(
        self,
        shot: Any,
        rows: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        start_candidates = [
            getattr(shot, "start_sec", None),
            getattr(shot, "start_time", None),
            getattr(shot, "start", None),
            getattr(shot, "t_start", None),
            (getattr(shot, "metadata", {}) or {}).get("start_sec"),
            (getattr(shot, "metadata", {}) or {}).get("start_time"),
            (getattr(shot, "metadata", {}) or {}).get("start"),
        ]
        end_candidates = [
            getattr(shot, "end_sec", None),
            getattr(shot, "end_time", None),
            getattr(shot, "end", None),
            getattr(shot, "t_end", None),
            (getattr(shot, "metadata", {}) or {}).get("end_sec"),
            (getattr(shot, "metadata", {}) or {}).get("end_time"),
            (getattr(shot, "metadata", {}) or {}).get("end"),
        ]

        start_sec = self._first_valid_float(start_candidates, default=None)
        end_sec = self._first_valid_float(end_candidates, default=None)

        if start_sec is None or end_sec is None or end_sec < start_sec:
            secs = [self._safe_float(x.get("sec", 0.0)) for x in rows if x.get("sec") is not None]
            if secs:
                start_sec = min(secs) if start_sec is None else start_sec
                end_sec = max(secs) if end_sec is None else end_sec
            else:
                start_sec = 0.0 if start_sec is None else start_sec
                end_sec = start_sec if end_sec is None else end_sec

        return float(start_sec), float(max(end_sec, start_sec))

    def _infer_process_like_hint(self, shot: Any, shot_summary: str = "") -> bool:
        text = " ".join(
            [
                str(getattr(shot, "shot_role", "") or ""),
                str(getattr(shot, "scene_type", "") or ""),
                str(getattr(shot, "informativeness_hint", "") or ""),
                str(shot_summary or ""),
            ]
        ).lower()
        keywords = [
            "prep", "preparation", "ingredient", "cooking", "recipe", "mixing", "chopping", "pouring",
            "assembly", "assemble", "tutorial", "repair", "fixing", "makeup", "craft", "procedure",
            "step", "washing", "cleaning", "demonstration", "how-to"
        ]
        return any(k in text for k in keywords)

    def _infer_likely_multi_step(
        self,
        process_like_hint: bool,
        duration_sec: float,
        temporal_span_sec: float,
        semantic_stage_count: int,
        high_flow_event_count: int,
        unique_object_labels_count: int,
    ) -> bool:
        evidence = 0
        if semantic_stage_count >= 2:
            evidence += 1
        if high_flow_event_count >= 2:
            evidence += 1
        if unique_object_labels_count >= 3:
            evidence += 1
        if temporal_span_sec >= 8.0:
            evidence += 1
        if duration_sec >= 12.0:
            evidence += 1
        if process_like_hint:
            evidence += 1
        return evidence >= 3

    def _semantic_diversity_hint(self, texts: List[str]) -> float:
        if len(texts) <= 1:
            return 0.0
        uniq = len(set(self._normalize_text(t) for t in texts if self._normalize_text(t)))
        return float(uniq) / float(max(1, len(texts)))

    def _topk_time_spread(self, rows_by_value: List[Dict[str, Any]], k: int = 3) -> float:
        secs = [
            self._safe_float(x.get("sec", 0.0))
            for x in rows_by_value[: max(1, k)]
            if x.get("sec") is not None
        ]
        if len(secs) <= 1:
            return 0.0
        return float(max(secs) - min(secs))

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        arr = sorted(float(x) for x in values)
        if len(arr) == 1:
            return arr[0]
        pos = max(0.0, min(1.0, q)) * (len(arr) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return arr[lo]
        alpha = pos - lo
        return arr[lo] * (1.0 - alpha) + arr[hi] * alpha

    def _normalize_text(self, text: str) -> str:
        text = str(text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    def _first_valid_float(self, candidates: List[Any], default: Any = 0.0) -> Any:
        for x in candidates:
            try:
                if x is None or x == "":
                    continue
                return float(x)
            except Exception:
                continue
        return default

    def _safe_nonneg_int(self, x: Any) -> int:
        try:
            val = int(x)
        except Exception:
            val = 0
        return max(0, val)

    def _safe_float(self, x: Any, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    def _safe_int(self, x: Any, default: int = 0) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _parse_json_obj(self, raw: Any) -> Dict[str, Any] | None:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        if not text:
            return None
        for candidate in [text, self._extract_json_block(text)]:
            if not candidate:
                continue
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return None

    def _extract_json_block(self, text: str) -> str | None:
        m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return text[s : e + 1]
        return None
