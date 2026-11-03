from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np

from orchestrator.structures import ChapterRecord, ShotRecord


class ChapterBuilder:
    """
    Turn ordered shots into chapters via pairwise merge decisions + smoothing.

    TODO:
    - replace local smoothing with DP/graph optimization if pairwise ambiguity stays high.
    - add optional change-point segmentation using ruptures on shot-level embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T1_chapter_build", {}) or {})
        self.merge_threshold = float(cfg.get("merge_threshold", 0.58) or 0.58)
        self.hard_boundary_threshold = float(cfg.get("hard_boundary_threshold", 0.35) or 0.35)
        self.min_chapter_shots = int(cfg.get("min_chapter_shots", 2) or 2)
        self.use_llm_for_pairwise = bool(cfg.get("use_llm_for_pairwise", True))

    def compute_pairwise_merge_decisions(
        self,
        shots: List[ShotRecord],
        model_manager: Any,
    ) -> Dict[str, Any]:
        decisions: List[Dict[str, Any]] = []
        for i in range(len(shots) - 1):
            left = shots[i]
            right = shots[i + 1]
            prior_score = self._pairwise_prior_score(left, right)
            llm_decision = self._llm_decide_same_chapter(left, right, model_manager)
            confidence = float(llm_decision.get("confidence", prior_score))
            if confidence <= 0 and prior_score > 0:
                confidence = prior_score
            same_chapter = bool(llm_decision.get("same_chapter", prior_score >= self.merge_threshold))
            hard_boundary = bool(
                llm_decision.get("hard_boundary", prior_score <= self.hard_boundary_threshold)
            )
            if left.shot_role == "title_card" or right.shot_role == "title_card":
                hard_boundary = True
                same_chapter = False
            decisions.append(
                {
                    "left_shot_id": left.shot_id,
                    "right_shot_id": right.shot_id,
                    "same_chapter": same_chapter,
                    "confidence": float(confidence),
                    "hard_boundary": hard_boundary,
                    "reason_type": llm_decision.get("reason_type", "prior_only"),
                    "notes": llm_decision.get("notes", ""),
                    "prior_score": float(prior_score),
                    "llm_fallback_used": bool(llm_decision.get("llm_fallback_used", False)),
                    "llm_fallback_reason": llm_decision.get("fallback_reason", ""),
                    "llm_raw": llm_decision.get("llm_raw", ""),
                }
            )
        return {
            "pairwise_merge_decisions": decisions,
            "pairwise_graph": self._build_pairwise_graph(shots, decisions),
        }

    def smooth_and_group(
        self,
        shots: List[ShotRecord],
        pairwise_merge_decisions: List[Dict[str, Any]],
        pairwise_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        decisions = self._apply_local_consistency_rules(pairwise_merge_decisions)
        decisions = self._apply_min_chapter_len_constraint(shots, decisions)
        decisions = self._merge_isolated_short_chapters(shots, decisions)
        chapters = self._group_shots_into_chapters(shots, decisions)
        return {
            "chapters": chapters,
            "smoothing_meta": {
                "decision_count": len(pairwise_merge_decisions),
                "final_decision_count": len(decisions),
                "graph": pairwise_graph,
            },
        }

    def _pairwise_prior_score(self, left: ShotRecord, right: ShotRecord) -> float:
        summary_overlap = self._token_overlap(left.vlm_summary, right.vlm_summary)
        scene_bonus = 0.12 if left.scene_type and left.scene_type == right.scene_type else 0.0
        role_bonus = 0.14 if left.shot_role == right.shot_role and left.shot_role else 0.0
        hint_bonus = 0.08 if left.informativeness_hint == right.informativeness_hint and left.informativeness_hint else 0.0
        boundary_penalty = 0.15 * max(left.boundary_confidence_out, right.boundary_confidence_in)
        score = 0.52 * summary_overlap + scene_bonus + role_bonus + hint_bonus - boundary_penalty
        return max(0.0, min(1.0, score))

    def _llm_decide_same_chapter(
        self, left_shot: ShotRecord, right_shot: ShotRecord, model_manager: Any
    ) -> Dict[str, Any]:
        if not self.use_llm_for_pairwise:
            return {"llm_fallback_used": True, "fallback_reason": "llm_pairwise_disabled"}
        try:
            llm = model_manager.get_resource("llm")
        except Exception:
            return {"llm_fallback_used": True, "fallback_reason": "llm_unavailable"}
        prompt = (
            "You are a strict JSON decision engine for merging two adjacent video shots into chapters.\n"
            "Decide whether the two shots belong to the same chapter.\n"
            "Return exactly one valid JSON object and nothing else.\n"
            "Do not use markdown fences. Do not add explanations before or after the JSON.\n"
            "The JSON schema is:\n"
            "{\n"
            '  "same_chapter": true,\n'
            '  "confidence": 0.0,\n'
            '  "hard_boundary": false,\n'
            '  "reason_type": "visual_continuation",\n'
            '  "notes": "short reason"\n'
            "}\n"
            "Rules:\n"
            "- same_chapter must be a JSON boolean: true or false.\n"
            "- confidence must be a JSON number between 0.0 and 1.0. Never output words like low, medium, or high.\n"
            "- hard_boundary must be a JSON boolean.\n"
            "- reason_type must be a short snake_case string.\n"
            "- notes must be a short plain string under 20 words.\n"
            "- If the evidence is mixed or uncertain, still return valid JSON and lower the numeric confidence.\n"
            "- Base the decision only on the provided shot summaries and hints.\n\n"
            f"LEFT role: {left_shot.shot_role}\n"
            f"LEFT vlm: {left_shot.vlm_summary}\n"
            f"LEFT scene_type: {left_shot.scene_type}\n"
            f"LEFT informativeness_hint: {left_shot.informativeness_hint}\n\n"
            f"RIGHT role: {right_shot.shot_role}\n"
            f"RIGHT vlm: {right_shot.vlm_summary}\n"
            f"RIGHT scene_type: {right_shot.scene_type}\n"
            f"RIGHT informativeness_hint: {right_shot.informativeness_hint}\n"
        )
        try:
            raw = llm.chat(prompt, max_new_tokens=256)
            import json
            import re

            m = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
            if not m:
                return {
                    "llm_fallback_used": True,
                    "fallback_reason": "llm_output_not_json",
                    "llm_raw": str(raw or ""),
                }
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                obj["llm_fallback_used"] = False
                obj["llm_raw"] = str(raw or "")
                return obj
        except Exception as e:
            return {
                "llm_fallback_used": True,
                "fallback_reason": "llm_pairwise_exception",
                "llm_raw": repr(e),
            }
        return {"llm_fallback_used": True, "fallback_reason": "llm_pairwise_empty", "llm_raw": ""}

    def _build_pairwise_graph(self, shots: List[ShotRecord], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "nodes": [x.shot_id for x in shots],
            "edges": [
                {
                    "u": x["left_shot_id"],
                    "v": x["right_shot_id"],
                    "same_chapter": x["same_chapter"],
                    "confidence": x["confidence"],
                }
                for x in decisions
            ],
        }

    def _apply_local_consistency_rules(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = [dict(x) for x in decisions]
        for i in range(1, len(out) - 1):
            prev_d = out[i - 1]
            cur_d = out[i]
            if prev_d["same_chapter"] and cur_d["same_chapter"]:
                prev_d["confidence"] = max(float(prev_d["confidence"]), 0.65)
                cur_d["confidence"] = max(float(cur_d["confidence"]), 0.65)
        return out

    def _apply_min_chapter_len_constraint(
        self, shots: List[ShotRecord], decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if len(shots) <= 2:
            return decisions
        out = [dict(x) for x in decisions]
        for i in range(len(out)):
            left_same = out[i - 1]["same_chapter"] if i - 1 >= 0 else False
            right_same = out[i]["same_chapter"] if i < len(out) else False
            if (not left_same) and (not right_same):
                middle_shot = shots[i]
                if middle_shot.shot_role not in {"title_card", "transition_bridge"}:
                    if i - 1 >= 0 and float(out[i - 1]["confidence"]) >= 0.45:
                        out[i - 1]["same_chapter"] = True
                    elif i < len(out) and float(out[i]["confidence"]) >= 0.45:
                        out[i]["same_chapter"] = True
        return out

    def _merge_isolated_short_chapters(
        self, shots: List[ShotRecord], decisions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        out = [dict(x) for x in decisions]
        if len(shots) < 3:
            return out
        for i in range(1, len(shots) - 1):
            left_break = not out[i - 1]["same_chapter"]
            right_break = not out[i]["same_chapter"]
            if left_break and right_break and shots[i].duration < 3.0 and shots[i].shot_role != "title_card":
                left_prior = float(out[i - 1].get("prior_score", 0.0))
                right_prior = float(out[i].get("prior_score", 0.0))
                if left_prior >= right_prior and left_prior >= 0.4:
                    out[i - 1]["same_chapter"] = True
                elif right_prior >= 0.4:
                    out[i]["same_chapter"] = True
        return out

    def _group_shots_into_chapters(
        self, shots: List[ShotRecord], decisions: List[Dict[str, Any]]
    ) -> List[ChapterRecord]:
        if not shots:
            return []
        groups: List[List[ShotRecord]] = [[shots[0]]]
        for i, decision in enumerate(decisions):
            if decision["same_chapter"]:
                groups[-1].append(shots[i + 1])
            else:
                groups.append([shots[i + 1]])
        chapters: List[ChapterRecord] = []
        for idx, group in enumerate(groups):
            start_sec = float(group[0].start_sec)
            end_sec = float(group[-1].end_sec)
            role = self._majority_role(group)
            chapters.append(
                ChapterRecord(
                    chapter_id=f"chapter_{idx:03d}",
                    shot_ids=[x.shot_id for x in group],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration=max(0.0, end_sec - start_sec),
                    chapter_role=role,
                    importance_hint=self._chapter_importance(group),
                    coverage_need="required" if role not in {"transition_bridge"} else "low",
                    metadata={"shot_count": len(group)},
                )
            )
        return chapters

    def _majority_role(self, group: List[ShotRecord]) -> str:
        counter: Dict[str, int] = {}
        for shot in group:
            key = shot.shot_role or "generic"
            counter[key] = counter.get(key, 0) + 1
        return max(counter.items(), key=lambda kv: kv[1])[0] if counter else "generic"

    def _chapter_importance(self, group: List[ShotRecord]) -> str:
        score = 0.0
        for shot in group:
            score += {"low": 0.5, "medium": 1.0, "high": 1.5}.get(shot.importance_hint, 1.0)
        score /= max(1, len(group))
        if score >= 1.3:
            return "high"
        if score <= 0.7:
            return "low"
        return "medium"

    def _token_overlap(self, a: str, b: str) -> float:
        sa = {x for x in (a or "").lower().split() if len(x) > 2}
        sb = {x for x in (b or "").lower().split() if len(x) > 2}
        if not sa or not sb:
            return 0.0
        return float(len(sa & sb) / max(1, len(sa | sb)))
