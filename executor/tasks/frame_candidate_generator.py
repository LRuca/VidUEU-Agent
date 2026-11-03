from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from orchestrator.structures import ShotRecord


class FrameCandidateGenerator:
    """
    Generate per-shot tool pools according to ToolPlan.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def generate(
        self,
        shots: List[ShotRecord],
        quota_plan: Dict[str, Any],
        tool_plan: Dict[str, Any],
        frame_items: List[Dict[str, Any]],
        model_manager: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        _ = quota_plan
        frame_groups = self._group_frames_by_shot(frame_items)
        shot_plans = (tool_plan.get("shot_plans", {}) or {}) if isinstance(tool_plan, dict) else {}
        out: Dict[str, List[Dict[str, Any]]] = {}

        for shot in shots:
            shot_frames = frame_groups.get(shot.shot_id, [])
            plan = shot_plans.get(shot.shot_id, {})
            max_candidates = int(plan.get("max_candidates", 3) or 3)
            sampling_density = str(plan.get("sampling_density", "medium") or "medium")
            candidate_tools = self._candidate_tools_for_shot(shot, sampling_density)

            candidates: List[Dict[str, Any]] = []
            for tool_item in candidate_tools:
                candidate_type = str(tool_item.get("tool_name", "") or "")
                cand = self._generate_candidate_by_type(shot, shot_frames, candidate_type, model_manager)
                if cand:
                    cand["tool_name"] = candidate_type
                    cand["tool_priority_level"] = str(tool_item.get("priority_level", "medium"))
                    cand["tool_priority_weight"] = int(tool_item.get("priority_weight", 2) or 2)
                    candidates.append(cand)

            candidates.extend(
                self._generate_uniform_support(
                    shot,
                    shot_frames,
                    max_candidates=max_candidates,
                    sampling_density=sampling_density,
                )
            )
            out[shot.shot_id] = self._dedup_candidates(candidates)[:max_candidates]
        return out

    def _candidate_tools_for_shot(self, shot: ShotRecord, sampling_density: str) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = [{"tool_name": "rep_peak", "priority_level": "high", "priority_weight": 3}]
        if shot.shot_role in {"screen_demo", "slide", "title_card"}:
            tools.append({"tool_name": "layout_peak", "priority_level": "high", "priority_weight": 3})
        if shot.shot_role in {"action", "broll"} or shot.scene_type in {"action", "sports", "dynamic_scene"}:
            tools.append({"tool_name": "change_peak", "priority_level": "high", "priority_weight": 3})
        if sampling_density != "low":
            tools.append({"tool_name": "coverage_peak", "priority_level": "medium", "priority_weight": 2})
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for tool in tools:
            name = str(tool.get("tool_name", ""))
            if name in seen:
                continue
            seen.add(name)
            deduped.append(tool)
        return deduped

    def _group_frames_by_shot(self, frame_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for item in frame_items:
            shot_id = str(item.get("shot_id", ""))
            out.setdefault(shot_id, []).append(item)
        return out

    def _generate_candidate_by_type(
        self,
        shot: ShotRecord,
        shot_frames: List[Dict[str, Any]],
        candidate_type: str,
        model_manager: Any,
    ) -> Optional[Dict[str, Any]]:
        if candidate_type == "rep_peak":
            return self._generate_rep_peak(shot, shot_frames)
        if candidate_type == "change_peak":
            return self._generate_change_peak(shot, shot_frames, model_manager)
        if candidate_type == "layout_peak":
            return self._generate_layout_peak(shot, shot_frames)
        if candidate_type == "action_peak":
            return self._generate_change_peak(shot, shot_frames, model_manager, label="action_peak")
        if candidate_type == "coverage_peak":
            return self._generate_coverage_peak(shot, shot_frames)
        return None

    def _generate_rep_peak(self, shot: ShotRecord, shot_frames: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not shot_frames:
            return None
        scores = [self._quality_score(x.get("img", "")) for x in shot_frames]
        idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        item = shot_frames[idx]
        return self._make_candidate(shot, item, "rep_peak", {"representativeness": scores[idx], "utility": scores[idx]})

    def _generate_change_peak(
        self,
        shot: ShotRecord,
        shot_frames: List[Dict[str, Any]],
        model_manager: Any,
        label: str = "change_peak",
    ) -> Optional[Dict[str, Any]]:
        if len(shot_frames) < 2:
            return None
        vals = self._change_scores(shot_frames, model_manager)
        idx = int(np.argmax(np.asarray(vals, dtype=np.float32)))
        item = shot_frames[idx]
        return self._make_candidate(shot, item, label, {"change_gain": vals[idx], "utility": vals[idx]})

    def _generate_layout_peak(self, shot: ShotRecord, shot_frames: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not shot_frames:
            return None
        scores = []
        for item in shot_frames:
            img = cv2.imread(str(item.get("img", "")), cv2.IMREAD_GRAYSCALE)
            if img is None:
                scores.append(0.0)
                continue
            small = cv2.resize(img, (192, 108), interpolation=cv2.INTER_AREA)
            edges = cv2.Canny(small, 60, 180)
            horiz = float(np.mean(np.sum(edges > 0, axis=1) > 18))
            vert = float(np.mean(np.sum(edges > 0, axis=0) > 12))
            score = 0.5 * horiz + 0.5 * vert
            scores.append(score)
        idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
        item = shot_frames[idx]
        return self._make_candidate(shot, item, "layout_peak", {"layout_signal": scores[idx], "utility": scores[idx]})

    def _generate_coverage_peak(self, shot: ShotRecord, shot_frames: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not shot_frames:
            return None
        idx = len(shot_frames) - 1 if len(shot_frames) > 1 else 0
        item = shot_frames[idx]
        utility = self._quality_score(item.get("img", ""))
        return self._make_candidate(shot, item, "coverage_peak", {"coverage_gain": 0.5, "utility": utility})

    def _generate_uniform_support(
        self,
        shot: ShotRecord,
        shot_frames: List[Dict[str, Any]],
        max_candidates: int,
        sampling_density: str,
    ) -> List[Dict[str, Any]]:
        if not shot_frames or max_candidates <= 1:
            return []
        picks: List[Dict[str, Any]] = []
        n = len(shot_frames)
        target = {"low": 1, "medium": 2, "high": 3}.get(sampling_density, 2)
        target = min(max_candidates, target)
        for k in range(target):
            idx = int(round((k + 1) * (n - 1) / float(target + 1)))
            item = shot_frames[idx]
            utility = self._quality_score(item.get("img", ""))
            cand = self._make_candidate(shot, item, "uniform_support", {"utility": utility, "coverage_gain": 0.2})
            cand["tool_name"] = "uniform_support"
            cand["tool_priority_level"] = "low"
            cand["tool_priority_weight"] = 1
            picks.append(cand)
        return picks

    def _change_scores(self, shot_frames: List[Dict[str, Any]], model_manager: Any) -> List[float]:
        paths = [str(x.get("img", "")) for x in shot_frames if x.get("img")]
        if len(paths) == len(shot_frames):
            try:
                embedder = model_manager.get_resource("embedder")
                out = embedder.compute_semantic_change_for_frame_paths(paths, return_similarity_matrix=False)
                scores = out.get("change_scores") or []
                if len(scores) == len(shot_frames):
                    return [float(x) for x in scores]
            except Exception:
                pass

        vals = []
        prev_small = None
        for item in shot_frames:
            img = cv2.imread(str(item.get("img", "")), cv2.IMREAD_GRAYSCALE)
            if img is None:
                vals.append(0.0)
                continue
            small = cv2.resize(img, (96, 54), interpolation=cv2.INTER_AREA)
            if prev_small is None:
                vals.append(0.0)
            else:
                vals.append(float(np.mean(np.abs(small.astype(np.float32) - prev_small.astype(np.float32))) / 255.0))
            prev_small = small
        return vals

    def _dedup_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for cand in sorted(candidates, key=lambda x: float((x.get("base_scores", {}) or {}).get("utility", 0.0)), reverse=True):
            key = int(cand.get("frame_index", -1))
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        return out

    def _make_candidate(
        self,
        shot: ShotRecord,
        item: Dict[str, Any],
        candidate_type: str,
        base_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "shot_id": shot.shot_id,
            "chapter_id": str(item.get("chapter_id", shot.metadata.get("chapter_id", ""))),
            "frame_index": int(item.get("index", -1)),
            "img": str(item.get("img", "")),
            "sec": float(item.get("sec", 0.0)),
            "candidate_type": candidate_type,
            "base_scores": dict(base_scores),
            "sampling_origin": candidate_type,
        }

    def _quality_score(self, img_path: str) -> float:
        if not img_path:
            return 0.0
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        small = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(small, cv2.CV_64F).var())
        return max(0.0, min(1.0, lap / 300.0))
