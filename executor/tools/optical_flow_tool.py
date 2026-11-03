from __future__ import annotations

from typing import Any, Dict, List

from executor.tools.base import BaseToolAdapter, ToolContext


class OpticalFlowToolAdapter(BaseToolAdapter):
    tool_name = "optical_flow"

    def run_shot(self, ctx: ToolContext) -> Dict[str, Any]:
        pre_gate_candidates = list(ctx.pre_gate_candidates or [])
        usable_candidates = list(ctx.candidates or [])
        if len(pre_gate_candidates) < 2:
            return {"raw": {"pair_hints": [], "raw_scores": []}, "semantic": {"summary": "", "frame_hints": []}, "diagnostics": {}}

        usable_indices = {
            int(item.get("frame_index", -1))
            for item in usable_candidates
            if int(item.get("frame_index", -1)) >= 0
        }
        candidate_frame_indices = [int(item.get("frame_index", -1)) for item in pre_gate_candidates]
        candidate_paths = [str(item.get("img", "") or "") for item in pre_gate_candidates]
        related_pair_positions: List[int] = []
        seen_positions = set()
        for pos in range(len(pre_gate_candidates) - 1):
            left_frame_index = candidate_frame_indices[pos]
            right_frame_index = candidate_frame_indices[pos + 1]
            if left_frame_index in usable_indices or right_frame_index in usable_indices:
                if pos not in seen_positions:
                    related_pair_positions.append(pos)
                    seen_positions.add(pos)
        if not related_pair_positions:
            return {"raw": {"pair_hints": [], "raw_scores": []}, "semantic": {"summary": "", "frame_hints": []}, "diagnostics": {}}

        def _run_raft(raft: Any) -> List[float]:
            scores: List[float] = []
            for pos in related_pair_positions:
                pair_paths = [candidate_paths[pos], candidate_paths[pos + 1]]
                if not pair_paths[0] or not pair_paths[1]:
                    scores.append(0.0)
                    continue
                pair_scores = list(
                    raft.compute_flow_for_frame_paths(pair_paths, iters=20, resize_short_side=384) or []
                )
                scores.append(float(pair_scores[0]) if pair_scores else 0.0)
            return scores

        raw_scores = ctx.model_manager.run_with_resource("raft", _run_raft)

        pair_hints: List[Dict[str, Any]] = []
        for local_idx, pos in enumerate(related_pair_positions):
            left = pre_gate_candidates[pos]
            right = pre_gate_candidates[pos + 1]
            score = float(raw_scores[local_idx]) if local_idx < len(raw_scores) else 0.0
            pair_hints.append(
                {
                    "from_frame_index": int(left.get("frame_index", -1)),
                    "to_frame_index": int(right.get("frame_index", -1)),
                    "from_sec": float(left.get("sec", 0.0) or 0.0),
                    "to_sec": float(right.get("sec", 0.0) or 0.0),
                    "flow_level": self._flow_level(score),
                    "mean_flow": score,
                }
            )

        hints_by_index: Dict[int, List[Dict[str, Any]]] = {}
        for pair in pair_hints:
            hints_by_index.setdefault(int(pair["from_frame_index"]), []).append(pair)
            hints_by_index.setdefault(int(pair["to_frame_index"]), []).append(pair)

        frame_hints: List[Dict[str, Any]] = []
        for candidate in usable_candidates:
            frame_index = int(candidate.get("frame_index", -1))
            related = list(hints_by_index.get(frame_index, []))
            levels = sorted({str(x.get("flow_level", "") or "") for x in related if str(x.get("flow_level", "") or "")})
            level_summary = ", ".join(levels)
            text = f"Motion level nearby: {level_summary}" if levels else ""
            frame_hints.append(
                {
                    "frame_key": self.frame_key(str(candidate.get("shot_id", "")), frame_index),
                    "img": str(candidate.get("img", "") or ""),
                    "text": text,
                    "score": max((self._flow_score(str(x.get("flow_level", "") or "")) for x in related), default=0.0),
                    "attributes": {
                        "pair_hints": related,
                        "flow_levels": levels,
                        "pair_count": len(related),
                    },
                    "evidence": {"raw_related_pairs": related},
                }
            )

        return {
            "raw": {"pair_hints": pair_hints, "raw_scores": raw_scores},
            "semantic": {
                "summary": " | ".join(x["text"] for x in frame_hints if x.get("text")),
                "frame_hints": frame_hints,
            },
            "diagnostics": {},
        }

    def _flow_level(self, score: float) -> str:
        if score < 0.15:
            return "low"
        if score < 0.45:
            return "medium"
        return "high"

    def _flow_score(self, level: str) -> float:
        return {"low": 0.2, "medium": 0.6, "high": 1.0}.get(str(level), 0.0)
