from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


class RepresentativeSelector:
    """
    Shot-wise representative subset selection.

    Objective (weighted facility location):
        F(S) = sum_i w_i * max_{j in S} sim(i, j)

    where:
      - w_i is derived from frame value / evidence
      - sim(i, j) is embedding cosine similarity (optionally time-decayed)

    This gives:
      - representativeness coverage
      - preference for high-value regions
      - no hard pre-clustering before selection

    Minimal update:
      - fuse image embedding + semantic text embedding
      - keep external API unchanged
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_representative_selection", {}) or {})

        self.selector_backend = str(
            cfg.get("selector_backend", "facility_location_greedy") or "facility_location_greedy"
        )
        self.paper_formula = bool(cfg.get("paper_formula", False))
        self.min_value_weight = float(cfg.get("min_value_weight", 0.25))
        self.value_power = float(cfg.get("value_power", 1.0))
        self.evidence_bonus_scale = float(cfg.get("evidence_bonus_scale", 0.10))
        self.tool_signal_scale = float(cfg.get("tool_signal_scale", 0.10))
        self.time_decay_tau_sec = cfg.get("time_decay_tau_sec", 18.0)
        self.time_decay_tau_sec = None if self.time_decay_tau_sec is None else float(self.time_decay_tau_sec)
        self.max_candidates_per_shot = int(cfg.get("max_candidates_per_shot", 512) or 512)

        # [NEW] image-text fusion config
        self.enable_text_fusion = bool(cfg.get("enable_text_fusion", True))
        self.image_fusion_weight = float(cfg.get("image_fusion_weight", 0.7))
        self.text_fusion_weight = float(cfg.get("text_fusion_weight", 0.3))

        # normalize weights safely
        total = self.image_fusion_weight + self.text_fusion_weight
        if total <= 1e-8:
            self.image_fusion_weight = 1.0
            self.text_fusion_weight = 0.0
        else:
            self.image_fusion_weight /= total
            self.text_fusion_weight /= total

    def select(
        self,
        value_frames: List[Dict[str, Any]],
        quota_plan: Dict[str, Any],
        model_manager: Any,
    ) -> Dict[str, Any]:
        if not value_frames:
            return {
                "selected_frames": [],
                "selection_debug": {"shots": {}},
                "fallback_flags": {
                    "embedding_fallback_used": False,
                    "embedding_backend": "none",
                    "selector_backend": self.selector_backend,
                },
            }

        embeddings, embedding_info = self._compute_embeddings(value_frames, model_manager)
        embedding_fallback_used = bool(embedding_info.get("fallback_used", False))
        embedding_backend = str(embedding_info.get("backend", "unknown") or "unknown")

        by_shot: Dict[str, List[int]] = {}
        for idx, frame in enumerate(value_frames):
            by_shot.setdefault(str(frame.get("shot_id", "")), []).append(idx)

        shot_plans = (quota_plan.get("shot_plans", {}) or {}) if isinstance(quota_plan, dict) else {}
        global_plan = (quota_plan.get("global_plan", {}) or {}) if isinstance(quota_plan, dict) else {}

        global_budget = self._safe_int(
            global_plan.get("global_budget", global_plan.get("total_budget", len(value_frames))),
            len(value_frames),
        )

        selected_frames: List[Dict[str, Any]] = []
        shot_debug: Dict[str, Any] = {}

        for shot_id, member_indices in by_shot.items():
            shot_plan = (shot_plans.get(shot_id, {}) or {})
            shot_quota = self._safe_int(
                shot_plan.get("target_quota", shot_plan.get("final_quota", 0)),
                0,
            )
            if shot_quota <= 0:
                continue

        global_budget = int(
            global_plan.get(
                "total_budget",
                global_plan.get("global_budget", len(value_frames))
            )
        )

        selected_frames: List[Dict[str, Any]] = []
        shot_debug: Dict[str, Any] = {}

        for shot_id, member_indices in by_shot.items():
            shot_cfg = (shot_plans.get(shot_id, {}) or {})
            shot_quota = int(
                shot_cfg.get(
                    "final_quota",
                    shot_cfg.get("target_quota", 0)
                )
            )
            if shot_quota <= 0:
                shot_debug[shot_id] = {
                    "quota": int(shot_quota),
                    "num_candidates": int(len(member_indices)),
                    "selected_local_indices": [],
                    "selected_frame_indices": [],
                    "selected_gains": [],
                }
                continue

            member_indices = sorted(
                member_indices,
                key=lambda i: (
                    self._safe_float(value_frames[i].get("sec", 0.0)),
                    self._safe_int(value_frames[i].get("frame_index", i)),
                    i,
                ),
            )

            if len(member_indices) > self.max_candidates_per_shot:
                member_indices = self._preprune_member_indices(
                    value_frames, member_indices, self.max_candidates_per_shot
                )

            local_frames = [value_frames[i] for i in member_indices]
            local_embeddings = np.stack([embeddings[i] for i in member_indices], axis=0)
            sim = self._compute_similarity_matrix(local_frames, local_embeddings)

            picked_local, gains, weights = self._select_one_shot(
                local_frames,
                local_embeddings,
                shot_quota,
                precomputed_sim=sim,
            )

            selected_rank_by_local = {int(local_idx): int(rank) for rank, local_idx in enumerate(picked_local)}
            selected_gain_by_local = {
                int(local_idx): float(gains[rank])
                for rank, local_idx in enumerate(picked_local)
            }

            candidate_rows: List[Dict[str, Any]] = []
            for local_idx, frame in enumerate(local_frames):
                vp = frame.get("value_profile", {}) or {}
                value_score = self._safe_float(frame.get("value_score", 0.0))
                evidence_bonus = self._safe_float(vp.get("evidence_bonus", 0.0))
                tool_signal = self._safe_float(vp.get("tool_signal", 0.0))
                selector_base_score = (
                    value_score
                    + self.evidence_bonus_scale * evidence_bonus
                    + self.tool_signal_scale * tool_signal
                )
                initial_selector_gain = float(np.sum(weights * sim[:, local_idx])) if sim.size else 0.0

                candidate_rows.append(
                    {
                        "frame_index": int(frame.get("frame_index", -1)),
                        "sec": self._safe_float(frame.get("sec", 0.0)),
                        "candidate_type": str(frame.get("candidate_type", "") or ""),
                        "value_score": float(value_score),
                        "evidence_bonus": float(evidence_bonus),
                        "tool_signal": float(tool_signal),
                        "selector_base_score": float(selector_base_score),
                        "selector_weight": float(weights[local_idx]),
                        "initial_selector_gain": float(initial_selector_gain),
                        "selected": bool(local_idx in selected_rank_by_local),
                        "selected_rank": int(selected_rank_by_local.get(local_idx, -1)),
                        "selected_gain": (
                            float(selected_gain_by_local[local_idx])
                            if local_idx in selected_gain_by_local
                            else None
                        ),
                    }
                )

            shot_selected = []
            for rank, local_idx in enumerate(picked_local):
                global_idx = member_indices[local_idx]
                row = dict(value_frames[global_idx])
                row["_selector_backend"] = self.selector_backend
                row["_selector_rank_in_shot"] = int(rank)
                row["_selector_gain"] = float(gains[rank])
                row["_selector_weight"] = float(weights[local_idx])
                shot_selected.append(row)

            selected_frames.extend(shot_selected)
            shot_debug[shot_id] = {
                "quota": int(shot_quota),
                "original_candidate_count": int(len(by_shot.get(shot_id, []))),
                "num_candidates": int(len(local_frames)),
                "prepruned": bool(len(by_shot.get(shot_id, [])) > len(local_frames)),
                "embedding_backend": embedding_backend,
                "selected_local_indices": [int(x) for x in picked_local],
                "selected_frame_indices": [int(local_frames[x].get("frame_index", -1)) for x in picked_local],
                "selected_gains": [float(x) for x in gains],
                "candidate_rows": candidate_rows,
            }

        # Apply pruning only for non-negative budgets; -1 means unlimited.
        if global_budget >= 0 and len(selected_frames) > global_budget:
            selected_frames = sorted(
                selected_frames,
                key=lambda x: (
                    float(x.get("_selector_gain", 0.0)),
                    float(x.get("value_score", 0.0)),
                ),
                reverse=True,
            )[:global_budget]

        return {
            "selected_frames": selected_frames,
            "selection_debug": {"shots": shot_debug},
            "fallback_flags": {
                "embedding_fallback_used": embedding_fallback_used,
                "embedding_backend": embedding_backend,
                "selector_backend": self.selector_backend,
            },
        }

    # =========================================================
    # selection core
    # =========================================================
    def _select_one_shot(
        self,
        frames: List[Dict[str, Any]],
        embeddings: np.ndarray,
        quota: int,
        precomputed_sim: np.ndarray | None = None,
    ) -> Tuple[List[int], List[float], np.ndarray]:
        n = len(frames)
        if n == 0 or quota <= 0:
            return [], [], np.zeros((0,), dtype=np.float32)
        if quota >= n:
            return list(range(n)), [1.0] * n, self._build_frame_weights(frames)

        sim = precomputed_sim if precomputed_sim is not None else self._compute_similarity_matrix(frames, embeddings)
        weights = self._build_frame_weights(frames)

        covered = np.zeros((n,), dtype=np.float32)
        selected: List[int] = []
        gains: List[float] = []
        chosen = np.zeros((n,), dtype=bool)

        for _ in range(min(quota, n)):
            best_j = -1
            best_gain = -1.0

            for j in range(n):
                if chosen[j]:
                    continue
                gain = float(np.sum(weights * np.maximum(0.0, sim[:, j] - covered)))
                gain += 1e-6 * float(frames[j].get("value_score", 0.0) or 0.0)  # stable tie-break
                if gain > best_gain:
                    best_gain = gain
                    best_j = j

            if best_j < 0:
                break

            chosen[best_j] = True
            covered = np.maximum(covered, sim[:, best_j])
            selected.append(best_j)
            gains.append(best_gain)

        return selected, gains, weights

    def _build_frame_weights(self, frames: List[Dict[str, Any]]) -> np.ndarray:
        if self.paper_formula:
            return np.asarray([max(0.0, self._safe_float(f.get("value_score", 0.0))) for f in frames], dtype=np.float32)
        raw = []
        for frame in frames:
            vp = frame.get("value_profile", {}) or {}
            value_score = self._safe_float(frame.get("value_score", 0.0))
            evidence_bonus = self._safe_float(vp.get("evidence_bonus", 0.0))
            tool_signal = self._safe_float(vp.get("tool_signal", 0.0))

            score = value_score
            score += self.evidence_bonus_scale * evidence_bonus
            score += self.tool_signal_scale * tool_signal
            raw.append(score)

        raw = np.asarray(raw, dtype=np.float32)
        if raw.size == 0:
            return raw

        min_v = float(raw.min())
        max_v = float(raw.max())
        if max_v - min_v < 1e-8:
            norm = np.ones_like(raw, dtype=np.float32)
        else:
            norm = (raw - min_v) / (max_v - min_v)

        norm = np.power(norm, self.value_power).astype(np.float32)
        weights = self.min_value_weight + (1.0 - self.min_value_weight) * norm
        return weights.astype(np.float32)

    def _compute_similarity_matrix(self, frames: List[Dict[str, Any]], embeddings: np.ndarray) -> np.ndarray:
        X = embeddings.astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim = X @ X.T
        sim = np.clip(sim, -1.0, 1.0)

        # use non-negative similarity only
        sim = np.maximum(sim, 0.0)

        if self.time_decay_tau_sec is not None and self.time_decay_tau_sec > 0:
            secs = np.asarray([self._safe_float(x.get("sec", 0.0)) for x in frames], dtype=np.float32)
            delta = np.abs(secs[:, None] - secs[None, :])
            decay = np.exp(-delta / float(self.time_decay_tau_sec)).astype(np.float32)
            sim = sim * decay

        np.fill_diagonal(sim, 1.0)
        return sim.astype(np.float32)

    # =========================================================
    # embedding / utility
    # =========================================================
    def _compute_embeddings(
        self,
        value_frames: List[Dict[str, Any]],
        model_manager: Any,
    ) -> Tuple[List[np.ndarray], Dict[str, Any]]:
        image_paths = [str(x.get("img", "") or "") for x in value_frames]
        semantic_texts = [self._extract_semantic_text(x) for x in value_frames]

        img_arr = None
        txt_arr = None
        img_ok = False
        txt_ok = False

        try:
            with model_manager.use_resource("embedder") as embedder:
                img_arr = embedder.compute_embeddings(image_paths)
                img_arr = np.asarray(img_arr, dtype=np.float32)
                if len(img_arr) == len(value_frames) and img_arr.ndim == 2 and img_arr.shape[1] > 0:
                    img_ok = True

                if self.enable_text_fusion:
                    txt_arr = embedder.compute_text_embeddings(semantic_texts)
                    txt_arr = np.asarray(txt_arr, dtype=np.float32)
                    if len(txt_arr) == len(value_frames) and txt_arr.ndim == 2 and txt_arr.shape[1] > 0:
                        txt_ok = True
        except Exception:
            img_arr = None
            img_ok = False
            txt_arr = None
            txt_ok = False

        if img_ok and txt_ok and self.text_fusion_weight > 0.0:
            fused = self._fuse_embeddings(img_arr, txt_arr)
            return [np.asarray(x, dtype=np.float32) for x in fused], {
                "fallback_used": False,
                "backend": "image_text_fused",
            }

        if img_ok:
            return [np.asarray(x, dtype=np.float32) for x in img_arr], {
                "fallback_used": False,
                "backend": "image_only",
            }

        if txt_ok:
            return [np.asarray(x, dtype=np.float32) for x in txt_arr], {
                "fallback_used": False,
                "backend": "text_only",
            }

        return [self._fallback_embedding(x) for x in value_frames], {
            "fallback_used": True,
            "backend": "value_profile_proxy",
        }

    def _fuse_embeddings(self, img_arr: np.ndarray, txt_arr: np.ndarray) -> np.ndarray:
        img_norm = self._l2_normalize_rows(img_arr)
        txt_norm = self._l2_normalize_rows(txt_arr)

        wi = np.sqrt(max(self.image_fusion_weight, 0.0)).astype(np.float32) if isinstance(
            np.sqrt(max(self.image_fusion_weight, 0.0)), np.ndarray
        ) else float(np.sqrt(max(self.image_fusion_weight, 0.0)))
        wt = np.sqrt(max(self.text_fusion_weight, 0.0)).astype(np.float32) if isinstance(
            np.sqrt(max(self.text_fusion_weight, 0.0)), np.ndarray
        ) else float(np.sqrt(max(self.text_fusion_weight, 0.0)))

        fused = np.concatenate([wi * img_norm, wt * txt_norm], axis=1).astype(np.float32)
        return fused

    def _extract_semantic_text(self, frame: Dict[str, Any]) -> str:
        sp = dict(frame.get("semantic_profile", {}) or {})
        vp = dict(frame.get("value_profile", {}) or {})
        tool_signals = dict(frame.get("tool_signals", {}) or {})

        text = str(sp.get("semantic_text", "") or "").strip()
        if text:
            return text

        text = str(vp.get("semantic_text", "") or "").strip()
        if text:
            return text

        text = str(tool_signals.get("semantic_summary", "") or "").strip()
        if text:
            return text

        return ""

    def _fallback_embedding(self, frame: Dict[str, Any]) -> np.ndarray:
        vp = frame.get("value_profile", {}) or {}
        sp = frame.get("semantic_profile", {}) or {}
        tool_signals = frame.get("tool_signals", {}) or {}
        signal_scores = (tool_signals.get("signal_scores", {}) or {})

        candidate_type = str(frame.get("candidate_type", "") or "")
        candidate_type_flags = {
            "layout_peak": 0.0,
            "change_peak": 0.0,
            "action_peak": 0.0,
            "coverage_peak": 0.0,
        }
        if candidate_type in candidate_type_flags:
            candidate_type_flags[candidate_type] = 1.0

        return np.asarray(
            [
                float(frame.get("value_score", 0.0) or 0.0),
                1.0 if str(sp.get("semantic_text", "") or "").strip() else 0.0,
                float(len([x for x in dict(sp.get("tool_descs", {}) or {}).values() if str(x or "").strip()]) > 0),
                float(signal_scores.get("ocr", 0.0) or 0.0),
                float(signal_scores.get("asr", 0.0) or 0.0),
                float(signal_scores.get("yolo", 0.0) or 0.0),
                float(signal_scores.get("optical_flow", 0.0) or 0.0),
                float(candidate_type_flags["layout_peak"]),
                float(candidate_type_flags["change_peak"]),
                float(candidate_type_flags["action_peak"]),
                float(candidate_type_flags["coverage_peak"]),
            ],
            dtype=np.float32,
        )

    def _l2_normalize_rows(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm = np.maximum(norm, eps)
        return x / norm

    def _preprune_member_indices(
        self,
        value_frames: List[Dict[str, Any]],
        member_indices: List[int],
        keep_n: int,
    ) -> List[int]:
        ranked = sorted(
            member_indices,
            key=lambda i: float(value_frames[i].get("value_score", 0.0) or 0.0),
            reverse=True,
        )
        return ranked[:keep_n]

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
