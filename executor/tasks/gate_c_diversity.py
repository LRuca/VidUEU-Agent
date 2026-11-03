from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import ruptures as rpt
except Exception:
    rpt = None


class GateCDiversity:
    """
    GateC v2:
    1) segment each shot into semantic steps with ruptures
    2) within each step, keep the highest-value frame
    3) optionally keep exception frames with strong extra evidence
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T2_gate_c", {}) or {})

        # legacy settings
        self.shot_dup_threshold = float(cfg.get("shot_dup_threshold", 0.94) or 0.94)
        self.chapter_dup_threshold = float(cfg.get("chapter_dup_threshold", 0.96) or 0.96)

        # new step-local mode
        self.mode = str(cfg.get("mode", "step_local") or "step_local").strip().lower()
        self.step_segmentation_enabled = bool(cfg.get("step_segmentation_enabled", True))
        self.step_model = str(cfg.get("step_model", "rbf") or "rbf").strip().lower()
        self.step_min_size = int(cfg.get("step_min_size", 6) or 6)
        self.step_jump = int(cfg.get("step_jump", 1) or 1)
        self.step_penalty = cfg.get("step_penalty", "auto")
        self.step_penalty_scale = float(cfg.get("step_penalty_scale", 1.6) or 1.6)
        self.step_embed_pca_dim = int(cfg.get("step_embed_pca_dim", 12) or 12)
        self.keep_exception_in_step = bool(cfg.get("keep_exception_in_step", False))

    def run(self, value_frames: List[Dict[str, Any]], model_manager: Any) -> Dict[str, Any]:
        if not value_frames:
            return {
                "clusters": [],
                "kept_frames": [],
                "dropped_frames": [],
                "exception_kept_frames": [],
                "steps": [],
                "fallback_flags": {
                    "embedding_fallback_used": False,
                    "embedding_backend": "none",
                    "step_segmentation_fallback_used": False,
                    "step_backend": "none",
                },
            }

        embeddings, embedding_fallback_used = self._compute_embeddings(value_frames, model_manager)

        if self.mode != "step_local" or not self.step_segmentation_enabled or rpt is None:
            out = self._run_legacy_cluster(value_frames, embeddings, embedding_fallback_used)
            out["fallback_flags"]["step_segmentation_fallback_used"] = True
            out["fallback_flags"]["step_backend"] = "legacy_cluster"
            return out

        clusters = []
        kept_frames = []
        dropped_frames = []
        exception_kept_frames = []
        steps = []

        shot_groups: Dict[str, List[int]] = {}
        for idx, frame in enumerate(value_frames):
            shot_groups.setdefault(str(frame.get("shot_id", "")), []).append(idx)

        cluster_idx = 0
        step_segmentation_fallback_used = False

        for shot_id, members_in_shot in shot_groups.items():
            members_in_shot = sorted(
                members_in_shot,
                key=lambda x: (
                    self._safe_float(value_frames[x].get("sec", 0.0)),
                    self._safe_int(value_frames[x].get("frame_index", x)),
                    x,
                ),
            )

            try:
                signal = self._build_step_signal(value_frames, embeddings, members_in_shot)
                bkps = self._detect_step_breaks(signal)
                ranges = self._bkps_to_ranges(len(members_in_shot), bkps)
            except Exception:
                step_segmentation_fallback_used = True
                ranges = [(0, len(members_in_shot))]

            for step_idx, (lo, hi) in enumerate(ranges):
                local_members = members_in_shot[lo:hi]
                if not local_members:
                    continue

                scored = sorted(
                    local_members,
                    key=lambda x: float(value_frames[x].get("value_score", 0.0)),
                    reverse=True,
                )
                keep_idx = scored[0]
                keep_frame = value_frames[keep_idx]
                kept_frames.append(keep_frame)

                step_record = {
                    "step_id": f"{shot_id}_step_{step_idx:03d}",
                    "shot_id": shot_id,
                    "start_sec": float(value_frames[local_members[0]].get("sec", 0.0) or 0.0),
                    "end_sec": float(value_frames[local_members[-1]].get("sec", 0.0) or 0.0),
                    "members": [int(value_frames[x].get("frame_index", -1)) for x in local_members],
                    "kept_frame": int(keep_frame.get("frame_index", -1)),
                    "kept_value_score": float(keep_frame.get("value_score", 0.0) or 0.0),
                    "exception_reason": "",
                }
                steps.append(step_record)

                clusters.append(
                    {
                        "cluster_id": f"cluster_{cluster_idx:03d}",
                        "shot_id": shot_id,
                        "members": [value_frames[x]["frame_index"] for x in local_members],
                        "kept_frame": keep_frame["frame_index"],
                        "drop_reasons": [],
                        "exception_reason": "",
                        "step_id": step_record["step_id"],
                    }
                )
                cluster_idx += 1

                for idx in scored[1:]:
                    cand = value_frames[idx]
                    if self.keep_exception_in_step and self._should_keep_as_exception(cand, keep_frame):
                        exception_kept_frames.append(cand)
                        kept_frames.append(cand)
                        step_record["exception_reason"] = "multi_evidence_gain"
                        clusters[-1]["exception_reason"] = "multi_evidence_gain"
                    else:
                        dropped_frames.append({**cand, "drop_reason": "same_step_lower_value"})
                        clusters[-1]["drop_reasons"].append(
                            {"frame_index": cand["frame_index"], "reason": "same_step_lower_value"}
                        )

        kept_unique = {}
        for frame in kept_frames:
            kept_unique[int(frame.get("frame_index", -1))] = frame

        return {
            "clusters": clusters,
            "kept_frames": list(kept_unique.values()),
            "dropped_frames": dropped_frames,
            "exception_kept_frames": exception_kept_frames,
            "steps": steps,
            "fallback_flags": {
                "embedding_fallback_used": bool(embedding_fallback_used),
                "embedding_backend": "value_profile_proxy" if embedding_fallback_used else "embedder",
                "step_segmentation_fallback_used": bool(step_segmentation_fallback_used),
                "step_backend": "ruptures_pelt_rbf" if not step_segmentation_fallback_used else "single_step_fallback",
            },
        }

    # =========================
    # step segmentation
    # =========================
    def _build_step_signal(
        self,
        value_frames: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        members_in_shot: List[int],
    ) -> np.ndarray:
        E = np.stack([np.asarray(embeddings[i], dtype=np.float32) for i in members_in_shot], axis=0)
        E = self._l2_normalize(E)

        E_red = self._reduce_dim(E, self.step_embed_pca_dim)

        # scalar signals from value_profile
        scalar_rows = []
        prev_e = None
        prev_vs = None
        prev_tool = None

        for idx in members_in_shot:
            frame = value_frames[idx]
            vp = frame.get("value_profile", {}) or {}

            tool_signal = float(vp.get("tool_signal", 0.0))
            evidence_bonus = float(vp.get("evidence_bonus", 0.0))
            change_gain = float(vp.get("change_gain", 0.0))
            coverage_gain = float(vp.get("coverage_gain", 0.0))
            informativeness = float(vp.get("informativeness", 0.0))
            value_score = float(frame.get("value_score", 0.0) or 0.0)

            cur_e = np.asarray(embeddings[idx], dtype=np.float32)
            cur_e = cur_e / (np.linalg.norm(cur_e) + 1e-8)

            emb_delta = 0.0 if prev_e is None else (1.0 - float(np.dot(prev_e, cur_e)))
            value_delta = 0.0 if prev_vs is None else abs(value_score - prev_vs)
            tool_delta = 0.0 if prev_tool is None else abs(tool_signal - prev_tool)

            prev_e = cur_e
            prev_vs = value_score
            prev_tool = tool_signal

            scalar_rows.append(
                [
                    emb_delta,
                    tool_signal,
                    evidence_bonus,
                    change_gain,
                    coverage_gain,
                    informativeness,
                    value_score,
                    value_delta,
                    tool_delta,
                ]
            )

        S = np.asarray(scalar_rows, dtype=np.float32)
        S = self._zscore(S)

        # weights: embedding主体，delta与证据信号辅助
        X = np.concatenate(
            [
                1.0 * E_red,
                0.8 * S,
            ],
            axis=1,
        ).astype(np.float32)

        return X

    def _detect_step_breaks(self, signal: np.ndarray) -> List[int]:
        n = int(signal.shape[0])
        if n <= max(2, self.step_min_size * 2):
            return [n]

        algo = rpt.Pelt(model=self.step_model, min_size=self.step_min_size, jump=self.step_jump).fit(signal)

        if isinstance(self.step_penalty, str) and self.step_penalty.lower() == "auto":
            pen = self.step_penalty_scale * np.log(n + 1.0) * max(1.0, signal.shape[1] ** 0.5)
        else:
            pen = float(self.step_penalty)

        bkps = algo.predict(pen=pen)
        bkps = sorted(set(int(x) for x in bkps if 0 < int(x) <= n))
        if not bkps or bkps[-1] != n:
            bkps.append(n)

        # enforce min size again, conservative merge
        cleaned = []
        prev = 0
        for b in bkps:
            if b - prev < self.step_min_size and cleaned:
                cleaned[-1] = b
            else:
                cleaned.append(b)
            prev = cleaned[-1]

        if cleaned[-1] != n:
            cleaned[-1] = n

        return cleaned

    def _bkps_to_ranges(self, n: int, bkps: List[int]) -> List[Tuple[int, int]]:
        out = []
        prev = 0
        for b in bkps:
            b = int(min(max(b, 0), n))
            if b > prev:
                out.append((prev, b))
            prev = b
        if not out:
            out = [(0, n)]
        return out

    # =========================
    # legacy fallback
    # =========================
    def _run_legacy_cluster(
        self,
        value_frames: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        embedding_fallback_used: bool,
    ) -> Dict[str, Any]:
        clusters = []
        kept_frames = []
        dropped_frames = []
        exception_kept_frames = []
        shot_groups: Dict[str, List[int]] = {}

        for idx, frame in enumerate(value_frames):
            shot_groups.setdefault(str(frame.get("shot_id", "")), []).append(idx)

        cluster_idx = 0
        for shot_id, members_in_shot in shot_groups.items():
            parent = {idx: idx for idx in members_in_shot}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

            for i_pos, i in enumerate(members_in_shot):
                for j in members_in_shot[i_pos + 1 :]:
                    sim = self._cosine(embeddings[i], embeddings[j])
                    if sim >= self.shot_dup_threshold:
                        union(i, j)

            grouped: Dict[int, List[int]] = {}
            for idx in members_in_shot:
                grouped.setdefault(find(idx), []).append(idx)

            for members in grouped.values():
                scored = sorted(members, key=lambda x: float(value_frames[x].get("value_score", 0.0)), reverse=True)
                keep_idx = scored[0]
                keep_frame = value_frames[keep_idx]
                cluster_record = {
                    "cluster_id": f"cluster_{cluster_idx:03d}",
                    "shot_id": shot_id,
                    "members": [value_frames[x]["frame_index"] for x in members],
                    "kept_frame": keep_frame["frame_index"],
                    "drop_reasons": [],
                    "exception_reason": "",
                }
                cluster_idx += 1
                kept_frames.append(keep_frame)

                for idx in scored[1:]:
                    cand = value_frames[idx]
                    if self._should_keep_as_exception(cand, keep_frame):
                        exception_kept_frames.append(cand)
                        kept_frames.append(cand)
                        cluster_record["exception_reason"] = "multi_evidence_gain"
                    else:
                        dropped_frames.append({**cand, "drop_reason": "near_duplicate_lower_value"})
                        cluster_record["drop_reasons"].append(
                            {"frame_index": cand["frame_index"], "reason": "near_duplicate_lower_value"}
                        )
                clusters.append(cluster_record)

        kept_unique = {}
        for frame in kept_frames:
            kept_unique[int(frame.get("frame_index", -1))] = frame

        return {
            "clusters": clusters,
            "kept_frames": list(kept_unique.values()),
            "dropped_frames": dropped_frames,
            "exception_kept_frames": exception_kept_frames,
            "steps": [],
            "fallback_flags": {
                "embedding_fallback_used": bool(embedding_fallback_used),
                "embedding_backend": "value_profile_proxy" if embedding_fallback_used else "embedder",
            },
        }

    # =========================
    # helpers
    # =========================
    def _compute_embeddings(self, value_frames: List[Dict[str, Any]], model_manager: Any) -> tuple[List[np.ndarray], bool]:
        paths = [str(x.get("img", "") or "") for x in value_frames]
        try:
            embedder = model_manager.get_resource("embedder")
            arr = embedder.compute_embeddings(paths)
            if len(arr) == len(value_frames):
                return [np.asarray(x, dtype=np.float32) for x in arr], False
        except Exception:
            pass
        return [self._fallback_embedding(x) for x in value_frames], True

    def _fallback_embedding(self, frame: Dict[str, Any]) -> np.ndarray:
        vp = frame.get("value_profile", {}) or {}
        return np.asarray(
            [
                float(vp.get("representativeness", 0.0)),
                float(vp.get("informativeness", 0.0)),
                float(vp.get("change_gain", 0.0)),
                float(vp.get("coverage_gain", 0.0)),
                float(vp.get("evidence_bonus", 0.0)),
            ],
            dtype=np.float32,
        )

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        if denom <= 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

    def _reduce_dim(self, x: np.ndarray, k: int) -> np.ndarray:
        n, d = x.shape
        k = max(1, min(int(k), n, d))
        if k >= d:
            return x.astype(np.float32)
        x0 = x - x.mean(axis=0, keepdims=True)
        try:
            u, s, _ = np.linalg.svd(x0, full_matrices=False)
            return (u[:, :k] * s[:k]).astype(np.float32)
        except Exception:
            return x[:, :k].astype(np.float32)

    def _zscore(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-6, 1.0, sd)
        return (x - mu) / sd

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

    def _should_keep_as_exception(self, candidate: Dict[str, Any], kept: Dict[str, Any]) -> bool:
        c = candidate.get("value_profile", {}) or {}
        k = kept.get("value_profile", {}) or {}
        if float(c.get("coverage_gain", 0.0)) >= float(k.get("coverage_gain", 0.0)) + 0.12:
            return True
        if float(c.get("change_gain", 0.0)) >= float(k.get("change_gain", 0.0)) + 0.18:
            return True
        if candidate.get("candidate_type") != kept.get("candidate_type") and float(c.get("evidence_bonus", 0.0)) >= 0.10:
            return True
        return False
