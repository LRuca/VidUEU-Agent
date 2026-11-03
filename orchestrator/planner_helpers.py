from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from orchestrator.structures import ChapterRecord, RepresentativeFrame, ShotRecord, VideoProfile


def parse_llm_json(text: str) -> Optional[Any]:
    """
    Best-effort extraction of a valid JSON object/array from LLM output.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"```json(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


@dataclass
class SegmentRecord:
    """Record per fixed-length segment during pre-splitting."""

    video_path: str
    segment_path: str
    index: int
    start_sec: float
    end_sec: float
    vlm_summary: str = ""
    vlm_type: str = ""
    tfidf_vector: Optional[np.ndarray] = None
    motion_score: float = 0.0


def to_jsonable_record(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return obj
    return dict(getattr(obj, "__dict__", {}) or {})


def shot_from_jsonable(obj: Dict[str, Any]) -> ShotRecord:
    rep_frames: List[RepresentativeFrame] = []
    for item in (obj.get("rep_frames") or []):
        if isinstance(item, RepresentativeFrame):
            rep_frames.append(item)
        elif isinstance(item, dict):
            rep_frames.append(RepresentativeFrame(**item))
    payload = dict(obj)
    payload["rep_frames"] = rep_frames
    return ShotRecord(**payload)


def chapter_from_jsonable(obj: Dict[str, Any]) -> ChapterRecord:
    return ChapterRecord(**dict(obj))


def video_profile_from_jsonable(obj: Dict[str, Any]) -> VideoProfile:
    return VideoProfile(**dict(obj))


def ensure_stage_workdir(temp_root: str, video_id: str, seg_id: str, stage_name: str) -> str:
    path = os.path.join(temp_root, video_id, seg_id, stage_name)
    os.makedirs(path, exist_ok=True)
    return path


def prepare_t1_frame_items(
    video_path: str,
    t0_frame_items: List[Dict[str, Any]],
    raw_shots: List[Dict[str, Any]],
    t1_workdir: str,
) -> List[Dict[str, Any]]:
    from executor.tasks.video_preprocess import extract_frames_with_timestamps

    if t0_frame_items:
        out = []
        for idx, item in enumerate(t0_frame_items):
            out.append(
                {
                    "index": idx,
                    "img": item.get("img") or item.get("path") or "",
                    "sec": float(item.get("sec", item.get("timestamp", 0.0)) or 0.0),
                }
            )
        if out:
            return out

    dense_fps = 2.0
    if raw_shots:
        avg_dur = float(np.mean([float(x.get("duration", 0.0) or 0.0) for x in raw_shots]))
        if avg_dur < 2.0:
            dense_fps = 3.0
    frame_dir = os.path.join(t1_workdir, "frames")
    items = extract_frames_with_timestamps(video_path, fps=dense_fps, out_dir=frame_dir)
    return [{"index": i, "img": x.get("img"), "sec": float(x.get("sec", 0.0) or 0.0)} for i, x in enumerate(items)]


def prepare_t2_frame_pool(
    video_path: str,
    shots: List[Any],
    t2_workdir: str,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    from executor.tasks.video_preprocess import extract_frames_with_timestamps

    t2_cfg = _load_t2_frame_sampling_config(config or {})
    frame_dir = os.path.join(t2_workdir, "frames")
    raw = extract_frames_with_timestamps(
        video_path,
        fps=float(t2_cfg.get("base_fps", 3.0) or 3.0),
        out_dir=frame_dir,
    )
    
    items = []
    last_shot = shots[-1] if shots else None
    for idx, item in enumerate(raw):
        sec = float(item.get("sec", 0.0) or 0.0)
        shot_id = ""
        chapter_id = ""
        for shot in shots:
            sid = getattr(shot, "shot_id", "")
            start_sec = float(getattr(shot, "start_sec", 0.0))
            end_sec = float(getattr(shot, "end_sec", 0.0))
            if (sec >= start_sec and sec < end_sec) or (abs(sec - end_sec) < 1e-6 and shot == last_shot):
                shot_id = sid
                chapter_id = (getattr(shot, "metadata", {}) or {}).get("chapter_id", "")
                break
        if not shot_id:
            continue
        items.append(
            {
                "index": idx,
                "frame_index": idx,
                "img": item.get("img"),
                "sec": sec,
                "shot_id": shot_id,
                "chapter_id": chapter_id,
                "candidate_type": "raw_pool",
                "sampling_origin": "prepare_t2_frame_pool",
                "base_scores": {},
            }
        )
    if not bool(t2_cfg.get("adaptive_sampling", True)):
        return items
    return _adaptive_subsample_t2_items(items, shots, t2_cfg)


def _load_t2_frame_sampling_config(config: Dict[str, Any]) -> Dict[str, Any]:
    tasks_cfg = (config.get("tasks", {}) or {}) if isinstance(config, dict) else {}
    t2_cfg = (tasks_cfg.get("T2_frames", {}) or {}) if isinstance(tasks_cfg, dict) else {}
    by_role = dict((t2_cfg.get("by_role", {}) or {}))
    by_sampling_hint = dict((t2_cfg.get("by_sampling_hint", {}) or {}))
    policy = ((config.get("runtime", {}) or {}).get("iteration_policy") or {}) if isinstance(config, dict) else {}
    sampling_density = policy.get("sampling_density", {}) if isinstance(policy, dict) else {}
    keyframe_budget = policy.get("keyframe_budget_multiplier", {}) if isinstance(policy, dict) else {}
    try:
        density_multiplier = float(sampling_density.get("default", 1.0) or 1.0)
    except (TypeError, ValueError):
        density_multiplier = 1.0
    try:
        budget_multiplier = float(keyframe_budget.get("default", 1.0) or 1.0)
    except (TypeError, ValueError):
        budget_multiplier = 1.0
    density_multiplier = max(0.1, min(4.0, density_multiplier))
    budget_multiplier = max(0.1, min(4.0, budget_multiplier))
    return {
        "adaptive_sampling": bool(t2_cfg.get("adaptive_sampling", True)),
        "base_fps": float(t2_cfg.get("base_fps", 3.0) or 3.0),
        "default_fps": float(t2_cfg.get("default_fps", 1.5) or 1.5),
        "min_fps": float(t2_cfg.get("min_fps", 0.5) or 0.5),
        "max_fps": float(t2_cfg.get("max_fps", 3.0) or 3.0),
        "min_frames_per_shot": max(1, int(t2_cfg.get("min_frames_per_shot", 1) or 1)),
        "long_shot_sec": float(t2_cfg.get("long_shot_sec", 8.0) or 8.0),
        "long_shot_min_frames": max(1, int(t2_cfg.get("long_shot_min_frames", 3) or 3)),
        "by_role": by_role,
        "by_sampling_hint": by_sampling_hint,
        "iteration_sampling_density": density_multiplier,
        "iteration_keyframe_budget_multiplier": budget_multiplier,
    }


def _adaptive_subsample_t2_items(
    items: List[Dict[str, Any]],
    shots: List[Any],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not items or not shots:
        return items

    by_shot: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        shot_id = str(item.get("shot_id", "") or "")
        if not shot_id:
            continue
        by_shot.setdefault(shot_id, []).append(item)

    shot_map = {str(getattr(shot, "shot_id", "") or ""): shot for shot in shots}
    kept: List[Dict[str, Any]] = []
    for shot in shots:
        shot_id = str(getattr(shot, "shot_id", "") or "")
        shot_items = sorted(by_shot.get(shot_id, []), key=lambda x: float(x.get("sec", 0.0) or 0.0))
        if not shot_items:
            continue
        target_fps = _resolve_t2_shot_sampling_fps(shot, cfg)
        min_keep = int(cfg.get("min_frames_per_shot", 1) or 1)
        budget_multiplier = float(cfg.get("iteration_keyframe_budget_multiplier", 1.0) or 1.0)
        min_keep = max(1, int(round(min_keep * budget_multiplier)))
        if float(getattr(shot, "duration", 0.0) or 0.0) >= float(cfg.get("long_shot_sec", 8.0) or 8.0):
            min_keep = max(min_keep, int(cfg.get("long_shot_min_frames", 3) or 3))
        kept.extend(_subsample_t2_items_for_shot(shot_items, target_fps=target_fps, min_keep=min_keep))

    kept.sort(key=lambda x: float(x.get("sec", 0.0) or 0.0))
    return kept


def _resolve_t2_shot_sampling_fps(shot: Any, cfg: Dict[str, Any]) -> float:
    default_fps = float(cfg.get("default_fps", 1.5) or 1.5)
    min_fps = float(cfg.get("min_fps", 0.5) or 0.5)
    max_fps = float(cfg.get("max_fps", 3.0) or 3.0)

    role = str(getattr(shot, "shot_role", "") or "generic").strip().lower() or "generic"
    scene_type = str(getattr(shot, "scene_type", "") or "").strip().lower()
    sampling_hint = str(getattr(shot, "candidate_sampling_hint", "") or "medium").strip().lower()
    informativeness = str(getattr(shot, "informativeness_hint", "") or "medium").strip().lower()
    duration = float(getattr(shot, "duration", 0.0) or 0.0)

    role_fps = cfg.get("by_role", {}).get(role)
    if role_fps is None and scene_type in {"action", "sports", "dynamic_scene"}:
        role_fps = cfg.get("by_role", {}).get("action")
    fps = float(role_fps if role_fps is not None else default_fps)
    fps *= float(cfg.get("iteration_sampling_density", 1.0) or 1.0)

    hint_scale = float(cfg.get("by_sampling_hint", {}).get(sampling_hint, 1.0) or 1.0)
    fps *= hint_scale

    if informativeness == "high":
        fps *= 1.10
    elif informativeness == "low":
        fps *= 0.90

    if duration <= 2.0:
        fps = min(fps, max(default_fps, 1.0))
    elif duration >= float(cfg.get("long_shot_sec", 8.0) or 8.0):
        fps = max(fps, min(default_fps, 1.5))

    return max(min_fps, min(max_fps, fps))


def _subsample_t2_items_for_shot(
    shot_items: List[Dict[str, Any]],
    *,
    target_fps: float,
    min_keep: int,
) -> List[Dict[str, Any]]:
    if not shot_items:
        return []
    if len(shot_items) <= min_keep:
        return list(shot_items)

    target_fps = max(float(target_fps or 0.0), 1e-6)
    min_interval = 1.0 / target_fps
    kept: List[Dict[str, Any]] = []
    last_sec: Optional[float] = None
    for item in shot_items:
        sec = float(item.get("sec", 0.0) or 0.0)
        if last_sec is None or (sec - last_sec) >= (min_interval - 1e-6):
            kept.append(item)
            last_sec = sec

    if len(kept) >= min_keep:
        return kept

    if min_keep == 1:
        return [shot_items[len(shot_items) // 2]]

    idxs = np.linspace(0, len(shot_items) - 1, num=min_keep, dtype=int).tolist()
    dedup = []
    seen = set()
    for idx in idxs:
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        dedup.append(shot_items[idx])
    return dedup

def chapters_to_split_points(chapters: List[Any], segment_start_sec: float, segment_end_sec: float) -> List[float]:
    points = []
    for chapter in chapters[:-1]:
        if isinstance(chapter, dict):
            end_sec = float(chapter.get("end_sec", 0.0) or 0.0)
        else:
            end_sec = float(getattr(chapter, "end_sec", 0.0) or 0.0)
        rel = max(0.0, min(segment_end_sec - segment_start_sec, end_sec - segment_start_sec))
        if rel > 0.0:
            points.append(round(rel, 3))
    uniq = []
    for value in points:
        if not uniq or abs(value - uniq[-1]) > 1e-3:
            uniq.append(value)
    return uniq


def estimate_structure_confidence(chapters: List[Any], pairwise_merge_decisions: List[Dict[str, Any]]) -> float:
    if not pairwise_merge_decisions:
        return 0.5 if chapters else 0.0
    confs = [float(x.get("confidence", 0.5) or 0.5) for x in pairwise_merge_decisions]
    base = float(np.mean(confs)) if confs else 0.5
    if len(chapters) <= 1:
        base *= 0.9
    return max(0.0, min(1.0, base))


def build_t1_debug_pack(
    shots: List[Any],
    chapters: List[Any],
    pairwise_merge_decisions: List[Dict[str, Any]],
    smoothing_meta: Dict[str, Any],
    shot_detector_meta: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "shot_count": len(shots),
        "chapter_count": len(chapters),
        "pairwise_decision_count": len(pairwise_merge_decisions),
        "shot_detector_meta": shot_detector_meta,
        "smoothing_meta": smoothing_meta,
    }


def decide_total_frame_budget(segment: Any, shots: List[Any], chapters: List[Any]) -> int:
    duration = max(0.0, float(getattr(segment, "end_sec", 0.0)) - float(getattr(segment, "start_sec", 0.0)))
    base = 4
    base += min(4, len(chapters))
    if duration >= 20:
        base += 2
    if len(shots) >= 8:
        base += 1
    return max(2, min(base, 16))

