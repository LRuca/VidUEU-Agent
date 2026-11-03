from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from orchestrator.planner_helpers import parse_llm_json
from orchestrator.structures import RepresentativeFrame, ShotRecord


class ShotProfiler:
    """
    Build shot profiles for T1 from VLM evidence only.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        cfg = ((self.config.get("tasks", {}) or {}).get("T1_shot_profile", {}) or {})
        self.paper_midpoint = bool(cfg.get("paper_midpoint", False))
        self.max_rep_frames = int(cfg.get("max_rep_frames", 3) or 3)
        self.long_shot_sec = float(cfg.get("long_shot_sec", 6.0) or 6.0)
        self.min_frames_for_second = int(cfg.get("min_frames_for_second", 6) or 6)
        self.enable_object_tags = bool(cfg.get("enable_object_tags", False))

    def build_shot_profiles(
        self,
        video_path: str,
        shots: List[Dict[str, Any]],
        frame_items: List[Dict[str, Any]],
        model_manager: Any,
        workdir: str = "",
    ) -> List[ShotRecord]:
        _ = video_path
        shot_frames = self._bind_frames_to_shots(shots, frame_items)
        shot_records: List[ShotRecord] = []
        shot_inputs = []

        for shot in shots:
            shot_id = str(shot["shot_id"])
            sf = shot_frames.get(shot_id, [])
            rep_frames = self._select_representative_frames(sf)
            shot_inputs.append(
                {
                    "shot": shot,
                    "shot_id": shot_id,
                    "shot_frames": sf,
                    "rep_frames": rep_frames,
                }
            )

        vlm_results = model_manager.run_vlm_jobs(
            shot_inputs,
            worker_fn=lambda vlm, item: self._profile_shot_with_vlm(item["rep_frames"], vlm),
        )

        for item, vlm_result in zip(shot_inputs, vlm_results):
            shot = item["shot"]
            shot_id = item["shot_id"]
            sf = item["shot_frames"]
            rep_frames = item["rep_frames"]
            vlm_payload, vlm_fallback_used, vlm_debug = vlm_result

            vlm_summary = self._compact_text(str(vlm_payload.get("vlm_summary", "") or ""), 220)
            role_fallback_used = False
            shot_role = str(vlm_payload.get("shot_role", "") or "").strip().lower()
            if not shot_role:
                shot_role = self._fallback_role(vlm_summary, sf)
                role_fallback_used = True
            scene_type = str(vlm_payload.get("scene_type", "") or "").strip().lower() or shot_role or "generic"
            visual_quality_hint = self._normalize_hint(vlm_payload.get("visual_quality_hint"), default="medium")
            informativeness_hint = self._normalize_hint(vlm_payload.get("informativeness_hint"), default="medium")
            candidate_sampling_hint = self._normalize_sampling_hint(
                vlm_payload.get("candidate_sampling_hint"),
                shot_duration=float(shot.get("duration", 0.0) or 0.0),
            )

            text_density = 0.0
            speech_density = 0.0
            object_tags = self._infer_object_tags(rep_frames, model_manager) if self.enable_object_tags else []
            importance_hint = self._infer_importance_hint(
                shot_role=shot_role,
                informativeness_hint=informativeness_hint,
                visual_quality_hint=visual_quality_hint,
            )

            debug_path = self._dump_vlm_debug(
                shot_id=shot_id,
                workdir=workdir,
                rep_frames=rep_frames,
                prompt=str(vlm_debug.get("prompt", "") or ""),
                raw=str(vlm_debug.get("raw", "") or ""),
                parsed=vlm_payload,
                fallback_used=bool(vlm_fallback_used),
                error=str(vlm_debug.get("error", "") or ""),
            )

            shot_records.append(
                ShotRecord(
                    shot_id=shot_id,
                    start_sec=float(shot["start_sec"]),
                    end_sec=float(shot["end_sec"]),
                    duration=float(shot["duration"]),
                    frame_indices=[int(x.get("index", -1)) for x in sf],
                    frame_paths=[str(x.get("img", "")) for x in sf if x.get("img")],
                    detector_backend=str(shot.get("detector_backend", "")),
                    boundary_confidence_in=float(shot.get("boundary_confidence_in", 0.0) or 0.0),
                    boundary_confidence_out=float(shot.get("boundary_confidence_out", 0.0) or 0.0),
                    rep_frames=rep_frames,
                    visual_embedding=None,
                    visual_novelty=0.0,
                    motion_score=0.0,
                    scene_type=scene_type,
                    visual_quality_hint=visual_quality_hint,
                    informativeness_hint=informativeness_hint,
                    candidate_sampling_hint=candidate_sampling_hint,
                    text_density=text_density,
                    speech_density=speech_density,
                    shot_role=shot_role,
                    vlm_summary=vlm_summary,
                    ocr_summary="",
                    asr_summary="",
                    object_tags=object_tags,
                    importance_hint=importance_hint,
                    metadata={
                        "rep_frame_count": len(rep_frames),
                        "workdir": workdir,
                        "vlm_debug_path": debug_path,
                        "fallback_flags": {
                            "vlm_profile_fallback_used": bool(vlm_fallback_used),
                            "role_fallback_used": bool(role_fallback_used),
                        },
                    },
                )
            )

        return shot_records

    def _bind_frames_to_shots(
        self, shots: List[Dict[str, Any]], frame_items: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {str(x["shot_id"]): [] for x in shots}
        for item in frame_items:
            sec = float(item.get("sec", 0.0) or 0.0)
            placed = False
            for shot in shots:
                sid = str(shot["shot_id"])
                s = float(shot["start_sec"])
                e = float(shot["end_sec"])
                if (sec >= s and sec < e) or (math.isclose(sec, e) and sid == shots[-1]["shot_id"]):
                    out[sid].append(item)
                    placed = True
                    break
            if not placed and shots:
                out[str(shots[-1]["shot_id"])].append(item)
        return out

    def _select_representative_frames(
        self,
        shot_frames: List[Dict[str, Any]],
    ) -> List[RepresentativeFrame]:
        if not shot_frames:
            return []
        if self.paper_midpoint:
            item = shot_frames[len(shot_frames) // 2]
            return [RepresentativeFrame(frame_index=int(item.get("index", 0)), sec=float(item.get("sec", 0)),
                img_path=str(item.get("img", "")), kind="midpoint", quality_score=0.0, change_score=0.0)]
        main_scores = self._score_main_frame(shot_frames)
        order = np.argsort(np.asarray(main_scores, dtype=np.float32))[::-1].tolist()
        picks = []
        max_pick = 1
        if self._need_second_frame(shot_frames):
            max_pick = min(self.max_rep_frames, 2)
        if len(shot_frames) >= max(8, self.min_frames_for_second * 2):
            max_pick = min(self.max_rep_frames, 3)
        seen = set()
        for idx in order:
            seen.add(idx)
            picks.append((idx, "main"))
            break
        if max_pick >= 2:
            mid_idx = len(shot_frames) // 2
            if mid_idx not in seen:
                seen.add(mid_idx)
                picks.append((mid_idx, "coverage_mid"))
        if max_pick >= 3:
            tail_idx = len(shot_frames) - 1
            if tail_idx not in seen:
                picks.append((tail_idx, "coverage_tail"))

        reps: List[RepresentativeFrame] = []
        for idx, kind in picks[: self.max_rep_frames]:
            item = shot_frames[idx]
            reps.append(
                RepresentativeFrame(
                    frame_index=int(item.get("index", idx)),
                    sec=float(item.get("sec", 0.0)),
                    img_path=str(item.get("img", "")),
                    kind=kind,
                    quality_score=float(main_scores[idx]) if idx < len(main_scores) else 0.0,
                    change_score=0.0,
                )
            )
        return reps

    def _score_main_frame(self, shot_frames: List[Dict[str, Any]]) -> List[float]:
        n = len(shot_frames)
        scores = []
        center = (n - 1) / 2.0 if n > 0 else 0.0
        for idx, item in enumerate(shot_frames):
            q = self._basic_quality(item.get("img", ""))
            d = abs(idx - center) / max(center, 1.0)
            centrality = 1.0 - min(1.0, d)
            scores.append(0.65 * q + 0.35 * centrality)
        return scores

    def _need_second_frame(self, shot_frames: List[Dict[str, Any]]) -> bool:
        if len(shot_frames) < self.min_frames_for_second:
            return False
        return (float(shot_frames[-1].get("sec", 0.0)) - float(shot_frames[0].get("sec", 0.0))) >= self.long_shot_sec

    def _basic_quality(self, img_path: str) -> float:
        if not img_path or not os.path.exists(img_path):
            return 0.0
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        small = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        lap_var = float(cv2.Laplacian(small, cv2.CV_64F).var())
        bright = float(small.mean())
        contrast = float(small.std())
        score = min(1.0, lap_var / 300.0) * 0.55 + min(1.0, contrast / 64.0) * 0.25
        if 35.0 <= bright <= 220.0:
            score += 0.20
        return max(0.0, min(1.0, score))

    def _profile_shot_with_vlm(
        self,
        rep_frames: List[RepresentativeFrame],
        vlm: Any,
    ) -> tuple[Dict[str, Any], bool, Dict[str, Any]]:
        paths = [x.img_path for x in rep_frames if x.img_path]
        if not paths:
            return {}, True, {"prompt": "", "raw": "", "error": "no_rep_frames"}
        prompt = (
            "You are building a visual-only shot profile for a video pipeline.\n"
            "Use only the images. Do not infer OCR or ASR content.\n"
            "Return strict JSON with keys: "
            "vlm_summary, shot_role, scene_type, visual_quality_hint, informativeness_hint, candidate_sampling_hint.\n"
            "Allowed hint values:\n"
            "- visual_quality_hint: low, medium, high\n"
            "- informativeness_hint: low, medium, high\n"
            "- candidate_sampling_hint: low, medium, high\n"
        )
        try:
            raw = vlm.chat(paths[: max(1, min(3, len(paths)))], prompt, max_new_tokens=180)
            obj = parse_llm_json(str(raw or ""))
            if isinstance(obj, dict):
                return obj, False, {"prompt": prompt, "raw": str(raw or ""), "error": ""}
            return {}, True, {"prompt": prompt, "raw": str(raw or ""), "error": "parse_failed"}
        except Exception as e:
            return {}, True, {"prompt": prompt, "raw": "", "error": repr(e)}

    def _dump_vlm_debug(
        self,
        *,
        shot_id: str,
        workdir: str,
        rep_frames: List[RepresentativeFrame],
        prompt: str,
        raw: str,
        parsed: Dict[str, Any],
        fallback_used: bool,
        error: str,
    ) -> str:
        if not workdir:
            return ""
        paths_cfg = (self.config.get("paths", {}) or {}) if isinstance(self.config, dict) else {}
        debug_root = str(paths_cfg.get("debug_dir") or "").strip()
        temp_root = str(paths_cfg.get("temp_dir") or "").strip()
        if debug_root:
            workdir_path = os.path.abspath(workdir)
            debug_root_path = os.path.abspath(debug_root)
            relative_parts: list[str]
            if temp_root:
                try:
                    relative = os.path.relpath(workdir_path, os.path.abspath(temp_root))
                    if not relative.startswith(".."):
                        relative_parts = [part for part in relative.split(os.sep) if part and part != "."]
                    else:
                        relative_parts = [os.path.basename(workdir_path)]
                except Exception:
                    relative_parts = [os.path.basename(workdir_path)]
            else:
                relative_parts = [os.path.basename(workdir_path)]
            debug_dir = os.path.join(debug_root_path, *relative_parts)
        else:
            debug_dir = os.path.join(workdir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        payload = {
            "shot_id": shot_id,
            "rep_frames": [
                {
                    "frame_index": int(x.frame_index),
                    "sec": float(x.sec),
                    "img_path": str(x.img_path),
                    "kind": str(x.kind),
                    "quality_score": float(x.quality_score),
                }
                for x in rep_frames
            ],
            "prompt": prompt,
            "raw_response": raw,
            "parsed_response": parsed,
            "fallback_used": bool(fallback_used),
            "error": error,
        }
        debug_path = os.path.join(debug_dir, f"{shot_id}_vlm_profile.json")
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return debug_path

    def _fallback_role(self, vlm_summary: str, shot_frames: List[Dict[str, Any]]) -> str:
        s = (vlm_summary or "").lower()
        if any(k in s for k in ("title", "opening card", "section card")):
            return "title_card"
        if any(k in s for k in ("screen", "ui", "software", "desktop", "interface")):
            return "screen_demo"
        if any(k in s for k in ("slide", "presentation")):
            return "slide"
        if any(k in s for k in ("speaker", "person speaking", "talking head")):
            return "talking_head"
        if any(k in s for k in ("action", "movement", "playing", "running")):
            return "action"
        if shot_frames and (float(shot_frames[-1].get("sec", 0.0)) - float(shot_frames[0].get("sec", 0.0))) < 1.0:
            return "transition_bridge"
        return "generic"

    def _normalize_hint(self, value: Any, default: str) -> str:
        s = str(value or "").strip().lower()
        if s in {"low", "medium", "high"}:
            return s
        return default

    def _normalize_sampling_hint(self, value: Any, shot_duration: float) -> str:
        s = str(value or "").strip().lower()
        if s in {"low", "medium", "high"}:
            return s
        if shot_duration >= 8.0:
            return "high"
        if shot_duration <= 3.0:
            return "low"
        return "medium"

    def _infer_object_tags(self, rep_frames: List[RepresentativeFrame], model_manager: Any) -> List[str]:
        paths = [x.img_path for x in rep_frames if x.img_path]
        if not paths:
            return []
        try:
            det = model_manager.run_with_resource("yolo", lambda yolo: yolo.run(paths))
        except Exception:
            return []
        counts: Dict[str, int] = {}
        for p in paths:
            for item in (det.get(p) or []):
                label = str(item.get("label", "") or "").strip().lower()
                if label:
                    counts[label] = counts.get(label, 0) + 1
        return [x for x, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6]]

    def _infer_importance_hint(
        self,
        shot_role: str,
        informativeness_hint: str,
        visual_quality_hint: str,
    ) -> str:
        if visual_quality_hint == "low":
            return "low"
        if shot_role in {"title_card", "screen_demo", "slide"}:
            return "high"
        if informativeness_hint == "high":
            return "high"
        if informativeness_hint == "low":
            return "low"
        return "medium"

    def _compact_text(self, text: str, max_chars: int = 240) -> str:
        t = " ".join((text or "").replace("\n", " ").split())
        if len(t) > max_chars:
            t = t[:max_chars].rstrip() + " ..."
        return t
