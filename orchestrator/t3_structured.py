from __future__ import annotations

import glob
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import traceback
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from orchestrator.planner_helpers import parse_llm_json
from executor.tasks.prompt_loader import render_prompt_text, structured_evidence_profile, current_prompt_profile
from orchestrator.metadata_admission import filter_signals, paper_profile
from orchestrator.prompt_audit import audited_text, audited_vision, paper_prompt_stage
from orchestrator.transition_validation import validate_transition, infer_transition_cause
from orchestrator.visual_state import validate_frame_observation, validate_pair_observation

@paper_prompt_stage("T3")
def run_t3_structured(
    planner: Any,
    t1_stage: Dict[str, Any],
    t2_stage: Dict[str, Any],
    segment: Any,         # current segment
) -> Dict[str, Any]:
    self = planner
    if getattr(planner, "model_manager", None) is not None:
        planner.model_manager.get_resource("vlm")
    """
    T3 local / temporal understanding.
    - Step1: reuse per-keyframe captions already produced in T2.
    - Step2: generate transition strings pairwise from keyframe i and i+1 only.

    Notes:
    - Transition generation no longer uses the old multi-image Round1 pass.
    - Atomic edit constraints are still enforced by prompt + local post-processing.
    - Cause labels are inferred locally from the final structured transition string.
    """
    import os
    import json
    import re
    from typing import Any, Dict, List, Optional
    from PIL import Image

    # ---- context ----
    t1_out = t1_stage.get("output", {}) or {}
    t2_out = t2_stage.get("output", {}) or {}

    video_type = t1_out.get("video_type", "other")
    summary = t1_out.get("global_summary", "")
    advice = t1_out.get("modality_hint", "mixed")

    signal_filter = {}
    if paper_profile(self.config):
        t2_out = copy.deepcopy(t2_out)
    keyframes: List[Dict[str, Any]] = t2_out.get("keyframes", []) or []
    mme_presence_cap = 0.40
    # ------- internal printer: T3 summary -------
    def _print_T3_summary(tag: str, segment_summary: str, frames: List[Dict[str, Any]], dataset_type: str):
        seg_idx = getattr(segment, "index", None)
        start_sec = getattr(segment, "start_sec", None)
        end_sec = getattr(segment, "end_sec", None)

        print("\n================ T3 LOCAL/TEMPORAL ======================")
        if seg_idx is not None and start_sec is not None and end_sec is not None:
            print(f"Segment: #{seg_idx}  [{start_sec:.1f}s -> {end_sec:.1f}s]")
        elif seg_idx is not None:
            print(f"Segment: #{seg_idx}")
        print(f"Tag   : {tag}")
        print("---------------------------------------------------------")
        print(f"video_type   : {video_type}")
        print(f"dataset_type : {dataset_type}")
        print("---------------------------------------------------------")
        print("segment_summary:")
        print(segment_summary if segment_summary else "(empty)")
        print("---------------------------------------------------------")
        print(f"num_frames: {len(frames)}")
        for i, fr in enumerate(frames):
            sec = fr.get("sec", None)
            cap = (fr.get("caption") or "").strip()
            trans = (fr.get("transition_to_next") or "").strip()
            cause = (fr.get("cause") or "").strip()
            cap_show = (cap[:77] + "...") if len(cap) > 80 else cap
            trans_show = (trans[:77] + "...") if len(trans) > 80 else trans

            if sec is not None:
                print(f"[{i}] t={sec:.2f}s")
            else:
                print(f"[{i}]")
            if cause:
                print(f"  cause             : {cause}")
            print(f"  caption           : {cap_show}")
            if trans_show:
                print(f"  transition_to_next: {trans_show}")
        print("=========================================================\n")

    # Skip T3 immediately if T2 rejected the segment.
    if not t2_out.get("quality_passed", False):
        print("[T3][WARN] quality_passed is False in T2, skip T3 captioning.")
        _print_T3_summary("rejected_by_T2", "", [], "none")
        return {"output": {"segment_summary": "", "frames": [], "dataset_type": "none"}}

    if not keyframes:
        print("[T3][WARN] No keyframes found, skip.")
        _print_T3_summary("no_keyframes", "", [], "none")
        return {"output": {"segment_summary": "", "frames": [], "dataset_type": "none"}}

    # ------- load keyframe images -------
    images: List[Image.Image] = []
    for item in keyframes:
        try:
            img = Image.open(item["img"]).convert("RGB")
            img.filename = str(item["img"])
            images.append(img)
        except Exception as e:
            for loaded in images:
                loaded.close()
            raise RuntimeError(f"T3 keyframe load failed; preserve image/caption bindings: {item.get('img')}") from e

    if not images:
        print("[T3][WARN] Failed to load any keyframe images.")
        _print_T3_summary("load_failed", "", [], "none")
        return {"output": {"segment_summary": "", "frames": [], "dataset_type": "none"}}

    num_frames = len(images)
    seg_idx = getattr(segment, "index", -1)

    # debug root
    debug_root = self.config.get("paths", {}).get("debug_dir", "logs/debug")
    os.makedirs(debug_root, exist_ok=True)
    seg_path = getattr(segment, "segment_path", "") or ""
    base_src = seg_path or getattr(segment, "video_path", "") or "unknown_video"
    base = os.path.splitext(os.path.basename(base_src))[0]

    # ---------------- helper: robust json parse ----------------
    def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        s = str(text).strip()
        s2 = s.strip().strip("`").strip()
        if s2.lower().startswith("json"):
            s2 = s2[4:].strip()
        try:
            obj = json.loads(s2)
        except Exception:
            try:
                obj = json.loads(s)
            except Exception:
                return None
        return obj if isinstance(obj, dict) else None

    def _normalize_transition_str(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return ""
        obj = _try_parse_json(s)
        if isinstance(obj, dict):
            if isinstance(obj.get("transition_to_next"), str):
                return obj["transition_to_next"].strip()
            for k in ("text", "transition", "transition_text"):
                if isinstance(obj.get(k), str):
                    return obj[k].strip()
        s = s.splitlines()[0].strip() if "\n" in s else s
        return s

    def _looks_like_structured_transition(s: str) -> bool:
        s = (s or "")
        return ("Edits(" in s) and ("Shot:" in s) and ("View:" in s) and ("Subject:" in s) and ("Uncertainty:" in s)

    # ---------------- [MODIFIED v2] atomicize post-processor ----------------
    def _split_obj_list(txt: str) -> List[str]:
        t = (txt or "").strip(" ,;.")
        if not t:
            return []
        t = re.sub(r"\s+", " ", t)
        t = t.replace(" & ", " and ")
        parts = [p.strip() for p in t.split(",") if p.strip()]
        out: List[str] = []
        for p in parts:
            if " and " in p:
                out.extend([x.strip() for x in p.split(" and ") if x.strip()])
            else:
                out.append(p)
        res, seen = [], set()
        for x in out:
            if x and x not in seen:
                res.append(x)
                seen.add(x)
        return res

    def _normalize_view_clause(vc: str) -> str:
        vc = (vc or "").strip(" ,;.")
        if not vc:
            return ""
        vc2 = re.sub(r"(?i)^(viewpoint|camera)\s*", "", vc).strip()
        vc2 = re.sub(r"(?i)\bpans\b", "pan", vc2)
        vc2 = re.sub(r"(?i)\bpan\s+to\b", "pan", vc2)
        vc2 = re.sub(r"(?i)\bshifts\b", "pan", vc2)
        vc2 = re.sub(r"(?i)\bmoves\b", "pan", vc2)
        vc2 = re.sub(r"(?i)\btilts\b", "tilt", vc2)
        vc2 = re.sub(r"(?i)\bzooms\b", "zoom", vc2)
        if not re.match(r"(?i)^(pan|zoom|tilt|cut)\b", vc2):
            vc2 = "pan " + vc2
        return vc2.strip()

    def _atomicize_transition(t: str) -> str:
        """
        Make edits truly atomic and fix Edits(k).
        - split 'and' objects
        - split 'replace X with Y' -> remove X; add Y
        - peel viewpoint/camera clause into its own edit
        """
        s = (t or "").strip()
        if not _looks_like_structured_transition(s):
            return s

        parts = [p.strip() for p in s.split("|")]
        head = parts[0] if parts else s
        tail = " | ".join(parts[1:]).strip()

        m = re.search(r"Edits\((\d+)\)\s*:\s*(.*)$", head, flags=re.I)
        if not m:
            return s
        edits_raw = (m.group(2) or "").strip()

        edits = [x.strip() for x in re.split(r";\s*", edits_raw) if x.strip()]
        cleaned: List[str] = []
        for e in edits:
            e = re.sub(r"^\d+\)\s*", "", e).strip()
            if e:
                cleaned.append(e)

        atoms: List[str] = []
        for e in cleaned:
            e0 = re.sub(r"\s+", " ", e).strip(" ;")
            low = e0.lower()

            view_clause = ""
            mm = re.search(r"(?i)([,;]\s*)(viewpoint|camera)\b.*$", e0)
            if mm:
                view_clause = e0[mm.start(2):].strip()
                e0 = e0[:mm.start(0)].rstrip(" ,;")
                low = e0.lower()

            if view_clause:
                vc = _normalize_view_clause(view_clause)
                if vc:
                    atoms.append(vc)

            m2 = re.search(r"(?i)\breplace\b(.+?)\bwith\b(.+)$", e0)
            if m2:
                x = (m2.group(1) or "").strip(" ,;")
                y = (m2.group(2) or "").strip(" ,;")
                for obj in _split_obj_list(x):
                    atoms.append(f"remove {obj}")
                for obj in _split_obj_list(y):
                    atoms.append(f"add {obj}")
                continue

            m3 = re.search(r"(?i)^\s*remove\b(.+)$", e0)
            if m3:
                for obj in _split_obj_list(m3.group(1)):
                    atoms.append(f"remove {obj}")
                continue

            m4 = re.search(r"(?i)^\s*add\b(.+)$", e0)
            if m4:
                for obj in _split_obj_list(m4.group(1)):
                    atoms.append(f"add {obj}")
                continue

            if e0:
                if " and " in e0.lower():
                    for verb in ["move", "rotate", "open", "close", "turn on", "turn off", "change text", "zoom", "pan", "cut", "tilt"]:
                        if e0.lower().startswith(verb + " "):
                            rest = e0[len(verb):].strip()
                            for obj in _split_obj_list(rest):
                                atoms.append(f"{verb} {obj}".strip())
                            break
                    else:
                        atoms.append(e0)
                else:
                    atoms.append(e0)

        out_atoms, seen = [], set()
        for a in atoms:
            a = (a or "").strip()
            if not a:
                continue
            if a not in seen:
                out_atoms.append(a)
                seen.add(a)
        out_atoms = out_atoms[:6]

        if not out_atoms:
            return 'Edits(0): (none) | Shot: none | View: none | Subject: same | Uncertainty: near-duplicate'

        k = len(out_atoms)
        edits_str = "; ".join([f"{i+1}) {out_atoms[i]}" for i in range(k)])
        if tail:
            return f"Edits({k}): {edits_str} | {tail}".strip(" |")
        return f"Edits({k}): {edits_str}".strip()

    # ---------------- [NEW v3] cause inference (rule-based fallback) ----------------
    _cause_enum = {"none", "edit", "object_motion", "occlusion", "camera_motion", "scene_cut", "unknown"}

    def _infer_cause_from_transition(tr: str) -> str:
        """
        Infer a coarse cause label from structured transition string.
        This is a fallback/consistency check.
        """
        s = (tr or "").lower()
        if not _looks_like_structured_transition(tr):
            return "unknown"

        # explicit scene cut in Shot or an edit "cut ..."
        if re.search(r"(?i)\bshot:\s*scene_cut\b", tr):
            return "scene_cut"
        if re.search(r"(?i)\b1\)\s*cut\b", tr) or re.search(r"(?i)\b;\s*\d\)\s*cut\b", tr):
            return "scene_cut"

        # camera motion keywords in edits or View label
        if re.search(r"(?i)\bview:\s*(minor|major)\b", tr):
            return "camera_motion"
        if re.search(r"(?i)\b(pan|zoom|tilt)\b", tr):
            return "camera_motion"

        # occlusion / out of frame hints (often show up as uncertainty notes)
        if re.search(r"(?i)\b(occlud|blocked|partially hidden|out of frame|off-screen)\b", tr):
            return "occlusion"

        # object motion: move verb or many "move ..." edits
        if re.search(r"(?i)\bmove\b", tr):
            return "object_motion"

        # edit: add/remove/change text/turn on/off/open/close/rotate
        if re.search(r"(?i)\b(add|remove|change text|turn on|turn off|open|close|rotate)\b", tr):
            return "edit"

        # no edits
        if re.search(r"(?i)\bedits\(0\):\s*\(none\)", tr):
            return "none"

        return "unknown"

    # =======================================================
    # Step 1) Reuse captions already generated for each keyframe in T2.
    # =======================================================
    frame_captions: List[str] = []
    for kf in keyframes:
        quality_profile = kf.get("quality_profile", {}) if isinstance(kf.get("quality_profile", {}), dict) else {}
        caption = str(
            kf.get("caption", "")
            or quality_profile.get("caption", "")
            or ""
        ).strip()
        if paper_profile(self.config):
            prompt = render_prompt_text("paper", "frame_caption", original_caption=caption, frame_id=f"F{len(frame_captions):02d}")
            caption_attempts = []
            request = prompt
            for attempt in range(3 if structured_evidence_profile() else 1):
                raw_caption = audited_vision(self, [kf["img"]], request,
                    max_new_tokens=6000 if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'} else 2400 if structured_evidence_profile() else 650)
                verified = parse_llm_json(raw_caption)
                try:
                    if not isinstance(verified, dict) or not isinstance(verified.get("caption"), str) or not verified["caption"].strip():
                        raise ValueError("Frame caption verification returned invalid JSON or empty caption")
                    for key in ["observations", "uncertainties", "unsupported_original_claims"]:
                        if not isinstance(verified.get(key), list) or any(not isinstance(v, str) for v in verified[key]):
                            raise ValueError(f"Frame caption verification lacks {key}")
                    if structured_evidence_profile():
                        observed = validate_frame_observation(verified, f"F{len(frame_captions):02d}", kf["img"])
                except ValueError as exc:
                    caption_attempts.append({"raw_response": raw_caption, "validation_error": str(exc)})
                    if not structured_evidence_profile() or attempt == 2:
                        raise
                    request = prompt + "\nThe previous reply failed the response contract: " + str(exc) + ". Use only the specified enum values and exact frame bindings. Return the complete JSON; inspect the image again, preserve uncertainty, and do not invent facts to repair a field.\nPREVIOUS REPLY:\n" + raw_caption
                else:
                    caption_attempts.append({"raw_response": raw_caption, "validation_error": None})
                    break
            kf["original_caption"] = caption
            kf["caption_validation"] = {**verified, "raw_response": raw_caption, "attempts": caption_attempts}
            if structured_evidence_profile():
                kf["visual_state"] = observed
            caption = verified["caption"].strip()
            kf["caption"] = caption
        frame_captions.append(caption)

    if paper_profile(self.config):
        t2_out, signal_filter = filter_signals(self, t1_out, t2_out)
        keyframes = t2_out.get("keyframes", [])

    captions_dump_path = None
    try:
        captions_dump_path = os.path.join(debug_root, f"t3_frame_captions_{base}_seg{seg_idx:03d}.txt")
        with open(captions_dump_path, "w", encoding="utf-8") as f:
            for i, cap in enumerate(frame_captions):
                f.write(f"[{i}] {cap}\n")
        print(f"[T3] Frame captions dumped to {captions_dump_path}")
    except Exception as e:
        print(f"[T3][WARN] Failed to dump frame captions: {e}")

    # =======================================================
    # Step 2) Transition generation now runs pairwise only; no multi-image Round1.
    # =======================================================
    dataset_type = "none"
    debug_path = None
    vlm_raw = ""
    frames_info: List[Dict[str, Any]] = [{"transition_to_next": "", "cause": "none"} for _ in range(num_frames)]

    # Round1 multi-image transition generation removed to avoid large VLM memory spikes.

    frames_info[-1]["transition_to_next"] = ""
    frames_info[-1]["cause"] = "none"

    # =======================================================
    # Step 2) Generate transition per pair (i, i+1) directly from the two images.
    # =======================================================
    t3_refine_max_new_tokens = int(
        self.config.get("tasks", {}).get("T3_local", {}).get("refine_max_new_tokens", 0) or 0
    )
    if t3_refine_max_new_tokens <= 0:
        t3_refine_max_new_tokens = 420
    if structured_evidence_profile():
        t3_refine_max_new_tokens = max(10000 if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'} else 3000, t3_refine_max_new_tokens)

    refined_transitions: List[str] = []
    refined_causes: List[str] = []
    transition_checks: List[Dict[str, Any]] = []

    for i in range(num_frames):
        if i == num_frames - 1:
            refined_transitions.append("")  # last must be empty
            refined_causes.append("none")
            transition_checks.append({"status": "not_applicable", "text": ""})
            continue

        cap_a = frame_captions[i] if i < len(frame_captions) else ""
        cap_b = frame_captions[i + 1] if (i + 1) < len(frame_captions) else ""

        state_context = {}
        if structured_evidence_profile():
            state_context = {"states_a": json.dumps(keyframes[i]["visual_state"], ensure_ascii=False),
                             "states_b": json.dumps(keyframes[i+1]["visual_state"], ensure_ascii=False)}
        rp = render_prompt_text(
            "t3_local",
            "pair_transition",
            video_type=video_type,
            summary="Verified keyframe states: " + " | ".join(frame_captions) if paper_profile(self.config) else summary,
            advice=advice,
            cap_a=cap_a,
            cap_b=cap_b,
            **state_context,
        )

        if paper_profile(self.config):
            rp += "\nRetained auxiliary tool signals (do not overrule images):\n" + json.dumps([keyframes[i].get("semantic_profile", {}), keyframes[i+1].get("semantic_profile", {})], ensure_ascii=False)

        raw2 = ""
        call_error = None
        pair_attempts = []
        request = rp
        if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
            from pathlib import Path
            from orchestrator.entity_identity import compare_bound_pair
            bound_source = {**keyframes[i], 'img': images[i]}
            bound_target = {**keyframes[i + 1], 'img': images[i + 1]}
            check = compare_bound_pair(self, bound_source, bound_target,
                Path(debug_root) / f't3_identity_{base}_seg{seg_idx:03d}_p{i:02d}_{i+1:02d}')
            raw2 = check['raw_response']
            pair_attempts = check['attempts']
        for attempt in range(3 if structured_evidence_profile() else 1):
            if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
                break
            try:
                raw2 = audited_vision(self,
                    [images[i], images[i + 1]],
                    request,
                    max_new_tokens=t3_refine_max_new_tokens,
                )
            except Exception as e:
                print(f"[T3][WARN] Refine VLM failed at pair {i}->{i+1}: {type(e).__name__}")
                raw2 = ""
                call_error = e
            if not structured_evidence_profile():
                break
            check = validate_pair_observation(raw2, keyframes[i]["visual_state"],
                                             keyframes[i+1]["visual_state"], error=call_error)
            pair_attempts.append({"raw_response": raw2, "status": check["status"],
                                  "validation_error": check.get("reason") or check.get("error_type"),
                                  "contract_details": check.get("contract_details")})
            if check["status"] != "invalid" or attempt == 2:
                break
            request = rp + "\nThe previous reply failed the response contract: " + check["reason"] + ". Return the complete JSON; inspect both images again, preserve uncertainty, and do not invent edits or facts to repair a field.\nPREVIOUS REPLY:\n" + raw2
            if check.get("contract_details"):
                request += "\nFAILED COMPARISON BINDINGS:\n" + json.dumps(check["contract_details"], ensure_ascii=False)

        if paper_profile(self.config):
            check = check if current_prompt_profile() in {'paper_v14', 'paper_v15'} else (validate_pair_observation(raw2, keyframes[i]["visual_state"],
                         keyframes[i+1]["visual_state"], error=call_error)
                     if structured_evidence_profile() else validate_transition(raw2, error=call_error))
            if structured_evidence_profile():
                check["attempts"] = pair_attempts
            raw2s = check["text"]
            transition_checks.append(check)
        else:
            raw2s = _normalize_transition_str(raw2 if isinstance(raw2, str) else str(raw2))
            transition_checks.append({"status": "legacy", "raw_response": raw2})

        # debug dump per pair
        try:
            p2 = os.path.join(debug_root, f"t3_refine_{base}_seg{seg_idx:03d}_p{i:02d}_{i+1:02d}.txt")
            with open(p2, "w", encoding="utf-8") as f:
                f.write(raw2 if isinstance(raw2, str) else str(raw2))
        except Exception:
            pass

        if not paper_profile(self.config) and raw2s and _looks_like_structured_transition(raw2s):
            raw2s = _atomicize_transition(raw2s)

        if raw2s and (not _looks_like_structured_transition(raw2s)):
            raw2s = ""

        refined_transitions.append(raw2s)

        cinf = infer_transition_cause(check) if paper_profile(self.config) else _infer_cause_from_transition(raw2s)
        if paper_profile(self.config) and check["status"] == "no_change":
            cinf = "none"
        if cinf not in _cause_enum:
            cinf = "unknown"
        refined_causes.append(cinf or "unknown")

    # 鐢?refined 瑕嗙洊 frames_info锛堝彧瑕嗙洊 transition_to_next + cause锛?
    for i in range(num_frames):
        if i == num_frames - 1:
            frames_info[i]["transition_to_next"] = ""
            frames_info[i]["cause"] = "none"
        else:
            frames_info[i]["transition_to_next"] = refined_transitions[i] if i < len(refined_transitions) else ""
            frames_info[i]["cause"] = refined_causes[i] if i < len(refined_causes) else "unknown"

    # ---- bind final transition/cause back to keyframes ----
    frames_out: List[Dict[str, Any]] = []
    for i, kf in enumerate(keyframes):
        info = frames_info[i] if i < len(frames_info) else {}
        caption = frame_captions[i] if i < len(frame_captions) else ""
        transition_to_next = info.get("transition_to_next", "") if isinstance(info, dict) else ""
        cause = info.get("cause", "") if isinstance(info, dict) else ""

        if i == num_frames - 1:
            transition_to_next = ""
            cause = "none"

        frames_out.append(
            {
                "img": kf.get("img"),
                "sec": kf.get("sec"),
                "caption": caption,
                **({"original_caption": kf.get("original_caption"), "caption_validation": kf.get("caption_validation"),
                    "transition_validation": transition_checks[i]} if paper_profile(self.config) else {}),
                **({"visual_state": kf["visual_state"]} if structured_evidence_profile() else {}),
                "transition_to_next": transition_to_next,
                "cause": (str(cause).strip().lower() if str(cause).strip().lower() in _cause_enum else ""),
                "origin_index": kf.get("origin_index", kf.get("frame_index", kf.get("index", i))),
                "shot_id": kf.get("shot_id", ""),
                "chapter_id": kf.get("chapter_id", ""),
            }
        )

    round1_segment_summary = ""

    def _fallback_summary(lines: List[str], limit: int = 3) -> str:
        uniq: List[str] = []
        seen = set()
        for line in lines:
            s = str(line or "").strip()
            if not s or s in seen:
                continue
            uniq.append(s)
            seen.add(s)
            if len(uniq) >= limit:
                break
        return " ".join(uniq)

    def _summarize_with_llm(level: str, lines: List[str], fallback: str = "") -> str:
        lines = [str(x or "").strip() for x in lines if str(x or "").strip()]
        if not lines:
            return fallback
        llm = getattr(self, "llm", None)
        if llm is None:
            return _fallback_summary(lines) or fallback
        prompt = render_prompt_text(
            "t3_local",
            "summary_by_level",
            level=level,
            level_title=level.title(),
            evidence_lines="\n".join(f"- {line}" for line in lines[:24]),
        )
        try:
            text = str(audited_text(llm, prompt, max_new_tokens=220) or "").strip()
        except Exception:
            if paper_profile(self.config):
                raise
            text = ""
        if paper_profile(self.config) and not text:
            raise ValueError("Paper summary generation returned empty response")
        return text or _fallback_summary(lines) or fallback

    shot_groups: Dict[str, List[Dict[str, Any]]] = {}
    shot_order: List[str] = []
    frame_positions_by_shot: Dict[str, List[int]] = {}
    for pos, fr in enumerate(frames_out):
        shot_id = str(fr.get("shot_id", "") or "")
        if shot_id not in shot_groups:
            shot_groups[shot_id] = []
            frame_positions_by_shot[shot_id] = []
            shot_order.append(shot_id)
        shot_groups[shot_id].append(fr)
        frame_positions_by_shot[shot_id].append(pos)

    shot_summaries: List[Dict[str, Any]] = []
    for shot_id in shot_order:
        shot_frames = shot_groups.get(shot_id, [])
        caption_lines = [str(fr.get("caption", "") or "").strip() for fr in shot_frames if str(fr.get("caption", "") or "").strip()]
        if paper_profile(self.config):
            for pos in frame_positions_by_shot.get(shot_id, []):
                caption_lines.extend(str(v) for v in keyframes[pos].get("semantic_profile", {}).get("tool_descs", {}).values())
                if structured_evidence_profile():
                    caption_lines.append(json.dumps(keyframes[pos]["visual_state"], ensure_ascii=False))
                    check = transition_checks[pos]
                    if check.get("status") in {"valid", "no_change"}:
                        caption_lines.append(check["text"])
        shot_summary = _summarize_with_llm("shot", caption_lines, fallback=_fallback_summary(caption_lines))
        shot_summaries.append(
            {
                "shot_id": shot_id,
                "chapter_id": str(shot_frames[0].get("chapter_id", "") or "") if shot_frames else "",
                "frame_positions": list(frame_positions_by_shot.get(shot_id, [])),
                "summary": shot_summary,
            }
        )

    chapter_groups: Dict[str, List[Dict[str, Any]]] = {}
    chapter_order: List[str] = []
    for shot_row in shot_summaries:
        chapter_id = str(shot_row.get("chapter_id", "") or "")
        if chapter_id not in chapter_groups:
            chapter_groups[chapter_id] = []
            chapter_order.append(chapter_id)
        chapter_groups[chapter_id].append(shot_row)

    chapter_summaries: List[Dict[str, Any]] = []
    for chapter_id in chapter_order:
        chapter_shots = chapter_groups.get(chapter_id, [])
        shot_lines = [str(row.get("summary", "") or "").strip() for row in chapter_shots if str(row.get("summary", "") or "").strip()]
        chapter_summary = _summarize_with_llm("chapter", shot_lines, fallback=_fallback_summary(shot_lines))
        chapter_summaries.append(
            {
                "chapter_id": chapter_id,
                "shot_ids": [str(row.get("shot_id", "") or "") for row in chapter_shots],
                "summary": chapter_summary,
            }
        )

    segment_summary = _summarize_with_llm(
        "segment",
        [str(row.get("summary", "") or "").strip() for row in chapter_summaries if str(row.get("summary", "") or "").strip()],
        fallback=_fallback_summary(
            [str(row.get("summary", "") or "").strip() for row in shot_summaries]
        ),
    )

    _print_T3_summary("ok", segment_summary, frames_out, dataset_type)

    # Close PIL handles to avoid accumulation on long videos.
    for im in images:
        try:
            im.close()
        except Exception:
            pass

    return {
        "output": {
            "segment_summary": segment_summary,
            "summary_scope": "segment",
            "signal_filter": signal_filter,
            "filtered_t2_output": t2_out if paper_profile(self.config) else None,
            "summary_hierarchy": {
                "shot_summaries": shot_summaries,
                "chapter_summaries": chapter_summaries,
                "segment_summary": segment_summary,
                "round1_segment_summary": round1_segment_summary,
            },
            "frames": frames_out,
            "dataset_type": dataset_type,
            "vlm_raw_path": debug_path,
        },
        "runtime": {
            "vlm_raw": vlm_raw,
            "vlm_raw_path": debug_path,
            "keyframes": keyframes,
            "video_type": video_type,
            "frame_captions": frame_captions,
            "refine_max_new_tokens": t3_refine_max_new_tokens,
            "round1_segment_summary": round1_segment_summary,
            "shot_summaries": shot_summaries,
            "chapter_summaries": chapter_summaries,
        },
    }
