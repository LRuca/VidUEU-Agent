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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from executor.tasks.prompt_loader import load_prompt_text, render_prompt_text, structured_evidence_profile, current_prompt_profile
from orchestrator.planner_helpers import parse_llm_json
from orchestrator.t4_evidence_pack import build_evidence_pack
from orchestrator.t4_generation_prompts import (
    build_dialog_evidence_pack,
    build_dialog_repair_prompt,
    build_fewshot_bank,
    build_instance_generation_prompt,
)
from orchestrator.t4_quota_decision import (
    apply_understanding_quota_decision_response,
    build_default_understanding_quota_decision,
    build_understanding_quota_decision_prompt,
)
from orchestrator.umm_task_producer import produce_task_samples
from align_unify_render_format import align_item
from orchestrator.metadata_admission import paper_profile, build_metadata, require_metadata_review_binding
from orchestrator.paper_validation import validate_refs, validate_single_image_item, validate_dialog, preserve_mmt_verifier_source
from orchestrator.paper_tasks import TASKS, route_tasks, label_paper_item
from orchestrator.umm_task_producer import (build_task_context, build_seed_candidates, build_weave_candidates,
    build_comm_c1_candidates, build_comm_c2_candidates, build_isg_candidates)
from orchestrator.iteration.labels import canonical_iteration_labels, propagate_iteration_labels

from orchestrator.prompt_audit import paper_prompt_stage, audited_text, audited_vision, AuditedTextClient

@paper_prompt_stage("T4")
def run_t4_structured(
    planner: Any,
    t1_stage: Dict[str, Any],
    t2_stage: Dict[str, Any],
    t3_stage: Dict[str, Any],
    segment: Any,
) -> Dict[str, Any]:
    self = planner
    is_paper = paper_profile(self.config)
    paper_rejections = []
    if getattr(planner, "model_manager", None) is not None:
        planner.model_manager.get_resource("llm")
    """
    T4 (NEW): Evidence -> Decide (LLM) -> Generate (LLM)
    - KEYFRAMES ONLY
    - T3.frames is already bound to T2.keyframes in temporal order
    - Align by POSITION (pos=0..N-1), NOT by path matching

    Key changes:
    1) MagicBrush candidate pair must be "small, controllable" edits:
    - we LLM-review each transition text + (caption_a, caption_b) to judge:
        viewpoint change too big? subject change too big? edits > 2? => reject.
    - we also ask the reviewer LLM to produce an instruction that covers ALL edits.
    2) Segment QA: always provide ALL keyframes to the generator (and attach them to item.meta),
    and require using segment-level background evidence (S_SEGEMENT_SUM) in evidence_refs.
    3) Still: no YOLO-based rule gating; YOLO is evidence only.
    4) YOLO evidence: top-K=5 must include bbox_01.
    5) Do NOT hard-filter endcard / continuity keywords; only use a soft preference via hints + prompt.
    HARD constraints remain: do not cross shot boundaries. Frame-level quality filtering is handled upstream in T2,
    so T4 only operates on the final keyframes selected by T2.

    [MODIFIED v2]
    - Allow subject change if (and only if) edit_count in {1,2}.
    - Only allow viewpoint change if View is none/minor. If View=minor, instruction MUST explicitly mention it.
    - Do NOT do fallback if LLM generation returns empty (MagicBrush may output 0).

    [MODIFIED v3]
    B) Robust edit_count: if edit_count != len(edits) and edits non-empty, override edit_count = len(edits).
    C) View=minor: NO hard filter "instruction must mention slight viewpoint"; keep it as prompt guidance only.

    [MODIFIED v4 - MINIMAL]
    - Batch review MagicBrush pairs in ONE LLM call to reduce latency/cost, preserving existing guards and cache format.

    [MODIFIED v5 - MINIMAL: VisDial/MMDU-style dialogs]
    - Add two new generation types:
        * dialog_visdial_like: single-image multi-turn dialog
        * dialog_mmdu_like: multi-image interleaved dialog (uses <ImageHere> placeholders)
    - Minimal integration:
        * extend candidate_types / quotas / parsing
        * add one dialog generator helper and two writers

    [MODIFIED v6 - MINIMAL: more dialogs + faster acceptance]
    - dialog rounds default = 3 (config can override)
    - visdial uses top3 anchors (diversified)
    - mmdu uses 3 groups pos_list
    - if turns insufficient -> 1-step LLM repair (append/make exact turns)
    - soften dialog "coherence" rules: keep only hard (turn count + grounded + no timeline)
    """
    import os
    import re
    import json
    from typing import Any, Dict, List, Optional, Tuple

    # --- YOLO evidence filtering (for evidence shaping only; not a frame-level quality gate) ---
    YOLO_MIN_CONF = 0.3
    
    # ---------------- MMT-bench category closed-set (EN only) ----------------
    # Scheme-1 (drop): generator must pick from closed-set; verifier drops out-of-set.
    MMT_ALLOWED_CATEGORIES = [
        "object",        # object existence/identity/part
        "attribute",     # color/material/shape/state
        "spatial",       # relative position / direction / layout
        "counting",      # number of objects
        "action",        # human/agent action / interaction
        "scene",         # place / environment / weather / time-of-day
        "text_ocr",      # visible text / signs / screens
        "logic",         # simple logic / comparison / constraint satisfaction
        "knowledge",     # common knowledge lightly grounded by image
        "other",         # fallback (still in closed-set)
    ]

    MMT_ALLOWED_L2 = [
        # object
        "object_existence", "object_identity", "object_part",
        # attribute
        "attribute_color", "attribute_shape", "attribute_material", "attribute_state",
        # spatial
        "spatial_left_right", "spatial_front_back", "spatial_in_on_under", "spatial_distance", "spatial_layout",
        # counting
        "counting_objects",
        # action
        "action_human", "action_interaction",
        # scene
        "scene_location", "scene_weather", "scene_time",
        # text/ocr
        "text_reading", "text_symbol",
        # logic / knowledge
        "logic_compare", "logic_condition",
        "knowledge_common",
        # fallback
        "other",
    ]

    # ---------------- MME category closed-set (EN only) ----------------
    # Key goal: prevent degeneration to "presence" by forcing closed-set choice + presence cap.
    MME_ALLOWED_CATEGORIES = [
        "presence",    # existence / yes-no
        "attribute",   # color/material/shape/state
        "counting",    # number of objects (ONLY when clearly countable)
        "spatial",     # relative position / layout (screen-frame only)
        "action",      # action / interaction / state change (single image)
        "scene",       # place/environment/time-of-day/weather (image-grounded)
        "text_ocr",    # visible large text/sign/screen (only if clearly readable)
        "logic",       # simple logic/compare within the image
        "other",       # fallback within closed-set
    ]
    _mme_cat_set = set([str(x).strip().lower() for x in MME_ALLOWED_CATEGORIES])



    # ---------------- 0) Collect context ----------------
    t1_out = t1_stage.get("output", {}) or {}
    t2_out = (t2_stage.get("output", {}) or {}) if isinstance(t2_stage, dict) else {}
    t3_out = (t3_stage.get("output", {}) or {}) if isinstance(t3_stage, dict) else {}

    if paper_profile(self.config) and isinstance(t3_out.get("filtered_t2_output"), dict):
        t2_out = t3_out["filtered_t2_output"]
    if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        gate = t3_out.get("gate_c") or {}
        if gate.get("decision") != "accept":
            raise ValueError("T4 requires a current accepted content-bound Gate C")
        metadata = build_metadata(t1_out, t2_out, t3_out, segment)
        require_metadata_review_binding(gate, metadata, current_prompt_profile(), getattr(getattr(self, "vlm", None), "model", None))

    # T4 writes items to DatasetBuilder before the planner flattens benchmark
    # buckets.  Freeze the T1/Qwen canonical label envelope here as well so
    # data.jsonl and the later flattened exports cannot diverge.
    iteration_labels = canonical_iteration_labels(t1_out)

    original_video_path: str = str(getattr(segment, "video_path", "") or "")
    segment_path: str = str(getattr(segment, "segment_path", None) or original_video_path)
    video_path: str = original_video_path or segment_path

    video_id = self._video_id(video_path)
    segment_index = int(getattr(segment, "index", 0) or 0)

    video_type = t1_out.get("video_type", "other")
    global_summary = t1_out.get("global_summary", "") or ""
    modality_hint = t1_out.get("modality_hint", "mixed")
    motion_hint = t1_out.get("motion_hint", "medium_motion")

    segment_summary = t3_out.get("segment_summary", "") or ""
    summary_hierarchy = t3_out.get("summary_hierarchy", {}) or {}
    shot_summaries_t3 = summary_hierarchy.get("shot_summaries", []) or []
    chapter_summaries_t3 = summary_hierarchy.get("chapter_summaries", []) or []
    frames_t3 = t3_out.get("frames", []) or []  # [{img,sec,caption,transition_to_next}, ...]

    # T2 signals
    shot_boundaries = t2_out.get("shot_boundaries", []) or []
    transition_indices = t2_out.get("transition_indices", []) or []
    tools_used = t2_out.get("tools", []) or []
    tool_reasons = t2_out.get("tool_reasons", {}) or {}

    # keyframes from T2
    keyframes_t2 = t2_out.get("keyframes", []) or []
    t1_shots = t1_out.get("shot_profiles") or t1_out.get("shots") or []
    t1_chapters = t1_out.get("chapters") or []

    # Auxiliary text evidence now comes from T2 tool-action outputs, not T0 caches.
    tool_execution = t2_out.get("tool_execution", {}) or {}
    semantic_by_shot = tool_execution.get("semantic_by_shot", {}) or {}
    t2_semantic_profiles = t2_out.get("semantic_profiles", []) or []
    t2_value_profiles = t2_out.get("value_profiles", []) or []

    def _safe_int(v: Any, default: int = -1) -> int:
        try:
            return int(v)
        except Exception:
            return default

    def _frame_keys_from_item(item: Dict[str, Any]) -> List[Tuple[str, int]]:
        shot_id = str(item.get("shot_id", "") or "").strip()
        out: List[Tuple[str, int]] = []
        for key in ("origin_index", "frame_index", "index"):
            idx = _safe_int(item.get(key), -1)
            if idx < 0:
                continue
            pair = (shot_id, idx)
            if pair not in out:
                out.append(pair)
        return out

    t2_frame_semantic_map: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in t2_semantic_profiles + t2_value_profiles:
        if not isinstance(row, dict):
            continue
        for pair in _frame_keys_from_item(row):
            t2_frame_semantic_map[pair] = row

    def _tool_evidence_suffix(tool_name: str) -> str:
        name = str(tool_name or "").strip().upper()
        name = re.sub(r"[^A-Z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "UNKNOWN"

    def _collect_tool_summary(tool_name: str) -> str:
        snippets: List[str] = []
        seen = set()
        for shot_payload in semantic_by_shot.values():
            summary = str((((shot_payload or {}).get(tool_name, {}) or {}).get("summary", "")) or "").strip()
            if not summary:
                continue
            if summary in seen:
                continue
            seen.add(summary)
            snippets.append(summary)
        return "\n".join(snippets).strip()

    tool_summary_by_name: Dict[str, str] = {}
    for shot_payload in semantic_by_shot.values():
        if not isinstance(shot_payload, dict):
            continue
        for tool_name in shot_payload.keys():
            tool_key = str(tool_name or "").strip()
            if not tool_key or tool_key in tool_summary_by_name:
                continue
            tool_summary_by_name[tool_key] = _collect_tool_summary(tool_key)
    tool_summary_by_name = {k: v for k, v in tool_summary_by_name.items() if str(v or "").strip()}

    asr_text = _collect_tool_summary("asr")
    ocr_text = _collect_tool_summary("ocr")
    use_asr_for_vlm = bool(asr_text)
    use_ocr_for_vlm = bool(ocr_text)

    # ---------------- 0.1) Config ----------------
    t4_conf = (self.config.get("tasks", {}) or {}).get("T4", {}) or {}
    max_llava = int(t4_conf.get("max_llava", 2))
    max_magicbrush = int(t4_conf.get("max_edit", t4_conf.get("max_magicbrush", 3)) or 3)
    max_mmt = int(t4_conf.get("max_mmt", 4))
    max_mme = int(t4_conf.get("max_mme", 10))
    max_unify = int(t4_conf.get("max_unify", 1))
    # presence cap: keep as fallback, not default
    mme_presence_cap = float(t4_conf.get("mme_presence_cap", 0.40) or 0.40)  # default 40%
    # NEW: only verify mmt (batch) to reduce ambiguity without verifying others
    enable_mmt_verification = bool(t4_conf.get("enable_mmt_verification", True))
    max_mmt_verify_batch = int(t4_conf.get("max_mmt_verify_batch", 12))


    # NEW: dialog knobs (safe defaults)
    max_dialog_visdial = int(t4_conf.get("max_dialog_visdial", 8))
    max_dialog_mmdu = int(t4_conf.get("max_dialog_mmdu", 8))

    enable_llm_decision = bool(t4_conf.get("enable_llm_decision", True))
    enable_llm_generation = bool(t4_conf.get("enable_llm_generation", True))
    enable_llm_magicbrush_instr = bool(t4_conf.get("enable_llm_edit_instruction", t4_conf.get("enable_llm_magicbrush_instruction", True)))
    only_unify = bool(t4_conf.get("only_unify", False))

    # new knobs (safe defaults)
    max_pair_review = int(t4_conf.get("max_edit_pair_review", t4_conf.get("max_magicbrush_pair_review", 12)) or 12)
    attach_all_keyframes_to_meta = bool(t4_conf.get("attach_all_keyframes_to_meta", True))
    max_generation = int(t4_conf.get("max_generation", 0))
    enable_generation_routing = bool(t4_conf.get("enable_generation_routing", False))
    generation_vlm_max_new_tokens = int(t4_conf.get("generation_vlm_max_new_tokens", 900) or 900)
    generation_ground_truth_dirname = str(
        t4_conf.get("generation_ground_truth_dirname", "generation_ground_truth") or "generation_ground_truth"
    ).strip() or "generation_ground_truth"

    # NEW: dialog formatting knobs (safe defaults)
    # [MODIFIED v6] default rounds = 3 (you asked)
    dialog_visdial_rounds = int(t4_conf.get("dialog_visdial_rounds", 3) or 3)  # number of QA pairs
    dialog_mmdu_rounds = int(t4_conf.get("dialog_mmdu_rounds", 3) or 3)        # number of QA pairs
    dialog_mmdu_image_count = int(t4_conf.get("dialog_mmdu_image_count", 3) or 3)
    dialog_mmdu_use_new_image_turn = bool(t4_conf.get("dialog_mmdu_use_new_image_turn", True))
    unify_comm_c1_k = int(t4_conf.get("unify_comm_c1_k", 3) or 3)
    unify_isg_k = int(t4_conf.get("unify_isg_k", 2) or 2)
    unify_llm_retry_attempts = int(t4_conf.get("unify_llm_retry_attempts", 1) or 1)

    # debug dump
    debug_root = self.config.get("paths", {}).get("debug_dir", "logs/debug")
    try:
        os.makedirs(debug_root, exist_ok=True)
    except Exception:
        pass

    def _dump_text(tag: str, text: str) -> Optional[str]:
        try:
            base = os.path.splitext(os.path.basename(segment_path or video_path or "video"))[0]
            fp = os.path.join(debug_root, f"t4_{tag}_{base}_seg{segment_index:03d}.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(text if isinstance(text, str) else str(text))
            return fp
        except Exception:
            return None

    def _dump_json(tag: str, obj: Any) -> Optional[str]:
        try:
            base = os.path.splitext(os.path.basename(segment_path or video_path or "video"))[0]
            fp = os.path.join(debug_root, f"t4_{tag}_{base}_seg{segment_index:03d}.json")
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
            return fp
        except Exception:
            return None

    # ===================== [PATCH 4.5] dialog drop debug counters (per segment) =====================
    dialog_debug = {
        "dialog_visdial_like": {
            "attempt": 0,
            "accept": 0,
            "drop_timeline": 0,
            "drop_turn_count": 0,
            "drop_alternation": 0,
            "drop_grounding": 0,
        },
        "dialog_mmdu_like": {
            "attempt": 0,
            "accept": 0,
            "drop_timeline": 0,
            "drop_turn_count": 0,
            "drop_alternation": 0,
            "drop_grounding": 0,
        },
    }
    # ===================== [END PATCH 4.5] =====================

    def _write_item_to_builder(item: Dict[str, Any]) -> None:
        enriched = propagate_iteration_labels(item, iteration_labels, provenance="T1_Global_Perception")
        item.clear()
        item.update(enriched)
        # NEW: convert path fields to relative (to source_root) for portability
        try:
            self._relativize_item_paths_inplace(item)
        except Exception:
            pass

        try:
            if hasattr(self.dataset_builder, "add_item"):
                self.dataset_builder.add_item(item)
            elif hasattr(self.dataset_builder, "append"):
                self.dataset_builder.append(item)
            elif hasattr(self.dataset_builder, "write"):
                self.dataset_builder.write(item)
        except Exception as e:
            raise RuntimeError("dataset_write_failed") from e

     # --- [NEW] clamp helper for soft scaling ---
    def _clamp(x: float, lo: float, hi: float) -> float:
        try:
            xf = float(x)
        except Exception:
            xf = 0.0
        if xf < lo:
            return lo
        if xf > hi:
            return hi
        return xf

    # ---------------- 0.1.1) Global benchmark mix controller (across all videos) ----------------
    mix = (t4_conf.get("global_mix") or {}) if isinstance(t4_conf.get("global_mix"), dict) else {}
    mix_warmup = int(mix.get("warmup_items", 40) or 40)
    single_cap = float(mix.get("single_cap", 0.20) or 0.20)
    edit_cap = float(mix.get("edit_cap", 0.30) or 0.30)
    dialog_min = float(mix.get("dialog_min", 0.40) or 0.40)
    dialog_max = float(mix.get("dialog_max", 0.50) or 0.50)

    gcounts = self.get_global_bench_counts_snapshot() if hasattr(self, "get_global_bench_counts_snapshot") else (getattr(self, "_global_bench_counts", {}) or {})
    try:
        g_single = int(gcounts.get("llava_bench", 0) or 0) + int(gcounts.get("mmt_bench", 0) or 0) + int(gcounts.get("mme", 0) or 0)
        g_edit = int(gcounts.get("edit", gcounts.get("magicbrush", 0)) or 0)
        g_dialog = int(gcounts.get("dialog_visdial_like", 0) or 0) + int(gcounts.get("dialog_mmdu_like", 0) or 0)
        g_unify = int(gcounts.get("unify", 0) or 0)
        g_total = int(g_single + g_edit + g_dialog + g_unify)
    except Exception:
        g_single = g_edit = g_dialog = g_total = 0

    cur_single_r = (g_single / g_total) if g_total > 0 else 0.0
    cur_edit_r = (g_edit / g_total) if g_total > 0 else 0.0
    cur_dialog_r = (g_dialog / g_total) if g_total > 0 else 0.0

    # if g_total >= mix_warmup:
    #     if cur_single_r >= single_cap:
    #         max_llava = 0
    #         max_mmt = 0
    #         max_mme = 0
    #     if cur_edit_r >= edit_cap:
    #         max_magicbrush = 0

    #     if cur_dialog_r < dialog_min:
    #         max_dialog_visdial = max(max_dialog_visdial, 1)
    #         max_dialog_mmdu = max(max_dialog_mmdu, 1)

    #     if cur_dialog_r > (dialog_max + 0.05):
    #         max_dialog_visdial = 0
    #         max_dialog_mmdu = 0
    # --- [MODIFIED] DO NOT hard-zero max_*; compute soft scaling factors instead ---
    single_scale = 1.0
    edit_scale = 1.0
    dialog_scale_up = 1.0
    dialog_scale_down = 1.0
    if g_total >= mix_warmup:
        if cur_single_r > 1e-6:
            single_scale = _clamp(single_cap / cur_single_r, 0.25, 1.0)
        if cur_edit_r > 1e-6:
            edit_scale = _clamp(edit_cap / cur_edit_r, 0.25, 1.0)
        if cur_dialog_r > 1e-6:
            dialog_scale_up = _clamp(dialog_min / cur_dialog_r, 1.0, 2.0)
            dialog_scale_down = _clamp(dialog_max / cur_dialog_r, 0.25, 1.0)
        else:
            # no dialog produced yet -> allow scale-up to max
            dialog_scale_up = 2.0
            dialog_scale_down = 1.0

    # global_mix_note = (
    #     f"GLOBAL_MIX: total={g_total}, single={g_single}({cur_single_r:.1%}), "
    #     f"edit={g_edit}({cur_edit_r:.1%}), dialog={g_dialog}({cur_dialog_r:.1%}) "
    #     f"| targets: single<= {single_cap:.0%}, edit<= {edit_cap:.0%}, dialog~{dialog_min:.0%}-{dialog_max:.0%}"
    # )
    global_mix_note = (
        f"GLOBAL_MIX: total={g_total}, single={g_single}({cur_single_r:.1%}), "
        f"edit={g_edit}({cur_edit_r:.1%}), dialog={g_dialog}({cur_dialog_r:.1%}) "
        f"| targets: single<= {single_cap:.0%}, edit<= {edit_cap:.0%}, dialog~{dialog_min:.0%}-{dialog_max:.0%} "
        f"| scales: single_scale={single_scale:.2f}, edit_scale={edit_scale:.2f}, "
        f"dialog_up={dialog_scale_up:.2f}, dialog_down={dialog_scale_down:.2f}"
    )

    # ---------------- 0.2) Text cleaning helpers ----------------
    def _clean_lines(text: str, max_chars: int, kind: str) -> str:
        if not text:
            return ""
        t = str(text)
        t = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", t)
        t = re.sub(r"[ \t]+", " ", t)
        lines = re.split(r"[\r\n]+", t)

        out, seen = [], set()
        for ln in lines:
            ln = (ln or "").strip()
            if not ln:
                continue
            if kind == "OCR":
                if len(ln) <= 1:
                    continue
                bad_ratio = sum(
                    1 for ch in ln
                    if not (ch.isalnum() or ("\u4e00" <= ch <= "\u9fff"))
                ) / max(1, len(ln))
                if bad_ratio > 0.6:
                    continue
            if ln in seen:
                continue
            seen.add(ln)
            out.append(ln)

        s = "\n".join(out).strip()
        if len(s) > max_chars:
            s = s[:max_chars].rstrip() + " ..."
        return s

    asr_snippet = _clean_lines(asr_text, 1200, "ASR") if use_asr_for_vlm else ""
    ocr_snippet = _clean_lines(ocr_text, 800, "OCR") if use_ocr_for_vlm else ""

    # ---------------- 1) Align KEYFRAMES (pos-aligned T3<->T2) ----------------
    n = min(len(frames_t3), len(keyframes_t2)) if keyframes_t2 else len(frames_t3)
    if n <= 0:
        evidence = {
            "video_id": video_id,
            "segment_index": segment_index,
            "video_type": video_type,
            "modality_hint": modality_hint,
            "motion_hint": motion_hint,
            "global_summary": global_summary,
            "segment_summary": segment_summary,
            "frames": [],
            "evidence_units": [],
            "evidence_pack_text": "",
            "tools_used": tools_used,
            "tool_reasons": tool_reasons,
            "structure": {
                "shot_boundaries": shot_boundaries,
                "transition_indices": transition_indices,
            },
        }
        return {"evidence": evidence, "decision": {"selected": []}, "instances": {}, "runtime": {}}

    shot_to_chapter: Dict[str, str] = {}
    origin_to_shot: Dict[int, str] = {}
    for shot in t1_shots:
        if not isinstance(shot, dict):
            continue
        shot_id = str(shot.get("shot_id", "") or "")
        meta = shot.get("metadata", {}) or {}
        if shot_id and meta.get("chapter_id"):
            shot_to_chapter[shot_id] = str(meta.get("chapter_id") or "")
        for idx in shot.get("frame_indices", []) or []:
            try:
                origin_to_shot[int(idx)] = shot_id
            except Exception:
                continue
    for chapter in t1_chapters:
        if not isinstance(chapter, dict):
            continue
        chapter_id = str(chapter.get("chapter_id", "") or "")
        for shot_id in chapter.get("shot_ids", []) or []:
            if shot_id and chapter_id and shot_id not in shot_to_chapter:
                shot_to_chapter[str(shot_id)] = chapter_id

    def _kf_origin_index(kf: Dict[str, Any], fr3: Dict[str, Any], pos: int) -> int:
        for key in ("index", "frame_index", "kf_index", "keyframe_index"):
            if isinstance(kf, dict) and kf.get(key) is not None:
                try:
                    return int(kf.get(key))
                except Exception:
                    pass
        for key in ("origin_index", "frame_index", "index"):
            if isinstance(fr3, dict) and fr3.get(key) is not None:
                try:
                    return int(fr3.get(key))
                except Exception:
                    pass
        return int(pos)

    def _resolve_shot_id(kf: Dict[str, Any], fr3: Dict[str, Any], origin_index: int) -> str:
        for src in (kf, fr3):
            if isinstance(src, dict):
                shot_id = str(src.get("shot_id", "") or "").strip()
                if shot_id:
                    return shot_id
        return str(origin_to_shot.get(int(origin_index), "") or "")

    def _resolve_chapter_id(kf: Dict[str, Any], fr3: Dict[str, Any], shot_id: str) -> str:
        for src in (kf, fr3):
            if isinstance(src, dict):
                chapter_id = str(src.get("chapter_id", "") or "").strip()
                if chapter_id:
                    return chapter_id
        return str(shot_to_chapter.get(str(shot_id), "") or "")

    frames_aligned: List[Dict[str, Any]] = []
    for pos in range(n):
        fr3 = frames_t3[pos] if pos < len(frames_t3) and isinstance(frames_t3[pos], dict) else {}
        kf2 = keyframes_t2[pos] if pos < len(keyframes_t2) and isinstance(keyframes_t2[pos], dict) else {}

        img = fr3.get("img") or kf2.get("img")
        sec = fr3.get("sec") if fr3.get("sec") is not None else kf2.get("sec")
        cap = (fr3.get("caption") or "").strip()
        trans = (fr3.get("transition_to_next") or "").strip()
        cause = (fr3.get("cause") or "").strip().lower()
        origin_index = _kf_origin_index(kf2, fr3, pos)
        shot_id = _resolve_shot_id(kf2, fr3, origin_index)
        chapter_id = _resolve_chapter_id(kf2, fr3, shot_id)
        shot_semantic = semantic_by_shot.get(shot_id, {}) if shot_id else {}
        frame_semantic_row = t2_frame_semantic_map.get((shot_id, origin_index), {}) if shot_id else {}
        frame_semantic_profile = dict((frame_semantic_row.get("semantic_profile", {}) or {})) if isinstance(frame_semantic_row, dict) else {}
        frame_tool_descs = dict(frame_semantic_profile.get("tool_descs", {}) or {})
        frame_semantic_text = str(frame_semantic_profile.get("semantic_text", "") or "") if frame_semantic_profile else ""


        frames_aligned.append(
            {
                "pos": pos,
                "frame_pos": pos,
                "segment_index": segment_index,
                "index": pos,
                "origin_index": origin_index,
                "img": img,
                "sec": sec,
                "shot_id": shot_id,
                "chapter_id": chapter_id,
                "caption": cap,
                "visual_state": fr3.get("visual_state"),
                "transition_validation": fr3.get("transition_validation"),
                "transition_to_next": trans,
                "score": kf2.get("score", None) if isinstance(kf2, dict) else None,
                "score_detail": kf2.get("score_detail", None) if isinstance(kf2, dict) else None,
                "cause": cause,
                "tool_semantic": shot_semantic if isinstance(shot_semantic, dict) else {},
                "frame_tool_descs": frame_tool_descs,
                "frame_semantic_text": frame_semantic_text,
            }
        )

    all_keyframes_meta = [{"pos": fr["pos"], "img": fr.get("img"), "sec": fr.get("sec")} for fr in frames_aligned]

    def _crosses_shot_boundary(i: int, j: int) -> bool:
        try:
            sbs = set(int(x) for x in (shot_boundaries or []))
        except Exception:
            sbs = set()
        if i < 0 or j < 0:
            return True
        if j <= i:
            return True
        for k in range(i, j):
            if k in sbs:
                return True
        return False

    def _same_shot_pair(i: int, j: int) -> bool:
        if i < 0 or j < 0 or i >= len(frames_aligned) or j >= len(frames_aligned):
            return False
        shot_i = str(frames_aligned[i].get("shot_id", "") or "").strip()
        shot_j = str(frames_aligned[j].get("shot_id", "") or "").strip()
        if shot_i and shot_j:
            return shot_i == shot_j
        return not _crosses_shot_boundary(i, j)

    # ---------------- 2) Build Evidence Units ----------------
    def _short(s: str, nmax: int) -> str:
        s = (s or "").strip()
        return s if len(s) <= nmax else (s[:nmax].rstrip() + " ...")

    def _looks_like_endcard(text: str) -> bool:
        t = (text or "").lower()
        if not t:
            return False
        kw = [
            "subscribe", "like and subscribe", "please like", "please subscribe",
            "thanks for watching", "thank you for watching", "end screen", "endcard",
            "follow", "notification", "bell icon", "turn on notifications",
            "support me", "patreon", "merch", "link in bio", "comment below",
        ]
        return any(k in t for k in kw)

    def _tokenize_simple(s: str) -> List[str]:
        s = (s or "").lower()
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s)
        toks = [x for x in s.split() if x]
        stop = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are"}
        return [t for t in toks if t not in stop]

    def _jaccard(a: str, b: str) -> float:
        ta = set(_tokenize_simple(a))
        tb = set(_tokenize_simple(b))
        if not ta and not tb:
            return 1.0
        if not ta or not tb:
            return 0.0
        return float(len(ta & tb)) / float(max(1, len(ta | tb)))

    def _parse_transition_struct(t: str) -> Dict[str, Any]:
        out = {"edit_count": None, "edits": [], "shot": "", "view": "", "subject": "", "uncertainty": ""}
        s = (t or "").strip()
        if not s:
            return out

        parts = [p.strip() for p in s.split("|") if p.strip()]
        for p in parts:
            m = re.search(r"Edits\((\d+)\)\s*:\s*(.*)$", p, flags=re.I)
            if m:
                try:
                    out["edit_count"] = int(m.group(1))
                except Exception:
                    out["edit_count"] = None
                rest = (m.group(2) or "").strip()
                rest_norm = re.sub(r"^\d+\)\s*", "", rest).strip(" .;").lower()
                if rest_norm in {"", "none", "(none)", "no change", "no changes"}:
                    out["edit_count"] = 0
                    out["edits"] = []
                    continue
                if rest and "(none)" not in rest.lower():
                    cand = [x.strip() for x in re.split(r";\s*", rest) if x.strip()]
                    edits = []
                    for c in cand:
                        c = re.sub(r"^\d+\)\s*", "", c).strip()
                        c_norm = c.strip(" .;").lower()
                        if c and c_norm not in {"none", "(none)", "no change", "no changes"}:
                            edits.append(c)
                    out["edits"] = edits[:6]
                continue

            m = re.search(r"^Shot:\s*(.*)$", p, flags=re.I)
            if m:
                out["shot"] = (m.group(1) or "").strip().lower()
                continue
            m = re.search(r"^View:\s*(.*)$", p, flags=re.I)
            if m:
                out["view"] = (m.group(1) or "").strip().lower()
                continue
            m = re.search(r"^Subject:\s*(.*)$", p, flags=re.I)
            if m:
                out["subject"] = (m.group(1) or "").strip().lower()
                continue
            m = re.search(r"^Uncertainty:\s*(.*)$", p, flags=re.I)
            if m:
                out["uncertainty"] = (m.group(1) or "").strip()
                continue

        try:
            ec = out.get("edit_count", None)
            edits = out.get("edits", [])
            if ec is not None and isinstance(edits, list) and edits and ec != len(edits):
                out["edit_count"] = len(edits)
            if ec is not None and isinstance(edits, list) and not edits:
                out["edit_count"] = 0
        except Exception:
            pass

        return out

    def _none_like_t3_edit(value: Any) -> bool:
        text = str(value or "").strip().strip(" .;").lower()
        return text in {"", "none", "(none)", "no change", "no changes", "no semantic change", "no semantic changes"}

    def _has_real_t3_edit(ts: Dict[str, Any]) -> bool:
        if not isinstance(ts, dict):
            return False
        ec = ts.get("edit_count", None)
        edits = ts.get("edits", [])
        if ec in (None, 0):
            return False
        if not isinstance(edits, list) or not edits:
            return False
        return any(not _none_like_t3_edit(edit) for edit in edits)

    ocr_lines: List[str] = []
    if ocr_snippet:
        for ln in re.split(r"[\r\n]+", ocr_snippet):
            ln = (ln or "").strip()
            if not ln:
                continue
            if len(ln) < 2:
                continue
            ocr_lines.append(ln)
            if len(ocr_lines) >= 12:
                break

    changes: List[Dict[str, Any]] = []
    change_map_struct: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for pos in range(len(frames_aligned) - 1):
        fr = frames_aligned[pos]
        trans = (fr.get("transition_to_next") or "").strip()
        if not trans:
            continue
        parsed = _parse_transition_struct(trans)
        changes.append({"from": pos, "to": pos + 1, "text": trans, "parsed": parsed})
        change_map_struct[(pos, pos + 1)] = parsed

    # ---------------- 2.1) MagicBrush candidate pairs (HARD constraints only) ----------------
    raw_pairs: List[Tuple[int, int]] = []
    pair_hints: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for ch in changes:
        i_raw = ch.get("from", -1)
        j_raw = ch.get("to", -1)
        try:
            i = int(i_raw if i_raw is not None else -1)
        except Exception:
            i = -1
        try:
            j = int(j_raw if j_raw is not None else -1)
        except Exception:
            j = -1
        if i < 0 or j < 0:
            continue
        if i >= len(frames_aligned) or j >= len(frames_aligned):
            continue
        if not _same_shot_pair(i, j):
            continue

        ch_text = (ch.get("text", "") or "").strip()
        cap_i = (frames_aligned[i].get("caption", "") or "").strip()
        cap_j = (frames_aligned[j].get("caption", "") or "").strip()

        end_like = _looks_like_endcard(ch_text) or _looks_like_endcard(cap_i) or _looks_like_endcard(cap_j)
        if ocr_lines and _looks_like_endcard(" | ".join(ocr_lines[:6])) and len(cap_j) <= 40:
            end_like = True

        jac = _jaccard(cap_i, cap_j)

        ts = ch.get("parsed", {}) if isinstance(ch.get("parsed", {}), dict) else {}
        pair_hints[(i, j)] = {
            "endcard_like": bool(end_like),
            "caption_jaccard": float(jac),
            "t3_edit_count": ts.get("edit_count", None),
            "t3_edits": ts.get("edits", []) if isinstance(ts.get("edits", []), list) else [],
            "t3_shot": str(ts.get("shot", "") or ""),
            "t3_view": str(ts.get("view", "") or ""),
            "t3_subject": str(ts.get("subject", "") or ""),
            "t3_uncertainty": str(ts.get("uncertainty", "") or ""),
            "same_shot": True,
        }

        cause_i = (frames_aligned[i].get("cause") or "").strip().lower()
        pair_hints[(i, j)]["t3_cause"] = cause_i

        raw_pairs.append((i, j))

    def _pair_rank(p: Tuple[int, int]) -> Tuple:
        h = pair_hints.get(p, {}) or {}
        ec = h.get("t3_edit_count", None)
        shot = (h.get("t3_shot", "") or "").lower()
        view = (h.get("t3_view", "") or "").lower()
        subj = (h.get("t3_subject", "") or "").lower()
        end_like = bool(h.get("endcard_like", False))
        jac = float(h.get("caption_jaccard", 0.0) or 0.0)

        if ec == 1:
            ec_rank = 0
        elif ec == 2:
            ec_rank = 1
        elif ec is None:
            ec_rank = 2
        elif ec == 0:
            ec_rank = 3
        else:
            ec_rank = 4

        subj_rank = 0 if (subj in ("same", "", "unknown")) else (2 if subj == "different" else 1)

        shot_rank = 0
        if "scene_cut" in shot or "cut" in shot:
            shot_rank = 3
        elif shot in ("major",):
            shot_rank = 2
        elif shot in ("minor", "none", ""):
            shot_rank = 0
        else:
            shot_rank = 1

        if view == "none":
            view_rank = 0
        elif view == "minor":
            view_rank = 1
        elif view == "":
            view_rank = 2
        else:
            view_rank = 3

        return (subj_rank, shot_rank, view_rank, ec_rank, 0 if not end_like else 1, -jac)

    raw_pairs_sorted = sorted(raw_pairs, key=_pair_rank)
    pairs_for_review = raw_pairs_sorted[: max(1, max_pair_review)]

    def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = parse_llm_json(s)
        except Exception:
            obj = None
        return obj if isinstance(obj, dict) else None
    

    # ---------------- [NEW] MMT batch verifier (ONLY for mmt_bench) ----------------
    def _batch_verify_mmt_items(mmt_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Verify MMT single-choice questions to reduce ambiguity:
        - exactly ONE correct option
        - question unambiguous for the single image
        - options same granularity/type
        Batch in ONE LLM call per segment.
        Returns filtered (and optionally fixed) items.
        """
        if not enable_llm_generation:
            return mmt_items
        if not enable_mmt_verification:
            return mmt_items
        if not isinstance(mmt_items, list) or not mmt_items:
            return mmt_items

        # ===================== [PATCH] closed-set lookup for drop =====================
        _mmt_cat_set = set([str(x).strip().lower() for x in MMT_ALLOWED_CATEGORIES])
        _mmt_l2_set = set([str(x).strip().lower() for x in MMT_ALLOWED_L2])
        # ===================== [END PATCH] =====================

        # cap batch to control cost
        items_in = mmt_items[: max(1, max_mmt_verify_batch)]

        # build minimal payload for LLM
        payload = []
        for it in items_in:
            if not isinstance(it, dict):
                continue
            try:
                p = int(it.get("image_pos", best_pos))
            except Exception:
                p = best_pos
            p = max(0, min(p, len(frames_aligned) - 1))

            payload.append({
                "image_pos": p,
                "question": str(it.get("question", "") or "").strip(),
                "A": str(it.get("A", "") or "").strip(),
                "B": str(it.get("B", "") or "").strip(),
                "C": str(it.get("C", "") or "").strip(),
                "D": str(it.get("D", "") or "").strip(),
                **({"perception_evidence": it.get("perception_evidence")} if structured_evidence_profile() else {}),
                "_answer": str(it.get("_answer", "") or "").strip(),
                "category": str(it.get("category", "general") or "general"),
                "l2-category": str(it.get("l2-category", "general") or "general"),
                "evidence_refs": it.get("evidence_refs", []),
            })

        if not payload:
            return []

        verifier_evidence = evidence_pack_text
        if paper_profile(self.config):
            positions = {it["image_pos"] for it in items_in}
            allowed_prefixes = tuple([f"F{pos:02d}_" for pos in positions] + ["S_SEGEMENT_SUM"])
            verifier_evidence = "\n".join(f"[{u['evidence_id']}] {u.get('text', '')}" for u in evidence_units
                                          if str(u.get("evidence_id", "")).startswith(allowed_prefixes))
        verifier_prompt = render_prompt_text(
            "t4_structured",
            "mmt_verifier",
            mmt_allowed_categories_json=json.dumps(MMT_ALLOWED_CATEGORIES),
            mmt_allowed_l2_json=json.dumps(MMT_ALLOWED_L2),
            payload_json=json.dumps(payload, ensure_ascii=False),
            evidence_pack_text=verifier_evidence,
        )

        raw = audited_text(self.llm, verifier_prompt, max_new_tokens=1200)
        _dump_text("mmt_verify_raw", raw if isinstance(raw, str) else str(raw))
        obj = _safe_parse_json(raw) if isinstance(raw, str) else None
        results = obj.get("results") if isinstance(obj, dict) else None
        if not isinstance(results, list):
            # on failure: be conservative -> return original items (do not wipe)
            if paper_profile(self.config):
                paper_rejections.append({"stage": "mmt_verifier", "reason": "invalid_verifier_response", "raw_response": raw})
                return []
            return mmt_items

        # map results by index order to keep stable
        out_items: List[Dict[str, Any]] = []
        for idx, r in enumerate(results):
            if idx >= len(items_in):
                break
            src = items_in[idx]
            if not isinstance(r, dict):
                continue
            ok = bool(r.get("ok", False))
            if not ok:
                continue

            fixed = r.get("fixed_item")
            use_it = fixed if isinstance(fixed, dict) else src
            if paper_profile(self.config):
                try:
                    if use_it.get("question") != src.get("question") or use_it.get("image_pos") != src.get("image_pos"):
                        raise ValueError("verifier_changed_question_or_image")
                    if structured_evidence_profile():
                        use_it = preserve_mmt_verifier_source(src, use_it)
                    validate_single_image_item(use_it, "mmt_bench", _evidence_id_set, src["image_pos"])
                except ValueError as exc:
                    paper_rejections.append({"stage": "mmt_verifier", "reason": str(exc), "candidate": src, "verifier_result": r})
                    continue

            # minimal sanity checks
            q = str(use_it.get("question", "") or "").strip()
            A = str(use_it.get("A", "") or "").strip()
            B = str(use_it.get("B", "") or "").strip()
            C = str(use_it.get("C", "") or "").strip()
            D = str(use_it.get("D", "") or "").strip()
            ans = str(use_it.get("_answer", "") or "").strip()
            if not q or not (A and B and C and D) or ans not in ("A", "B", "C", "D"):
                continue

            try:
                p = int(use_it.get("image_pos", src.get("image_pos", best_pos)))
            except Exception:
                p = int(src.get("image_pos", best_pos) or best_pos)
            p = max(0, min(p, len(frames_aligned) - 1))
            use_it["image_pos"] = p

            # ===================== [PATCH] closed-set category drop (scheme-1) =====================
            cat = str(use_it.get("category", src.get("category", "general")) or "general").strip().lower()
            l2cat = str(use_it.get("l2-category", src.get("l2-category", "general")) or "general").strip().lower()

            # scheme-1 drop: category / l2-category must be in closed-set
            if cat not in _mmt_cat_set:
                continue
            if l2cat not in _mmt_l2_set:
                continue

            # normalize back
            use_it["category"] = cat
            use_it["l2-category"] = l2cat
            # ===================== [END PATCH] =====================


            evr = use_it.get("evidence_refs", src.get("evidence_refs", []))
            if not isinstance(evr, list):
                evr = []
            use_it["evidence_refs"] = _ensure_single_image_frame_ref(p, evr)

            out_items.append(use_it)

        # append any leftover original items beyond batch cap without verification (optional behavior)
        # Here we keep behavior conservative: verified subset replaces the first batch; rest stays as-is.
        if not paper_profile(self.config) and len(mmt_items) > len(items_in):
            out_items.extend(mmt_items[len(items_in):])

        return out_items


    review_cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
    reviewed_ok_pairs: List[Tuple[int, int]] = []

    def _batch_review_magicbrush_pairs(pairs: List[Tuple[int, int]]) -> None:
        to_llm: List[Dict[str, Any]] = []
        pair_order: List[Tuple[int, int]] = []

        for (a, b) in pairs:
            key = (int(a), int(b))
            if key in review_cache:
                continue

            ts = change_map_struct.get(key, {}) if isinstance(change_map_struct.get(key, {}), dict) else {}
            ts_shot = (ts.get("shot") or "").lower()
            ts_view = (ts.get("view") or "").lower()
            ts_subj = (ts.get("subject") or "").lower()
            ts_ec = ts.get("edit_count", None)

            if not _has_real_t3_edit(ts):
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "t3_no_semantic_edit"}
                continue
            if is_paper and ts_ec not in (1, 2):
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "too_many_observed_changes_for_local_edit"}
                continue
            if ts_view and ts_view not in ("none", "minor"):
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "t3_view_not_none"}
                continue
            if "scene_cut" in ts_shot or "cut" in ts_shot:
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "t3_struct_scene_cut"}
                continue
            if ts_subj == "different":
                if isinstance(ts_ec, int) and ts_ec not in (1, 2):
                    review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "t3_subject_diff_not_1to2_edits"}
                    continue

            fr_a = frames_aligned[a]
            fr_b = frames_aligned[b]
            cap_a = (fr_a.get("caption", "") or "").strip()
            cap_b = (fr_b.get("caption", "") or "").strip()
            ch_text = (fr_a.get("transition_to_next", "") or "").strip()
            hint = pair_hints.get(key, {}) or {}
            end_like = bool(hint.get("endcard_like", False))
            jac = float(hint.get("caption_jaccard", 0.0))
            cause = (hint.get("t3_cause", "") or "").lower()
            if cause == "camera_motion":
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "t3_cause_camera_motion"}
                continue

            to_llm.append({
                "from": a,
                "to": b,
                "pair_hints": {"caption_jaccard": jac, "endcard_like": end_like},
                "t3_struct": ts,
                "caption_a": _short(cap_a, 420),
                "caption_b": _short(cap_b, 420),
                "transition_text": _short(ch_text, 420),
                "t3_cause": str(hint.get("t3_cause", "") or ""),
            })
            pair_order.append(key)

        if not to_llm:
            return

        batch_prompt = render_prompt_text(
            "t4_structured",
            "magicbrush_pair_batch_review",
            segment_summary_short=_short(segment_summary, 360),
            input_pairs_json=json.dumps(to_llm, ensure_ascii=False),
        )


        raw = audited_text(self.llm, batch_prompt, max_new_tokens=1400)
        _dump_text("mb_review_batch_raw", raw if isinstance(raw, str) else str(raw))

        obj = _safe_parse_json(raw) if isinstance(raw, str) else None
        pairs_out = obj.get("pairs") if isinstance(obj, dict) else None
        if not isinstance(pairs_out, list):
            for key in pair_order:
                if key not in review_cache:
                    review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "parse_failed"}
            return

        out_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for it in pairs_out:
            if not isinstance(it, dict):
                continue
            try:
                a = int(it.get("from"))
                b = int(it.get("to"))
            except Exception:
                continue
            out_map[(a, b)] = it

        for key in pair_order:
            if key in review_cache:
                continue

            a, b = key
            it = out_map.get(key)
            if not isinstance(it, dict):
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "missing_in_batch_output"}
                continue

            ok = bool(it.get("ok", False))
            try:
                ec = int(it.get("edit_count", 0) or 0)
            except Exception:
                ec = 0
            edits = it.get("edits", [])
            if not isinstance(edits, list):
                edits = []
            edits = [str(x).strip() for x in edits if str(x).strip()]
            instr = str(it.get("instruction", "") or "").strip()
            reason = str(it.get("reason", "") or "").strip()

            if not ok:
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": reason[:320]}
                continue

            if edits and ec != len(edits):
                if is_paper:
                    ok = False
                else:
                    ec = len(edits)

            if ec not in (1, 2):
                ok = False
            if not instr:
                ok = False

            if ok:
                if not re.search(r"(?i)do not change anything else\.?$", instr.strip()):
                    ok = False

            ts = change_map_struct.get(key, {}) if isinstance(change_map_struct.get(key, {}), dict) else {}
            ts_shot = (ts.get("shot") or "").lower()
            ts_view = (ts.get("view") or "").lower()
            ts_subj = (ts.get("subject") or "").lower()
            ts_ec = ts.get("edit_count", None)

            if not _has_real_t3_edit(ts):
                ok = False
            if ts_view and ts_view not in ("none", "minor"):
                ok = False
            if "scene_cut" in ts_shot or "cut" in ts_shot:
                ok = False
            if ok and ts_subj == "different":
                if isinstance(ts_ec, int) and ts_ec not in (1, 2):
                    ok = False
                if ec not in (1, 2):
                    ok = False

            if len(instr) > 320 and not is_paper:
                instr = instr[:320]

            hint = pair_hints.get(key, {}) or {}
            if not ok:
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": "not_small_or_struct_break"}
                continue

            cause = (hint.get("t3_cause","") or "").lower()

            # keep strict rejects
            if cause in ("scene_cut", "occlusion", "camera_motion"):
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": f"t3_cause_{cause}"}
                continue

            review_cache[key] = {
                "ok": True,
                "edit_count": ec,
                "edits": edits[:2],
                "instruction": instr,
                "reason": reason[:480],
                "pair_hints": {
                    "caption_jaccard": float(hint.get("caption_jaccard", 0.0) or 0.0),
                    "endcard_like": bool(hint.get("endcard_like", False)),
                    "t3_struct": ts,
                    "t3_struct_edit_count": ts_ec,
                    "t3_struct_shot": (ts.get("shot") or "").lower(),
                    "t3_struct_view": (ts.get("view") or "").lower(),
                    "t3_struct_subject": (ts.get("subject") or "").lower(),
                    "t3_cause": str(hint.get("t3_cause", "") or ""),
                },
            }

    try:
        _batch_review_magicbrush_pairs(pairs_for_review)
    except Exception as e:
        for (a, b) in pairs_for_review:
            key = (int(a), int(b))
            if key not in review_cache:
                review_cache[key] = {"ok": False, "edit_count": 0, "edits": [], "instruction": "", "reason": f"batch_exception:{str(e)[:120]}"}

    for (a, b) in pairs_for_review:
        key = (int(a), int(b))
        rv = review_cache.get(key, {})
        if isinstance(rv, dict) and rv.get("ok", False):
            reviewed_ok_pairs.append(key)

    candidate_pairs: List[Tuple[int, int]] = reviewed_ok_pairs[:]

    # ---------------- 2.4) Evidence units ----------------
    evidence_units: List[Dict[str, Any]] = []

    def _add_ev(
        eid: str,
        level: str,
        text: str,
        source: str,
        frame_pos: Optional[int] = None,
        *,
        signal_type: str = "",
        shot_id: Optional[str] = None,
        chapter_id: Optional[str] = None,
        origin_index: Optional[int] = None,
        pair_from_pos: Optional[int] = None,
        pair_to_pos: Optional[int] = None,
        pair_key: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        frame_data = None
        if frame_pos is not None:
            try:
                idx = int(frame_pos)
            except Exception:
                idx = -1
            if 0 <= idx < len(frames_aligned):
                frame_data = frames_aligned[idx]
        final_shot_id = shot_id if shot_id is not None else (frame_data.get("shot_id") if isinstance(frame_data, dict) else None)
        final_chapter_id = chapter_id if chapter_id is not None else (frame_data.get("chapter_id") if isinstance(frame_data, dict) else None)
        final_origin_index = origin_index if origin_index is not None else (frame_data.get("origin_index") if isinstance(frame_data, dict) else None)
        evidence_units.append(
            {
                "evidence_id": eid,
                "level": level,
                "source": source,
                "signal_type": signal_type or source,
                "segment_index": segment_index,
                "frame_pos": frame_pos,
                "origin_index": final_origin_index,
                "shot_id": final_shot_id,
                "chapter_id": final_chapter_id,
                "pair_from_pos": pair_from_pos,
                "pair_to_pos": pair_to_pos,
                "pair_key": pair_key,
                "text": text,
                "payload": payload or {},
            }
        )

    evidence_pack = build_evidence_pack(
        video_id=video_id,
        segment_index=segment_index,
        video_type=video_type,
        modality_hint=modality_hint,
        motion_hint=motion_hint,
        global_summary=global_summary,
        segment_summary=segment_summary,
        shot_summaries_t3=shot_summaries_t3,
        chapter_summaries_t3=chapter_summaries_t3,
        frames_aligned=frames_aligned,
        all_keyframes_meta=all_keyframes_meta,
        evidence_units=evidence_units,
        tools_used=tools_used,
        tool_reasons=tool_reasons,
        shot_boundaries=shot_boundaries,
        transition_indices=transition_indices,
        raw_pairs_sorted=raw_pairs_sorted,
        pair_hints=pair_hints,
        change_map_struct=change_map_struct,
        reviewed_ok_pairs=reviewed_ok_pairs,
        review_cache=review_cache,
        candidate_pairs=candidate_pairs,
        add_ev=_add_ev,
        dump_text=_dump_text,
        short_text=_short,
        tool_evidence_suffix=_tool_evidence_suffix,
    )
    pair_records = evidence_pack["pair_records"]
    reviewed_pair_to_pid = evidence_pack["reviewed_pair_to_pid"]
    reviewed_pairs_text = evidence_pack["reviewed_pairs_text"]
    evidence_pack_text = evidence_pack["evidence_pack_text"]
    evidence = evidence_pack["evidence"]

    # ---------------- 3) Initialize decision state ----------------
    decision = {
        "edit_pairs": [],
    }

    llm_decision_raw = ""

    is_paper = paper_profile(self.config)
    if is_paper:
        t3_out["task_profile"] = "paper"
    unify_bundle = {
        "segment": {"index": segment_index, "video_id": video_id, "video_path": video_path, "segment_path": segment_path},
        "t1_out": t1_out, "t2_out": t2_out, "t3_out": t3_out,
        "frames_aligned": frames_aligned, "evidence": evidence, "evidence_units": evidence_units,
        "pair_records": pair_records,
        "candidate_pairs": [{"from": a, "to": b} for a, b in candidate_pairs],
        "reviewed_ok_pairs": [{"from": a, "to": b} for a, b in reviewed_ok_pairs],
    }
    if is_paper:
        if not enable_llm_decision:
            raise ValueError("Paper profile requires automatic LLM routing")
        ctx = build_task_context(unify_bundle)
        eligible = {
            "fine_grained_perception": bool(frames_aligned),
            "single_image_reasoning": bool(frames_aligned),
            "multi_image_logic": len(frames_aligned) >= 2,
            "localized_editing": bool(candidate_pairs),
            "grounded_edit_to_select": bool(build_seed_candidates(ctx)),
            "understand_recall_generate_select": bool(build_weave_candidates(ctx)),
            "stepwise_sequence": bool(build_comm_c1_candidates(ctx, k=unify_comm_c1_k)),
            "middle_step_gap": bool(build_comm_c2_candidates(ctx)),
            "instructional_future_sequence": bool(build_isg_candidates(ctx, k=unify_isg_k)),
        }
        limits = {name: max_unify for name in TASKS}
        limits.update(fine_grained_perception=max_mmt, single_image_reasoning=max_llava,
                      multi_image_logic=max_dialog_mmdu, localized_editing=max_magicbrush)
        iteration_policy = ((self.config.get("runtime", {}) or {}).get("iteration_policy") or {})
        routing_bias = iteration_policy.get("routing_quota_bias", {}) if isinstance(iteration_policy, dict) else {}
        if isinstance(routing_bias, dict):
            for task_name, (_, benchmark_name) in TASKS.items():
                bias = routing_bias.get(task_name, routing_bias.get(benchmark_name, 1.0))
                try:
                    bias_value = max(0.0, min(4.0, float(bias)))
                except (TypeError, ValueError):
                    bias_value = 1.0
                limits[task_name] = max(0, int(round(limits.get(task_name, 0) * bias_value)))
        enabled_capabilities = self.config.get("runtime", {}).get("capabilities", ["understanding", "editing", "unified"])
        for task_name, (capability, _) in TASKS.items():
            if capability not in enabled_capabilities:
                limits[task_name] = 0
                eligible[task_name] = False
        try:
            routed = route_tasks(self.llm, evidence_text=evidence_pack_text,
                                 evidence_ids={u["evidence_id"] for u in evidence_units},
                                 limits=limits, eligible=eligible, edit_pairs=candidate_pairs,
                                 evidence_units=evidence_units, frames=frames_aligned)
        except ValueError as exc:
            _dump_json("paper_routing_failed", {
                "error": str(exc), "attempts": getattr(exc, "routing_attempts", [])
            })
            raise
        decision.update(routed)
        if isinstance(iteration_policy, dict):
            decision["iteration_policy"] = {
                "iteration": iteration_policy.get("iteration"),
                "task_type_distribution": iteration_policy.get("task_type", {}),
                "routing_quota_bias": routing_bias,
            }
        llm_decision_raw = routed["raw"]
        selected_modules = [{"editing": "edit", "unified": "unify"}.get(c, c) for c in routed["capabilities"]]
        decision["selected"] = selected_modules
        decision["module_route_reason"] = "metadata_llm_automatic_routing"
        task_quotas = {TASKS[name][1]: row["quota"] for name, row in routed["tasks"].items()}
        understanding_quotas = {name: task_quotas.get(name, 0) for name in ["llava_bench", "mmt_bench", "dialog_mmdu_like"]}
        edit_quota, generation_quota = task_quotas.get("edit", 0), 0
        unify_quota = sum(task_quotas.get(bench, 0) for capability, bench in TASKS.values() if capability == "unified")
        _dump_json("paper_routing", decision)
    else:
        # ---- module-level routing ----
        # Stable routing contract for T4:
        # 1) generation is a first-class module in decision["selected"]. When enabled by config,
        #    it should always be selected here, then executed below via generation review ->
        #    prompt generation. It does not depend on candidate edit pairs.
        # 2) edit and unify are pair-dependent modules. They are enabled only when this segment
        #    has usable candidate edit pairs; otherwise both must be skipped together.
        # 3) understanding is not part of the final production route. Related quotas/config may
        #    still exist in legacy code paths, but T4 should not select understanding as an
        #    output-producing module in the current contract.
        available_edit_pair_count = len(candidate_pairs)
        generation_enabled = bool(enable_generation_routing and int(max_generation or 0) > 0)
        selected_modules: List[str] = ["generation"] if generation_enabled else []
        generation_quota = int(max_generation or 0) if generation_enabled else 0
        edit_quota = 0
        unify_quota = 0
        if available_edit_pair_count <= 0:
            decision["edit_pairs"] = []
            decision["module_route_reason"] = "generation_only_no_edit_pairs" if generation_enabled else "no_edit_pairs_skip_segment"
        else:
            selected_modules = ["edit", "unify"] + (["generation"] if generation_enabled else [])
            edit_quota = min(max_magicbrush, available_edit_pair_count)
            decision["edit_pairs"] = [{"from": a, "to": b} for (a, b) in candidate_pairs[: edit_quota]]
            if reviewed_ok_pairs:
                unify_quota = min(max_unify, len(reviewed_ok_pairs))
            decision["module_route_reason"] = "edit_pairs_available_with_generation" if generation_enabled else "edit_pairs_available"

        if only_unify:
            selected_modules = ["unify"] if unify_quota > 0 else []
            edit_quota = 0
            generation_quota = 0
            decision["edit_pairs"] = []
            decision["module_route_reason"] = "only_unify" if unify_quota > 0 else "only_unify_no_reviewed_pairs"

        decision["selected"] = selected_modules

        # Understanding quotas are computed only after module routing and only when
        # understanding is explicitly selected. This avoids quota/API cost for segments
        # whose active route is generation-only or edit/unify-only.
        if "understanding" in selected_modules:
            quota_setup = build_default_understanding_quota_decision(
                t4_conf=t4_conf,
                g_total=g_total,
                mix_warmup=mix_warmup,
                single_scale=single_scale,
                dialog_scale_up=dialog_scale_up,
                dialog_scale_down=dialog_scale_down,
                cur_dialog_r=cur_dialog_r,
                dialog_min=dialog_min,
                max_llava=max_llava,
                max_mmt=max_mmt,
                max_mme=max_mme,
                max_dialog_visdial=max_dialog_visdial,
                max_dialog_mmdu=max_dialog_mmdu,
            )
            candidate_types = quota_setup["candidate_types"]
            quota_limits = quota_setup["quota_limits"]
            default_quotas = quota_setup["default_quotas"]
            understanding_decision = quota_setup["decision"]
            understanding_quotas = dict(understanding_decision.get("quotas", {}) or {})
            if enable_llm_decision:
                decision_prompt = build_understanding_quota_decision_prompt(
                    candidate_types=candidate_types,
                    quota_limits=quota_limits,
                    evidence_pack_text=evidence_pack_text,
                    global_mix_note=global_mix_note,
                )
                llm_decision_raw = audited_text(self.llm, decision_prompt, max_new_tokens=1200)
                _dump_text("decision_raw", llm_decision_raw if isinstance(llm_decision_raw, str) else str(llm_decision_raw))
                parsed = _safe_parse_json(llm_decision_raw) if isinstance(llm_decision_raw, str) else None
                understanding_decision = apply_understanding_quota_decision_response(
                    parsed=parsed,
                    decision=understanding_decision,
                    candidate_types=candidate_types,
                    default_quotas=default_quotas,
                    max_llava=max_llava,
                    max_mmt=max_mmt,
                    max_mme=max_mme,
                    max_dialog_visdial=max_dialog_visdial,
                    max_dialog_mmdu=max_dialog_mmdu,
                    llm_decision_raw=llm_decision_raw,
                )
                understanding_quotas = dict(understanding_decision.get("quotas", {}) or {})
        else:
            understanding_quotas = {}


    # ---------------- 4) Generation ----------------
    instances: Dict[str, List[Dict[str, Any]]] = {
        "understanding.llava_bench": [],
        "edit.magicbrush": [],
        "understanding.mmt_bench": [],
        "understanding.mme": [],
        "understanding.dialog_visdial_like": [],
        "understanding.dialog_mmdu_like": [],
        "generation.geneval": [],
        "generation.wise": [],
        "unify.seed.l3.gts": [],
        "unify.weave.uie_select_recall": [],
        "unify.comm.task4.c1_steps": [],
        "unify.comm.task4.c2_fill_gap": [],
        "unify.isg.i2.seq_select": [],
    }
    selected_modules = set(decision.get("selected", []) or [])

    def _instance_key_for_item(item: Dict[str, Any]) -> str:
        module = str(item.get("module", "") or "").strip()
        benchmark = str(item.get("benchmark", "") or "").strip()
        if module == "edit":
            return "edit.magicbrush" if benchmark in {"", "edit", "magicbrush"} else f"edit.{benchmark}"
        if module == "generation":
            return f"generation.{benchmark or 'generation'}"
        if module == "understanding":
            return f"understanding.{benchmark or 'unknown'}"
        if module == "unify":
            return benchmark if benchmark.startswith("unify.") else f"unify.{benchmark or 'task'}"
        return benchmark or module or "unknown.unknown"

    def _append_instance(item: Dict[str, Any]) -> None:
        if is_paper:
            label_paper_item(item)
            item.setdefault("meta", {}).update(metadata_id=t3_out.get("metadata_id"), summary_scope="segment",
                metadata_content_sha256=(t3_out.get("gate_c") or {}).get("review_binding", {}).get("metadata_content_sha256"),
                gate_c=t3_out.get("gate_c"), routing_task=decision["tasks"].get(item["task_type"]),
                model_evidence_refs=list(item.get("evidence_refs", [])), citation_origin="model", training_approved=False)
        instances.setdefault(_instance_key_for_item(item), []).append(item)

    def _instance_count(key: str) -> int:
        return len(instances.get(key, []) or [])

    def _count_prefix(prefix: str) -> int:
        return sum(len(v or []) for k, v in instances.items() if str(k).startswith(prefix))

    def _pick_best_pos() -> int:
        best_pos = 0
        best_s = None
        for fr in frames_aligned:
            pos = int(fr.get("pos", 0) or 0)
            s = fr.get("score", None)
            if s is None:
                continue
            try:
                fs = float(s)
            except Exception:
                continue
            if (best_s is None) or (fs > best_s):
                best_s = fs
                best_pos = pos
        return best_pos

    best_pos = _pick_best_pos()
    best_frame = frames_aligned[best_pos] if 0 <= best_pos < len(frames_aligned) else (frames_aligned[0] if frames_aligned else None)

    # build evidence id set for safe injection
    _evidence_id_set = set()
    try:
        for ev in (evidence_units or []):
            if isinstance(ev, dict) and ev.get("evidence_id"):
                _evidence_id_set.add(str(ev["evidence_id"]))
    except Exception:
        _evidence_id_set = set()

    def _ensure_segment_refs(refs: List[Any]) -> List[str]:
        if is_paper:
            return list(refs) if isinstance(refs, list) else []
        rs = [str(x) for x in refs] if isinstance(refs, list) else []
        if "S_SEGEMENT_SUM" not in rs and "S_SEGEMENT_SUM" in _evidence_id_set:
            rs.append("S_SEGEMENT_SUM")
        out = []
        seen = set()
        for x in rs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out[:24]

    def _preferred_frame_ref_candidates(image_pos: int) -> List[str]:
        base = [f"F{int(image_pos):02d}_CAP"]
        tool_refs = sorted(
            x for x in _evidence_id_set
            if isinstance(x, str) and x.startswith(f"F{int(image_pos):02d}_TOOL_")
        )
        return base + tool_refs

    def _ensure_single_image_frame_ref(image_pos: int, refs: List[Any]) -> List[str]:
        if is_paper:
            return list(refs) if isinstance(refs, list) else []
        rs = _ensure_segment_refs(refs)
        try:
            pfx = f"F{int(image_pos):02d}_"
        except Exception:
            pfx = "F00_"

        has_frame_ref = any(isinstance(x, str) and x.startswith(pfx) for x in rs)
        if has_frame_ref:
            return rs

        cand = _preferred_frame_ref_candidates(int(image_pos))
        for c in cand:
            if c in _evidence_id_set and c not in rs:
                rs.append(c)
                break

        out = []
        seen = set()
        for x in rs:
            x = str(x)
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out[:24]

    def _ensure_any_of_frame_refs(image_pos_list: List[int], refs: List[Any]) -> List[str]:
        if is_paper:
            return list(refs) if isinstance(refs, list) else []
        rs = _ensure_segment_refs(refs)
        if any(isinstance(x, str) and re.match(r"^F\d{2}_", str(x)) for x in rs):
            return rs[:24]
        for p in image_pos_list:
            cand = _preferred_frame_ref_candidates(int(p))
            for c in cand:
                if c in _evidence_id_set and c not in rs:
                    rs.append(c)
                    return rs[:24]
        return rs[:24]

    def _generation_ground_truth_dir() -> str:
        output_file = str(getattr(getattr(self, "dataset_builder", None), "output_file", "") or "").strip()
        if output_file:
            return str(Path(output_file).resolve().parent / generation_ground_truth_dirname)
        output_dir = str((((self.config.get("paths", {}) or {}).get("output_dir")) or "")).strip()
        if output_dir:
            return str(Path(output_dir).resolve() / generation_ground_truth_dirname)
        return str(Path("dataset") / "output" / generation_ground_truth_dirname)

    def _copy_generation_ground_truth(src_path: str, image_pos: int, rank: int) -> str:
        src = str(src_path or "").strip()
        if not src:
            return ""
        src_obj = Path(src)
        ext = src_obj.suffix or ".jpg"
        dest_dir = Path(_generation_ground_truth_dir())
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{video_id}_seg{segment_index:03d}_gen_{int(rank):03d}_p{int(image_pos):02d}{ext}"
        dest = dest_dir / dest_name
        shutil.copy2(str(src_obj), str(dest))
        return str(dest)

    def _coerce_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        text = str(v or "").strip().lower()
        return text in {"1", "true", "yes", "y"}

    def _normalize_generation_review(parsed: Any, *, image_pos: int, raw_response: str) -> Dict[str, Any]:
        obj = parsed if isinstance(parsed, dict) else {}
        subject_clarity = str(obj.get("subject_clarity", "") or "").strip().lower()
        if subject_clarity not in {"high", "medium", "low"}:
            subject_clarity = "low"

        suitable = _coerce_bool(obj.get("suitable", False))
        composition_clean = _coerce_bool(obj.get("composition_clean", False))
        text_overlay_heavy = _coerce_bool(obj.get("text_overlay_heavy", False))
        background_clean = _coerce_bool(obj.get("background_clean", False))
        generation_reason = str(obj.get("generation_reason", "") or "").strip()
        benchmark_style = str(obj.get("benchmark_style", "") or "").strip().lower()
        generation_prompt = str(obj.get("generation_prompt", "") or "").strip()
        if benchmark_style not in {"geneval", "wise"}:
            benchmark_style = ""

        if suitable and (not generation_prompt or not benchmark_style):
            suitable = False
            if generation_reason:
                missing = []
                if not benchmark_style:
                    missing.append("benchmark_style")
                if not generation_prompt:
                    missing.append("generation_prompt")
                generation_reason += f" Missing {', '.join(missing)}."
            else:
                generation_reason = "Suitable flag was true but benchmark_style or generation_prompt was missing."

        if not suitable:
            benchmark_style = ""
            generation_prompt = ""
        if not generation_reason:
            generation_reason = "The model did not provide a valid suitability rationale."

        return {
            "image_pos": int(image_pos),
            "suitable": bool(suitable),
            "subject_clarity": subject_clarity,
            "composition_clean": bool(composition_clean),
            "text_overlay_heavy": bool(text_overlay_heavy),
            "background_clean": bool(background_clean),
            "generation_reason": generation_reason,
            "benchmark_style": benchmark_style,
            "generation_prompt": generation_prompt,
            "raw_response": raw_response,
            "parse_ok": isinstance(parsed, dict),
        }

    _single_image_bad_q = re.compile(
        r"(?i)\b(overall process|video segment|this segment|leading up|prior to|before|after|earlier|later|next|previous|step-by-step|sequence|timeline)\b"
    )

    generation_reviews: List[Dict[str, Any]] = []
    selected_generation_reviews: List[Dict[str, Any]] = []

    def _run_generation_routing() -> Dict[str, List[Dict[str, Any]]]:
        if not enable_generation_routing:
            return {"reviews": [], "selected": []}
        if max_generation <= 0:
            return {"reviews": [], "selected": []}

        jobs: List[Dict[str, Any]] = []
        for fr in frames_aligned:
            img = str(fr.get("img", "") or "").strip()
            if not img:
                continue
            jobs.append(
                {
                    "image_pos": int(fr.get("pos", 0) or 0),
                    "image_path": img,
                    "caption": str(fr.get("caption", "") or "").strip(),
                    "score": fr.get("score", None),
                }
            )
        if not jobs:
            return {"reviews": [], "selected": []}

        def _run_single(single_vlm: Any, job: Dict[str, Any]) -> Dict[str, Any]:
            prompt = render_prompt_text(
                "t4_structured",
                "generation_single_frame_review",
                frame_caption=_short(str(job.get("caption", "") or ""), 600),
            )
            raw_response = ""
            parsed = None
            try:
                raw_response = str(
                    single_vlm.chat(
                        images=[str(job["image_path"])],
                        prompt=prompt,
                        max_new_tokens=generation_vlm_max_new_tokens,
                    )
                )
                parsed = _safe_parse_json(raw_response) if isinstance(raw_response, str) else None
            except Exception as exc:
                raw_response = f"ERROR: {type(exc).__name__}: {exc}"
            review = _normalize_generation_review(
                parsed,
                image_pos=int(job.get("image_pos", 0) or 0),
                raw_response=raw_response,
            )
            review["image_path"] = str(job.get("image_path", "") or "")
            review["caption"] = str(job.get("caption", "") or "")
            review["score"] = job.get("score", None)
            return review

        model_manager = getattr(self, "model_manager", None)
        if model_manager is not None:
            reviews = model_manager.run_vlm_jobs(
                jobs,
                worker_fn=lambda single_vlm, job: _run_single(single_vlm, job),
            )
        else:
            reviews = [_run_single(self.vlm, job) for job in jobs]

        normalized = [rv for rv in reviews if isinstance(rv, dict)]
        for rv in normalized:
            try:
                score_value = float(rv.get("score", 0.0) or 0.0)
            except Exception:
                score_value = 0.0
            rv["_sort_key"] = (
                0 if str(rv.get("subject_clarity", "low")) == "high" else (1 if str(rv.get("subject_clarity", "low")) == "medium" else 2),
                0 if bool(rv.get("composition_clean", False)) else 1,
                0 if not bool(rv.get("text_overlay_heavy", False)) else 1,
                0 if bool(rv.get("background_clean", False)) else 1,
                -score_value,
                int(rv.get("image_pos", 0) or 0),
            )

        suitable = [
            rv for rv in normalized
            if bool(rv.get("suitable", False))
            and str(rv.get("generation_prompt", "") or "").strip()
        ]
        suitable.sort(key=lambda rv: rv.get("_sort_key", (9, 9, 9, 9, 0.0, 9999)))
        _dump_json(
            "generation_routing",
            {
                "reviewed_count": len(normalized),
                "suitable_count": len(suitable),
                "selected_count": min(max_generation, len(suitable)),
                "reviews": [
                    {
                        k: v for k, v in rv.items()
                        if k != "_sort_key"
                    }
                    for rv in normalized
                ],
            },
        )
        return {
            "reviews": normalized,
            "selected": suitable[: max_generation],
        }

    # [MODIFIED v6] hard "no timeline" for dialogs
    _dialog_timeline_bad = re.compile(
        r"(?i)\b(before|after|earlier|later|previous|next|prior to|leading up|timeline|sequence|step-by-step|frame\s*\d+|image\s*sequence)\b"
    )

    # [PATCH 4] Dialog grounding:
    # - Per-turn: MUST have at least one frame-level evidence ref (Fxx_*)
    # - Dialog-level: MUST have at least one S_* in a USER turn AND at least one S_* in an ASSISTANT turn
    def _dialog_turn_has_frame_ref(evr: List[str]) -> bool:
        if not isinstance(evr, list):
            return False
        return any(isinstance(x, str) and re.match(r"^F\d{2}_", str(x)) for x in evr)

    def _dialog_has_segment_refs_by_role(turns: List[Dict[str, Any]]) -> bool:
        if not isinstance(turns, list):
            return False
        has_user_s = False
        has_asst_s = False
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("role", "") or "").strip().lower()
            evr = t.get("evidence_refs", [])
            if not isinstance(evr, list):
                evr = []
            has_s = any(isinstance(x, str) and x.startswith("S_") for x in evr)
            if role == "user" and has_s:
                has_user_s = True
            if role == "assistant" and has_s:
                has_asst_s = True
            if has_user_s and has_asst_s:
                return True
        return bool(has_user_s and has_asst_s)

    # [PATCH 4.1] deterministic fallback: inject missing S_* by role (user + assistant)
    def _best_segment_ref_fallback() -> str:
        for k in ("S_SEGEMENT_SUM",):
            if k in _evidence_id_set:
                return k
        return ""

    def _ensure_dialog_segment_refs_by_role(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure dialog-level segment grounding without making every turn carry S_*.
        - If no USER turn has S_*, inject one into the first USER turn.
        - If no ASSISTANT turn has S_*, inject one into the first ASSISTANT turn.
        Deterministic and minimal changes.
        """
        if not isinstance(turns, list) or not turns:
            return turns

        if is_paper:
            return turns
        need_user = True
        need_asst = True
        for t in turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("role", "") or "").strip().lower()
            evr = t.get("evidence_refs", [])
            if not isinstance(evr, list):
                evr = []
            has_s = any(isinstance(x, str) and str(x).startswith("S_") for x in evr)
            if role == "user" and has_s:
                need_user = False
            if role == "assistant" and has_s:
                need_asst = False

        seg_id = _best_segment_ref_fallback()
        if not seg_id:
            return turns

        if need_user:
            for t in turns:
                if isinstance(t, dict) and str(t.get("role", "") or "").strip().lower() == "user":
                    evr = t.get("evidence_refs", [])
                    if not isinstance(evr, list):
                        evr = []
                    evr = [str(x) for x in evr if str(x).strip()]
                    if seg_id not in evr:
                        evr = [seg_id] + evr
                    t["evidence_refs"] = evr[:24]
                    break

        if need_asst:
            for t in turns:
                if isinstance(t, dict) and str(t.get("role", "") or "").strip().lower() == "assistant":
                    evr = t.get("evidence_refs", [])
                    if not isinstance(evr, list):
                        evr = []
                    evr = [str(x) for x in evr if str(x).strip()]
                    if seg_id not in evr:
                        evr = [seg_id] + evr
                    t["evidence_refs"] = evr[:24]
                    break

        return turns

    # ---------------- 4.1) (unchanged) _llm_generate_instances ----------------
    def _llm_generate_instances(
        benchmark: str,
        quota: int,
        *,
        evidence_pack_text: str,
        candidate_pairs: Optional[List[Tuple[int, int]]] = None,
        best_frame: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        quota = int(quota or 0)
        if quota <= 0:
            return []
        
        # ===================== [PATCH 4.5] attempt counter =====================
        if benchmark  in dialog_debug:
            dialog_debug[benchmark]["attempt"] += 1
        # ===================== [END PATCH 4.5] =====================

        seg_meta = {
            "video_id": video_id,
            "segment_index": segment_index,
            "video_type": video_type,
            "modality_hint": modality_hint,
            "motion_hint": motion_hint,
            "keyframe_count": len(frames_aligned),
            "quota": quota,
            "all_keyframes": all_keyframes_meta,
        }

        default_img = ""
        default_pos = 0
        if isinstance(best_frame, dict):
            default_img = best_frame.get("img") or ""
            try:
                default_pos = int(best_frame.get("pos", 0) or 0)
            except Exception:
                default_pos = 0

        if is_paper and benchmark in ("mme", "mmt_bench", "llava_bench"):
            allowed = {eid for eid in _evidence_id_set if eid.startswith((f"F{default_pos:02d}_", "S_SEGEMENT_SUM"))}
            evidence_pack_text = "\n".join(line for line in evidence_pack_text.splitlines()
                if any(line.startswith("[" + eid + "]") for eid in allowed))
            seg_meta = {"video_id": video_id, "segment_index": segment_index, "summary_scope": "segment",
                        "solver_image_pos": default_pos, "quota": quota}

        pairs_txt = ""
        if candidate_pairs:
            pairs_txt = json.dumps([{"from": a, "to": b} for (a, b) in candidate_pairs], ensure_ascii=False)

        if benchmark == "mme":
            schema = (
                "{\n"
                "  \"items\": [\n"
                "    {\n"
                "      \"image_pos\": int,\n"
                "      \"category\": str,\n"
                "      \"question\": str,\n"
                "      \"answer\": \"Yes\"|\"No\",\n"
                "      \"evidence_refs\": [\"S_SEGEMENT_SUM\", \"F00_CAP\", ...]\n"
                "    }\n"
                "  ]\n"
                "}\n"
            )
        elif benchmark == "mmt_bench":
            schema = (
                "{\n"
                "  \"items\": [\n"
                "    {\n"
                "      \"image_pos\": int,\n"
                "      \"question\": str,\n"
                "      \"A\": str, \"B\": str, \"C\": str, \"D\": str,\n"
                "      \"_answer\": \"A\"|\"B\"|\"C\"|\"D\",\n"
                "      \"category\": str,\n"
                "      \"l2-category\": str,\n"
                "      \"evidence_refs\": [\"S_SEGEMENT_SUM\", \"F00_CAP\", ...]\n"
                "    }\n"
                "  ]\n"
                "}\n"
            )
        elif benchmark == "llava_bench":
            schema = (
                "{\n"
                "  \"items\": [\n"
                "    {\n"
                "      \"image_pos\": int,\n"
                "      \"category\": \"conv\"|\"detail\"|\"complex\",\n"
                "      \"question\": str,\n"
                "      \"gpt_answer\": str,\n"
                + ("      \"reasoning_category\": \"reasoning\"|\"function\"|\"activity\"|\"relation\",\n" if is_paper else "")
                + "      \"evidence_refs\": [\"S_SEGEMENT_SUM\", \"F00_CAP\", ...]\n"
                "    }\n"
                "  ]\n"
                "}\n"
            )
        elif benchmark == "edit":
            schema = (
                "{\n"
                "  \"items\": [\n"
                "    {\n"
                "      \"from_pos\": int,\n"
                "      \"to_pos\": int,\n"
                "      \"instruction\": str,\n"
                "      \"evidence_refs\": [\"P00\", \"S_SEGEMENT_SUM\", \"F00_CAP\", \"F01_CAP\", ...]\n"
                "    }\n"
                "  ]\n"
                "}\n"
            )
        else:
            return []

        if structured_evidence_profile() and benchmark in {"mmt_bench", "llava_bench"}:
            field = "perception_evidence" if benchmark == "mmt_bench" else "reasoning_evidence"
            schema += f"\nEach item MUST also include {field}, with the full fields specified in the rules below."

        single_image_rules = load_prompt_text("t4_structured", "single_image_rules")

        complex_question_rules = load_prompt_text("t4_structured", "complex_question_rules")



        mmt_option_rules = load_prompt_text("t4_structured", "mmt_option_rules")
        mmt_category_rules = render_prompt_text(
            "t4_structured",
            "mmt_category_rules",
            mmt_allowed_categories_json=json.dumps(MMT_ALLOWED_CATEGORIES),
            mmt_allowed_l2_json=json.dumps(MMT_ALLOWED_L2),
        )
        llava_answer_rules = load_prompt_text("t4_structured", "llava_answer_rules")

        yes_n = quota // 2
        no_n = quota - yes_n

        # ===================== [PATCH 1] segment-level category coverage + hot-category throttling =====================
        def _top_hot_cats(d: Dict[str, int], k: int = 3) -> List[str]:
            if not isinstance(d, dict) or not d:
                return []
            items = []
            for kk, vv in d.items():
                try:
                    items.append((int(vv), str(kk)))
                except Exception:
                    continue
            items.sort(key=lambda x: (-x[0], x[1]))
            return [c for _, c in items[: max(0, int(k or 0))]]

        def _cat_hist_text(d: Dict[str, int], max_k: int = 10) -> str:
            if not isinstance(d, dict) or not d:
                return "(empty)"
            items = []
            for kk, vv in d.items():
                try:
                    items.append((int(vv), str(kk)))
                except Exception:
                    continue
            items.sort(key=lambda x: (-x[0], x[1]))
            items = items[: max(1, int(max_k or 10))]
            return ", ".join([f"{c}:{n}" for (n, c) in items]) if items else "(empty)"

        # "hot" categories are throttled globally; also enforce per-segment diversity.
        global_mmt_counts = self.get_global_category_counts_snapshot("mmt_bench") if hasattr(self, "get_global_category_counts_snapshot") else (getattr(self, "_global_mmt_cat_counts", {}) or {})
        global_llava_counts = self.get_global_category_counts_snapshot("llava_bench") if hasattr(self, "get_global_category_counts_snapshot") else (getattr(self, "_global_llava_cat_counts", {}) or {})
        global_mme_counts = self.get_global_category_counts_snapshot("mme") if hasattr(self, "get_global_category_counts_snapshot") else (getattr(self, "_global_mme_cat_counts", {}) or {})

        hot_mmt = _top_hot_cats(global_mmt_counts, k=3)
        hot_llava = _top_hot_cats(global_llava_counts, k=3)
        hot_mme = _top_hot_cats(global_mme_counts, k=3)

        segment_diversity_hard = load_prompt_text("t4_structured", "segment_diversity_hard")

        mmt_balance_rules = render_prompt_text(
            "t4_structured",
            "mmt_balance_rules",
            global_hist_text=_cat_hist_text(global_mmt_counts, max_k=12),
            hot_categories=(hot_mmt if hot_mmt else "(none)"),
            segment_diversity_hard=segment_diversity_hard,
        )

        llava_balance_rules = render_prompt_text(
            "t4_structured",
            "llava_balance_rules",
            global_hist_text=_cat_hist_text(global_llava_counts, max_k=12),
            hot_categories=(hot_llava if hot_llava else "(none)"),
            segment_diversity_hard=segment_diversity_hard,
        )

        mme_balance_rules = render_prompt_text(
            "t4_structured",
            "mme_balance_rules",
            global_hist_text=_cat_hist_text(global_mme_counts, max_k=12),
            hot_categories=(hot_mme if hot_mme else "(none)"),
            segment_diversity_hard=segment_diversity_hard,
        )
        # ===================== [END PATCH 1] =====================

        magicbrush_rules = render_prompt_text(
            "t4_structured",
            "magicbrush_rules",
            pairs_txt=pairs_txt,
        )

        # ===================== [PATCH] Published-style few-shots (per benchmark) =====================
        fewshot_text = build_fewshot_bank().get(benchmark, "")
        # ===================== [END PATCH] =====================


        prompt_body = ""

        if benchmark in ("mme", "mmt_bench", "llava_bench"):
            prompt_body += single_image_rules
            prompt_body += complex_question_rules
            if benchmark == "mme":
                prompt_body += mme_balance_rules
                prompt_body += render_prompt_text(
                    "t4_structured",
                    "mme_category_presence_rules",
                    mme_allowed_categories_json=json.dumps(MME_ALLOWED_CATEGORIES),
                    mme_presence_cap_percent=int(round(mme_presence_cap * 100)),
                )

                prompt_body += render_prompt_text(
                    "t4_structured",
                    "mme_answer_balance_rules",
                    yes_n=yes_n,
                    no_n=no_n,
                )


            elif benchmark == "mmt_bench":
                prompt_body += mmt_option_rules
                prompt_body += mmt_balance_rules
                prompt_body += mmt_category_rules
                prompt_body += load_prompt_text("t4_structured", "mmt_multi_choice_quality")
            elif benchmark == "llava_bench":
                prompt_body += llava_answer_rules
                prompt_body += llava_balance_rules
                prompt_body += load_prompt_text("t4_structured", "llava_reasoning_rules")

        if benchmark == "edit":
            prompt_body += magicbrush_rules

        prompt = build_instance_generation_prompt(
            benchmark=benchmark,
            fewshot_text=fewshot_text,
            seg_meta=seg_meta,
            default_pos=default_pos,
            default_img=default_img,
            body=prompt_body,
            evidence_pack_text=evidence_pack_text,
            quota=quota,
            schema=schema,
        )

        raw = audited_text(self.llm, prompt, max_new_tokens=1600)
        _dump_text(f"gen_{benchmark}_raw", raw if isinstance(raw, str) else str(raw))
        obj = _safe_parse_json(raw) if isinstance(raw, str) else None
        if not isinstance(obj, dict):
            return []

        items = obj.get("items")
        if not isinstance(items, list):
            return []

        out_items = []
        cand_set = set(candidate_pairs or [])

        for it in items:
            if not isinstance(it, dict):
                continue
            evr = it.get("evidence_refs", [])
            if not isinstance(evr, list) or not evr:
                continue

            if benchmark in ("mme", "mmt_bench", "llava_bench"):
                if is_paper:
                    try:
                        validate_single_image_item(it, benchmark, _evidence_id_set, default_pos)
                    except ValueError as exc:
                        paper_rejections.append({"stage": "generation", "reason": str(exc), "candidate": it})
                        continue
                qtxt = str(it.get("question", "") or "")
                if _single_image_bad_q.search(qtxt):
                    continue
                try:
                    pos0 = int(it.get("image_pos", default_pos))
                except Exception:
                    pos0 = default_pos
                pos0 = max(0, min(pos0, len(frames_aligned) - 1))
                it["image_pos"] = pos0
                it["evidence_refs"] = _ensure_single_image_frame_ref(pos0, evr)

            elif benchmark == "edit":
                try:
                    a = int(it.get("from_pos", -1))
                    b = int(it.get("to_pos", -1))
                except Exception:
                    continue
                if (a, b) not in cand_set:
                    continue

                instr = str(it.get("instruction", "") or "").strip()
                if is_paper and (not instr or not re.search(r"(?i)do not change anything else\.?$", instr)):
                    paper_rejections.append({"stage": "editing", "reason": "missing_invariant_suffix", "candidate": it})
                    continue
                if instr and (not re.search(r"(?i)do not change anything else\.?$", instr.strip())):
                    instr = instr.rstrip(".") + ". Do not change anything else."
                it["instruction"] = instr if is_paper else (instr[:320] if instr else "")

                ts = change_map_struct.get((a, b), {}) if isinstance(change_map_struct.get((a, b), {}), dict) else {}
                ts_view = (ts.get("view") or "").lower()
                if ts_view and ts_view not in ("none" , "minor"):
                    continue

                if is_paper:
                    try:
                        validate_refs(evr, _evidence_id_set, [a, b], all_frames=True)
                        directional_ids = {u["evidence_id"] for u in evidence_units
                                           if u.get("pair_from_pos") == a and u.get("pair_to_pos") == b}
                        if not directional_ids.intersection(evr):
                            raise ValueError("editing_missing_change_citation")
                    except ValueError as exc:
                        paper_rejections.append({"stage": "editing", "reason": str(exc), "candidate": it})
                        continue
                it["evidence_refs"] = _ensure_segment_refs(evr)

            else:
                it["evidence_refs"] = _ensure_segment_refs(evr)

            out_items.append(it)
            if len(out_items) >= quota:
                break

        # ---------------- [NEW] MME closed-set category normalize + presence cap (post-filter) ----------------
        if benchmark == "mme" and out_items:
            # 1) normalize / drop out-of-set categories
            norm = []
            for x in out_items:
                if not isinstance(x, dict):
                    continue
                c = str(x.get("category", "") or "").strip().lower()
                if not c:
                    c = "other"
                if c not in _mme_cat_set:
                    # scheme-1 drop: out-of-set categories are rejected to prevent open-ended drift
                    continue
                x["category"] = c
                norm.append(x)
            out_items = norm

            # 2) presence cap (best-effort): keep presence only up to cap if we have non-presence candidates
            if out_items:
                max_presence = int(round(float(quota) * float(mme_presence_cap)))
                max_presence = max(0, min(max_presence, int(quota)))
                pres = [x for x in out_items if str(x.get("category", "")).strip().lower() == "presence"]
                nonp = [x for x in out_items if str(x.get("category", "")).strip().lower() != "presence"]

                # Only cap if we have any non-presence to use (do not destroy yield)
                if nonp and len(pres) > max_presence:
                    pres = pres[:max_presence]
                    out_items = (nonp + pres)[:quota]

            # 3) per-call minimal diversity: if quota>=4, try to keep >=2 distinct categories when possible
            if len(out_items) >= 4:
                cats = []
                for x in out_items:
                    cats.append(str(x.get("category", "") or "other"))
                distinct = sorted(set(cats))
                if len(distinct) < 2:
                    # try to rescue diversity if there exist candidates of other categories in the raw items list
                    # (best-effort, no re-generation)
                    pass
        # ---------------- [END NEW] ----------------

        # --- [PATCH] enforce MME yes/no balance deterministically ---
        if benchmark == "mme" and out_items:
            yes_items = [x for x in out_items if str(x.get("answer","")).strip() == "Yes"]
            no_items  = [x for x in out_items if str(x.get("answer","")).strip() == "No"]

            # keep up to target
            yes_items = yes_items[:yes_n]
            no_items  = no_items[:no_n]

            # if one side short, backfill from the other side (but keep total<=quota)
            merged = yes_items + no_items
            if len(merged) < min(quota, len(out_items)):
                rest = [x for x in out_items if x not in merged]
                merged.extend(rest[: (quota - len(merged))])

            out_items = merged[:quota]

        return out_items


    # ---------------- [NEW] 4.X Dialog generator ----------------
    def _dialog_evidence_pack(pos_list: List[int], *, max_chars: int = 2400) -> str:
        return build_dialog_evidence_pack(
            pos_list=pos_list,
            evidence_id_set=_evidence_id_set,
            segment_summary=segment_summary,
            frames_aligned=frames_aligned,
            short_text=_short,
            tool_evidence_suffix=_tool_evidence_suffix,
            max_chars=max_chars,
            evidence_units=evidence_units,
        )
        lines = []
        if "S_SEGEMENT_SUM" in _evidence_id_set:
            lines.append(f"[S_SEGEMENT_SUM] {_short(segment_summary, 500)}")

        for p in pos_list:
            p = int(p)
            if 0 <= p < len(frames_aligned):
                fr = frames_aligned[p]
                cap = (fr.get("caption") or "").strip()
                if cap and f"F{p:02d}_CAP" in _evidence_id_set:
                    lines.append(f"[F{p:02d}_CAP] Frame F{p:02d}: {cap}")
                frame_tool_descs = fr.get("frame_tool_descs", {}) if isinstance(fr.get("frame_tool_descs", {}), dict) else {}
                for tool_name, tool_summary in frame_tool_descs.items():
                    eid = f"F{p:02d}_TOOL_{_tool_evidence_suffix(tool_name)}"
                    if eid not in _evidence_id_set:
                        continue
                    tool_summary = str(tool_summary or "").strip()
                    if tool_summary:
                        lines.append(f"[{eid}] Frame F{p:02d}: {tool_name} semantic hint: {_short(tool_summary, 180)}")

        s = "\n".join(lines).strip()
        if len(s) > max_chars:
            s = s[:max_chars].rstrip() + " ..."
        return s

    # [MODIFIED v6] visdial top3 anchors (diverse)
    def _pick_visdial_anchors(topk: int = 3) -> List[int]:
        topk = max(1, int(topk or 1))
        cands = []
        for fr in frames_aligned:
            p = int(fr.get("pos", 0) or 0)
            if not fr.get("img"):
                continue
            sc = fr.get("score", None)
            try:
                scf = float(sc) if sc is not None else 0.0
            except Exception:
                scf = 0.0
            cap = (fr.get("caption") or "").strip()

        #     cands.append((scf, p, cap))

        # cands.sort(key=lambda x: (-x[0], x[1]))

        # --- [NEW] dialog suitability score (anchor should be "askable") ---
            tool_desc_count = 0
            try:
                tool_descs = fr.get("frame_tool_descs", {}) or {}
                if isinstance(tool_descs, dict):
                    tool_desc_count = sum(1 for v in tool_descs.values() if str(v or "").strip())
            except Exception:
                tool_desc_count = 0
            semantic_text = str(fr.get("frame_semantic_text", "") or "").strip()
            cap_len = len(cap)
            # heuristic: richer caption + frame semantic evidence => more "dialog-friendly"
            dlg_score = 0.0
            dlg_score += min(1.0, cap_len / 120.0) * 1.2
            dlg_score += min(1.0, float(tool_desc_count) / 3.0) * 0.8
            dlg_score += 0.4 if semantic_text else 0.0
            # mild preference for high visual quality score, but not dominant
            dlg_score += _clamp(scf, 0.0, 1.0) * 0.5
            # endcard-like penalty if caption looks like CTA
            if _looks_like_endcard(cap):
                dlg_score -= 1.5
            cands.append((dlg_score, scf, p, cap))
        # rank by dialog suitability first, then score, then position
        cands.sort(key=lambda x: (-x[0], -x[1], x[2]))

        out = []
        caps = []
        for _, __, p, cap in cands:
            if len(out) >= topk:
                break
            # simple diversity by caption jaccard (avoid near duplicates)
            ok = True
            for c0 in caps:
                if _jaccard(cap, c0) >= 0.85:
                    ok = False
                    break
            if not ok:
                continue
            out.append(int(p))
            caps.append(cap)
        if not out:
            return [best_pos]
        if best_pos not in out and len(out) < topk and 0 <= best_pos < len(frames_aligned) and bool(frames_aligned[best_pos].get("img")):
            out = [best_pos] + out
            out = out[:topk]
        return out[:topk]

    # helper: pick mmdu positions from a chosen start (keep previous logic)
    def _pick_mmdu_positions_from_start(start_pos: int, k: int) -> List[int]:
        k = max(2, min(int(k or 3), 5))
        if len(frames_aligned) <= 0:
            return []

        def _is_good(p: int) -> bool:
            return (0 <= p < len(frames_aligned)) and bool(frames_aligned[p].get("img"))

        start = int(start_pos)
        if not _is_good(start):
            # nearest good around start
            found = None
            for d in range(len(frames_aligned)):
                a = start - d
                b = start + d
                if _is_good(a):
                    found = a
                    break
                if _is_good(b):
                    found = b
                    break
            if found is None:
                return []
            start = found

        picked = [int(start)]

        def _can_add(p: int) -> bool:
            if not _is_good(p):
                return False
            mn = min(picked)
            mx = max(picked)
            if p > mx:
                return not _crosses_shot_boundary(mx, p)
            if p < mn:
                return not _crosses_shot_boundary(p, mn)
            return True

        cur = start
        step = 1
        while len(picked) < k:
            p = cur + step
            if _can_add(p):
                picked.append(int(p))
                cur = p
                step = 1
                continue
            step += 1
            if start + step >= len(frames_aligned):
                break

        cur = start
        step = 1
        while len(picked) < k:
            p = cur - step
            if _can_add(p):
                picked.append(int(p))
                cur = p
                step = 1
                continue
            step += 1
            if start - step < 0:
                break

        picked = sorted(set(picked))
        if len(picked) > k:
            picked.sort(key=lambda x: (abs(x - start), x))
            picked = sorted(picked[:k])
        return picked

    # keep old API used elsewhere
    def _pick_mmdu_positions(k: int) -> List[int]:
        return _pick_mmdu_positions_from_start(best_pos, k)

    # [MODIFIED v6] mmdu 3 groups
    def _pick_mmdu_groups(num_groups: int = 3, k: int = 3) -> List[List[int]]:
        num_groups = max(1, int(num_groups or 1))
        k = max(2, min(int(k or 3), 5))

        # choose multiple distinct start positions by score (diverse in pos)
        starts = []
        scored = []
        for fr in frames_aligned:
            p = int(fr.get("pos", 0) or 0)
            if not fr.get("img"):
                continue
            sc = fr.get("score", None)
            try:
                scf = float(sc) if sc is not None else 0.0
            except Exception:
                scf = 0.0
            scored.append((scf, p))
        scored.sort(key=lambda x: (-x[0], x[1]))

        for _, p in scored:
            if len(starts) >= num_groups:
                break
            if not starts:
                starts.append(p)
                continue
            # keep starts apart
            if min(abs(p - s) for s in starts) < 3:
                continue
            starts.append(p)

        if best_pos not in starts:
            starts = [best_pos] + starts
        starts = starts[:num_groups]

        groups = []
        seen = set()
        for s in starts:
            pos_list = _pick_mmdu_positions_from_start(s, k)
            key = tuple(pos_list)
            if len(pos_list) >= 2 and key not in seen:
                groups.append(pos_list)
                seen.add(key)
            if len(groups) >= num_groups:
                break

        if not groups:
            fallback = _pick_mmdu_positions(k)
            if len(fallback) >= 2:
                groups = [fallback]
        return groups[:num_groups]

    # [MODIFIED v6] repair helper: if turns insufficient, attempt one-shot repair
    def _repair_dialog_turns(kind: str, rounds: int, pos_list: List[int], turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        want = int(rounds) * 2
        # only attempt repair if we have at least 2 QA already
        if not isinstance(turns, list) or len(turns) < 4:
            return turns

        dlg_ev = _dialog_evidence_pack(pos_list, max_chars=2000)
        schema = (
            "{\n"
            "  \"turns\": [\n"
            "    {\"role\": \"user\"|\"assistant\", \"text\": str, \"evidence_refs\": [str, ...]}\n"
            "  ]\n"
            "}\n"
        )

        repair_prompt = build_dialog_repair_prompt(
            kind=kind,
            rounds=rounds,
            pos_list=pos_list,
            dlg_ev=dlg_ev,
            turns=turns,
        )
        raw = audited_text(self.llm, repair_prompt, max_new_tokens=700)
        _dump_text(f"repair_{kind}_raw", raw if isinstance(raw, str) else str(raw))
        obj = _safe_parse_json(raw) if isinstance(raw, str) else None
        if not isinstance(obj, dict):
            return turns
        out_turns = obj.get("turns", [])
        if not isinstance(out_turns, list) or len(out_turns) != want:
            return turns

        # [PATCH 4.3] normalize + enforce grounding injection (same as main path)
        norm_turns = []
        for t in out_turns:
            if not isinstance(t, dict):
                continue
            role = str(t.get("role", "") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            text = str(t.get("text", "") or "").strip()
            if not text:
                continue
            evr = t.get("evidence_refs", [])
            if not isinstance(evr, list):
                evr = []
            evr = [str(x) for x in evr if str(x).strip()]

            # enforce frame grounding injection
            if kind == "dialog_visdial_like":
                evr = _ensure_single_image_frame_ref(pos_list[0], evr)
            else:
                evr = _ensure_any_of_frame_refs(pos_list, evr)

            norm_turns.append({"role": role, "text": text[:600], "evidence_refs": evr})

        if len(norm_turns) != want:
            return turns

        # deterministic dialog-level S_* injection by role
        norm_turns = _ensure_dialog_segment_refs_by_role(norm_turns)
        return norm_turns


    def _llm_generate_dialogs(kind: str, quota: int, *, pos_list: List[int]) -> List[Dict[str, Any]]:
        quota = int(quota or 0)
        if quota <= 0:
            return []
        
        # ===================== [PATCH] dialog attempt counter =====================
        if kind in dialog_debug:
            dialog_debug[kind]["attempt"] += 1
        # ===================== [END PATCH] =====================

        pos_list = [int(p) for p in (pos_list or []) if (isinstance(p, int) or str(p).isdigit())]
        pos_list = [p for p in pos_list if 0 <= p < len(frames_aligned)]
        pos_list = sorted(dict.fromkeys(pos_list).keys())

        if kind == "dialog_visdial_like":
            if not pos_list:
                pos_list = [best_pos]
            pos_list = [pos_list[0]]
            rounds = max(2, min(int(dialog_visdial_rounds or 3), 10))
        elif kind == "dialog_mmdu_like":
            if len(pos_list) < 2:
                pos_list = _pick_mmdu_positions(dialog_mmdu_image_count)
            pos_list = [p for p in pos_list if 0 <= p < len(frames_aligned)]
            if len(pos_list) < 2:
                return []
            pos_list = pos_list[:5]
            rounds = max(2, min(int(dialog_mmdu_rounds or 3), 10))
        else:
            return []

        dlg_ev = _dialog_evidence_pack(pos_list)
        _dump_text(f"dialog_{kind}_evidence_pack", dlg_ev)

        schema = (
            "{\n"
            "  \"dialogs\": [\n"
            "    {\n"
            "      \"turns\": [\n"
            "        {\"role\": \"user\", \"text\": str, \"evidence_refs\": [str, ...]},\n"
            "        {\"role\": \"assistant\", \"text\": str, \"evidence_refs\": [str, ...]}\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        schema = (
            "{\n"
            "  \"dialogs\": [\n"
            "    {\n"
            "      \"turns\": [\n"
            "        {\"role\": \"user\", \"text\": str, \"evidence_refs\": [str, ...]},\n"
            "        {\"role\": \"assistant\", \"text\": str, \"evidence_refs\": [str, ...]}\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        # [PATCH v7] wording normalization: ban "I will"/discourse starters; provide replacements + a shot
        wording_norm = load_prompt_text("t4_structured", "dialog_wording_normalization")

        if kind == "dialog_visdial_like":
            kind_rules = render_prompt_text(
                "t4_structured",
                "dialog_visdial_kind_rules",
                rounds=rounds,
                total_turns=(rounds * 2),
            ) + "\n" + wording_norm

            image_block = (
                f"IMAGE: image_pos={pos_list[0]} image_path={frames_aligned[pos_list[0]].get('img')}\n"
                "The image is implicit; do not use <ImageHere>.\n"
            )

        else:
            kind_rules = render_prompt_text(
                "t4_structured",
                "dialog_mmdu_kind_rules",
                rounds=rounds,
                total_turns=(rounds * 2),
            ) + "\n" + wording_norm

            image_block = (
                "IMAGES (map to placeholders):\n"
                + "\n".join(
                    [f"- Image{idx+1}: image_pos={p} image_path={frames_aligned[p].get('img')}" for idx, p in enumerate(pos_list)]
                )
                + "\n"
            )

        base_rules = render_prompt_text(
            "t4_structured",
            "dialog_base_rules",
            total_turns=(rounds * 2),
            distinct_user_foci=rounds,
        )

        prompt = render_prompt_text(
            "t4_structured",
            "dialog_generation_prompt",
            kind_rules=kind_rules,
            base_rules=base_rules,
            image_block=image_block,
            dlg_ev=dlg_ev,
            quota=quota,
            kind=kind,
            schema=schema,
        )

        raw = audited_text(self.llm, prompt, max_new_tokens=1400)
        _dump_text(f"gen_{kind}_raw", raw if isinstance(raw, str) else str(raw))

        obj = _safe_parse_json(raw) if isinstance(raw, str) else None
        if not isinstance(obj, dict):
            return []
        dialogs = obj.get("dialogs")
        if not isinstance(dialogs, list):
            return []

        out = []
        want = rounds * 2

        for d in dialogs:
            if not isinstance(d, dict):
                continue
            turns = d.get("turns", [])
            if is_paper:
                try:
                    validate_dialog(turns, pos_list, _evidence_id_set, rounds)
                except ValueError as exc:
                    paper_rejections.append({"stage": "dialog", "reason": str(exc), "candidate": d})
                    continue
                # Preserve validated paper turns; legacy repairs can change the task.
                out.append(d)
                if kind in dialog_debug:
                    dialog_debug[kind]["accept"] += 1
                continue
            if not isinstance(turns, list) or len(turns) < 4:
                continue

            # normalize turns
            norm_turns = []
            for t in turns:
                if not isinstance(t, dict):
                    continue
                role = str(t.get("role", "") or "").strip().lower()
                if role not in ("user", "assistant"):
                    continue
                text = str(t.get("text", "") or "").strip()
                if not text:
                    continue
                evr = t.get("evidence_refs", [])
                if not isinstance(evr, list):
                    evr = []
                evr = [str(x) for x in evr if str(x).strip()]

                # enforce grounding injection
                if kind == "dialog_visdial_like":
                    evr = _ensure_single_image_frame_ref(pos_list[0], evr)
                else:
                    evr = _ensure_any_of_frame_refs(pos_list, evr)

                norm_turns.append({"role": role, "text": text[:600], "evidence_refs": evr})

            if len(norm_turns) < 4:
                continue

            # role alternation check (hard)
            ok_alt = True
            for i in range(1, len(norm_turns)):
                if norm_turns[i]["role"] == norm_turns[i-1]["role"]:
                    ok_alt = False
                    break
            if not ok_alt:
                # try repair once
                norm_turns = _repair_dialog_turns(kind, rounds, pos_list, norm_turns)
                ok_alt = True
                for i in range(1, len(norm_turns)):
                    if norm_turns[i]["role"] == norm_turns[i-1]["role"]:
                        ok_alt = False
                        break
                if not ok_alt:

                    # ===================== [PATCH 4.5] drop_alternation =====================
                    if kind in dialog_debug:
                        dialog_debug[kind]["drop_alternation"] += 1
                    # ===================== [END PATCH 4.5] =====================

                    continue

            # trim or repair to exact turns
            if len(norm_turns) > want:
                norm_turns = norm_turns[:want]
            if len(norm_turns) < want:
                # [MODIFIED v6] do NOT drop; attempt repair once
                norm_turns = _repair_dialog_turns(kind, rounds, pos_list, norm_turns)

            if not isinstance(norm_turns, list) or len(norm_turns) != want:

                # ===================== [PATCH 4.5] drop_turn_count =====================
                if kind in dialog_debug:
                    dialog_debug[kind]["drop_turn_count"] += 1
                # ===================== [END PATCH 4.5] =====================

                continue


            # hard timeline ban
            if any(_dialog_timeline_bad.search(str(t.get("text", "") or "")) for t in norm_turns if isinstance(t, dict)):
                # try repair once to remove timeline words
                norm_turns = _repair_dialog_turns(kind, rounds, pos_list, norm_turns)
                if any(_dialog_timeline_bad.search(str(t.get("text", "") or "")) for t in norm_turns if isinstance(t, dict)):

                    # ===================== [PATCH 4.5] drop_timeline =====================
                    if kind in dialog_debug:
                        dialog_debug[kind]["drop_timeline"] += 1
                    # ===================== [END PATCH 4.5] =====================

                    continue


            # [PATCH 4] hard groundedness:
            # - per turn: must have Fxx_*
            # - dialog level: must have S_* in at least one user turn AND one assistant turn
            ok_ground = True
            for t in norm_turns:
                evr = t.get("evidence_refs", [])
                if not _dialog_turn_has_frame_ref(evr):
                    ok_ground = False
                    break

            ok_seg = _dialog_has_segment_refs_by_role(norm_turns)

            if not (ok_ground and ok_seg):
                # try repair once
                norm_turns = _repair_dialog_turns(kind, rounds, pos_list, norm_turns)

                # deterministic fallback: inject missing S_* by role (harmless if already satisfied)
                norm_turns = _ensure_dialog_segment_refs_by_role(norm_turns)

                ok_ground = True
                for t in norm_turns:
                    if not _dialog_turn_has_frame_ref(t.get("evidence_refs", [])):
                        ok_ground = False
                        break

                ok_seg = _dialog_has_segment_refs_by_role(norm_turns)

                if not (ok_ground and ok_seg):
                    # ===================== [PATCH 4.5] drop_grounding =====================
                    if kind in dialog_debug:
                        dialog_debug[kind]["drop_grounding"] += 1
                    # ===================== [END PATCH 4.5] =====================
                    continue


            # always run final deterministic S_* injection by role (safe)
            norm_turns = _ensure_dialog_segment_refs_by_role(norm_turns)



            out.append({"turns": norm_turns})

            # ===================== [PATCH 4.5] accept counter =====================
            if kind in dialog_debug:
                dialog_debug[kind]["accept"] += 1
            # ===================== [END PATCH 4.5] =====================

            if len(out) >= quota:
                break


        return out

    # ---------------- [NEW] Global category counter bump ----------------
    def _bump_cat(counter: Dict[str, int], cat: Any):
        """Increment global category count safely."""
        try:
            c = str(cat or "").strip()
        except Exception:
            c = ""
        if not c:
            return
        counter[c] = int(counter.get(c, 0) or 0) + 1

    def _bump_item_category(benchmark_name: str, item_obj: Dict[str, Any]):
        """Update per-benchmark global category counters based on item['category']."""
        if not isinstance(item_obj, dict):
            return
        if hasattr(self, "bump_global_item_category"):
            self.bump_global_item_category(benchmark_name, item_obj)
            return
        cat_val = item_obj.get("category", None)
        if benchmark_name == "mme":
            _bump_cat(self._global_mme_cat_counts, cat_val)
        elif benchmark_name == "mmt_bench":
            _bump_cat(self._global_mmt_cat_counts, cat_val)
        elif benchmark_name == "llava_bench":
            _bump_cat(self._global_llava_cat_counts, cat_val)
    # ---------------- [END NEW] ----------------

    generation_route_result = _run_generation_routing() if "generation" in selected_modules else {"reviews": [], "selected": []}
    generation_reviews = list((generation_route_result or {}).get("reviews", []) or [])
    selected_generation_reviews = list((generation_route_result or {}).get("selected", []) or [])
    generation_quota = len(selected_generation_reviews)

    # ---------------- 4.A Edit module ----------------
    if "edit" in selected_modules:
        quota = int(edit_quota or 0)
        selected_edit_pairs = [(int(p["from"]), int(p["to"])) for p in (decision.get("edit_pairs") or []) if isinstance(p, dict) and "from" in p and "to" in p]
        if enable_llm_generation and quota > 0 and selected_edit_pairs:
            gen_items = _llm_generate_instances(
                "edit",
                quota,
                evidence_pack_text=evidence_pack_text,
                candidate_pairs=selected_edit_pairs,
                best_frame=best_frame,
            )
        else:
            gen_items = []

        made = 0
        cand_set = set(selected_edit_pairs)
        for it in gen_items:
            if made >= quota:
                break
            try:
                a = int(it.get("from_pos", -1))
                b = int(it.get("to_pos", -1))
            except Exception:
                continue
            if (a, b) not in cand_set:
                continue

            fr_a = frames_aligned[a]
            fr_b = frames_aligned[b]

            rv = review_cache.get((a, b), {}) or {}
            instr = str(it.get("instruction", "") or "").strip()
            if enable_llm_magicbrush_instr and len(instr) > 320:
                instr = instr[:320]
            if not instr:
                continue

            if not re.search(r"(?i)do not change anything else\.?$", instr.strip()):
                instr = instr.rstrip(".") + ". Do not change anything else."

            evr = it.get("evidence_refs", [])
            if not isinstance(evr, list):
                evr = []
            evr = _ensure_segment_refs(evr)

            item = {
                "module": "edit",
                "benchmark": "edit",
                "img_id": f"{video_id}_seg{segment_index:03d}_mb_{made:03d}",
                "turn_index": 1,
                "source_img": fr_a.get("img"),
                "mask_img": None,
                "instruction": instr,
                "target_img": fr_b.get("img"),
                "evidence_refs": evr,
                "meta": {
                    "module": "edit",
                    "source": "video_segment",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "from_keyframe_pos": a,
                    "to_keyframe_pos": b,
                    "from_origin_index": fr_a.get("origin_index"),
                    "to_origin_index": fr_b.get("origin_index"),
                    "pair_hints": pair_hints.get((a, b), {}),
                    "review": rv,
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _write_item_to_builder(item)
            made += 1

    # ---------------- 4.A.1 Generation module ----------------
    if selected_generation_reviews:
        for rank, review in enumerate(selected_generation_reviews):
            try:
                pos = int(review.get("image_pos", 0) or 0)
            except Exception:
                pos = 0
            if pos < 0 or pos >= len(frames_aligned):
                continue
            fr = frames_aligned[pos]
            src_img = str(fr.get("img", "") or "")
            if not src_img:
                continue
            try:
                gt_img = _copy_generation_ground_truth(src_img, pos, rank)
            except Exception:
                gt_img = src_img

            query = str(review.get("generation_prompt", "") or "").strip()
            if not query:
                continue

            item = {
                "module": "generation",
                "benchmark": str(review.get("benchmark_style", "") or "").strip() or "generation",
                "generation_id": f"{video_id}_seg{segment_index:03d}_gen_{rank:03d}",
                "image": gt_img,
                "query": query,
                "prompt": query,
                "caption": str(fr.get("caption", "") or ""),
                "evidence_refs": _ensure_single_image_frame_ref(pos, [f"F{pos:02d}_CAP"]),
                "meta": {
                    "module": "generation",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "keyframe_pos": pos,
                    "origin_index": fr.get("origin_index"),
                    "source_image": src_img,
                    "ground_truth_image": gt_img,
                    "generation_review": {
                        "suitable": bool(review.get("suitable", False)),
                        "subject_clarity": str(review.get("subject_clarity", "") or ""),
                        "composition_clean": bool(review.get("composition_clean", False)),
                        "text_overlay_heavy": bool(review.get("text_overlay_heavy", False)),
                        "background_clean": bool(review.get("background_clean", False)),
                        "generation_reason": str(review.get("generation_reason", "") or ""),
                        "benchmark_style": str(review.get("benchmark_style", "") or ""),
                    },
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _write_item_to_builder(item)

    # ---------------- [MODIFIED v6] Dialog writers: visdial top3 anchors + mmdu 3 groups ----------------
    if "understanding" in selected_modules:
        quota = int(understanding_quotas.get("dialog_visdial_like", 0) or 0)
        anchors = _pick_visdial_anchors(topk=min(max(3, quota), 8))  # <-- top3 anchors
        gen_dialogs_all = []
        made = 0

        # minimal change: generate 1 dialog per anchor until quota met
        if enable_llm_generation and quota > 0:
            for p in anchors:
                if made >= quota:
                    break
                ds = _llm_generate_dialogs("dialog_visdial_like", 1, pos_list=[p])
                if ds:
                    gen_dialogs_all.append((p, ds[0]))
                    made += 1

        made = 0
        for (p, d) in gen_dialogs_all:
            if made >= quota:
                break
            turns = d.get("turns", [])
            if not isinstance(turns, list) or not turns:
                continue
            fr = frames_aligned[p]
            item = {
                "module": "understanding",
                "benchmark": "dialog_visdial_like",
                "dialog_id": f"{video_id}_seg{segment_index:03d}_vd_{made:03d}",
                "image": fr.get("img"),
                "turns": turns,
                "meta": {
                    "module": "understanding",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "keyframe_pos": int(p),
                    "origin_index": fr.get("origin_index"),
                    "rounds": int(dialog_visdial_rounds),
                    "anchors": anchors,
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _write_item_to_builder(item)
            made += 1

    if "understanding" in selected_modules:
        quota = int(understanding_quotas.get("dialog_mmdu_like", 0) or 0)
        groups = _pick_mmdu_groups(num_groups=min(max(3, quota), 8), k=dialog_mmdu_image_count)  # <-- 3 groups
        gen_dialogs_all = []
        made = 0

        if enable_llm_generation and quota > 0:
            for pos_list in groups:
                if made >= quota:
                    break
                if len(pos_list) < 2:
                    continue
                ds = _llm_generate_dialogs("dialog_mmdu_like", 1, pos_list=pos_list)
                if ds:
                    gen_dialogs_all.append((pos_list, ds[0]))
                    made += 1

        made = 0
        for (pos_list, d) in gen_dialogs_all:
            if made >= quota:
                break
            turns = d.get("turns", [])
            if not isinstance(turns, list) or not turns:
                continue

            images = []
            for p in pos_list:
                fr = frames_aligned[p]
                images.append({"pos": int(p), "image": fr.get("img"), "origin_index": fr.get("origin_index")})

            item = {
                "module": "understanding",
                "benchmark": "dialog_mmdu_like",
                "dialog_id": f"{video_id}_seg{segment_index:03d}_mmdu_{made:03d}",
                "images": images,
                "turns": turns,
                "meta": {
                    "module": "understanding",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "image_positions": [int(p) for p in pos_list],
                    "rounds": int(dialog_mmdu_rounds),
                    "groups": groups,
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _write_item_to_builder(item)
            made += 1

    # ---------------- 4.B / 4.C / 4.D: QA writers (unchanged) ----------------
    if "understanding" in selected_modules:
        quota = int(understanding_quotas.get("mme", 0) or 0)
        gen_items = []
        if enable_llm_generation and quota > 0 and best_frame:
            gen_items = _llm_generate_instances("mme", quota, evidence_pack_text=evidence_pack_text, best_frame=best_frame)

        for it in gen_items:
            try:
                pos = int(it.get("image_pos", best_pos))
            except Exception:
                pos = best_pos
            pos = max(0, min(pos, len(frames_aligned) - 1))
            fr = frames_aligned[pos]

            q = str(it.get("question", "") or "").strip()
            a = str(it.get("answer", "") or "").strip()
            if a not in ("Yes", "No") or not q:
                continue

            item = {
                "module": "understanding",
                "benchmark": "mme",
                "question_id": f"{video_id}_seg{segment_index:03d}_mme_{_instance_count('understanding.mme'):04d}",
                "image": fr.get("img"),
                "category": str(it.get("category", "general") or "general"),
                "question": q,
                "answer": a,
                "evidence_refs": _ensure_single_image_frame_ref(pos, it.get("evidence_refs", [])),
                "meta": {
                    "module": "understanding",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "keyframe_pos": pos,
                    "origin_index": fr.get("origin_index"),
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _bump_item_category("mme", item)
            _write_item_to_builder(item)

    if "understanding" in selected_modules:
        quota = int(understanding_quotas.get("mmt_bench", 0) or 0)
        gen_items = []
        if enable_llm_generation and quota > 0 and best_frame:
            gen_items = _llm_generate_instances("mmt_bench", quota, evidence_pack_text=evidence_pack_text, best_frame=best_frame)

        # [NEW] verify ONLY mmt items (batch) to reduce ambiguity
        if gen_items and quota > 0:
            gen_items = _batch_verify_mmt_items(gen_items)

        for it in gen_items:
            try:
                pos = int(it.get("image_pos", best_pos))
            except Exception:
                pos = best_pos
            pos = max(0, min(pos, len(frames_aligned) - 1))
            fr = frames_aligned[pos]

            q = str(it.get("question", "") or "").strip()
            A = str(it.get("A", "") or "").strip()
            B = str(it.get("B", "") or "").strip()
            C = str(it.get("C", "") or "").strip()
            D = str(it.get("D", "") or "").strip()
            ans = str(it.get("_answer", "A") or "A").strip()
            if ans not in ("A", "B", "C", "D"):
                continue
            if not q or not (A and B and C and D):
                continue

            item = {
                "module": "understanding",
                "benchmark": "mmt_bench",
                "index": int(f"{segment_index}{_instance_count('understanding.mmt_bench')+1}"),
                "image": fr.get("img"),
                "question": q,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "category": str(it.get("category", "general") or "general"),
                "l2-category": str(it.get("l2-category", "general") or "general"),
                "split": "VAL",
                "_answer": ans,
                **({"perception_evidence": it["perception_evidence"]} if structured_evidence_profile() else {}),
                "evidence_refs": _ensure_single_image_frame_ref(pos, it.get("evidence_refs", [])),
                "meta": {
                    "module": "understanding",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "keyframe_pos": pos,
                    "origin_index": fr.get("origin_index"),
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _bump_item_category("mmt_bench", item)
            _write_item_to_builder(item)


    if "understanding" in selected_modules:
        quota = int(understanding_quotas.get("llava_bench", 0) or 0)
        gen_items = []
        if enable_llm_generation and quota > 0 and best_frame:
            gen_items = _llm_generate_instances("llava_bench", quota, evidence_pack_text=evidence_pack_text, best_frame=best_frame)

        for it in gen_items:
            try:
                pos = int(it.get("image_pos", best_pos))
            except Exception:
                pos = best_pos
            pos = max(0, min(pos, len(frames_aligned) - 1))
            fr = frames_aligned[pos]

            cat = str(it.get("category", "conv") or "conv").strip()
            if cat not in ("conv", "detail", "complex"):
                cat = "conv"
            q = str(it.get("question", "") or "").strip()
            a = str(it.get("gpt_answer", "") or "").strip()
            if not q or not a:
                continue

            item = {
                "module": "understanding",
                "benchmark": "llava_bench",
                "question_id": int(f"{segment_index}{_instance_count('understanding.llava_bench')+1}"),
                "image_id": os.path.basename(fr.get("img") or f"{video_id}.jpg"),
                "image": fr.get("img"),
                "category": cat,
                "question": q,
                "caption": (fr.get("caption") or ""),
                "gpt_answer": a,
                **({"reasoning_category": it["reasoning_category"]} if is_paper else {}),
                **({"reasoning_evidence": it["reasoning_evidence"]} if structured_evidence_profile() else {}),
                "evidence_refs": _ensure_single_image_frame_ref(pos, it.get("evidence_refs", [])),
                "meta": {
                    "module": "understanding",
                    "video_path": video_path,
                    "segment_path": segment_path,
                    "segment_index": segment_index,
                    "keyframe_pos": pos,
                    "origin_index": fr.get("origin_index"),
                    "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                },
            }
            _append_instance(item)
            _bump_item_category("llava_bench", item)
            _write_item_to_builder(item)

    unify_result = {}
    if "unify" in selected_modules:
        quota = int(unify_quota or 0)
        if quota > 0:
            unify_bundle = {
                "segment": {
                    "index": segment_index,
                    "video_id": video_id,
                    "video_path": video_path,
                    "segment_path": segment_path,
                },
                "t1_out": t1_out,
                "t2_out": t2_out,
                "t3_out": t3_out,
                "frames_aligned": frames_aligned,
                "evidence": evidence,
                "evidence_units": evidence_units,
                "pair_records": pair_records,
                "candidate_pairs": [{"from": a, "to": b} for (a, b) in candidate_pairs],
                "reviewed_ok_pairs": [{"from": a, "to": b} for (a, b) in reviewed_ok_pairs],
            }
            unify_result = produce_task_samples(
                unify_bundle,
                llm=AuditedTextClient(self.llm) if enable_llm_generation else None,
                comm_c1_k=unify_comm_c1_k,
                isg_k=unify_isg_k,
                llm_retry_attempts=max(1, unify_llm_retry_attempts),
                selected_tasks=[bench for capability, bench in TASKS.values() if capability == "unified" and task_quotas.get(bench, 0)] if is_paper else None,
                strict_generation=is_paper,
            )
            unify_samples = unify_result.get("samples", {}) if isinstance(unify_result, dict) else {}
            for task_type, task_items in (unify_samples.items() if isinstance(unify_samples, dict) else []):
                if not isinstance(task_items, list):
                    continue
                for idx, sample in enumerate(task_items):
                    if not isinstance(sample, dict):
                        continue
                    item = {
                        "module": "unify",
                        "benchmark": str(task_type),
                        "unify_id": f"{video_id}_seg{segment_index:03d}_{str(task_type).replace('.', '_')}_{idx:03d}",
                        "turns": sample.get("turns", {}),
                        "answer": sample.get("answer", {}),
                        "render": sample.get("render", {}),
                        "evidence_refs": _ensure_segment_refs(sample.get("evidence_refs", [])),
                        "meta": {
                            **(sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}),
                            "module": "unify",
                            "video_path": video_path,
                            "segment_path": segment_path,
                            "segment_index": segment_index,
                            "source": "t4_unify_submodule",
                            "segment_keyframes": all_keyframes_meta if attach_all_keyframes_to_meta else None,
                        },
                    }
                    item = align_item(item, profile="paper" if is_paper else "legacy")
                    _append_instance(item)
                    _write_item_to_builder(item)


    # ---------------- 5) Runtime + Return ----------------
    runtime = {
        "paper_rejections": paper_rejections,
        "unify_drops": unify_result.get("drops", {}),
        "pending_generation": unify_result.get("pending", {}),
        "execution_pending": any(unify_result.get("pending", {}).values()),
        "gate_c": t3_out.get("gate_c"),
        "paper_tasks": decision.get("tasks", {}),
        "segment_index": segment_index,
        "segment_path": segment_path,
        "video_path": video_path,
        "tools_used": tools_used,
        "use_asr_for_vlm": use_asr_for_vlm,
        "use_ocr_for_vlm": use_ocr_for_vlm,
        "llm_decision_enabled": enable_llm_decision,
        "llm_generation_enabled": enable_llm_generation,
        "llm_edit_instruction_enabled": enable_llm_magicbrush_instr,
        "llm_decision_raw": llm_decision_raw if isinstance(llm_decision_raw, str) else "",
        "keyframe_count": len(frames_aligned),
        "keyframe_index_space": "pos_0_to_N-1",
        "t2_quality_gate_note": "T4 receives only T2-selected final keyframes; no frame-level filtered_indices chain is used in T4.",
        "evidence_unit_count": len(evidence_units),
        "raw_pair_count": len(raw_pairs),
        "reviewed_pair_count": len(reviewed_ok_pairs),
        "candidate_pair_count": len(candidate_pairs),
        "max_pair_review": max_pair_review,
        "attach_all_keyframes_to_meta": attach_all_keyframes_to_meta,
        # NEW dialog runtime
        "max_dialog_visdial": max_dialog_visdial,
        "max_dialog_mmdu": max_dialog_mmdu,
        "dialog_visdial_rounds": dialog_visdial_rounds,
        "dialog_mmdu_rounds": dialog_mmdu_rounds,
        "dialog_mmdu_image_count": dialog_mmdu_image_count,
        "generated_dialog_visdial": _instance_count("understanding.dialog_visdial_like"),
        "generated_dialog_mmdu": _instance_count("understanding.dialog_mmdu_like"),
        "generated_generation": _count_prefix("generation."),
        "generation_reviewed_count": len(generation_reviews),
        "generation_suitable_count": len([rv for rv in generation_reviews if bool(rv.get("suitable", False))]),
        "generated_unify": (
            _count_prefix("unify.")
        ),
        "selected_modules": list(decision.get("selected", []) or []),
        "module_route_reason": str(decision.get("module_route_reason", "") or ""),
        "dialog_debug": dialog_debug,

    }

    # ===================== [PATCH 4.5] dump dialog_debug =====================
    try:
        _dump_json("dialog_debug", dialog_debug)
    except Exception:
        pass
    # ===================== [END PATCH 4.5] =====================

    return {
        "evidence": evidence,
        "decision": decision,
        "instances": instances,
        "runtime": runtime,
    }









# =====================================================================
# Orchestrate full pipeline (across segments)
# =====================================================================
