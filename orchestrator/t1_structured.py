from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from orchestrator.planner_helpers import (
    build_t1_debug_pack,
    ensure_stage_workdir,
    estimate_structure_confidence,
    prepare_t1_frame_items,
    to_jsonable_record,
)


def _build_no_shots_output(segment: Any, shot_backend: str, shot_meta: Dict[str, Any]) -> Dict[str, Any]:
    fallback_used = bool(shot_backend in {"opencv", "fixed"})
    return {
        "output": {
            "shots": [],
            "shot_profiles": [],
            "chapters": [],
            "video_profile": {},
            "gate_a": {
                "segment_summary": {"total_shots": 0, "kept_shots": 0, "dropped_shots": 0},
                "shot_decisions": [],
                "kept_shot_ids": [],
                "dropped_shot_ids": [],
            },
            "split_points_sec": [],
            "should_split": False,
            "structure_confidence": 0.0,
            "video_type": "unknown",
            "global_summary": "",
            "summary": "",
            "motion_hint": "medium_motion",
            "suitable_for_dataset": False,
            "dataset_reason": "no_shots",
            "fallback_flags": {
                "shot_detector_fallback_used": fallback_used,
                "shot_detector_empty_output": True,
            },
        },
        "runtime": {
            "t1_workdir": "",
            "shot_detector_backend": shot_backend,
            "shot_detector_meta": shot_meta,
            "pairwise_merge_decisions": [],
            "smoothing_meta": {},
            "debug_pack": {"reason": "no_shots"},
            "fallback_summary": {
                "shot_detector_backend": shot_backend,
                "shot_detector_fallback_used": fallback_used,
                "no_shots_fallback_used": True,
                "shot_profile_vlm_fallback_shots": [],
                "shot_role_fallback_shots": [],
                "chapter_pairwise_llm_fallback_pairs": [],
            },
        },
    }


from orchestrator.prompt_audit import paper_prompt_stage, audited_text, audited_vision, AuditedTextClient
from orchestrator.iteration.taxonomy import domain_from_video_type
from orchestrator.iteration.metadata_audit import audit_domain

@paper_prompt_stage("T1")
def run_t1_structured(planner: Any, segment: Any) -> Dict[str, Any]:
    """
    T1 flow:
    shot detection -> local frame sampling -> VLM-only shot profiles -> chapter/video profiles -> GateA.
    """
    video_path = segment.segment_path
    video_id = planner._video_id(segment.video_path)
    seg_id = f"seg_{segment.index:03d}"
    t1_workdir = ensure_stage_workdir(planner.temp_root, video_id, seg_id, "T1")

    shot_detector = planner.model_manager.get_resource("shot_detector")
    shot_result = shot_detector.detect(
        video_path=video_path,
        prefer_backend="pyscenedetect",
        allow_fallback_backend=True,
        video_type_hint="",
    )
    raw_shots = shot_result.get("shots", []) or []
    shot_backend = shot_result.get("backend", "none")
    shot_meta = shot_result.get("meta", {}) or {}
    for shot in raw_shots:
        shot["detector_backend"] = shot_backend

    if not raw_shots:
        return _build_no_shots_output(segment, shot_backend, shot_meta)

    frame_items = prepare_t1_frame_items(
        video_path=video_path,
        t0_frame_items=[],
        raw_shots=raw_shots,
        t1_workdir=t1_workdir,
    )

    shot_records = planner.modules.shot_profiler.build_shot_profiles(
        video_path=video_path,
        shots=raw_shots,
        frame_items=frame_items,
        model_manager=planner.model_manager,
        workdir=t1_workdir,
    )

    merge_result = planner.modules.chapter_builder.compute_pairwise_merge_decisions(
        shots=shot_records,
        model_manager=planner.model_manager,
    )
    pairwise_merge_decisions = merge_result.get("pairwise_merge_decisions", []) or []
    pairwise_graph = merge_result.get("pairwise_graph", {}) or {}

    chapter_result = planner.modules.chapter_builder.smooth_and_group(
        shots=shot_records,
        pairwise_merge_decisions=pairwise_merge_decisions,
        pairwise_graph=pairwise_graph,
    )
    chapter_records = chapter_result.get("chapters", []) or []
    smoothing_meta = chapter_result.get("smoothing_meta", {}) or {}

    shot_id_to_chapter = {}
    for chapter in chapter_records:
        for shot_id in chapter.shot_ids:
            shot_id_to_chapter[shot_id] = chapter.chapter_id
    for shot in shot_records:
        shot.metadata["chapter_id"] = shot_id_to_chapter.get(shot.shot_id, "")

    profile_result = planner.modules.structure_profiler.profile_video_and_chapters(
        shots=shot_records,
        chapters=chapter_records,
        model_manager=planner.model_manager,
        video_id=video_id,
    )
    video_profile = profile_result.get("video_profile")
    chapter_records = profile_result.get("chapters", chapter_records)

    gate_a = planner.modules.gate_a_shot_admission.run(
        video_profile=video_profile,
        chapters=chapter_records,
        shots=shot_records,
        model_manager=planner.model_manager,
    )

    structure_confidence = estimate_structure_confidence(chapter_records, pairwise_merge_decisions)
    debug_pack = build_t1_debug_pack(shot_records, chapter_records, pairwise_merge_decisions, smoothing_meta, shot_meta)

    video_type = getattr(video_profile, "video_type", "") if video_profile else ""
    global_summary = getattr(video_profile, "video_summary", "") if video_profile else ""
    runtime_policy = ((getattr(planner, "config", {}) or {}).get("runtime", {}) or {}).get("iteration_policy")
    policy_domain = domain_from_video_type(video_type)
    domain_source = "global_perception" if policy_domain else None
    domain_confidence = None
    domain_audit_evidence = []
    domain_audit_status = "not_needed"
    domain_audit_model_identity = {}
    domain_audit_prompt_sha256 = None
    if isinstance(runtime_policy, dict) and not policy_domain:
        audit_images = []
        for shot in shot_records:
            for frame in list(getattr(shot, "rep_frames", []) or []):
                image = str(getattr(frame, "img_path", "") or "")
                if image and Path(image).exists() and image not in audit_images:
                    audit_images.append(image)
                if len(audit_images) >= 4:
                    break
            if len(audit_images) >= 4:
                break
        audited = audit_domain(planner, audit_images, global_summary, video_type)
        if audited:
            video_type = audited["video_type"]
            policy_domain = audited["policy_domain"]
            domain_source = audited["domain_source"]
            domain_confidence = audited["domain_confidence"]
            domain_audit_evidence = audited["domain_audit_evidence"]
            domain_audit_status = audited.get("domain_audit_status", "unknown")
            domain_audit_model_identity = audited.get("domain_audit_model_identity", {})
            domain_audit_prompt_sha256 = audited.get("domain_audit_prompt_sha256")
    high_sampling = sum(1 for x in shot_records if str(x.candidate_sampling_hint or "").lower() == "high")
    low_sampling = sum(1 for x in shot_records if str(x.candidate_sampling_hint or "").lower() == "low")
    if high_sampling >= max(1, len(shot_records) // 2):
        motion_hint = "high_motion"
    elif low_sampling >= max(1, len(shot_records) // 2):
        motion_hint = "low_motion"
    else:
        motion_hint = "medium_motion"

    output = {
        "shots": [to_jsonable_record(x) for x in shot_records],
        "shot_profiles": [to_jsonable_record(x) for x in shot_records],
        "chapters": [to_jsonable_record(x) for x in chapter_records],
        "video_profile": to_jsonable_record(video_profile) if video_profile else {},
        "gate_a": gate_a,
        "split_points_sec": [],
        "should_split": False,
        "structure_confidence": float(structure_confidence),
        "video_type": video_type or "generic",
        "global_summary": global_summary,
        "summary": global_summary,
        "motion_hint": motion_hint,
        "suitable_for_dataset": bool(gate_a.get("kept_shot_ids")),
        "dataset_reason": "" if gate_a.get("kept_shot_ids") else "gate_a_rejected_all_shots",
        "split_reason": "disabled_fixed_presplit_only",
        "split_paths": [],
        "fallback_flags": {
            "shot_detector_fallback_used": bool(shot_backend in {"opencv", "fixed"}),
            "shot_profile_vlm_fallback_used": any(
                bool(((getattr(x, "metadata", {}) or {}).get("fallback_flags", {}) or {}).get("vlm_profile_fallback_used", False))
                for x in shot_records
            ),
            "shot_role_fallback_used": any(
                bool(((getattr(x, "metadata", {}) or {}).get("fallback_flags", {}) or {}).get("role_fallback_used", False))
                for x in shot_records
            ),
            "chapter_pairwise_llm_fallback_used": any(
                bool((d.get("llm_fallback_used", False))) for d in pairwise_merge_decisions
            ),
        },
    }
    if isinstance(runtime_policy, dict):
        output["policy_domain"] = policy_domain
        output["domain_source"] = domain_source or "global_perception_unresolved"
        output["domain_confidence"] = domain_confidence if domain_confidence is not None else float(
            ((getattr(video_profile, "metadata", {}) or {}).get("domain_confidence", 0.0) or 0.0)
        ) if video_profile else None
        if domain_audit_evidence:
            output["domain_audit_evidence"] = domain_audit_evidence
        output["domain_audit_status"] = domain_audit_status
        output["domain_audit_model_identity"] = domain_audit_model_identity
        output["domain_audit_prompt_sha256"] = domain_audit_prompt_sha256
        if not policy_domain:
            output["domain_unknown_reason"] = f"metadata_audit:{domain_audit_status}"
        output["iteration_policy"] = {
            "iteration": runtime_policy.get("iteration"),
            "video_type_weight": (runtime_policy.get("video_type_weights", {}) or {}).get(video_type, 1.0),
        }
    return {
        "output": output,
        "runtime": {
            "t1_workdir": t1_workdir,
            "shot_detector_backend": shot_backend,
            "shot_detector_meta": shot_meta,
            "pairwise_merge_decisions": pairwise_merge_decisions,
            "smoothing_meta": smoothing_meta,
            "debug_pack": debug_pack,
            "fallback_summary": {
                "shot_detector_backend": shot_backend,
                "shot_detector_fallback_used": bool(shot_backend in {"opencv", "fixed"}),
                "no_shots_fallback_used": False,
                "shot_profile_vlm_fallback_shots": [
                    x.shot_id
                    for x in shot_records
                    if bool(((getattr(x, "metadata", {}) or {}).get("fallback_flags", {}) or {}).get("vlm_profile_fallback_used", False))
                ],
                "shot_role_fallback_shots": [
                    x.shot_id
                    for x in shot_records
                    if bool(((getattr(x, "metadata", {}) or {}).get("fallback_flags", {}) or {}).get("role_fallback_used", False))
                ],
                "chapter_pairwise_llm_fallback_pairs": [
                    f"{d.get('left_shot_id', '')}->{d.get('right_shot_id', '')}"
                    for d in pairwise_merge_decisions
                    if bool(d.get("llm_fallback_used", False))
                ],
            },
        },
    }
