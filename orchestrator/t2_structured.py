from __future__ import annotations

from typing import Any, Dict, List
import os
from orchestrator.planner_helpers import (
    chapter_from_jsonable,
    ensure_stage_workdir,
    prepare_t2_frame_pool,
    shot_from_jsonable,
    video_profile_from_jsonable,
)


def _build_missing_t1_structure_output() -> Dict[str, Any]:
    return {
        "output": {
            "keyframes": [],
            "quality_passed": False,
            "reason": "missing_t1_structure",
            "tool_plan": {},
            "tool_pool": [],
            "candidate_frames": [],
            "usable_pool": [],
            "candidate_pool": [],
            "gate_b": {},
            "semantic_profiles": [],
            "value_profiles": [],
            "quota_plan": {},
            "representative_selection": {},
            "gate_d": {},
            "fallback_flags": {
                "missing_t1_structure": True,
            },
        },
        "runtime": {"fallback_summary": {"missing_t1_structure": True}},
    }


def _flatten_candidates(candidates_by_shot: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for _, items in (candidates_by_shot or {}).items():
        out.extend(items or [])
    out.sort(key=lambda x: (str(x.get("shot_id", "")), float(x.get("sec", 0.0))))
    return out


def _build_final_keyframes(selected_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for item in selected_frames:
        value_profile = item.get("value_profile", {}) or {}
        origin_index = item.get("origin_index", None)
        if origin_index is None:
            origin_index = item.get("frame_index", item.get("index", None))
        rows.append(
            {
                "img": item.get("img"),
                "sec": float(item.get("sec", 0.0) or 0.0),
                "score": float(item.get("value_score", 0.0) or 0.0),
                "score_detail": value_profile,
                "shot_id": item.get("shot_id", ""),
                "chapter_id": item.get("chapter_id", ""),
                "candidate_type": item.get("candidate_type", ""),
                "quality_profile": item.get("quality_profile", {}),
                "origin_index": origin_index,
                "frame_index": item.get("frame_index", origin_index),
                "index": item.get("frame_index", origin_index),
            }
        )
    rows.sort(key=lambda x: float(x.get("sec", 0.0)))
    return rows


def _summarize_tools(tool_plan: Dict[str, Any]) -> tuple[List[str], Dict[str, str]]:
    tool_order = ["omniparser", "ocr", "asr", "yolo", "optical_flow"]
    seen = []
    reasons: Dict[str, str] = {}
    for chapter_plan in list((tool_plan.get("chapter_plans", {}) or {}).values()):
        actions = list((chapter_plan or {}).get("tool_actions", []) or [])
        for action in actions:
            tool_name = str((action or {}).get("tool_name", "") or "")
            plan_reason = str((action or {}).get("reason", "") or (chapter_plan or {}).get("plan_reason", "") or "")
            if not tool_name:
                continue
            if tool_name not in seen:
                seen.append(tool_name)
            reasons.setdefault(tool_name, plan_reason)
    for shot_plan in list((tool_plan.get("shot_plans", {}) or {}).values()):
        actions = list((shot_plan or {}).get("tool_actions", []) or [])
        for action in actions:
            tool_name = str((action or {}).get("tool_name", "") or "")
            plan_reason = str((action or {}).get("reason", "") or (shot_plan or {}).get("plan_reason", "") or "")
            if not tool_name:
                continue
            if tool_name not in seen:
                seen.append(tool_name)
            reasons.setdefault(tool_name, plan_reason)
    ordered = [x for x in tool_order if x in seen]
    ordered.extend([x for x in seen if x not in ordered])
    return ordered, reasons


def _filter_frames_to_kept_shots(frames: List[Dict[str, Any]], kept_shot_ids: set[str]) -> List[Dict[str, Any]]:
    if not frames or not kept_shot_ids:
        return []
    return [x for x in frames if str(x.get("shot_id", "") or "") in kept_shot_ids]


def _apply_final_gate_rules(
    selected_frames: List[Dict[str, Any]],
    shot_objs: List[Any],
    candidate_pool_count: int,
) -> tuple[bool, str]:
    if not selected_frames:
        return False, "no_frames_after_quota_and_representative_selection"

    selected_shot_ids = {str(x.get("shot_id", "")) for x in selected_frames if x.get("shot_id")}
    role_set = {str(getattr(x, "shot_role", "") or "generic") for x in shot_objs}
    high_info_count = sum(1 for x in shot_objs if str(getattr(x, "informativeness_hint", "") or "") == "high")

    if (
        len(selected_frames) <= 1
        and len(selected_shot_ids) <= 1
        and candidate_pool_count <= 1
        and high_info_count == 0
        and role_set.issubset({"generic", "talking_head"})
    ):
        return False, "single-shot low-diversity monologue-like content is not sufficient"

    return True, ""


def _select_final_frames_by_shot(frames: List[Dict[str, Any]], quota_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not frames:
        return []
    shot_plans = (quota_plan.get("shot_plans", {}) or {}) if isinstance(quota_plan, dict) else {}
    total_budget = int((quota_plan.get("global_plan", {}) or {}).get("total_budget", len(frames)))
    by_shot: Dict[str, List[Dict[str, Any]]] = {}
    for frame in frames:
        by_shot.setdefault(str(frame.get("shot_id", "")), []).append(frame)

    selected: List[Dict[str, Any]] = []
    for shot_id, shot_frames in by_shot.items():
        shot_quota = int((shot_plans.get(shot_id, {}) or {}).get("final_quota", len(shot_frames)))
        if shot_quota <= 0:
            continue
        picks = sorted(shot_frames, key=lambda x: float(x.get("value_score", 0.0)), reverse=True)[:shot_quota]
        selected.extend(picks)

    selected.sort(key=lambda x: float(x.get("value_score", 0.0)), reverse=True)
    return selected[:total_budget]


from orchestrator.prompt_audit import paper_prompt_stage, audited_text, audited_vision, AuditedTextClient

@paper_prompt_stage("T2")
def run_t2_structured(planner: Any, t1_stage: Dict[str, Any], segment: Any) -> Dict[str, Any]:
    """
    T2 flow:
    ToolPlan -> GateB(on raw T2 frame pool) -> Value -> Quota -> Representative select -> GateD.
    """
    t1_out = t1_stage.get("output", {}) or {}

    shot_objs = [shot_from_jsonable(x) for x in (t1_out.get("shot_profiles") or t1_out.get("shots") or [])]
    chapter_objs = [chapter_from_jsonable(x) for x in (t1_out.get("chapters") or [])]
    video_profile = video_profile_from_jsonable(t1_out.get("video_profile") or {}) if t1_out.get("video_profile") else None
    gate_a = t1_out.get("gate_a", {}) or {}
    kept_shot_ids = set(gate_a.get("kept_shot_ids") or [])

    if not shot_objs or not chapter_objs or video_profile is None:
        return _build_missing_t1_structure_output()

    shot_objs = [x for x in shot_objs if x.shot_id in kept_shot_ids] if kept_shot_ids else []
    if not shot_objs:
        return {
            "output": {
                "keyframes": [],
                "quality_passed": False,
                "reason": "gate_a_rejected_all_shots",
                "tool_plan": {},
                "tool_pool": [],
                "candidate_frames": [],
                "usable_pool": [],
                "candidate_pool": [],
                "gate_b": {"usable_frames": [], "rejected_frames": [], "all_frame_quality_profiles": []},
                "semantic_profiles": [],
                "value_profiles": [],
                "quota_plan": {},
                "gate_d": {"selected_frames": [], "selected_shot_ids": [], "final_pass": False, "final_reason": "no_kept_shots"},
                "fallback_flags": {
                    "gate_a_rejected_all_shots": True,
                },
            },
            "runtime": {"fallback_summary": {"gate_a_rejected_all_shots": True}},
        }
    chapter_id_by_shot = {}
    filtered_chapters = []
    for chapter in chapter_objs:
        kept_members = [sid for sid in chapter.shot_ids if sid in kept_shot_ids]
        if not kept_members:
            continue
        chapter.shot_ids = kept_members
        filtered_chapters.append(chapter)
        for shot_id in kept_members:
            chapter_id_by_shot[shot_id] = chapter.chapter_id
    chapter_objs = filtered_chapters
    for shot in shot_objs:
        shot.metadata["chapter_id"] = chapter_id_by_shot.get(shot.shot_id, "")

    video_path = segment.segment_path
    video_id = planner._video_id(segment.video_path)
    seg_id = f"seg_{segment.index:03d}"
    t2_workdir = ensure_stage_workdir(planner.temp_root, video_id, seg_id, "T2")
    tool_plan = planner.modules.tool_planner.plan(
        video_profile=video_profile,
        chapters=chapter_objs,
        shots=shot_objs,
        kept_shot_ids=list(kept_shot_ids),
        llm=planner.llm,
    )

    frame_pool = prepare_t2_frame_pool(
        video_path=video_path,
        shots=shot_objs,
        t2_workdir=t2_workdir,
        config=getattr(planner, "config", {}) or {},
    )

    frame_pool = _filter_frames_to_kept_shots(frame_pool, kept_shot_ids)
    for item in frame_pool:
        if not item.get("chapter_id"):
            item["chapter_id"] = chapter_id_by_shot.get(item.get("shot_id", ""), "")

    candidate_frames = list(frame_pool)
    candidates_by_shot = planner.modules.tool_executor.group_by_shot(candidate_frames)

    gate_b = planner.modules.gate_b_quality.run(candidate_frames, planner.model_manager)
    usable_pool = gate_b.get("usable_frames", []) or []
    usable_by_shot = planner.modules.tool_executor.group_by_shot(usable_pool)

    tool_execution = planner.modules.tool_executor.execute(
        shots=shot_objs,
        candidates_by_shot=usable_by_shot,
        pre_gate_candidates_by_shot=candidates_by_shot,
        tool_plan=tool_plan,
        video_path=video_path,
        t2_workdir=t2_workdir,
        model_manager=planner.model_manager,
    )

    enriched_candidates_by_shot = tool_execution.get("enriched_candidates_by_shot", {}) or {}
    tool_pool = _flatten_candidates(enriched_candidates_by_shot)
    tool_pool = _filter_frames_to_kept_shots(tool_pool, kept_shot_ids)

    semantic_profiles = planner.modules.frame_signal_semanticizer.build(
        enriched_frames=tool_pool,
        shots=shot_objs,
        chapters=chapter_objs,
        video_profile=video_profile,
        tool_plan=tool_plan,
        llm=planner.llm,
    )

    value_profiles = planner.modules.llm_value_scorer.analyze(
        semanticized_frames=semantic_profiles,
        shots=shot_objs,
        chapters=chapter_objs,
        video_profile=video_profile,
        tool_plan=tool_plan,
        llm=planner.llm,
    )

    candidate_pool = _filter_frames_to_kept_shots(value_profiles, kept_shot_ids)
    tools_used, tool_reasons = _summarize_tools(tool_plan)

    quota_plan = planner.modules.quota_planner.plan(
        video_profile=video_profile,
        chapters=chapter_objs,
        shots=shot_objs,
        value_frames=candidate_pool,
        llm=planner.llm,
    )

    representative_selection = planner.modules.representative_selector.select(
        value_frames=candidate_pool,
        quota_plan=quota_plan,
        model_manager=planner.model_manager,
    )

    selected_frames = representative_selection.get("selected_frames", []) or []
    selected_frames = _filter_frames_to_kept_shots(selected_frames, kept_shot_ids)

    keyframes = _build_final_keyframes(selected_frames)
    shot_boundary_source = t1_out.get("shot_boundaries") or []
    transition_indices = list(t1_out.get("transition_indices") or [])
    aux = dict((t1_out.get("aux") or {}) if isinstance(t1_out.get("aux"), dict) else {})
    aux.setdefault("per_frame_flow", list(aux.get("per_frame_flow") or []))
    final_pass, final_reason = _apply_final_gate_rules(
        selected_frames=selected_frames,
        shot_objs=shot_objs,
        candidate_pool_count=len(candidate_pool),
    )

    gate_d = {
        "selected_frames": selected_frames,
        "selected_frame_indices": [int(x.get("frame_index", -1)) for x in selected_frames],
        "selected_shot_ids": sorted(list({str(x.get("shot_id", "")) for x in selected_frames if x.get("shot_id")})),
        "final_pass": bool(final_pass),
        "final_reason": final_reason,
        "coverage_summary": {
            "selected_count": len(selected_frames),
            "pre_gate_candidate_count": len(candidate_frames),
            "tool_pool_count": len(tool_pool),
            "usable_count": len(usable_pool),
            "candidate_count": len(candidate_pool),
        },
    }

    return {
        "output": {
            "keyframes": keyframes,
            "quality_passed": bool(gate_d["final_pass"]),
            "reason": str(gate_d["final_reason"] or ""),
            "tool_plan": tool_plan,
            "tool_execution": tool_execution,
            "tools": tools_used,
            "tool_reasons": tool_reasons,
            "tool_pool": tool_pool,
            "candidate_frames": candidate_frames,
            "usable_pool": usable_pool,
            "candidate_pool": candidate_pool,
            "gate_b": gate_b,
            "semantic_profiles": semantic_profiles,
            "value_profiles": value_profiles,
            "quota_plan": quota_plan,
            "representative_selection": representative_selection,
            "gate_d": gate_d,
            "shot_boundaries": list(shot_boundary_source) if isinstance(shot_boundary_source, list) else [],
            "transition_indices": transition_indices,
            "flow_pairs": list(aux.get("flow_pairs") or []),
            "optical_flow_pairs": list(aux.get("optical_flow_pairs") or []),
            "aux": aux,
            "fallback_flags": {
                "gate_b_model_stage_fallback_used": bool(
                    (gate_b.get("fallback_flags", {}) or {}).get("model_stage_fallback_used", False)
                ),
                "representative_selector_embedding_fallback_used": bool(
                    (representative_selection.get("fallback_flags", {}) or {}).get("embedding_fallback_used", False)
                ),
            },
        },
        "runtime": {
            "t2_workdir": t2_workdir,
            "fallback_summary": {
                "gate_b": gate_b.get("fallback_flags", {}) or {},
                "representative_selection": representative_selection.get("fallback_flags", {}) or {},
            },
        },
    }
