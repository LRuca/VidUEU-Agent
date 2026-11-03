from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple
from executor.tasks.prompt_loader import current_prompt_profile


def build_evidence_pack(
    *,
    video_id: str,
    segment_index: int,
    video_type: str,
    modality_hint: str,
    motion_hint: str,
    global_summary: str,
    segment_summary: str,
    shot_summaries_t3: List[Dict[str, Any]],
    chapter_summaries_t3: List[Dict[str, Any]],
    frames_aligned: List[Dict[str, Any]],
    all_keyframes_meta: List[Dict[str, Any]],
    evidence_units: List[Dict[str, Any]],
    tools_used: List[str],
    tool_reasons: Dict[str, Any],
    shot_boundaries: List[Any],
    transition_indices: List[Any],
    raw_pairs_sorted: List[Tuple[int, int]],
    pair_hints: Dict[Tuple[int, int], Dict[str, Any]],
    change_map_struct: Dict[Tuple[int, int], Dict[str, Any]],
    reviewed_ok_pairs: List[Tuple[int, int]],
    review_cache: Dict[Tuple[int, int], Dict[str, Any]],
    candidate_pairs: List[Tuple[int, int]],
    add_ev: Callable[..., None],
    dump_text: Callable[[str, str], None],
    short_text: Callable[[str, int], str],
    tool_evidence_suffix: Callable[[str], str],
) -> Dict[str, Any]:
    change_map = {
        (int(fr.get("pos", 0) or 0), int(fr.get("pos", 0) or 0) + 1): (fr.get("transition_to_next", "") or "").strip()
        for fr in frames_aligned[:-1]
    }

    for idx, shot_row in enumerate(shot_summaries_t3):
        if not isinstance(shot_row, dict):
            continue
        shot_summary = str(shot_row.get("summary", "") or "").strip()
        if not shot_summary:
            continue
        add_ev(
            f"S_SHOT_{idx:02d}_SUM",
            "segment",
            f"shot_summary[{str(shot_row.get('shot_id', '') or '')}]: {short_text(shot_summary, 700)}",
            "t3",
            None,
            signal_type="summary",
            shot_id=str(shot_row.get("shot_id", "") or ""),
            chapter_id=str(shot_row.get("chapter_id", "") or ""),
            payload=shot_row,
        )

    for idx, chapter_row in enumerate(chapter_summaries_t3):
        if not isinstance(chapter_row, dict):
            continue
        chapter_summary = str(chapter_row.get("summary", "") or "").strip()
        if not chapter_summary:
            continue
        add_ev(
            f"S_CHAPTER_{idx:02d}_SUM",
            "segment",
            f"chapter_summary[{str(chapter_row.get('chapter_id', '') or '')}]: {short_text(chapter_summary, 800)}",
            "t3",
            None,
            signal_type="summary",
            chapter_id=str(chapter_row.get("chapter_id", "") or ""),
            payload=chapter_row,
        )

    add_ev("S_SEGEMENT_SUM", "segment", short_text(segment_summary, 900), "t3", None, signal_type="summary")

    for fr in frames_aligned:
        pos = int(fr.get("pos", 0) or 0)
        sec = fr.get("sec", None)
        cap = (fr.get("caption", "") or "").strip()
        if cap:
            add_ev(f"F{pos:02d}_CAP", "frame", f"Frame F{pos:02d} (t={sec}): caption: {cap}", "t3", pos, signal_type="caption")
        state = fr.get("visual_state")
        if isinstance(state, dict) and state.get("validation_status") == "contract_valid":
            add_ev(f"F{pos:02d}_STATE", "frame", json.dumps(state, ensure_ascii=False), "t3", pos,
                   signal_type="entity_state", payload=state)

        frame_tool_descs = fr.get("frame_tool_descs", {}) if isinstance(fr.get("frame_tool_descs", {}), dict) else {}
        for tool_name in sorted(frame_tool_descs.keys()):
            tool_summary = str(frame_tool_descs.get(tool_name, "") or "").strip()
            if not tool_summary:
                continue
            suffix = tool_evidence_suffix(tool_name)
            add_ev(
                f"F{pos:02d}_TOOL_{suffix}",
                "frame",
                f"Frame F{pos:02d}: {tool_name} semantic hint: {short_text(tool_summary, 360)}",
                f"tool_{tool_name}",
                pos,
                signal_type=str(tool_name or "").strip().lower(),
                payload={"tool_name": tool_name, "semantic_text": fr.get("frame_semantic_text", "")},
            )

    pair_records: List[Dict[str, Any]] = []
    for (a, b) in raw_pairs_sorted[:30]:
        txt = change_map.get((a, b), "") or ""
        hint = pair_hints.get((a, b), {}) or {}
        ts = change_map_struct.get((a, b), {}) if isinstance(change_map_struct.get((a, b), {}), dict) else {}
        pair_key = f"F{a:02d}->F{b:02d}"
        pair_records.append(
            {
                "segment_index": segment_index,
                "pair_from_pos": a,
                "pair_to_pos": b,
                "pair_key": pair_key,
                "from_shot_id": frames_aligned[a].get("shot_id", ""),
                "to_shot_id": frames_aligned[b].get("shot_id", ""),
                "chapter_id": frames_aligned[a].get("chapter_id", "") or frames_aligned[b].get("chapter_id", ""),
                "transition_text": txt,
                "cause": str(hint.get("t3_cause", "") or ""),
                "t3_struct": ts,
                "entity_comparison": (frames_aligned[a].get("transition_validation") or {}).get("entity_comparison"),
                "review": None,
            }
        )

        if txt:
            add_ev(
                f"P{len(pair_records)-1:02d}_TRANS",
                "pair",
                f"Pair {pair_key}: {txt}",
                "t3",
                a,
                signal_type="transition",
                shot_id=frames_aligned[a].get("shot_id", ""),
                chapter_id=frames_aligned[a].get("chapter_id", ""),
                pair_from_pos=a,
                pair_to_pos=b,
                pair_key=pair_key,
                payload={"t3_struct": ts, "pair_hints": hint},
            )
        comparison = pair_records[-1].get("entity_comparison")
        if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"} and isinstance(comparison, dict):
            add_ev(f"P{len(pair_records)-1:02d}_COMPARE", "pair",
                   json.dumps(comparison, ensure_ascii=False), "t3", a,
                   signal_type="entity_comparison", pair_from_pos=a, pair_to_pos=b,
                   pair_key=pair_key, payload=comparison)

    reviewed_pair_to_pid: Dict[Tuple[int, int], str] = {}
    reviewed_pairs_text: List[str] = []
    pcount = 0
    for (a, b) in reviewed_ok_pairs[:30]:
        rv = review_cache.get((a, b), {}) or {}
        instr = (rv.get("instruction", "") or "").strip()
        edits = rv.get("edits", []) if isinstance(rv.get("edits", []), list) else []
        hint = rv.get("pair_hints", {}) if isinstance(rv.get("pair_hints", {}), dict) else pair_hints.get((a, b), {})
        if not instr:
            continue
        pair_key = f"F{a:02d}->F{b:02d}"
        line = (
            f"Reviewed pair P{pcount:02d}: F{a:02d}->F{b:02d} ok=true; "
            f"edit_count={int(rv.get('edit_count', 0) or 0)}; edits={edits}; "
            f"instruction={instr}; "
            f"(pair_hints: endcard_like={bool(hint.get('endcard_like', False))}; caption_jaccard={float(hint.get('caption_jaccard', 0.0)):.3f})"
        )
        add_ev(
            f"P{pcount:02d}_REVIEW",
            "pair",
            line,
            "review",
            a,
            signal_type="decision",
            shot_id=frames_aligned[a].get("shot_id", ""),
            chapter_id=frames_aligned[a].get("chapter_id", ""),
            pair_from_pos=a,
            pair_to_pos=b,
            pair_key=pair_key,
            payload=rv,
        )
        reviewed_pair_to_pid[(a, b)] = f"P{pcount:02d}_REVIEW"
        for rec in pair_records:
            if rec.get("pair_key") == pair_key:
                rec["review"] = rv
                break
        reviewed_pairs_text.append(line)
        pcount += 1

    evidence_pack_text = "\n".join(
        [f"[{ev['evidence_id']}] {ev['text']}" for ev in evidence_units if ev.get("evidence_id") and ev.get("text")]
    ).strip()
    dump_text("evidence_pack", evidence_pack_text)

    evidence = {
        "video_id": video_id,
        "segment_index": segment_index,
        "video_type": video_type,
        "modality_hint": modality_hint,
        "motion_hint": motion_hint,
        "global_summary": global_summary,
        "segment_summary": segment_summary,
        "summary_hierarchy": {
            "shot_summaries": shot_summaries_t3,
            "chapter_summaries": chapter_summaries_t3,
            "segment_summary": segment_summary,
        },
        "frames": [
            {
                "pos": fr["pos"],
                "frame_pos": fr["frame_pos"],
                "origin_index": fr["origin_index"],
                "img": fr["img"],
                "sec": fr.get("sec"),
                "shot_id": fr.get("shot_id", ""),
                "chapter_id": fr.get("chapter_id", ""),
                "caption": fr.get("caption", ""),
                "cause": fr.get("cause", ""),
                "transition_to_next": fr.get("transition_to_next", ""),
                "score": fr.get("score", None),
                "tool_semantic": fr.get("tool_semantic", {}),
                "frame_tool_descs": fr.get("frame_tool_descs", {}),
                "frame_semantic_text": fr.get("frame_semantic_text", ""),
            }
            for fr in frames_aligned
        ],
        "all_keyframes": all_keyframes_meta,
        "evidence_units": evidence_units,
        "evidence_pack_text": evidence_pack_text,
        "tools_used": tools_used,
        "tool_reasons": tool_reasons,
        "structure": {
            "shot_boundaries": shot_boundaries,
            "transition_indices": transition_indices,
        },
        "aligned_pairs": pair_records,
        "candidate_pairs": [{"from": a, "to": b} for (a, b) in candidate_pairs],
        "reviewed_ok_pairs": [{"from": a, "to": b, "review": review_cache.get((a, b), {})} for (a, b) in reviewed_ok_pairs],
    }

    return {
        "pair_records": pair_records,
        "reviewed_pair_to_pid": reviewed_pair_to_pid,
        "reviewed_pairs_text": reviewed_pairs_text,
        "evidence_pack_text": evidence_pack_text,
        "evidence": evidence,
    }
