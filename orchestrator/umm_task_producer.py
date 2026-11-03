from __future__ import annotations

from executor.tasks.prompt_loader import current_prompt_profile

import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from orchestrator import umm_task_prompts
from orchestrator.planner_helpers import parse_llm_json
from orchestrator.evidence_selection import complete_evidence_text


TASK_SEED = "unify.seed.l3.gts"
TASK_WEAVE = "unify.weave.uie_select_recall"
TASK_COMM_C1 = "unify.comm.task4.c1_steps"
TASK_COMM_C2 = "unify.comm.task4.c2_fill_gap"
TASK_ISG = "unify.isg.i2.seq_select"
SEGMENT_EVIDENCE_ID = "S_SEGEMENT_SUM"
ALT_SEGMENT_EVIDENCE_IDS = ("S_SEGEMENT_SUM", "S_SUM", "S_GSUM")


def _short_text(text: Any, limit: int = 240) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    head = value[: max(0, limit - 3)].rstrip()
    if " " in head:
        head = head.rsplit(" ", 1)[0].rstrip()
    return head + "..."


def _as_pair_tuple(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None
    if isinstance(value, dict):
        try:
            return int(value.get("from")), int(value.get("to"))
        except Exception:
            return None
    return None


def _pair_key(a: int, b: int) -> str:
    return f"F{int(a):02d}->F{int(b):02d}"


def _frame_cap_id(pos: int) -> str:
    return f"F{int(pos):02d}_CAP"


@dataclass
class EvidenceClosure:
    evidence_ids: List[str]
    evidence_text: str
    frame_positions: List[int]
    pair_keys: List[str]


@dataclass
class TaskContext:
    segment: Dict[str, Any]
    t1_out: Dict[str, Any]
    t2_out: Dict[str, Any]
    t3_out: Dict[str, Any]
    evidence: Dict[str, Any]
    evidence_units: List[Dict[str, Any]]
    frames: List[Dict[str, Any]]
    pair_records: List[Dict[str, Any]]
    candidate_pairs: List[Tuple[int, int]]
    reviewed_ok_pairs: List[Tuple[int, int]]
    evidence_by_id: Dict[str, Dict[str, Any]]
    pair_record_by_tuple: Dict[Tuple[int, int], Dict[str, Any]]
    pair_transition_id_by_tuple: Dict[Tuple[int, int], str]
    pair_review_id_by_tuple: Dict[Tuple[int, int], str]
    frame_by_pos: Dict[int, Dict[str, Any]]
    segment_evidence_id: str
    shot_summary_ids: Dict[str, str] = field(default_factory=dict)
    chapter_summary_ids: Dict[str, str] = field(default_factory=dict)


@dataclass
class PairCandidate:
    from_pos: int
    to_pos: int
    pair_key: str
    pair_ref: str
    review_ref: Optional[str]
    frame_refs: List[str]
    score: float
    pair_record: Dict[str, Any]


@dataclass
class CommC1Candidate:
    frame_positions: List[int]
    pair_refs: List[str]
    review_refs: List[str]
    score: float


@dataclass
class CommC2Candidate:
    frame_positions: List[int]
    pair_refs: List[str]
    review_refs: List[str]
    score: float


@dataclass
class ISGCandidate:
    context_positions: List[int]
    future_positions: List[int]
    pair_refs: List[str]
    review_refs: List[str]
    score: float


def build_task_context(bundle: Dict[str, Any]) -> TaskContext:
    segment = dict(bundle.get("segment", {}) or {})
    t1_out = dict(bundle.get("t1_out", {}) or {})
    t2_out = dict(bundle.get("t2_out", {}) or {})
    t3_out = dict(bundle.get("t3_out", {}) or {})
    evidence = dict(bundle.get("evidence", {}) or {})
    evidence_units = list(bundle.get("evidence_units") or evidence.get("evidence_units") or [])
    frames = list(bundle.get("frames_aligned") or evidence.get("frames") or [])
    pair_records = list(bundle.get("pair_records") or evidence.get("aligned_pairs") or [])

    evidence_by_id: Dict[str, Dict[str, Any]] = {}
    shot_summary_ids: Dict[str, str] = {}
    chapter_summary_ids: Dict[str, str] = {}
    for unit in evidence_units:
        if not isinstance(unit, dict):
            continue
        evidence_id = str(unit.get("evidence_id", "") or "").strip()
        if not evidence_id:
            continue
        evidence_by_id[evidence_id] = unit
        shot_id = str(unit.get("shot_id", "") or "").strip()
        chapter_id = str(unit.get("chapter_id", "") or "").strip()
        if evidence_id.startswith("S_SHOT_") and shot_id and shot_id not in shot_summary_ids:
            shot_summary_ids[shot_id] = evidence_id
        if evidence_id.startswith("S_CHAPTER_") and chapter_id and chapter_id not in chapter_summary_ids:
            chapter_summary_ids[chapter_id] = evidence_id

    segment_evidence_id = ""
    for candidate_id in ALT_SEGMENT_EVIDENCE_IDS:
        if candidate_id in evidence_by_id:
            segment_evidence_id = candidate_id
            break
    if not segment_evidence_id:
        for evidence_id in evidence_by_id:
            if evidence_id.startswith("S_"):
                segment_evidence_id = evidence_id
                break
    if not segment_evidence_id:
        segment_evidence_id = SEGMENT_EVIDENCE_ID

    candidate_pairs: List[Tuple[int, int]] = []
    for item in list(bundle.get("candidate_pairs") or evidence.get("candidate_pairs") or []):
        key = _as_pair_tuple(item)
        if key is not None and key not in candidate_pairs:
            candidate_pairs.append(key)

    reviewed_ok_pairs: List[Tuple[int, int]] = []
    for item in list(bundle.get("reviewed_ok_pairs") or evidence.get("reviewed_ok_pairs") or []):
        key = _as_pair_tuple(item)
        if key is not None and key not in reviewed_ok_pairs:
            reviewed_ok_pairs.append(key)

    frame_by_pos: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        try:
            pos = int(frame.get("pos", frame.get("index", -1)))
        except Exception:
            continue
        normalized = dict(frame)
        normalized["pos"] = pos
        frame_by_pos[pos] = normalized

    pair_record_by_tuple: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for record in pair_records:
        if not isinstance(record, dict):
            continue
        key = _as_pair_tuple({"from": record.get("pair_from_pos"), "to": record.get("pair_to_pos")})
        if key is not None:
            pair_record_by_tuple[key] = record

    pair_transition_id_by_tuple: Dict[Tuple[int, int], str] = {}
    pair_review_id_by_tuple: Dict[Tuple[int, int], str] = {}
    for evidence_id, unit in evidence_by_id.items():
        key = _as_pair_tuple({"from": unit.get("pair_from_pos"), "to": unit.get("pair_to_pos")})
        if key is None:
            continue
        if evidence_id.endswith("_TRANS"):
            pair_transition_id_by_tuple[key] = evidence_id
        elif evidence_id.endswith("_REVIEW"):
            pair_review_id_by_tuple[key] = evidence_id

    # Backfill simple consecutive pairs from frame transitions when the T4 pack
    # did not materialize aligned_pairs/candidate_pairs, which happens in some
    # older runs or non-edit-friendly segments.
    ordered_positions = sorted(
        int(frame.get("pos", frame.get("index", -1)))
        for frame in frames
        if isinstance(frame, dict) and str(frame.get("caption", "") or "").strip()
    )
    for idx in range(len(ordered_positions) - 1):
        a = ordered_positions[idx]
        b = ordered_positions[idx + 1]
        frame_a = frame_by_pos.get(a, {})
        frame_b = frame_by_pos.get(b, {})
        transition_text = str(frame_a.get("transition_to_next", "") or "").strip()
        if not transition_text:
            continue
        pair_key = (a, b)
        if pair_key not in pair_record_by_tuple:
            pair_record_by_tuple[pair_key] = {
                "pair_from_pos": a,
                "pair_to_pos": b,
                "pair_key": _pair_key(a, b),
                "transition_text": transition_text,
                "from_shot_id": frame_a.get("shot_id", ""),
                "to_shot_id": frame_b.get("shot_id", ""),
                "chapter_id": frame_a.get("chapter_id", "") or frame_b.get("chapter_id", ""),
                "review": {},
            }
        if pair_key not in pair_transition_id_by_tuple and _frame_cap_id(a) in evidence_by_id and _frame_cap_id(b) in evidence_by_id:
            pair_transition_id_by_tuple[pair_key] = f"DERIVED_{_pair_key(a, b)}_TRANS"
            evidence_by_id[pair_transition_id_by_tuple[pair_key]] = {
                "evidence_id": pair_transition_id_by_tuple[pair_key],
                "text": f"Pair {_pair_key(a, b)}: {transition_text}",
                "pair_from_pos": a,
                "pair_to_pos": b,
                "level": "pair",
                "source": "derived_transition",
                "shot_id": frame_a.get("shot_id", ""),
                "chapter_id": frame_a.get("chapter_id", "") or frame_b.get("chapter_id", ""),
            }

    return TaskContext(
        segment=segment,
        t1_out=t1_out,
        t2_out=t2_out,
        t3_out=t3_out,
        evidence=evidence,
        evidence_units=evidence_units,
        frames=frames,
        pair_records=pair_records,
        candidate_pairs=candidate_pairs,
        reviewed_ok_pairs=reviewed_ok_pairs,
        evidence_by_id=evidence_by_id,
        pair_record_by_tuple=pair_record_by_tuple,
        pair_transition_id_by_tuple=pair_transition_id_by_tuple,
        pair_review_id_by_tuple=pair_review_id_by_tuple,
        frame_by_pos=frame_by_pos,
        segment_evidence_id=segment_evidence_id,
        shot_summary_ids=shot_summary_ids,
        chapter_summary_ids=chapter_summary_ids,
    )


def _score_pair(ctx: TaskContext, pair: Tuple[int, int], prefer_review: bool) -> float:
    a, b = pair
    frame_a = ctx.frame_by_pos.get(a, {})
    frame_b = ctx.frame_by_pos.get(b, {})
    score = 0.0
    if prefer_review and pair in ctx.reviewed_ok_pairs:
        score += 5.0
    if ctx.pair_transition_id_by_tuple.get(pair):
        score += 2.0
    if ctx.pair_review_id_by_tuple.get(pair):
        score += 1.5
    if frame_a.get("shot_id") and frame_a.get("shot_id") == frame_b.get("shot_id"):
        score += 1.0
    elif frame_a.get("chapter_id") and frame_a.get("chapter_id") == frame_b.get("chapter_id"):
        score += 0.5
    if frame_a.get("caption") and frame_b.get("caption"):
        score += 1.0
    if frame_a.get("frame_tool_descs") or frame_b.get("frame_tool_descs"):
        score += 0.2
    return score


def _build_pair_candidates_from_pairs(
    ctx: TaskContext,
    ordered_pairs: List[Tuple[int, int]],
    *,
    prefer_review: bool,
) -> List[PairCandidate]:
    seen: List[Tuple[int, int]] = []
    for pair in ordered_pairs:
        if pair not in seen:
            seen.append(pair)

    candidates: List[PairCandidate] = []
    for pair in seen:
        a, b = pair
        if pair not in ctx.pair_transition_id_by_tuple:
            continue
        frame_a = ctx.frame_by_pos.get(a)
        frame_b = ctx.frame_by_pos.get(b)
        if not frame_a or not frame_b:
            continue
        if not frame_a.get("caption") or not frame_b.get("caption"):
            continue
        pair_record = ctx.pair_record_by_tuple.get(pair, {})
        candidates.append(
            PairCandidate(
                from_pos=a,
                to_pos=b,
                pair_key=_pair_key(a, b),
                pair_ref=ctx.pair_transition_id_by_tuple[pair],
                review_ref=ctx.pair_review_id_by_tuple.get(pair),
                frame_refs=[_frame_cap_id(a), _frame_cap_id(b)],
                score=_score_pair(ctx, pair, prefer_review=prefer_review),
                pair_record=pair_record,
            )
        )
    candidates.sort(key=lambda item: (-item.score, item.from_pos, item.to_pos))
    return candidates


def _build_edit_pair_candidates(ctx: TaskContext) -> List[PairCandidate]:
    return _build_pair_candidates_from_pairs(ctx, list(ctx.reviewed_ok_pairs), prefer_review=True)


def _build_context_pair_candidates(ctx: TaskContext) -> List[PairCandidate]:
    seen: List[Tuple[int, int]] = []
    ordered_pairs = list(ctx.reviewed_ok_pairs) + list(ctx.candidate_pairs) + list(ctx.pair_record_by_tuple.keys())
    for pair in ordered_pairs:
        if pair not in seen:
            seen.append(pair)
    candidates = _build_pair_candidates_from_pairs(ctx, seen, prefer_review=True)
    return [c for c in candidates if _meaningful_temporal_pair(ctx, c)] if ctx.t3_out.get("task_profile") == "paper" else candidates


def _meaningful_temporal_pair(ctx: TaskContext, candidate: PairCandidate) -> bool:
    from orchestrator.transition_validation import leading_edit_verb
    a, b = candidate.from_pos, candidate.to_pos
    fa, fb = ctx.frame_by_pos[a], ctx.frame_by_pos[b]
    text = str(candidate.pair_record.get("transition_text", ""))
    if (fa.get("shot_id") != fb.get("shot_id") or "scene_cut" in text.lower() or
            re.search(r"\bView:\s*major\b", text, re.I)):
        return False
    if str(fa.get("cause", "")).lower() in {"camera_motion", "scene_cut", "occlusion"}:
        return False
    edit_text = _normalize_transition_text(text).strip(" .;").lower()
    if edit_text in {"", "none", "(none)", "no change", "no changes", "unchanged"}:
        return False
    match = re.search(r"Edits\((\d+)\)", text, re.I)
    body = re.sub(r"^Edits\(\d+\):\s*", "", text.split("|")[0], flags=re.I)
    atoms = [re.sub(r"^\d+\)\s*", "", part.strip()) for part in body.split(";") if part.strip()]
    verbs = [leading_edit_verb(atom) for atom in atoms]
    return bool(match and int(match.group(1)) == len(atoms) > 0 and
                all(verbs) and not set(verbs) & {"zoom", "pan", "tilt", "cut"})


def _has_review_anchor(candidate: PairCandidate) -> bool:
    return bool(candidate.review_ref)


def _chain_has_review_anchor(chain: List[PairCandidate]) -> bool:
    return any(_has_review_anchor(item) for item in chain)


def build_seed_candidates(ctx: TaskContext) -> List[PairCandidate]:
    return _build_edit_pair_candidates(ctx)


def build_weave_candidates(ctx: TaskContext) -> List[PairCandidate]:
    out: List[PairCandidate] = []
    pairs = _build_context_pair_candidates(ctx) if ctx.t3_out.get("task_profile") == "paper" else _build_edit_pair_candidates(ctx)
    for candidate in pairs:
        record = candidate.pair_record or {}
        transition_text = str(record.get("transition_text", "") or "").strip()
        review = record.get("review", {}) if isinstance(record.get("review"), dict) else {}
        if transition_text or review.get("instruction") or review.get("edits"):
            out.append(candidate)
    return out


def build_comm_c1_candidates(ctx: TaskContext, k: int = 3) -> List[CommC1Candidate]:
    if k < 2:
        k = 2
    pairs = sorted(_build_context_pair_candidates(ctx), key=lambda item: (item.from_pos, item.to_pos))
    pair_map = {(cand.from_pos, cand.to_pos): cand for cand in pairs}
    out: List[CommC1Candidate] = []
    for candidate in pairs:
        chain = [candidate]
        current = candidate.to_pos
        while len(chain) < k:
            nxt = pair_map.get((current, current + 1))
            if nxt is None:
                break
            chain.append(nxt)
            current = nxt.to_pos
        if len(chain) < 2:
            continue
        if ctx.t3_out.get("task_profile") != "paper" and not _chain_has_review_anchor(chain):
            continue
        frame_positions = [chain[0].from_pos] + [item.to_pos for item in chain]
        out.append(
            CommC1Candidate(
                frame_positions=frame_positions,
                pair_refs=[item.pair_ref for item in chain],
                review_refs=[item.review_ref for item in chain if item.review_ref],
                score=sum(item.score for item in chain),
            )
        )
    out.sort(key=lambda item: (-item.score, item.frame_positions))
    return out


def build_comm_c2_candidates(ctx: TaskContext) -> List[CommC2Candidate]:
    pair_map = {(cand.from_pos, cand.to_pos): cand for cand in _build_context_pair_candidates(ctx)}
    out: List[CommC2Candidate] = []
    for (a, b), left in pair_map.items():
        right = pair_map.get((b, b + 1))
        if right is None:
            continue
        if ctx.t3_out.get("task_profile") != "paper" and not _chain_has_review_anchor([left, right]):
            continue
        out.append(
            CommC2Candidate(
                frame_positions=[a, b, right.to_pos],
                pair_refs=[left.pair_ref, right.pair_ref],
                review_refs=[ref for ref in [left.review_ref, right.review_ref] if ref],
                score=left.score + right.score,
            )
        )
    out.sort(key=lambda item: (-item.score, item.frame_positions))
    return out


def build_isg_candidates(ctx: TaskContext, k: int = 2) -> List[ISGCandidate]:
    future_count = max(1, int(k))
    c1_candidates = build_comm_c1_candidates(ctx, k=future_count + 1)
    out: List[ISGCandidate] = []
    for chain in c1_candidates:
        if len(chain.frame_positions) < 3:
            continue
        actual_future_count = min(future_count, max(1, len(chain.frame_positions) - 2))
        out.append(
            ISGCandidate(
                context_positions=chain.frame_positions[:2],
                future_positions=chain.frame_positions[2 : 2 + actual_future_count],
                pair_refs=chain.pair_refs,
                review_refs=chain.review_refs,
                score=chain.score,
            )
        )
    return out


def attach_evidence_refs(ctx: TaskContext, task_type: str, candidate: Any) -> EvidenceClosure:
    evidence_ids: List[str] = [ctx.segment_evidence_id]
    frame_positions: List[int] = []
    pair_keys: List[str] = []

    if isinstance(candidate, PairCandidate):
        frame_positions = [candidate.from_pos, candidate.to_pos]
        pair_keys = [candidate.pair_key]
        evidence_ids.extend(candidate.frame_refs)
        evidence_ids.append(candidate.pair_ref)
        if candidate.review_ref:
            evidence_ids.append(candidate.review_ref)
    elif isinstance(candidate, CommC1Candidate):
        frame_positions = list(candidate.frame_positions)
        pair_keys = [_pair_key(frame_positions[i], frame_positions[i + 1]) for i in range(len(frame_positions) - 1)]
        evidence_ids.extend([_frame_cap_id(pos) for pos in frame_positions])
        evidence_ids.extend(candidate.pair_refs)
        evidence_ids.extend(candidate.review_refs)
    elif isinstance(candidate, CommC2Candidate):
        frame_positions = list(candidate.frame_positions)
        pair_keys = [_pair_key(frame_positions[i], frame_positions[i + 1]) for i in range(2)]
        evidence_ids.extend([_frame_cap_id(pos) for pos in frame_positions])
        evidence_ids.extend(candidate.pair_refs)
        evidence_ids.extend(candidate.review_refs)
    elif isinstance(candidate, ISGCandidate):
        frame_positions = list(candidate.context_positions) + list(candidate.future_positions)
        pair_keys = [_pair_key(frame_positions[i], frame_positions[i + 1]) for i in range(len(frame_positions) - 1)]
        evidence_ids.extend([_frame_cap_id(pos) for pos in frame_positions])
        evidence_ids.extend(candidate.pair_refs)
        evidence_ids.extend(candidate.review_refs)

    if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        evidence_ids.extend(f"F{p:02d}_STATE" for p in frame_positions)
        for row in ctx.evidence_by_id.values():
            if row.get("signal_type") == "entity_comparison" and row.get("pair_key") in pair_keys:
                evidence_ids.append(row["evidence_id"])

    shot_ids = []
    chapter_ids = []
    for pos in frame_positions:
        frame = ctx.frame_by_pos.get(pos, {})
        shot_id = str(frame.get("shot_id", "") or "").strip()
        chapter_id = str(frame.get("chapter_id", "") or "").strip()
        if shot_id and shot_id not in shot_ids:
            shot_ids.append(shot_id)
        if chapter_id and chapter_id not in chapter_ids:
            chapter_ids.append(chapter_id)
    for shot_id in shot_ids:
        shot_ref = ctx.shot_summary_ids.get(shot_id)
        if shot_ref:
            evidence_ids.append(shot_ref)
    for chapter_id in chapter_ids:
        chapter_ref = ctx.chapter_summary_ids.get(chapter_id)
        if chapter_ref:
            evidence_ids.append(chapter_ref)

    unique_ids: List[str] = []
    for evidence_id in evidence_ids:
        if evidence_id in ctx.evidence_by_id and evidence_id not in unique_ids:
            unique_ids.append(evidence_id)
    evidence_lines = []
    for evidence_id in unique_ids:
        unit = ctx.evidence_by_id.get(evidence_id, {})
        text = str(unit.get("text", "") or "").strip()
        if text:
            evidence_lines.append(f"[{evidence_id}] {_short_text(text, 500)}")
    if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        required = {key for key in unique_ids if key.endswith(("_CAP", "_STATE", "_TRANS", "_COMPARE"))}
        evidence_text = complete_evidence_text([ctx.evidence_by_id[key] for key in unique_ids], required, max_chars=24000)
    else:
        evidence_text = "\n".join(evidence_lines).strip()
    return EvidenceClosure(
        evidence_ids=unique_ids,
        evidence_text=evidence_text,
        frame_positions=frame_positions,
        pair_keys=pair_keys,
    )


def _maybe_call_llm(llm: Any, prompt: str) -> Optional[Dict[str, Any]]:
    if llm is None:
        return None
    raw = None
    if hasattr(llm, "chat"):
        raw = llm.chat(prompt, max_new_tokens=1200)
    elif callable(llm):
        raw = llm(prompt)
    parsed = parse_llm_json(raw if isinstance(raw, str) else str(raw))
    return parsed if isinstance(parsed, dict) else None


def _segment_summary(ctx: TaskContext) -> str:
    return str(ctx.evidence.get("segment_summary") or ctx.t3_out.get("segment_summary") or "").strip()


def _frame_caption(ctx: TaskContext, pos: int) -> str:
    return str((ctx.frame_by_pos.get(pos) or {}).get("caption", "") or "").strip()


def _transition_text(ctx: TaskContext, a: int, b: int) -> str:
    record = ctx.pair_record_by_tuple.get((a, b), {})
    return _normalize_transition_text(record.get("transition_text", ""))


def _default_select(question: str, answer: str, rationale: str) -> Dict[str, Any]:
    return {"question": question, "answer": answer, "rationale": rationale}


def _target_answer(pos: int) -> str:
    return f"target:F{int(pos):02d}"


def _coerce_text(value: Any) -> str:
    return str(value or "").strip()


def _prefer_generated_text(value: Any, default: str, *, banned_markers: Tuple[str, ...] = ()) -> str:
    text = _coerce_text(value)
    if current_prompt_profile() != "legacy":
        if not text:
            raise ValueError("formal_generation_empty_text")
        return text
    lowered = text.lower()
    if not text or any(marker in lowered for marker in banned_markers):
        return default.strip()
    return text


def _clause_text(value: Any, limit: int = 240) -> str:
    return _compact_phrase(value, limit).rstrip(" .;,:")


def _strip_frame_prefix(text: Any) -> str:
    value = _coerce_text(text)
    if value.startswith("Frame F") and ":" in value:
        return value.split(":", 1)[1].strip()
    return value


def _join_natural_list(items: List[str]) -> str:
    parts = [item.strip(" .;,:") for item in items if item.strip(" .;,:")]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _to_descriptive_edit(text: str) -> str:
    value = text.strip(" .;,:")
    lowered = value.lower()
    if lowered.startswith("remove "):
        rest = value[7:].strip()
        return f"{rest} is removed"
    if lowered.startswith("add "):
        rest = value[4:].strip()
        return f"{rest} appears"
    if lowered.startswith("move "):
        rest = value[5:].strip()
        match = re.match(r"(.+?)\s+from\s+(.+?)\s+to\s+(.+)", rest, flags=re.IGNORECASE)
        if match:
            subject = match.group(1).strip()
            src = match.group(2).strip()
            dst = match.group(3).strip()
            return f"{subject} moves from {src} to {dst}"
        return f"{rest} moves"
    if lowered.startswith("change "):
        rest = value[7:].strip()
        return f"{rest} changes"
    if lowered.startswith("open "):
        rest = value[5:].strip()
        return f"{rest} opens"
    if lowered.startswith("lower "):
        rest = value[6:].strip()
        return f"{rest} lowers"
    if lowered.startswith("raise "):
        rest = value[6:].strip()
        return f"{rest} rises"
    return value


def _normalize_transition_text(text: Any) -> str:
    value = _coerce_text(text)
    if not value:
        return ""
    value = value.split("|", 1)[0].strip()
    if value.lower().startswith("edits("):
        value = re.sub(r"^Edits\(\d+\):\s*", "", value, flags=re.IGNORECASE).strip()
        parts = [seg.strip() for seg in re.split(r"\s*\d+\)\s*", value) if seg.strip()]
        joined = _join_natural_list([_to_descriptive_edit(part) for part in parts])
        if joined:
            return joined
    return value


def _compact_phrase(text: Any, limit: int = 160) -> str:
    value = _normalize_transition_text(_strip_frame_prefix(text))
    if len(value) <= limit:
        return value
    for delimiter in (". ", "; ", ", while ", ", and ", ", but ", ", with ", ","):
        idx = value.rfind(delimiter, 0, limit + 1)
        if idx >= max(24, limit // 2):
            return value[:idx].strip()
    return _short_text(value, limit)


def _clean_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for value in values:
        text = _coerce_text(value)
        if text:
            out.append(text)
    return out


def _is_target_answer(value: Any) -> bool:
    text = _coerce_text(value)
    return len(text) == 10 and text.startswith("target:F") and text[8:].isdigit()


def _has_grounded_rationale_signal(value: Any) -> bool:
    if current_prompt_profile() != "legacy":
        return bool(_coerce_text(value))
    lowered = _coerce_text(value).lower()
    return any(marker in lowered for marker in ("because", "grounded", "supported", "evidence"))


def _has_selection_question(value: Any) -> bool:
    text = _coerce_text(value)
    return bool(text) if current_prompt_profile() != "legacy" else "candidate" in text.lower()


def _use_generated_question(value: Any, default: str, *, required_markers: Tuple[str, ...] = ("candidate",)) -> str:
    text = _prefer_generated_text(
        value,
        default,
        banned_markers=("maybe", "possibly", "not explicitly shown", "hypothetical", "unseen"),
    )
    if current_prompt_profile() != "legacy":
        return text
    lowered = text.lower()
    if all(marker in lowered for marker in required_markers):
        return text
    return default


def _use_generated_rationale(value: Any, default: str) -> str:
    text = _prefer_generated_text(
        value,
        default,
        banned_markers=("maybe", "possibly", "not explicitly shown", "hypothetical", "unseen", "not yet"),
    )
    if current_prompt_profile() != "legacy":
        return text
    lowered = text.lower()
    if "because" in lowered or "grounded" in lowered or "supported" in lowered:
        return text
    return default


def _use_generated_text_with_markers(
    value: Any,
    default: str,
    *,
    required_markers: Tuple[str, ...] = (),
    banned_markers: Tuple[str, ...] = ("maybe", "possibly", "not explicitly shown", "hypothetical", "unseen"),
) -> str:
    text = _prefer_generated_text(value, default, banned_markers=banned_markers)
    if current_prompt_profile() != "legacy":
        return text
    lowered = text.lower()
    if all(marker in lowered for marker in required_markers):
        return text
    return default


def _benchmark_select(
    *,
    pos: int,
    question_default: str,
    rationale_default: str,
    select_obj: Optional[Dict[str, Any]] = None,
    required_question_markers: Tuple[str, ...] = ("candidate",),
) -> Dict[str, Any]:
    select_obj = select_obj if isinstance(select_obj, dict) else {}
    return _default_select(
        _use_generated_question(select_obj.get("question"), question_default, required_markers=required_question_markers),
        select_obj.get("answer") if current_prompt_profile() != "legacy" else _target_answer(pos),
        _use_generated_rationale(select_obj.get("rationale"), rationale_default),
    )


def _seed_goal_default(target_caption: str, transition: str) -> str:
    if transition:
        return f"The correct image shows {_clause_text(target_caption, 140)}."
    return f"The correct image shows {_clause_text(target_caption, 140)}."


def _seed_summary_default(stable_caption: str, transition: str, target_caption: str) -> str:
    if transition:
        return (
            f"The scene stays the same overall, but now {_clause_text(target_caption, 110)}. "
            f"This matches the visible change: {_clause_text(transition, 120)}."
        )
    return f"The scene stays the same overall, and the target image shows {_clause_text(target_caption, 110)}."


def _seed_select_question_default() -> str:
    return "Which candidate image best matches the updated scene?"


def _seed_select_rationale_default(pos: int, stable_caption: str, target_caption: str, transition: str) -> str:
    if transition:
        return (
            f"Choose F{pos:02d} because it keeps the same overall scene '{_clause_text(stable_caption, 90)}' while showing "
            f"'{_clause_text(target_caption, 100)}', which matches the visible change '{_clause_text(transition, 100)}'."
        )
    return f"Choose F{pos:02d} because it best matches '{_clause_text(target_caption, 100)}'."


def _seed_invariants_default(from_pos: int, stable_caption: str) -> List[str]:
    return [
        f"Keep the same core scene from F{from_pos:02d}: {_clause_text(stable_caption, 100)}.",
        "Do not introduce unrelated objects, actions, or scene changes.",
    ]


def _normalize_seed_invariants(values: Any, from_pos: int, stable_caption: str) -> List[str]:
    invariants = _clean_text_list(values)
    if current_prompt_profile() != "legacy":
        if not invariants:
            raise ValueError("formal_generation_empty_invariants")
        return invariants
    if len(invariants) < 2:
        return _seed_invariants_default(from_pos, stable_caption)
    return invariants


def _weave_fact_defaults(from_caption: str, to_caption: str) -> List[str]:
    return [
        f"Earlier image: {_clause_text(from_caption, 150)}.",
        f"Later image: {_clause_text(to_caption, 150)}.",
    ]


def _weave_question_default() -> str:
    return "After looking at the earlier image, which recalled later detail identifies the correct candidate?"


def _weave_answer_default(from_pos: int, to_pos: int, transition: str) -> str:
    if transition:
        return f"Compared with F{from_pos:02d}, the later image F{to_pos:02d} is identified by {_clause_text(transition, 150)}."
    return f"Compared with F{from_pos:02d}, F{to_pos:02d} contains the recalled later-state detail."


def _weave_rubric_default() -> List[str]:
    return [
        "Stay consistent with the earlier-versus-later contrast.",
        "Mention only recalled changes that are visible in the evidence.",
        "Do not add unsupported attributes or events.",
    ]


def _normalize_weave_rubric(values: Any) -> List[str]:
    rubric = _clean_text_list(values)
    if current_prompt_profile() != "legacy":
        if not rubric:
            raise ValueError("formal_generation_empty_rubric")
        return rubric
    if len(rubric) < 3:
        return _weave_rubric_default()
    return rubric


def _weave_select_question_default(from_pos: int) -> str:
    return f"Using the earlier state in F{from_pos:02d}, which candidate frame matches the recalled later state?"


def _weave_select_rationale_default(pos: int, transition: str) -> str:
    if transition:
        return f"Choose F{pos:02d} because the recalled later state matches the visible change '{_clause_text(transition, 120)}'."
    return f"Choose F{pos:02d} because it matches the recalled later state."


def _comm_c1_overall_default(frame_positions: List[int], transitions: List[str]) -> str:
    if not transitions:
        return f"The sequence continues from F{frame_positions[0]:02d} through a few clear intermediate states."
    parts = [_clause_text(text, 100) for text in transitions if _coerce_text(text)]
    return f"The sequence unfolds step by step: {'; '.join(parts)}."


def _comm_c1_step_default(prev_pos: int, pos: int, transition: str) -> str:
    if transition:
        return f"From F{prev_pos:02d} to F{pos:02d}, {_clause_text(transition, 150)}."
    return f"From F{prev_pos:02d} to F{pos:02d}, the scene moves to the next visible state."


def _comm_c1_select_question_default(step_idx: int, prev_pos: int, pos: int) -> str:
    return f"For step {step_idx}, which candidate frame matches the next visible state after F{prev_pos:02d}?"


def _comm_c1_select_rationale_default(pos: int, transition: str) -> str:
    if transition:
        return f"Choose F{pos:02d} because this is the frame where {_clause_text(transition, 120)}."
    return f"Choose F{pos:02d} because it is the correct next state for this step."


def _comm_c2_default_question(a: int, c: int) -> str:
    return f"Given F{a:02d} and F{c:02d}, which candidate frame is the real middle step between them?"


def _comm_c2_default_select_question(a: int, c: int) -> str:
    return f"Which candidate frame fits after F{a:02d} and before F{c:02d} as the middle step?"


def _comm_c2_default_explanation(
    *,
    earlier_pos: int,
    middle_pos: int,
    later_pos: int,
    left_transition: str,
    right_transition: str,
    earlier_caption: str,
    middle_caption: str,
    later_caption: str,
) -> str:
    return (
        f"F{middle_pos:02d} is the middle step because F{earlier_pos:02d}->F{middle_pos:02d} shows "
        f"'{_clause_text(left_transition, 140)}', and F{middle_pos:02d}->F{later_pos:02d} then shows "
        f"'{_clause_text(right_transition, 140)}'. Its caption ('{_clause_text(middle_caption, 120)}') fits between "
        f"the earlier state ('{_clause_text(earlier_caption, 90)}') and the later state ('{_clause_text(later_caption, 90)}')."
    )


def _comm_c2_default_rationale(earlier_pos: int, middle_pos: int, later_pos: int, left_transition: str, right_transition: str) -> str:
    return (
        f"Choose F{middle_pos:02d} because it matches both sides of the sequence: "
        f"F{earlier_pos:02d}->F{middle_pos:02d} shows '{_clause_text(left_transition, 110)}' and "
        f"F{middle_pos:02d}->F{later_pos:02d} shows '{_clause_text(right_transition, 110)}'."
    )


def _isg_default_overall(context_positions: List[int], future_positions: List[int], future_transitions: List[str]) -> str:
    next_target = future_positions[0]
    next_transition = _clause_text(future_transitions[0] if future_transitions else "", 140)
    if next_transition:
        return (
            f"Given context frames F{context_positions[0]:02d} and F{context_positions[1]:02d}, the immediate next future step toward F{next_target:02d} is: "
            f"{next_transition}."
        )
    return f"Given context frames F{context_positions[0]:02d} and F{context_positions[1]:02d}, pick the immediate next future step toward F{next_target:02d}."


def _isg_default_edit(previous_pos: int, pos: int, transition_text: str) -> str:
    if transition_text:
        return f"After F{previous_pos:02d}, the next visible change is {_clause_text(transition_text, 160)}."
    return f"After F{previous_pos:02d}, move to the next future state F{pos:02d}."


def _isg_default_because(previous_pos: int, pos: int, transition_text: str) -> str:
    if transition_text:
        return f"Because the next transition after the context is F{previous_pos:02d}->F{pos:02d}: {_clause_text(transition_text, 150)}."
    return f"Because F{pos:02d} is the first future state after the provided context."


def _isg_default_so_that(previous_caption: str, future_caption: str) -> str:
    return (
        f"So the sequence moves from '{_clause_text(previous_caption, 100)}' to '{_clause_text(future_caption, 120)}' in the next step."
    )


def _isg_default_check(previous_pos: int, pos: int, future_caption: str) -> str:
    return (
        f"Keep the context from F{previous_pos:02d}. Add the next-step detail seen in F{pos:02d} "
        f"('{_clause_text(future_caption, 140)}'). Avoid later changes or unsupported details."
    )


def _normalize_isg_check(value: Any, previous_pos: int, pos: int, future_caption: str) -> str:
    default = _isg_default_check(previous_pos, pos, future_caption)
    return _prefer_generated_text(value, default)


def _base_meta(ctx: TaskContext, task_type: str, closure: EvidenceClosure) -> Dict[str, Any]:
    return {
        "benchmark": task_type,
        "segment_index": int(ctx.segment.get("index", ctx.evidence.get("segment_index", 0)) or 0),
        "video_id": str(ctx.evidence.get("video_id", "") or ctx.segment.get("video_id", "")),
        "frame_positions": closure.frame_positions,
        "pair_keys": closure.pair_keys,
        "segment_keyframes": [
            {
                "pos": int(frame.get("pos", frame.get("index", -1))),
                "img": str(frame.get("img", "") or "").strip(),
            }
            for frame in ctx.frames
            if isinstance(frame, dict) and str(frame.get("img", "") or "").strip()
        ],
    }


def _render_msg(role: str, text: Any) -> Dict[str, str]:
    return {"role": role, "type": "text", "content": _coerce_text(text)}


def _frame_image_map_from_sample(sample: Dict[str, Any]) -> Dict[int, str]:
    src = sample.get("src", {}) if isinstance(sample.get("src"), dict) else {}
    frames = list(src.get("frames") or [])
    images = list(src.get("images") or [])
    out: Dict[int, str] = {}
    for idx, pos in enumerate(frames):
        try:
            key = int(pos)
        except Exception:
            continue
        if idx < len(images):
            img = _coerce_text(images[idx])
            if img:
                out[key] = img
    meta = sample.get("meta", {}) if isinstance(sample.get("meta"), dict) else {}
    for frame in list(meta.get("segment_keyframes") or []):
        if not isinstance(frame, dict):
            continue
        try:
            key = int(frame.get("pos"))
        except Exception:
            continue
        img = _coerce_text(frame.get("img"))
        if img and key not in out:
            out[key] = img
    return out


def _sample_images_for_render(sample: Dict[str, Any], render: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    frame_map = _frame_image_map_from_sample(sample)
    query_positions: List[int] = []
    answer_positions: List[int] = []
    kind = _coerce_text(render.get("kind"))
    if kind in {"edit", "vlm_understand"}:
        for key in ("source_frame",):
            if key in render:
                query_positions.append(int(render[key]))
        for key in ("target_frame",):
            if key in render:
                answer_positions.append(int(render[key]))
    elif kind == "steps":
        for key in ("earlier_frame", "later_frame"):
            if key in render:
                query_positions.append(int(render[key]))
        if "middle_frame" in render:
            answer_positions.append(int(render["middle_frame"]))
    elif "context_positions" in render or "future_positions" in render:
        query_positions.extend([int(x) for x in list(render.get("context_positions") or [])])
        answer_positions.extend([int(x) for x in list(render.get("future_positions") or [])])
    elif "frame_positions" in render:
        positions = [int(x) for x in list(render.get("frame_positions") or [])]
        if positions:
            query_positions.append(positions[0])
            answer_positions.extend(positions[1:])
    query_images = [frame_map[pos] for pos in query_positions if pos in frame_map]
    answer_images = [frame_map[pos] for pos in answer_positions if pos in frame_map]
    return query_images, answer_images


def _rulebased_render_messages(sample: Dict[str, Any]) -> Dict[str, Any]:
    render = sample.get("render", {}) if isinstance(sample.get("render"), dict) else {}
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    benchmark = _coerce_text(sample.get("benchmark"))
    query_messages: List[Dict[str, str]] = []
    answer_messages: List[Dict[str, str]] = []

    def q(text: Any, role: str = "user") -> None:
        value = _coerce_text(text)
        if value:
            query_messages.append(_render_msg(role, value))

    def a(text: Any, role: str = "assistant") -> None:
        value = _coerce_text(text)
        if value:
            answer_messages.append(_render_msg(role, value))

    if benchmark == TASK_SEED:
        plan = turns.get("plan", {}) if isinstance(turns.get("plan"), dict) else {}
        gen = turns.get("gen", {}) if isinstance(turns.get("gen"), dict) else {}
        select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
        q(plan.get("goal"))
        q(gen.get("summary"), role="assistant")
        q(select.get("question"))
        a(select.get("answer"))
        a(select.get("rationale"))
    elif benchmark == TASK_WEAVE:
        understand = turns.get("understand", {}) if isinstance(turns.get("understand"), dict) else {}
        qa_open = turns.get("qa_open", {}) if isinstance(turns.get("qa_open"), dict) else {}
        select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
        for fact in _clean_text_list(understand.get("facts")):
            q(fact, role="assistant")
        q(qa_open.get("question"))
        q(qa_open.get("answer"), role="assistant")
        q(select.get("question"))
        a(select.get("answer"))
        a(select.get("rationale"))
    elif benchmark == TASK_COMM_C1:
        q(turns.get("question"))
        q(turns.get("overall"), role="assistant")
        steps = _clean_text_list(turns.get("steps"))
        selects = list(turns.get("select") or [])
        for idx, step in enumerate(steps):
            q(step, role="assistant")
            select = selects[idx] if idx < len(selects) and isinstance(selects[idx], dict) else {}
            q(select.get("question"))
        for idx, _step in enumerate(steps):
            select = selects[idx] if idx < len(selects) and isinstance(selects[idx], dict) else {}
            a(select.get("answer"))
            a(select.get("rationale"))
    elif benchmark == TASK_COMM_C2:
        select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
        q(turns.get("question"))
        q(turns.get("explanation"), role="assistant")
        q(select.get("question"))
        a(select.get("answer"))
        a(select.get("rationale"))
    elif benchmark == TASK_ISG:
        q(turns.get("overall"), role="assistant")
        edits = _clean_text_list(turns.get("edits"))
        because = _clean_text_list(turns.get("because"))
        so_that = _clean_text_list(turns.get("so_that"))
        check = _clean_text_list(turns.get("check"))
        selects = list(turns.get("select") or [])
        for idx, select in enumerate(selects):
            if not isinstance(select, dict):
                continue
            if idx < len(edits):
                q(edits[idx], role="assistant")
            if idx < len(because):
                q(because[idx], role="assistant")
            if idx < len(so_that):
                q(so_that[idx], role="assistant")
            if idx < len(check):
                q(check[idx], role="assistant")
            q(select.get("question"))
        for idx, select in enumerate(selects):
            if not isinstance(select, dict):
                continue
            a(select.get("answer"))
            a(select.get("rationale"))

    query_images, answer_images = _sample_images_for_render(sample, render)
    render["query"] = {"messages": query_messages, "images": query_images}
    render["answer"] = {"messages": answer_messages, "images": answer_images}
    sample["render"] = render
    return sample


def generate_seed_sample(ctx: TaskContext, candidate: PairCandidate, llm: Any = None) -> Dict[str, Any]:
    closure = attach_evidence_refs(ctx, TASK_SEED, candidate)
    payload = {
        "from_pos": candidate.from_pos,
        "to_pos": candidate.to_pos,
        "from_caption": _frame_caption(ctx, candidate.from_pos),
        "to_caption": _frame_caption(ctx, candidate.to_pos),
        "transition": _transition_text(ctx, candidate.from_pos, candidate.to_pos),
    }
    obj = _maybe_call_llm(llm, umm_task_prompts.build_seed_prompt(payload, closure.evidence_text)) or {}
    transition = payload["transition"] or f"Update frame {candidate.from_pos} toward frame {candidate.to_pos}."
    plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
    gen = obj.get("gen") if isinstance(obj.get("gen"), dict) else {}
    select = obj.get("select") if isinstance(obj.get("select"), dict) else {}
    stable_caption = payload["from_caption"]
    target_caption = payload["to_caption"]
    invariants = _normalize_seed_invariants(plan.get("invariants"), candidate.from_pos, stable_caption)
    edits = _clean_text_list(plan.get("edits"))
    if not edits:
        edits = [transition]
    return {
        "benchmark": TASK_SEED,
        "turns": {
            "plan": {
                "goal": _prefer_generated_text(plan.get("goal"), _seed_goal_default(target_caption, transition)),
                "invariants": invariants,
                "edits": edits,
            },
            "gen": {
                "summary": _prefer_generated_text(gen.get("summary"), _seed_summary_default(stable_caption, transition, target_caption))
            },
            "select": _benchmark_select(
                pos=candidate.to_pos,
                question_default=_seed_select_question_default(),
                rationale_default=_seed_select_rationale_default(candidate.to_pos, stable_caption, target_caption, transition),
                select_obj=select,
                required_question_markers=("candidate", "image"),
            ),
        },
        "answer": {"target_frame_pos": candidate.to_pos},
        "render": {"kind": "edit", "source_frame": candidate.from_pos, "target_frame": candidate.to_pos},
        "evidence_refs": closure.evidence_ids,
        "meta": _base_meta(ctx, TASK_SEED, closure),
    }


def generate_weave_sample(ctx: TaskContext, candidate: PairCandidate, llm: Any = None) -> Dict[str, Any]:
    closure = attach_evidence_refs(ctx, TASK_WEAVE, candidate)
    payload = {
        "from_pos": candidate.from_pos,
        "to_pos": candidate.to_pos,
        "from_caption": _frame_caption(ctx, candidate.from_pos),
        "to_caption": _frame_caption(ctx, candidate.to_pos),
        "transition": _transition_text(ctx, candidate.from_pos, candidate.to_pos),
    }
    obj = _maybe_call_llm(llm, umm_task_prompts.build_weave_prompt(payload, closure.evidence_text)) or {}
    transition = payload["transition"] or f"Recall the change from F{candidate.from_pos:02d} to F{candidate.to_pos:02d}."
    understand = obj.get("understand") if isinstance(obj.get("understand"), dict) else {}
    plan = obj.get("plan") if isinstance(obj.get("plan"), dict) else {}
    qa_open = obj.get("qa_open") if isinstance(obj.get("qa_open"), dict) else {}
    select = obj.get("select") if isinstance(obj.get("select"), dict) else {}
    return {
        "benchmark": TASK_WEAVE,
        "turns": {
            "understand": {
                "facts": list(
                    _clean_text_list(understand.get("facts"))
                    or _weave_fact_defaults(payload["from_caption"], payload["to_caption"])
                )
            },
            "plan": {"edits": _clean_text_list(plan.get("edits")) or [transition]},
            "qa_open": {
                "question": _use_generated_question(
                    qa_open.get("question"),
                    _weave_question_default(),
                    required_markers=("later",),
                ),
                "answer": _prefer_generated_text(
                    qa_open.get("answer"),
                    _weave_answer_default(candidate.from_pos, candidate.to_pos, transition),
                ),
                "rubric": _normalize_weave_rubric(qa_open.get("rubric")),
            },
            "select": _benchmark_select(
                pos=candidate.to_pos,
                question_default=_weave_select_question_default(candidate.from_pos),
                rationale_default=_weave_select_rationale_default(candidate.to_pos, transition),
                select_obj=select,
                required_question_markers=("candidate", "later"),
            ),
        },
        "answer": {"target_frame_pos": candidate.to_pos},
        "render": {"kind": "vlm_understand", "source_frame": candidate.from_pos, "target_frame": candidate.to_pos},
        "evidence_refs": closure.evidence_ids,
        "meta": _base_meta(ctx, TASK_WEAVE, closure),
    }


def generate_comm_c1_sample(ctx: TaskContext, candidate: CommC1Candidate, llm: Any = None) -> Dict[str, Any]:
    closure = attach_evidence_refs(ctx, TASK_COMM_C1, candidate)
    transitions = [
        _transition_text(ctx, candidate.frame_positions[i], candidate.frame_positions[i + 1])
        for i in range(len(candidate.frame_positions) - 1)
    ]
    payload = {"frame_positions": candidate.frame_positions, "transitions": transitions}
    obj = _maybe_call_llm(llm, umm_task_prompts.build_comm_c1_prompt(payload, closure.evidence_text, len(transitions))) or {}
    if current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
        endpoints = [candidate.frame_positions[0], candidate.frame_positions[-1]]
        ids = {f"F{p:02d}_{suffix}" for p in endpoints for suffix in ("CAP", "STATE")}
        endpoint_evidence = complete_evidence_text([ctx.evidence_by_id[key] for key in sorted(ids)], ids)
        from executor.tasks.prompt_loader import render_prompt_text
        overall = llm.author_overall(render_prompt_text("umm_task_prompts", "comm_c1_overall",
            evidence_text=endpoint_evidence, positions_json=json.dumps(endpoints)), endpoints)
        obj = {**obj, "overall": overall["overall"]}
    step_texts = _clean_text_list(obj.get("step_texts"))
    if not step_texts:
        step_texts = [
            _comm_c1_step_default(candidate.frame_positions[idx], candidate.frame_positions[idx + 1], text)
            for idx, text in enumerate(transitions)
        ]
    selects = obj.get("select") if isinstance(obj.get("select"), list) else []
    while len(step_texts) < len(transitions):
        idx = len(step_texts)
        step_texts.append(_comm_c1_step_default(candidate.frame_positions[idx], candidate.frame_positions[idx + 1], transitions[idx]))
    select_items = []
    for idx, text in enumerate(transitions):
        src = selects[idx] if idx < len(selects) and isinstance(selects[idx], dict) else {}
        select_items.append(
            _benchmark_select(
                pos=candidate.frame_positions[idx + 1],
                question_default=_comm_c1_select_question_default(idx + 1, candidate.frame_positions[idx], candidate.frame_positions[idx + 1]),
                rationale_default=_comm_c1_select_rationale_default(candidate.frame_positions[idx + 1], text or step_texts[idx]),
                select_obj=src,
                required_question_markers=("candidate", "step"),
            )
        )
    return {
        "benchmark": TASK_COMM_C1,
        "turns": {
            "question": _use_generated_question(
                obj.get("question"),
                "For each grounded step in the coherent sequence, which candidate frame matches the required intermediate state?",
                required_markers=("candidate", "step"),
            ),
            "overall": _prefer_generated_text(obj.get("overall"), _comm_c1_overall_default(candidate.frame_positions, transitions)),
            "steps": step_texts[: len(transitions)],
            "select": select_items,
        },
        "answer": {"target_frame_positions": candidate.frame_positions[1:]},
        "render": {"kind": "interleaved_mm", "frame_positions": candidate.frame_positions},
        "evidence_refs": closure.evidence_ids,
        "meta": _base_meta(ctx, TASK_COMM_C1, closure),
    }


def generate_comm_c2_sample(ctx: TaskContext, candidate: CommC2Candidate, llm: Any = None) -> Dict[str, Any]:
    closure = attach_evidence_refs(ctx, TASK_COMM_C2, candidate)
    earlier_pos, middle_pos, later_pos = candidate.frame_positions
    left = _transition_text(ctx, earlier_pos, middle_pos)
    right = _transition_text(ctx, middle_pos, later_pos)
    payload = {"frame_positions": candidate.frame_positions, "left_transition": left, "right_transition": right}
    obj = _maybe_call_llm(llm, umm_task_prompts.build_comm_c2_prompt(payload, closure.evidence_text)) or {}
    left_caption = _frame_caption(ctx, earlier_pos)
    middle_caption = _frame_caption(ctx, middle_pos)
    right_caption = _frame_caption(ctx, later_pos)
    default_explanation = _comm_c2_default_explanation(
        earlier_pos=earlier_pos,
        middle_pos=middle_pos,
        later_pos=later_pos,
        left_transition=left,
        right_transition=right,
        earlier_caption=left_caption,
        middle_caption=middle_caption,
        later_caption=right_caption,
    )
    explanation = _prefer_generated_text(
        obj.get("explanation"),
        default_explanation,
        banned_markers=("not explicitly shown", "hypothetical unseen", "cannot be the intermediate", "might be", "not yet"),
    )
    select = obj.get("select") if isinstance(obj.get("select"), dict) else {}
    default_question = _comm_c2_default_question(earlier_pos, later_pos)
    default_select_question = _comm_c2_default_select_question(earlier_pos, later_pos)
    default_rationale = _comm_c2_default_rationale(earlier_pos, middle_pos, later_pos, left, right)
    return {
        "benchmark": TASK_COMM_C2,
        "turns": {
            "question": _use_generated_question(
                obj.get("question"),
                default_question,
                required_markers=("candidate",),
            ),
            "explanation": explanation,
            "select": _benchmark_select(
                pos=middle_pos,
                question_default=default_select_question,
                rationale_default=default_rationale,
                select_obj=select,
                required_question_markers=("candidate", "middle"),
            ),
        },
        "answer": {"target_frame_pos": middle_pos},
        "render": {
            "kind": "steps",
            "earlier_frame": earlier_pos,
            "middle_frame": middle_pos,
            "later_frame": later_pos,
        },
        "evidence_refs": closure.evidence_ids,
        "meta": _base_meta(ctx, TASK_COMM_C2, closure),
    }


def generate_isg_sample(ctx: TaskContext, candidate: ISGCandidate, llm: Any = None) -> Dict[str, Any]:
    closure = attach_evidence_refs(ctx, TASK_ISG, candidate)
    full_positions = candidate.context_positions + candidate.future_positions
    transitions = [
        _transition_text(ctx, full_positions[i], full_positions[i + 1])
        for i in range(len(full_positions) - 1)
    ]
    payload = {
        "context_positions": candidate.context_positions,
        "future_positions": candidate.future_positions,
        "future_transitions": transitions[len(candidate.context_positions) - 1 :],
    }
    obj = _maybe_call_llm(llm, umm_task_prompts.build_isg_prompt(payload, closure.evidence_text, len(candidate.future_positions))) or {}
    future_transitions = transitions[len(candidate.context_positions) - 1 :]
    default_overall = _isg_default_overall(
        candidate.context_positions,
        candidate.future_positions,
        future_transitions,
    )
    edits = []
    because = []
    so_that = []
    check = []
    for idx, pos in enumerate(candidate.future_positions):
        transition_text = future_transitions[idx] if idx < len(future_transitions) else ""
        previous_pos = full_positions[idx + len(candidate.context_positions) - 1]
        previous_caption = _frame_caption(ctx, previous_pos)
        future_caption = _frame_caption(ctx, pos)
        edits.append(
            str((obj.get("edits") or [])[idx]).strip()
            if isinstance(obj.get("edits"), list) and idx < len(obj.get("edits"))
            else _isg_default_edit(previous_pos, pos, transition_text)
        )
        because.append(
            str((obj.get("because") or [])[idx]).strip()
            if isinstance(obj.get("because"), list) and idx < len(obj.get("because"))
            else _isg_default_because(previous_pos, pos, transition_text)
        )
        so_that.append(
            str((obj.get("so_that") or [])[idx]).strip()
            if isinstance(obj.get("so_that"), list) and idx < len(obj.get("so_that"))
            else _isg_default_so_that(previous_caption, future_caption)
        )
        check.append(
            _normalize_isg_check(
                (obj.get("check") or [])[idx] if isinstance(obj.get("check"), list) and idx < len(obj.get("check")) else "",
                previous_pos,
                pos,
                future_caption,
            )
        )
    selects = obj.get("select") if isinstance(obj.get("select"), list) else []
    select_items = []
    for idx, pos in enumerate(candidate.future_positions):
        src = selects[idx] if idx < len(selects) and isinstance(selects[idx], dict) else {}
        transition_text = future_transitions[idx] if idx < len(future_transitions) else ""
        select_items.append(
            _benchmark_select(
                pos=pos,
                question_default=f"After context frames F{candidate.context_positions[0]:02d} and F{candidate.context_positions[1]:02d}, which candidate frame matches the immediate next future step {idx + 1}?",
                rationale_default=(
                    f"Choose F{pos:02d} because it is the next frame after the context"
                    + (f" where {_clause_text(transition_text, 120)}." if transition_text else ".")
                ),
                select_obj=src,
                required_question_markers=("candidate", "future"),
            )
        )
    return {
        "benchmark": TASK_ISG,
        "turns": {
            "overall": _use_generated_text_with_markers(
                obj.get("overall"),
                default_overall,
                required_markers=("immediate next future step",),
            ),
            "edits": edits[: len(candidate.future_positions)],
            "because": because[: len(candidate.future_positions)],
            "so_that": so_that[: len(candidate.future_positions)],
            "check": check[: len(candidate.future_positions)],
            "select": select_items,
        },
        "answer": {"future_frame_positions": candidate.future_positions},
        "render": {
            "kind": "interleaved_mm",
            "context_positions": candidate.context_positions,
            "future_positions": candidate.future_positions,
        },
        "evidence_refs": closure.evidence_ids,
        "meta": _base_meta(ctx, TASK_ISG, closure),
    }


def gate_seed_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    plan = turns.get("plan", {}) if isinstance(turns.get("plan"), dict) else {}
    select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
    refs = list(sample.get("evidence_refs") or [])
    question = _coerce_text(select.get("question")).lower()
    ok = bool(plan.get("goal")) and bool(plan.get("invariants")) and bool(plan.get("edits")) and _is_target_answer(select.get("answer"))
    ok = ok and any(ref in refs for ref in ALT_SEGMENT_EVIDENCE_IDS)
    ok = ok and _has_selection_question(question) and _has_grounded_rationale_signal(select.get("rationale"))
    return {"ok": ok, "reason": "" if ok else "seed fields incomplete"}


def gate_weave_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    understand = turns.get("understand", {}) if isinstance(turns.get("understand"), dict) else {}
    plan = turns.get("plan", {}) if isinstance(turns.get("plan"), dict) else {}
    select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
    qa_open = turns.get("qa_open", {}) if isinstance(turns.get("qa_open"), dict) else {}
    question = _coerce_text(select.get("question")).lower()
    ok = bool(understand.get("facts")) and bool(plan.get("edits")) and _is_target_answer(select.get("answer"))
    if qa_open:
        ok = ok and bool(qa_open.get("question")) and bool(qa_open.get("answer")) and bool(qa_open.get("rubric"))
    ok = ok and _has_selection_question(question) and _has_grounded_rationale_signal(select.get("rationale"))
    return {"ok": ok, "reason": "" if ok else "weave fields incomplete"}


def gate_comm_c1_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    steps = list(turns.get("steps") or [])
    selects = list(turns.get("select") or [])
    ok = bool(steps) and len(steps) == len(selects) and all(
        isinstance(item, dict)
        and _is_target_answer(item.get("answer"))
        and _has_selection_question(item.get("question"))
        and _has_grounded_rationale_signal(item.get("rationale"))
        for item in selects
    )
    return {"ok": ok, "reason": "" if ok else "comm c1 steps/select mismatch"}


def gate_comm_c2_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    select = turns.get("select", {}) if isinstance(turns.get("select"), dict) else {}
    question = _coerce_text(select.get("question")).lower()
    ok = bool(turns.get("explanation")) and _is_target_answer(select.get("answer"))
    render = sample.get("render", {}) if isinstance(sample.get("render"), dict) else {}
    ok = ok and all(key in render for key in ("earlier_frame", "middle_frame", "later_frame"))
    explanation = str(turns.get("explanation", "") or "").lower()
    if any(marker in explanation for marker in ("not explicitly shown", "hypothetical unseen", "cannot be the intermediate")):
        ok = False
    ok = ok and _has_selection_question(question) and _has_grounded_rationale_signal(select.get("rationale"))
    return {"ok": ok, "reason": "" if ok else "comm c2 explanation/render incomplete"}


def gate_isg_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    turns = sample.get("turns", {}) if isinstance(sample.get("turns"), dict) else {}
    edits = list(turns.get("edits") or [])
    because = list(turns.get("because") or [])
    so_that = list(turns.get("so_that") or [])
    check = list(turns.get("check") or [])
    selects = list(turns.get("select") or [])
    ok = bool(edits) and len(edits) == len(because) == len(so_that) == len(check) == len(selects)
    ok = ok and all(
        isinstance(item, dict)
        and _is_target_answer(item.get("answer"))
        and _has_selection_question(item.get("question"))
        and _has_grounded_rationale_signal(item.get("rationale"))
        for item in selects
    )
    return {"ok": ok, "reason": "" if ok else "isg step arrays incomplete"}


class GenerationCallPending(ValueError):
    def __init__(self, cause: Exception):
        self.error_type = type(cause).__name__
        from executor.models.api_budget import ApiBudgetExhausted
        self.reason = "api_budget_exhausted" if isinstance(cause, ApiBudgetExhausted) else "model_call_pending"
        super().__init__(self.reason + ": " + self.error_type)


class StrictTaskLLM:
    def __init__(self, llm: Any, task: str, ctx: TaskContext, candidate: Any):
        self.llm, self.task, self.ctx, self.candidate = llm, task, ctx, candidate
        self.raw_response = None
        self.model_evidence_refs = []
        self.overall_response = None
        self.generation_attempts = []
        self.citation_source_object = None

    def author_overall(self, prompt: str, positions: list[int]) -> dict:
        try:
            raw = self.llm.chat(prompt, max_new_tokens=650)
        except Exception as exc:
            raise GenerationCallPending(exc) from exc
        self.overall_response = raw
        obj = parse_llm_json(raw)
        refs = obj.get("evidence_refs") if isinstance(obj, dict) else None
        allowed = {f"F{p:02d}_{suffix}" for p in positions for suffix in ("CAP", "STATE")}
        if (not isinstance(obj, dict) or not isinstance(obj.get("overall"), str) or not obj["overall"].strip()
                or not isinstance(refs, list) or not refs or any(ref not in allowed for ref in refs)
                or any(not any(ref.startswith(f"F{p:02d}_") for ref in refs) for p in positions)):
            raise ValueError("formal_overall_outside_endpoint_evidence")
        self.model_evidence_refs = list(dict.fromkeys([*self.model_evidence_refs, *refs]))
        return obj

    def chat(self, prompt: str, max_new_tokens: int = 1200) -> str:
        if self.llm is None:
            raise ValueError("formal_generation_requires_llm")
        try:
            raw = self.llm.chat(prompt, max_new_tokens=max_new_tokens)
        except Exception as exc:
            raise GenerationCallPending(exc) from exc
        self.raw_response = raw
        self.generation_attempts.append(raw)
        obj = parse_llm_json(raw)
        if self.citation_source_object is not None and isinstance(obj, dict):
            if {k: v for k, v in obj.items() if k != "evidence_refs"} != self.citation_source_object:
                raise ValueError("citation_repair_changed_task")
        required = {
            TASK_SEED: ["plan", "gen", "select"],
            TASK_WEAVE: ["understand", "plan", "qa_open", "select"],
            TASK_COMM_C1: ["question", "overall", "step_texts", "select"],
            TASK_COMM_C2: ["question", "explanation", "select"],
            TASK_ISG: ["overall", "edits", "because", "so_that", "check", "select"],
        }[self.task]
        if self.task == TASK_COMM_C1 and current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
            required = [field for field in required if field != "overall"]
        def nonempty(value: Any) -> bool:
            if isinstance(value, dict):
                return bool(value) and all(nonempty(v) for v in value.values())
            if isinstance(value, list):
                return bool(value) and all(nonempty(v) for v in value)
            return isinstance(value, str) and bool(value.strip())
        if not isinstance(obj, dict) or any(not nonempty(obj.get(k)) for k in required):
            raise ValueError("formal_generation_missing_fields")
        nested = {
            TASK_SEED: {"plan": ["goal", "invariants", "edits"], "gen": ["summary"]},
            TASK_WEAVE: {"understand": ["facts"], "plan": ["edits"], "qa_open": ["question", "answer", "rubric"]},
        }.get(self.task, {})
        if any(not isinstance(obj.get(key), dict) or any(not nonempty(obj[key].get(field)) for field in fields)
               for key, fields in nested.items()):
            raise ValueError("formal_generation_missing_nested_fields")
        array_fields = {TASK_SEED: [], TASK_WEAVE: [], TASK_COMM_C1: ["step_texts"],
                        TASK_COMM_C2: [], TASK_ISG: ["edits", "because", "so_that", "check"]}[self.task]
        if any(not isinstance(obj[key], list) or any(not isinstance(v, str) for v in obj[key]) for key in array_fields):
            raise ValueError("formal_generation_invalid_text_array")
        for key, fields in nested.items():
            for field in fields:
                value = obj[key][field]
                if field in {"invariants", "edits", "facts", "rubric"}:
                    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                        raise ValueError("formal_generation_invalid_text_array")
                elif not isinstance(value, str):
                    raise ValueError("formal_generation_invalid_text")
        closure = attach_evidence_refs(self.ctx, self.task, self.candidate)
        refs = obj.get("evidence_refs")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in closure.evidence_ids for r in refs):
            raise ValueError("formal_generation_invalid_model_citations")
        needed = {self.ctx.segment_evidence_id, *[_frame_cap_id(p) for p in closure.frame_positions]}
        if current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"}:
            needed.discard(self.ctx.segment_evidence_id)
        if isinstance(self.candidate, PairCandidate):
            needed.add(self.candidate.pair_ref)
        else:
            needed.update(self.candidate.pair_refs)
        if not needed.issubset(refs):
            if current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v13", "paper_v15"} and self.citation_source_object is None:
                self.citation_source_object = {k: v for k, v in obj.items() if k != "evidence_refs"}
                # A bounded response-format correction, with both replies and
                # the exact final prompt retained by the caller's audit.
                correction = (prompt + "\nPREVIOUS RESPONSE:\n" + raw +
                    "\nResponse citation contract failed. Missing required supplied frame/transition IDs: " +
                    json.dumps(sorted(needed - set(refs))) +
                    ". Return the complete same JSON, changing ONLY evidence_refs. Cite an ID only if the supplied "
                    "evidence actually supports this task; do not add unrelated citations. Do not alter the question, "
                    "answer, steps, targets, temporal roles or task purpose. If support is absent, return no task.")
                return self.chat(correction, max_new_tokens=max_new_tokens)
            raise ValueError("formal_generation_missing_endpoint_or_transition_citations")
        if self.task in {TASK_SEED, TASK_WEAVE}:
            targets = [self.candidate.to_pos]
        elif self.task == TASK_COMM_C1:
            targets = self.candidate.frame_positions[1:]
            if len(obj["step_texts"]) != len(targets):
                raise ValueError("formal_generation_step_count_mismatch")
        elif self.task == TASK_COMM_C2:
            targets = [self.candidate.frame_positions[1]]
        else:
            targets = self.candidate.future_positions
            if any(len(obj[k]) != len(targets) for k in ["edits", "because", "so_that", "check"]):
                raise ValueError("formal_generation_future_count_mismatch")
        selects = obj["select"] if isinstance(obj["select"], list) else [obj["select"]]
        if len(selects) != len(targets) or any(not isinstance(row, dict) or row.get("answer") != _target_answer(target) for row, target in zip(selects, targets)):
            raise ValueError("formal_generation_target_mismatch")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for row in selects for k in ["question", "rationale"]):
            raise ValueError("formal_generation_incomplete_selection")
        self.model_evidence_refs = list(refs)
        return raw


def produce_task_samples(
    bundle: Dict[str, Any],
    *,
    llm: Any = None,
    comm_c1_k: int = 3,
    isg_k: int = 2,
    llm_retry_attempts: int = 1,
    selected_tasks: Optional[List[str]] = None,
    strict_generation: bool = False,
) -> Dict[str, Any]:
    ctx = build_task_context(bundle)
    samples: Dict[str, List[Dict[str, Any]]] = {
        TASK_SEED: [],
        TASK_WEAVE: [],
        TASK_COMM_C1: [],
        TASK_COMM_C2: [],
        TASK_ISG: [],
    }
    drops: Dict[str, List[str]] = {key: [] for key in samples}
    pending: Dict[str, List[dict]] = {key: [] for key in samples}
    retry_attempts = max(1, int(llm_retry_attempts or 1)) if llm is not None else 1

    def _attempt_task(task_type: str, candidate: Any, generator: Any, gate_fn: Any) -> None:
        if selected_tasks is not None and task_type not in selected_tasks:
            drops[task_type].append("not_routed")
            return
        last_reason = "no_attempt"
        last_pending = None
        for _ in range(retry_attempts):
            try:
                generation_llm = StrictTaskLLM(llm, task_type, ctx, candidate) if strict_generation else llm
                if strict_generation:
                    from executor.tasks.prompt_loader import prompt_profile_scope
                    with prompt_profile_scope(current_prompt_profile() if current_prompt_profile() != "legacy" else "paper_v1"):
                        sample = generator(ctx, candidate, llm=generation_llm)
                    sample["meta"]["available_evidence_refs"] = sample["evidence_refs"]
                    sample["meta"]["model_evidence_refs"] = generation_llm.model_evidence_refs
                    sample["meta"]["raw_generation_response"] = generation_llm.raw_response
                    sample["meta"]["raw_generation_attempts"] = generation_llm.generation_attempts
                    if generation_llm.overall_response is not None:
                        sample["meta"]["raw_overall_response"] = generation_llm.overall_response
                    sample["meta"]["citation_origin"] = "model"
                    sample["evidence_refs"] = generation_llm.model_evidence_refs
                else:
                    sample = generator(ctx, candidate, llm=generation_llm)
            except GenerationCallPending as exc:
                last_pending = {"status": "pending", "semantic_decision": None,
                                "error_type": exc.error_type, "reason": exc.reason}
                continue
            except ValueError as exc:
                last_pending = None
                last_reason = str(exc)
                continue
            last_pending = None
            if isinstance(sample, dict):
                sample = _rulebased_render_messages(sample)
            if strict_generation:
                with prompt_profile_scope(current_prompt_profile() if current_prompt_profile() != "legacy" else "paper_v1"):
                    gate = gate_fn(sample)
            else:
                gate = gate_fn(sample)
            if gate["ok"]:
                samples[task_type].append(sample)
                return
            last_reason = gate["reason"]
        if last_pending is not None:
            pending[task_type].append(last_pending)
        else:
            drops[task_type].append(last_reason)

    seed_candidates = build_seed_candidates(ctx)
    if seed_candidates:
        _attempt_task(TASK_SEED, seed_candidates[0], generate_seed_sample, gate_seed_sample)
    else:
        drops[TASK_SEED].append("no_candidate")

    weave_candidates = build_weave_candidates(ctx)
    if weave_candidates:
        _attempt_task(TASK_WEAVE, weave_candidates[0], generate_weave_sample, gate_weave_sample)
    else:
        drops[TASK_WEAVE].append("no_candidate")

    comm_c1_candidates = build_comm_c1_candidates(ctx, k=comm_c1_k)
    if comm_c1_candidates:
        _attempt_task(TASK_COMM_C1, comm_c1_candidates[0], generate_comm_c1_sample, gate_comm_c1_sample)
    else:
        drops[TASK_COMM_C1].append("no_candidate")

    comm_c2_candidates = build_comm_c2_candidates(ctx)
    if comm_c2_candidates:
        _attempt_task(TASK_COMM_C2, comm_c2_candidates[0], generate_comm_c2_sample, gate_comm_c2_sample)
    else:
        drops[TASK_COMM_C2].append("no_candidate")

    isg_candidates = build_isg_candidates(ctx, k=isg_k)
    if isg_candidates:
        _attempt_task(TASK_ISG, isg_candidates[0], generate_isg_sample, gate_isg_sample)
    else:
        drops[TASK_ISG].append("no_candidate")

    return {"context": ctx, "samples": samples, "drops": drops, "pending": pending}
