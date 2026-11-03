"""Paper task taxonomy and evidence-bound automatic routing."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from orchestrator.planner_helpers import parse_llm_json
from executor.tasks.prompt_loader import render_prompt_text, current_prompt_profile
from orchestrator.prompt_audit import audited_text

TASKS = {
    "fine_grained_perception": ("understanding", "mmt_bench"),
    "single_image_reasoning": ("understanding", "llava_bench"),
    "multi_image_logic": ("understanding", "dialog_mmdu_like"),
    "localized_editing": ("editing", "edit"),
    "grounded_edit_to_select": ("unified", "unify.seed.l3.gts"),
    "understand_recall_generate_select": ("unified", "unify.weave.uie_select_recall"),
    "stepwise_sequence": ("unified", "unify.comm.task4.c1_steps"),
    "middle_step_gap": ("unified", "unify.comm.task4.c2_fill_gap"),
    "instructional_future_sequence": ("unified", "unify.isg.i2.seq_select"),
}

# The paper uses capability-level names while the runtime keeps shorter,
# benchmark-oriented names.  Accepting these explicit aliases at the routing
# boundary lets the model follow the paper terminology without changing the
# downstream schema.
TASK_ALIASES = {
    "visual_reasoning": "single_image_reasoning",
    "editing_and_infilling": "localized_editing",
    "interleaved_understand_recall_generate_select": "understand_recall_generate_select",
    "stepwise_sequence_generate_select": "stepwise_sequence",
    "middle_step_gap_generate_fill": "middle_step_gap",
    "instructional_sequential_generate_select": "instructional_future_sequence",
}

# Segment summaries provide context, but the task itself must be grounded in
# the evidence type that can establish its claim.  These rules are deliberately
# about the routing decision only; generated samples receive stricter checks in
# paper_validation.py.
TASK_EVIDENCE_RULES = {
    "fine_grained_perception": "single_frame",
    "single_image_reasoning": "single_frame",
    "multi_image_logic": "multi_frame",
    "localized_editing": "reviewed_edit_pair",
    "grounded_edit_to_select": "reviewed_edit_pair",
    "understand_recall_generate_select": "temporal",
    "stepwise_sequence": "temporal",
    "middle_step_gap": "temporal_gap",
    "instructional_future_sequence": "future_sequence",
}

# Explicit compatibility spellings only; never guess a nearby frame or pair.
EVIDENCE_ALIASES = {
    "S_SUM": "S_SEGEMENT_SUM",
    "S_GSUM": "S_SEGEMENT_SUM",
    "S_SEGMENT_SUM": "S_SEGEMENT_SUM",
}

ROUTING_EVIDENCE_CONTRACT = (
    "\n\nROUTING EVIDENCE CONTRACT (task-specific; overrides generic summary wording):\n"
    "Route each positive quota using evidence appropriate to that task. "
    "Single-image tasks require a concrete frame reference (F*_CAP or F*_STATE). "
    "Multi-image logic requires at least two distinct frame positions and a transition when the pack exposes a relevant change. "
    "Editing tasks require a reviewed pair (P*_REVIEW), its transition (P*_TRANS), and source/target frame references. "
    "Match actual pair endpoints, never the numeric P ID; selected edit_pairs must match those citations. "
    "Temporal unified tasks require ordered frame references with increasing times and a semantic transition connecting them. "
    "Middle-step gaps require before/after frames and the transition path between them. "
    "Future-sequence tasks require two context states plus a real later target, established by a transition or a cited F*_STATE. "
    "S_SEGEMENT_SUM is optional context and is never a universal hard requirement. "
    "Never invent an evidence ID. A legal but incomplete closure means quota 0 for that task only; unknown IDs and malformed JSON remain hard errors. "
    "Routing proves task suitability only; generated samples must independently pass their stricter format, scope, evidence and target checks."
)


class RouteEvidenceError(ValueError):
    """Hard routing error (unknown IDs or malformed routing structure)."""


def _canonical_task_name(name: Any) -> str:
    name = str(name or "").strip()
    return TASK_ALIASES.get(name, name)


def _evidence_alias_map(evidence_ids: set[str]) -> dict[str, str]:
    """Return only aliases whose target exists in this evidence pack."""
    aliases = {alias: target for alias, target in EVIDENCE_ALIASES.items() if target in evidence_ids}
    # ``Fxx_IMAGE`` was used by an early export.  Generate only the exact
    # one-to-one aliases backed by this pack; this is not nearest-frame fuzzy
    # matching and remains visible in aliases_applied.
    for evidence_id in evidence_ids:
        match = re.match(r"^F(\d+)_CAP$", str(evidence_id))
        if match:
            aliases.setdefault(f"F{int(match.group(1)):02d}_IMAGE", evidence_id)
    if "S_SEGEMENT_SUM" in evidence_ids:
        aliases.setdefault("S_SEGMENT_SUM", "S_SEGEMENT_SUM")
    return aliases


def _normalize_evidence_refs(raw_refs: Any, evidence_ids: set[str], *, task: str) -> tuple[list[str], list[str], list[str]]:
    """Normalize explicit compatibility spellings while rejecting unknown IDs."""
    if not isinstance(raw_refs, list) or any(not isinstance(ref, str) for ref in raw_refs):
        raise RouteEvidenceError(f"Invalid routing evidence_refs for {task}")
    aliases = _evidence_alias_map(evidence_ids)
    raw = list(raw_refs)
    normalized: list[str] = []
    applied: list[str] = []
    for value in raw:
        stripped = value.strip()
        # Case normalization is safe because internal IDs are uppercase and it
        # does not choose a nearby frame or pair.
        candidate = stripped if stripped in evidence_ids else stripped.upper()
        target = candidate if candidate in evidence_ids else aliases.get(candidate, candidate)
        if target != stripped:
            applied.append(f"{stripped}->{target}")
        if target not in evidence_ids:
            raise RouteEvidenceError(f"Unknown routing evidence for {task}: {stripped}")
        if target not in normalized:
            normalized.append(target)
    return raw, normalized, applied


def _frame_positions(refs: list[str]) -> set[int]:
    positions: set[int] = set()
    for ref in refs:
        # CAP/STATE are concrete visual observations.  Tool-only frame hints
        # (Fxx_TOOL_*) are auxiliary context and cannot establish a visual
        # route on their own.
        match = re.match(r"^F(\d+)_(?:CAP|STATE)$", ref)
        if match:
            positions.add(int(match.group(1)))
    return positions


@dataclass
class _EvidenceIndex:
    pairs: dict[str, tuple[int, int]]
    semantic_transitions: set[str]
    times: dict[int, float]
    positions: set[int]

    def ordered(self, a: int, b: int) -> bool:
        return a < b and (a not in self.times or b not in self.times or self.times[a] < self.times[b])


def _evidence_index(evidence_ids: set[str], evidence_text: str, evidence_units: list[dict] | None,
                    frames: list[dict] | None) -> _EvidenceIndex:
    """Use existing structured endpoints; text parsing is for old trace replay only."""
    from orchestrator.umm_task_producer import build_task_context, build_weave_candidates

    if evidence_units is None:
        units = []
        for match in re.finditer(r"^\[([^]\r\n]+)\]\s*(.*?)(?=^\[|\Z)", evidence_text, re.M | re.S):
            evidence_id, text = match.groups()
            if evidence_id not in evidence_ids:
                continue
            unit = {"evidence_id": evidence_id, "text": text.strip()}
            pair = re.match(r"(?:Pair|Reviewed pair P\d+:)\s+F(\d+)->F(\d+)", text)
            if pair:
                unit.update(pair_from_pos=int(pair[1]), pair_to_pos=int(pair[2]))
            units.append(unit)
    else:
        units = evidence_units
    pairs, records = {}, []
    frame_rows = {f["pos"]: dict(f) for f in frames or []}
    for unit in units:
        evidence_id = unit.get("evidence_id")
        if evidence_id not in evidence_ids:
            continue
        for pos in _frame_positions([evidence_id]):
            frame = frame_rows.setdefault(pos, {"pos": pos})
            frame.setdefault("caption", unit.get("text", ""))
            frame.setdefault("shot_id", "")
            if not frames:
                time_match = re.search(r"\(t=([\d.]+)\)", str(unit.get("text", "")))
                if time_match:
                    frame["sec"] = float(time_match[1])
        a, b = unit.get("pair_from_pos"), unit.get("pair_to_pos")
        if type(a) is not int or type(b) is not int:
            continue
        pairs[evidence_id] = (a, b)
        if evidence_id.endswith("_TRANS"):
            text = str(unit.get("text", ""))
            text = re.sub(r"^Pair F\d+->F\d+:\s*", "", text)
            records.append({"pair_from_pos": a, "pair_to_pos": b, "transition_text": text})
    times = {pos: float(f["sec"]) for pos, f in frame_rows.items()
             if isinstance(f.get("sec"), (float, int)) and not isinstance(f["sec"], bool)}
    ctx = build_task_context({"evidence_units": units, "frames_aligned": list(frame_rows.values()),
                              "pair_records": records, "t3_out": {"task_profile": "paper"}})
    # Reuse the existing temporal eligibility rule.  Do not add a second,
    # stricter semantic classifier in the router or infer semantics from IDs.
    semantic = {c.pair_ref for c in build_weave_candidates(ctx) if c.pair_ref in evidence_ids}
    return _EvidenceIndex(pairs, semantic, times, set(frame_rows))


def _connected(a: int, b: int, pairs: set[tuple[int, int]]) -> bool:
    reached = {a}
    for source, target in sorted(pairs):
        if source in reached:
            reached.add(target)
    return b in reached


def _validate_task_evidence(task: str, refs: list[str], *, index: _EvidenceIndex,
                            edit_pairs: list[tuple[int, int]]) -> str | None:
    """Return a task-specific admission error, or None when the closure passes."""
    rule = TASK_EVIDENCE_RULES[task]
    positions = _frame_positions(refs)
    transitions = [ref for ref in refs if ref.endswith("_TRANS")]
    reviews = [ref for ref in refs if re.match(r"^P\d+_REVIEW$", ref)]
    pair_positions = index.pairs
    transition_pairs = {pair_positions[t] for t in transitions if t in pair_positions}

    if rule == "single_frame":
        return None if positions else "missing_frame_evidence"
    if rule == "multi_frame":
        if len(positions) < 2:
            return "requires_two_distinct_frame_positions"
        changed_pairs = {pair_positions[t] for t in index.semantic_transitions if t in pair_positions
                         and set(pair_positions[t]) <= positions}
        if changed_pairs and not changed_pairs.intersection(transition_pairs):
            return "missing_comparison_transition_evidence"
        if transitions and not any({a, b} <= positions for a, b in transition_pairs):
            return "comparison_transition_does_not_match_frames"
        return None
    if rule == "reviewed_edit_pair":
        if not reviews:
            return "missing_reviewed_pair_evidence"
        if not transitions:
            return "missing_pair_transition_evidence"
        review_pairs = {pair_positions[r] for r in reviews if r in pair_positions}
        if any(ref not in pair_positions for ref in reviews + transitions):
            return "missing_pair_endpoint_binding"
        matching_pairs = review_pairs.intersection(transition_pairs)
        if not matching_pairs:
            return "review_and_transition_reference_different_pairs"
        cited = positions
        candidate_set = {tuple(pair) for pair in edit_pairs}
        matching_pairs.intersection_update(candidate_set)
        if not matching_pairs:
            return "reviewed_pair_not_in_candidate_pairs"
        if len(cited) < 2:
            return "missing_source_target_frame_evidence"
        if not any(a in cited and b in cited and index.ordered(a, b) for a, b in matching_pairs):
            return "source_target_frames_do_not_match_reviewed_pair"
        return None
    if rule in {"temporal", "temporal_gap", "future_sequence"}:
        ordered_positions = sorted(positions)
        if len(positions) < 2 or not all(index.ordered(a, b) for a, b in zip(ordered_positions, ordered_positions[1:])):
            return "requires_two_distinct_ordered_frames"
        if not transitions:
            if rule == "future_sequence" and len(positions) >= 3 and f"F{max(positions):02d}_STATE" in refs:
                return None
            return "missing_semantic_transition_evidence"
        semantic_pairs = {pair_positions[t] for t in transitions if t in index.semantic_transitions
                          and t in pair_positions and index.ordered(*pair_positions[t])}
        if not semantic_pairs:
            return "transition_is_camera_only_or_no_change"
        if rule == "future_sequence":
            # Two observed context states plus a genuinely later target; its
            # transition can establish the target at routing time.
            for a, b in semantic_pairs:
                if a in positions and b in index.positions and any(p < a and index.ordered(p, a) for p in positions):
                    return None
            return "missing_future_state_after_two_context_frames"
        before, after = min(positions), max(positions)
        if rule == "temporal_gap":
            if not _connected(before, after, semantic_pairs) or not any(before < p < after for p in index.positions):
                return "missing_transition_path_between_gap_endpoints"
        elif not any({a, b} <= positions for a, b in semantic_pairs):
            return "transition_does_not_match_cited_frames"
        return None
    return "unknown_task_evidence_rule"


def validate_route_evidence(task: str, refs: Any, *, evidence_ids: set[str], evidence_text: str = "",
                            edit_pairs: list[tuple[int, int]] | None = None,
                            evidence_units: list[dict] | None = None, frames: list[dict] | None = None) -> dict[str, Any]:
    """Public, side-effect-free validator used by routing and regression tests."""
    canonical = _canonical_task_name(task)
    if canonical not in TASKS:
        raise RouteEvidenceError(f"Unknown paper task: {task}")
    raw, normalized, aliases = _normalize_evidence_refs(refs, evidence_ids, task=canonical)
    index = _evidence_index(evidence_ids, evidence_text, evidence_units, frames)
    reason = _validate_task_evidence(canonical, normalized, index=index, edit_pairs=edit_pairs or [])
    return {"task": canonical, "raw_refs": raw, "normalized_refs": normalized,
            "aliases_applied": aliases, "valid": reason is None,
            "validation_error": reason}


def route_tasks(llm: Any, *, evidence_text: str, evidence_ids: set[str],
                limits: dict[str, int], eligible: dict[str, bool],
                edit_pairs: list[tuple[int, int]], evidence_units: list[dict] | None = None,
                frames: list[dict] | None = None) -> dict[str, Any]:
    prompt = render_prompt_text("paper", "routing", profile=current_prompt_profile() if current_prompt_profile() != "legacy" else "paper_v2",
        tasks_json=json.dumps(TASKS), limits_json=json.dumps(limits), eligible_json=json.dumps(eligible),
        pairs_json=json.dumps(edit_pairs), evidence_text=evidence_text)
    # Keep the versioned prompt registry immutable while removing the old
    # universal-S_* sentence from the effective request.  Older profiles are
    # historical artifacts; this replacement makes every active profile obey
    # the same task-specific contract below.
    prompt = prompt.replace(
        "Read S_SEGEMENT_SUM before concrete frame/transition evidence.\n"
        "For every task give quota, reason, evidence_refs. Positive quota requires an existing S_* ID and relevant frame/transition IDs. Explain zero quotas.",
        "Use concrete frame, pair, review, and transition records to establish each task.\n"
        "For every task give quota, reason, evidence_refs. S_SEGEMENT_SUM is optional context; positive quotas require task-specific evidence closure. Explain zero quotas.",
    )
    prompt += ROUTING_EVIDENCE_CONTRACT
    attempts = []
    request = prompt
    for attempt in range(2):
        raw = audited_text(llm, request, max_new_tokens=2400)
        try:
            result = _validate_routes(raw, evidence_ids, limits, eligible, edit_pairs, evidence_text=evidence_text,
                                      evidence_units=evidence_units, frames=frames)
        except ValueError as exc:
            attempts.append({"raw": raw, "validation_error": str(exc)})
            if hasattr(exc, "tasks"):
                attempts[-1]["tasks"] = exc.tasks
            if attempt:
                exc.routing_attempts = attempts
                raise
            if current_prompt_profile() in {'paper_v10', 'paper_v11', 'paper_v13', 'paper_v15'}:
                correction = render_prompt_text('paper', 'routing_correction',
                    validation_error=str(exc), previous_response=raw)
                correction = correction.replace(
                    "Every positive quota must cite an existing S_* summary and concrete relevant frame/transition evidence. Add only genuinely supporting existing IDs.",
                    "Each positive quota must cite task-specific evidence: concrete frames for single-image tasks, multiple frames for multi-image tasks, reviewed pair plus transition and endpoints for editing, and ordered frames plus a semantic transition for temporal tasks. S_SEGEMENT_SUM is optional context. Add only genuinely supporting existing IDs.",
                )
                request = prompt + '\n' + correction + ROUTING_EVIDENCE_CONTRACT
            else:
                request = (
                prompt + "\nYour previous response failed validation: " + str(exc) +
                "\nPREVIOUS RESPONSE:\n" + raw +
                "\nReturn the complete corrected JSON. Cite evidence appropriate to each task: "
                "single-image tasks need frame evidence; multi-image tasks need multiple frame "
                "positions; editing tasks need a reviewed pair, its transition and source/target "
                "frames; temporal tasks need ordered frames and a semantic transition. "
                "S_SEGEMENT_SUM is optional context. Do not invent IDs; use quota zero when "
                "task-specific support is insufficient."
                )
        else:
            attempts.append({"raw": raw, "validation_error": None})
            if current_prompt_profile() in {'paper_v10', 'paper_v11', 'paper_v13', 'paper_v15'}:
                previous = parse_llm_json(attempts[0]['raw']) if len(attempts) > 1 else None
                current = parse_llm_json(raw)
                changes = []
                if isinstance(previous, dict) and isinstance(previous.get('tasks'), dict):
                    for task in TASKS:
                        before = previous['tasks'].get(task)
                        if before is None:
                            before = next((previous['tasks'].get(alias) for alias, canonical in TASK_ALIASES.items()
                                           if canonical == task and alias in previous['tasks']), None)
                        before = before if isinstance(before, dict) else {}
                        after = current['tasks'].get(task) if isinstance(current, dict) and isinstance(current.get('tasks'), dict) else None
                        if after is None and isinstance(current, dict) and isinstance(current.get('tasks'), dict):
                            after = next((current['tasks'].get(alias) for alias, canonical in TASK_ALIASES.items()
                                          if canonical == task and alias in current['tasks']), None)
                        after = after if isinstance(after, dict) else {}
                        for field in ['quota', 'reason']:
                            if before.get(field) != after.get(field):
                                changes.append({'task': task, 'field': field, 'before': before.get(field), 'after': after.get(field)})
                result['correction_semantic_changes'] = changes
            return {**result, "attempts": attempts}


def _validate_routes(raw: str, evidence_ids: set[str], limits: dict[str, int],
                     eligible: dict[str, bool], edit_pairs: list[tuple[int, int]],
                     evidence_text: str = "", evidence_units: list[dict] | None = None,
                     frames: list[dict] | None = None) -> dict[str, Any]:
    parsed = parse_llm_json(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tasks"), dict):
        raise RouteEvidenceError("Automatic task routing returned invalid JSON; no default tasks are admitted")
    routes = {}
    supplied_tasks = {}
    for name, row in parsed["tasks"].items():
        canonical = _canonical_task_name(name)
        if canonical not in TASKS or canonical in supplied_tasks:
            raise RouteEvidenceError(f"Unknown or duplicate routing task: {name}")
        supplied_tasks[canonical] = row
    index = _evidence_index(evidence_ids, evidence_text, evidence_units, frames)
    for task in TASKS:
        row = supplied_tasks.get(task)
        if not isinstance(row, dict) or not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise RouteEvidenceError(f"Missing routing decision/reason for {task}")
        quota = row.get("quota")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise RouteEvidenceError(f"Invalid quota for {task}")
        raw_refs, refs, aliases = _normalize_evidence_refs(row.get("evidence_refs", []), evidence_ids, task=task)
        accepted = min(quota, max(0, limits.get(task, 0))) if eligible.get(task, False) else 0
        validation_error = None
        if accepted:
            validation_error = _validate_task_evidence(task, refs, index=index, edit_pairs=edit_pairs)
        if validation_error:
            # A valid ID set with an incomplete task closure is a data-quality
            # decision for this task, not a reason to discard every other route.
            accepted = 0
        routes[task] = {"quota": accepted, "requested_quota": quota, "reason": row["reason"],
                        "evidence_refs": refs, "raw_refs": raw_refs,
                        "normalized_refs": refs, "aliases_applied": aliases,
                        "eligibility": bool(eligible.get(task, False)),
                        "admission_reason": (
                            "ineligible" if not eligible.get(task, False) else
                            "insufficient_task_specific_evidence" if validation_error else
                            "quota_limit" if quota > accepted else "model_decision"
                        ),
                        "validation_error": validation_error}
    selected_pairs = []
    raw_selected_pairs = parsed.get("edit_pairs", [])
    if not isinstance(raw_selected_pairs, list):
        raise RouteEvidenceError("Invalid edit_pairs in routing response")
    for row in raw_selected_pairs:
        if (not isinstance(row, dict) or type(row.get("from")) is not int or type(row.get("to")) is not int
                or (row["from"], row["to"]) not in edit_pairs):
            raise RouteEvidenceError("Routing selected an unreviewed edit pair")
        if row not in selected_pairs:
            selected_pairs.append(row)
    editing = routes["localized_editing"]
    if editing["quota"]:
        refs = editing["evidence_refs"]
        cited_reviews = {index.pairs[r] for r in refs if r.endswith("_REVIEW") and r in index.pairs}
        cited_transitions = {index.pairs[r] for r in refs if r.endswith("_TRANS") and r in index.pairs}
        cited_positions = _frame_positions(refs)
        supported_pairs = {pair for pair in cited_reviews & cited_transitions if set(pair) <= cited_positions}
        if not selected_pairs or any((r["from"], r["to"]) not in supported_pairs for r in selected_pairs):
            editing.update(quota=0, admission_reason="insufficient_task_specific_evidence",
                           validation_error="selected_pair_not_supported_by_cited_closure")
    if any(v["validation_error"] for v in routes.values()) and not any(v["quota"] for v in routes.values()):
        exc = RouteEvidenceError("All positive routing decisions lack task-specific evidence")
        exc.tasks = routes
        raise exc
    return {"tasks": routes, "edit_pairs": selected_pairs, "raw": raw,
            "capabilities": list(dict.fromkeys(TASKS[k][0] for k, v in routes.items() if v["quota"])),
            "routing_evidence": {
                "policy": "task_specific_closure_v1",
                "segment_summary_optional": True,
                "unknown_ids_hard_fail": True,
                "task_level_downgrades": [k for k, v in routes.items() if v.get("validation_error")],
            }}


def label_paper_item(item: dict[str, Any]) -> dict[str, Any]:
    benchmark = item.get("benchmark", "")
    task = next((name for name, (_, b) in TASKS.items() if b == benchmark), None)
    if benchmark == "mme":
        task = "fine_grained_perception"
    if task is None:
        raise ValueError(f"Non-paper output benchmark: {benchmark}")
    item["capability"], item["task_type"] = TASKS[task][0], task
    return item
