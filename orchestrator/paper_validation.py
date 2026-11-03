"""Evidence-scope checks and independent visual validation of paper candidates."""
from __future__ import annotations

import copy

import json
import re
from typing import Any

from executor.tasks.prompt_loader import render_prompt_text, current_prompt_profile, structured_evidence_profile
from orchestrator.planner_helpers import parse_llm_json
from orchestrator.prompt_audit import audited_vision, paper_prompt_stage
from orchestrator.independent_observation import candidate_claims, ancillary_claims, validate_claim_checks
from orchestrator.review_projection import compact_review_input, task_review_contract, review_response_schema, separate_provenance_refs
from orchestrator.review_consistency import validate_bound_review_errors


def validate_refs(refs: Any, evidence_ids: set[str], positions: list[int], *, all_frames: bool = False,
                  single_image: bool = False) -> list[str]:
    if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in evidence_ids for r in refs):
        raise ValueError("missing_or_unknown_model_citations")
    prefixes = [f"F{p:02d}_" for p in positions]
    if not any(any(r.startswith(pfx) for pfx in prefixes) for r in refs):
        raise ValueError("missing_frame_citation")
    if all_frames and any(not any(r.startswith(pfx) for r in refs) for pfx in prefixes):
        raise ValueError("missing_endpoint_citation")
    if single_image and any(not r.startswith((*prefixes, "S_")) for r in refs):
        raise ValueError("cross_frame_evidence_in_single_image_item")
    return list(refs)


def validate_single_image_item(item: dict, benchmark: str, evidence_ids: set[str], position: int) -> None:
    if item.get("image_pos") != position or isinstance(item.get("image_pos"), bool):
        raise ValueError("generated_image_pos_outside_prompt_scope")
    validate_refs(item.get("evidence_refs"), evidence_ids, [position], single_image=True)
    if benchmark == "mmt_bench":
        options = [str(item.get(k, "")).strip() for k in "ABCD"]
        if not all(options) or len({re.sub(r"\W+", "", v.lower()) for v in options}) != 4:
            raise ValueError("empty_or_duplicate_options")
        if item.get("_answer") not in {"A", "B", "C", "D"}:
            raise ValueError("invalid_choice_answer")
        if current_prompt_profile() in {"paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
            directions = [re.sub(r"^(to |the |of |in |at )+", "", value.lower()).strip(' .') for value in options]
            axes = []
            for value in directions:
                found = {axis for axis, pattern in {
                    "horizontal": r"\b(left|right)\b", "vertical": r"\b(above|below)\b",
                    "depth": r"\b(behind|front)\b"}.items() if re.search(pattern, value)}
                axes.append(found)
            if all(len(found) == 1 for found in axes) and len(set.union(*axes)) > 1:
                raise ValueError("nonexclusive_spatial_axes")
    if benchmark == "llava_bench":
        if item.get("reasoning_category") not in {"reasoning", "function", "activity", "relation"}:
            raise ValueError("missing_paper_reasoning_category")
        answer = str(item.get("gpt_answer", "")).strip()
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s]
        if not 2 <= len(sentences) <= 4:
            raise ValueError("reasoning_answer_requires_2_to_4_sentences")
    if structured_evidence_profile() and benchmark in {"mmt_bench", "llava_bench"}:
        validate_understanding_evidence(item, benchmark, evidence_ids, position)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing_{field}")
    return value.strip()


def _canonical(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold())


def _validate_visible_claim_refs(refs: Any, evidence_ids: set[str], position: int) -> None:
    validate_refs(refs, evidence_ids, [position], single_image=True)
    if any(r.startswith("S_") for r in refs):
        raise ValueError("summary_cannot_establish_single_image_claim")
    # Tool IDs alone do not establish whether a signal came from these pixels,
    # audio, or an adjacent frame. Bind claims to the verified image observation.
    allowed = {f"F{position:02d}_CAP", f"F{position:02d}_STATE"}
    if any(r not in allowed for r in refs):
        raise ValueError("auxiliary_signal_cannot_establish_single_image_claim")


def _claim_refs(row: dict) -> Any:
    primary, explicit = row.get("evidence_refs"), row.get("same_image_evidence_refs")
    if primary is not None and explicit is not None and primary != explicit:
        raise ValueError("conflicting_same_image_citations")
    return primary if primary is not None else explicit


def preserve_mmt_verifier_source(source: dict, fixed: dict) -> dict:
    fields = ["question", "image_pos", "A", "B", "C", "D", "_answer", "evidence_refs"]
    if any(fixed.get(k) != source.get(k) for k in fields):
        raise ValueError("verifier_changed_question_options_answer_or_evidence")
    if "perception_evidence" in fixed and fixed["perception_evidence"] != source.get("perception_evidence"):
        raise ValueError("verifier_changed_author_evidence_contract")
    # The verifier acknowledges the producer's item. It does not author a new
    # evidence contract; an omitted optional contract does not erase the source.
    return source


def validate_understanding_evidence(item: dict, benchmark: str, evidence_ids: set[str], position: int) -> None:
    """Require inspectable author evidence, without treating it as visual proof."""
    field = "reasoning_evidence" if benchmark == "llava_bench" else "perception_evidence"
    contract = item.get(field)
    if not isinstance(contract, dict):
        raise ValueError(f"missing_{field}")
    cue_key = "visible_cues" if benchmark == "llava_bench" else "premise_cues"
    cues = contract.get(cue_key)
    if not isinstance(cues, list) or not cues:
        raise ValueError("missing_visible_premise_cues")
    for cue in cues:
        if not isinstance(cue, dict):
            raise ValueError("invalid_visible_cue")
        _required_text(cue.get("claim"), "visible_cue_claim")
        _validate_visible_claim_refs(_claim_refs(cue), evidence_ids, position)
    if benchmark == "mmt_bench":
        checks = contract.get("option_checks")
        if not isinstance(checks, dict) or set(checks) != set("ABCD"):
            raise ValueError("missing_per_option_checks")
        for letter, check in checks.items():
            if not isinstance(check, dict) or check.get("status") != ("supported" if letter == item["_answer"] else "excluded"):
                raise ValueError("ambiguous_or_unsupported_choice")
            _required_text(check.get("reason"), "option_exclusion_reason")
            _validate_visible_claim_refs(_claim_refs(check), evidence_ids, position)
        return
    conclusion = _required_text(contract.get("conclusion"), "reasoning_conclusion")
    link = _required_text(contract.get("inference_link"), "inference_link")
    direct_claims = {_canonical(cue["claim"]) for cue in cues}
    if _canonical(conclusion) in direct_claims or _canonical(link) in direct_claims | {_canonical(conclusion)}:
        raise ValueError("description_repeated_as_inference")
    claims = contract.get("answer_claims")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", item["gpt_answer"].strip()) if s.strip()]
    if not isinstance(claims, list) or len(claims) != len(sentences):
        raise ValueError("answer_claims_do_not_cover_every_sentence")
    for sentence, claim in zip(sentences, claims):
        if not isinstance(claim, dict) or claim.get("claim") != sentence or claim.get("support") not in {"visible", "inferred"}:
            raise ValueError("answer_claim_binding_mismatch")
        _validate_visible_claim_refs(_claim_refs(claim), evidence_ids, position)
    if not any(c["support"] == "inferred" for c in claims):
        raise ValueError("reasoning_has_no_inferred_claim")


def validate_dialog(turns: Any, positions: list[int], evidence_ids: set[str], rounds: int) -> None:
    if not isinstance(turns, list) or len(turns) != rounds * 2:
        raise ValueError("dialog_turn_count_mismatch")
    used = set()
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict) or turn.get("role") != ("user" if i % 2 == 0 else "assistant") or not str(turn.get("text", "")).strip():
            raise ValueError("dialog_role_or_text_invalid")
        refs = validate_refs(turn.get("evidence_refs"), evidence_ids, positions)
        frame_refs = [r for r in refs if re.match(r"^F\d+_", r)]
        if any(not any(r.startswith(f"F{p:02d}_") for p in positions) for r in frame_refs):
            raise ValueError("dialog_cites_unseen_frame")
        used.update(p for p in positions if any(r.startswith(f"F{p:02d}_") for r in refs))
    if len(used) < 2:
        raise ValueError("dialog_does_not_bind_multiple_images")


def review_turn_ids(item: dict) -> list[str]:
    turns = item.get("turns")
    if isinstance(turns, list) and turns:
        return [str(turn.get("t") or f"turn_{i + 1}") for i, turn in enumerate(turns)]
    return ["task"]


def blind_selection_keys(value: Any) -> Any:
    result = copy.deepcopy(value)
    if not isinstance(result, dict):
        return result
    if isinstance(result.get("eval"), dict):
        result["eval"].pop("answer", None)
    for key in ("turns", "turn_inputs"):
        for turn in result.get(key, []) or []:
            if not isinstance(turn, dict) or not str(turn.get("t", "")).startswith("select"):
                continue
            if isinstance(turn.get("gt"), dict):
                turn["gt"].pop("answer", None)
            if isinstance(turn.get("render"), dict):
                turn["render"].pop("answer", None)
            turn.pop("rationale", None)
    return result


def validate_round_answers(result: dict, item: dict) -> None:
    turns = item.get("turns", [])
    checks = result.get("turn_checks", [])
    for turn, check in zip(turns, checks):
        answer = _required_text(check.get("independent_answer"), "independent_turn_answer").strip()
        if not str(turn.get("t", "")).startswith("select"):
            if answer in {"A", "B", "C", "D"} and result["decision"] == "accept":
                result.update(decision="hold", answer_comparison="choice_letter_cannot_replace_open_round")
            continue
        count = len(turn.get("in", {}).get("images", []))
        letters = [chr(65 + i) for i in range(count)]
        options = check.get("option_checks")
        if not letters or not isinstance(options, dict) or set(options) != set(letters):
            raise ValueError("Independent selection lacks actual per-round option checks")
        for option in options.values():
            if not isinstance(option, dict) or option.get("status") not in {"supported", "excluded", "uncertain"} or not str(option.get("reason", "")).strip():
                raise ValueError("Invalid per-round option check")
        expected = turn.get("gt", {}).get("answer")
        if result["decision"] == "accept" and (
                answer != expected or any(options[key]["status"] != ("supported" if key == expected else "excluded") for key in letters)):
            result.update(decision="hold", answer_comparison="independent_selection_mismatch_or_ambiguity")


@paper_prompt_stage("IndependentVisualReview")
def review_paper_item(planner: Any, item: dict, images: list, scope: dict, retry_note: str = "") -> dict:
    review_item = item
    if current_prompt_profile() in {"paper_v3", "paper_v4", "paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        fields = ["module", "benchmark", "capability", "task_type", "question", "A", "B", "C", "D", "instruction", "turns", "eval"]
        if item.get("benchmark") != "mmt_bench":
            fields += ["gpt_answer", "answer"]
        review_item = {key: item[key] for key in fields if key in item}
    if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        scope = {**scope, "required_turn_checks": review_turn_ids(item),
                 "history_policy": "Only previously shown solver images and answers; reviewer-only targets are not solver evidence"}
    if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        review_item, scope = blind_selection_keys(review_item), blind_selection_keys(scope)
    replacements = {"item_json": json.dumps(review_item, ensure_ascii=False), "scope_json": json.dumps(scope)}
    if current_prompt_profile() in {"paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        observations = scope.get('independent_observations')
        if not isinstance(observations, list) or len(observations) != len(images):
            raise ValueError('independent_review_requires_answer_blind_observations')
        claims = candidate_claims(review_item)
        replacements.update(observations_json=json.dumps(observations, ensure_ascii=False),
                            claims_json=json.dumps(claims, ensure_ascii=False))
    if current_prompt_profile() in {'paper_v12', 'paper_v13', 'paper_v14', 'paper_v15'}:
        projection = compact_review_input(item, scope)
        claims = projection['claims']
        replacements = {'projection_json': json.dumps(projection, ensure_ascii=False),
                        'task_contract': task_review_contract(item),
                        'response_schema': json.dumps(review_response_schema(item, claims, profile=current_prompt_profile()))}
    prompt = render_prompt_text("paper", "visual_review", profile=current_prompt_profile(), **replacements)
    if retry_note:
        prompt += "\nThe previous reply failed the response contract: " + retry_note + ". Return a complete judgment; do not change the task or claim acceptance to repair the format."
    api_tokens = 7000 if current_prompt_profile() in {"paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'} else 4000 if getattr(getattr(planner, "vlm", None), "wire_profile", "") == "openai" else 1800 if structured_evidence_profile() else 1400
    if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
        api_tokens = 12000
    raw = audited_vision(planner, images, prompt, max_new_tokens=api_tokens)
    result = parse_llm_json(raw)
    fields = ["premise_supported", "answer_supported", "unique_answer", "options_exclusive"]
    if not isinstance(result, dict) or result.get("decision") not in {"accept", "reject", "hold"}:
        raise ValueError("Invalid independent visual review")
    raw_decision = result["decision"]
    if any(not isinstance(result.get(k), bool) for k in fields) or not isinstance(result.get("visible_cues"), list) or not isinstance(result.get("unsupported_claims"), list) or not result.get("reason"):
        raise ValueError("Incomplete independent visual review")
    if not isinstance(result.get("independent_answer"), str) or not result["independent_answer"].strip():
        raise ValueError("Independent visual review lacks its own answer")
    if result["decision"] == "accept" and (not all(result[k] for k in fields) or result["unsupported_claims"] or not result["visible_cues"]):
        result["decision"] = "hold"
    if result["decision"] == "accept" and item.get("benchmark") == "mmt_bench":
        independent = result["independent_answer"].strip()
        expected = item.get("_answer")
        if independent != expected and independent.casefold() != str(item.get(expected, "")).strip().casefold():
            result["decision"] = "hold"
            result["answer_comparison"] = "independent_answer_does_not_match_candidate"
    elif (result["decision"] == "accept" and result["independent_answer"].strip() in {"A", "B", "C", "D"}
          and not (current_prompt_profile() in {"paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'} and isinstance(item.get("turns"), list)
                   and any(str(turn.get("t", "")).startswith("select") for turn in item["turns"]))):
        # v9 validates every round below. A top-level selection letter must not
        # erase complete independent answers to the dialogue's open rounds.
        result["decision"] = "hold"
        result["answer_comparison"] = "choice_letter_is_not_an_open_response_answer"
    if structured_evidence_profile():
        if not isinstance(result.get("task_type_supported"), bool):
            raise ValueError("Independent review lacks task-type judgment")
        if not result["task_type_supported"] and result["decision"] == "accept":
            result["decision"] = "hold"
        if item.get("benchmark") == "mmt_bench":
            checks = result.get("option_checks")
            if not isinstance(checks, dict) or set(checks) != set("ABCD"):
                raise ValueError("Independent review lacks per-option judgment")
            for check in checks.values():
                if not isinstance(check, dict) or check.get("status") not in {"supported", "excluded", "uncertain"} or not str(check.get("reason", "")).strip():
                    raise ValueError("Invalid independent option judgment")
            if result["decision"] == "accept" and any(checks[k]["status"] != ("supported" if k == item["_answer"] else "excluded") for k in "ABCD"):
                result["decision"] = "hold"
                result["answer_comparison"] = "independent_option_checks_not_unique"
        elif item.get("benchmark") == "llava_bench":
            inference = result.get("inference_check")
            if not isinstance(inference, dict) or not isinstance(inference.get("substantive_inference"), bool):
                raise ValueError("Independent review lacks inference judgment")
            conclusion = _required_text(inference.get("conclusion"), "independent_conclusion")
            _required_text(inference.get("inference_link"), "independent_inference_link")
            cues = inference.get("visible_cues")
            if not isinstance(cues, list) or not cues or any(not isinstance(c, str) or not c.strip() for c in cues):
                raise ValueError("Independent review lacks inference cues")
            if result["decision"] == "accept" and (not inference["substantive_inference"] or _canonical(conclusion) in {_canonical(c) for c in cues}):
                result["decision"] = "hold"
                result["answer_comparison"] = "direct_description_is_not_reasoning"
    if current_prompt_profile() in {"paper_v5", "paper_v6", "paper_v7", "paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        if not isinstance(result.get("scope_supported"), bool):
            raise ValueError("Independent review lacks solver-scope judgment")
        if item.get("benchmark") not in {"mmt_bench", "llava_bench"}:
            checks = result.get("turn_checks")
            expected = review_turn_ids(item)
            if (not isinstance(checks, list) or len(checks) != len(expected) or
                    [check.get("turn") for check in checks if isinstance(check, dict)] != expected):
                raise ValueError("Independent review lacks complete ordered turn checks")
            fields = ["premise_supported", "answer_supported", "scope_supported", "target_supported", "direction_supported"]
            for check in checks:
                if any(not isinstance(check.get(key), bool) for key in fields) or not str(check.get("reason", "")).strip():
                    raise ValueError("Invalid independent turn judgment")
            if result["decision"] == "accept" and any(not all(check[key] for key in fields) for check in checks):
                result["decision"] = "hold"
                result["answer_comparison"] = "unsupported_turn_claim_scope_or_target"
        if result["decision"] == "accept" and not result["scope_supported"]:
            result["decision"] = "hold"
            result["answer_comparison"] = "solver_scope_not_supported"
    if current_prompt_profile() in {"paper_v8", "paper_v9", "paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'} and isinstance(item.get("turns"), list):
        validate_round_answers(result, item)
        result["raw_model_decision"] = raw_decision
    if current_prompt_profile() in {"paper_v10", "paper_v11", "paper_v12", 'paper_v13', 'paper_v14', 'paper_v15'}:
        validate_claim_checks(result, claims, observations)
        result['raw_model_decision'] = raw_decision
    if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'}:
        validate_bound_review_errors(result, claims, mcq=item.get('benchmark') == 'mmt_bench')
    return {**result, "raw_response": raw, "training_approved": False, "review_scope": scope}


@paper_prompt_stage('AncillaryExplanationReview')
def review_ancillary_item(planner: Any, item: dict, images: list, scope: dict, bindings: list,
                          retry_note: str = '') -> dict:
    claims = ancillary_claims(item, decode_render_json=current_prompt_profile() in {'paper_v12', 'paper_v13', 'paper_v14', 'paper_v15'})
    if not claims:
        return {'decision': 'accept', 'reason': 'No additional retained explanation claims.', 'claim_checks': [], 'api_calls': 0}
    observations = scope.get('independent_observations')
    if not isinstance(observations, list) or len(observations) != len(images):
        raise ValueError('ancillary_review_requires_answer_blind_observations')
    # Freeze independent selection before disclosing its retained explanation.
    replacements = dict(claims_json=json.dumps(claims, ensure_ascii=False),
        scope_json=json.dumps(blind_selection_keys(scope), ensure_ascii=False),
        bindings_json=json.dumps(bindings, ensure_ascii=False),
        observations_json=json.dumps(observations, ensure_ascii=False))
    if current_prompt_profile() in {'paper_v12', 'paper_v13', 'paper_v14', 'paper_v15'}:
        projection = compact_review_input(blind_selection_keys(item), blind_selection_keys(scope))
        replacements = {'projection_json': json.dumps({'task': projection['task'], 'rounds': projection['rounds'],
            'history_claims': projection['claims'],
            'scope': projection['scope'], 'claims': claims, 'neutral_bindings': bindings,
            'observations': projection['observations']}, ensure_ascii=False),
            'task_contract': task_review_contract(item)}
    if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
        if current_prompt_profile() == 'paper_v15':
            from orchestrator.spatial_grounding_v15 import spatial_schema
        else:
            from orchestrator.spatial_grounding import spatial_schema
        replacements['spatial_schema'] = json.dumps(spatial_schema())
    prompt = render_prompt_text('paper', 'ancillary_review', **replacements)
    if retry_note:
        prompt += '\nResponse contract error: ' + retry_note + '. Return complete ordered checks without inventing facts.'
    raw = audited_vision(planner, images, prompt, max_new_tokens=6000 if current_prompt_profile() in {'paper_v14', 'paper_v15'} else 3500)
    result = parse_llm_json(raw)
    if not isinstance(result, dict) or result.get('decision') not in {'accept', 'reject', 'hold'} or not str(result.get('reason', '')).strip():
        raise ValueError('invalid_ancillary_review')
    result['raw_model_decision'] = result['decision']
    if current_prompt_profile() in {'paper_v12', 'paper_v13', 'paper_v14', 'paper_v15'}:
        separate_provenance_refs(result, bindings)
    from orchestrator.independent_observation import structural_frame_proofs
    validate_claim_checks(result, claims, observations,
                          structural_proofs=structural_frame_proofs(claims, item, scope))
    result['raw_response'] = raw
    result['claims'] = claims
    return result
