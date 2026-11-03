"""Answer-blind image observations and complete candidate claim checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from executor.tasks.prompt_loader import current_prompt_profile, render_prompt_text
from orchestrator.planner_helpers import parse_llm_json
from orchestrator.prompt_audit import audited_vision, image_identity, paper_prompt_stage


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def service_binding(client: Any) -> dict:
    return {key: getattr(client, attribute, None) for key, attribute in {
        'requested_model': 'model', 'endpoint': 'base_url', 'reasoning_effort': 'reasoning_effort',
        'wire_profile': 'wire_profile'}.items()}


def validate_observation(value: Any, frame_id: str) -> dict:
    if not isinstance(value, dict) or value.get('frame_id') != frame_id:
        raise ValueError('independent_observation_frame_mismatch')
    entities = value.get('entities')
    if not isinstance(entities, list) or not entities:
        raise ValueError('independent_observation_missing_entities')
    ids = []
    for row in entities:
        if (not isinstance(row, dict) or not isinstance(row.get('id'), str)
                or not row['id'].startswith(frame_id + '_O') or not str(row.get('label', '')).strip()
                or not str(row.get('visible_cues', '')).strip() or not isinstance(row.get('uncertainties'), list)):
            raise ValueError('independent_observation_invalid_entity')
        box = row.get('bbox_estimate')
        if box is not None and (not isinstance(box, list) or len(box) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not 0 <= x <= 1000 for x in box)
                or not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError('independent_observation_invalid_estimated_box')
        ids.append(row['id'])
    if len(set(ids)) != len(ids):
        raise ValueError('independent_observation_duplicate_entity')
    allowed = {'horizontal': {'left_of', 'right_of', 'overlapping', 'uncertain'},
               'vertical': {'above', 'below', 'overlapping', 'uncertain'},
               'depth': {'in_front_of', 'behind', 'uncertain'},
               'contact': {'touching', 'inside', 'on', 'separate', 'uncertain'}}
    relations = value.get('relations')
    if not isinstance(relations, list):
        raise ValueError('independent_observation_missing_relations')
    for row in relations:
        if (not isinstance(row, dict) or row.get('subject') not in ids or row.get('anchor') not in ids
                or row.get('subject') == row.get('anchor') or row.get('axis') not in allowed
                or row.get('relation') not in allowed[row['axis']]
                or not all(str(row.get(k, '')).strip() for k in ['id', 'subject_part', 'anchor_part', 'support'])
                or row.get('certainty') not in {'supported', 'uncertain'}
                # A named relation can remain a tentative hypothesis. Only
                # supported relations may establish downstream spatial facts.
                or (row['relation'] == 'uncertain' and row['certainty'] != 'uncertain')):
            raise ValueError('independent_observation_invalid_relation')
    events = value.get('events')
    if not isinstance(events, list) or any(not isinstance(row, dict)
            or row.get('status') not in {'visible', 'not_visible', 'uncertain'}
            or not all(str(row.get(k, '')).strip() for k in ['id', 'claim', 'support']) for row in events):
        raise ValueError('independent_observation_invalid_event')
    all_ids = ids + [row['id'] for row in relations] + [row['id'] for row in events]
    if current_prompt_profile() in {'paper_v13', 'paper_v14', 'paper_v15'}:
        from orchestrator.visual_state import CATEGORIES
        if any(row.get('category') not in CATEGORIES for row in entities):
            raise ValueError('independent_observation_requires_entity_categories')
        roles = value.get('roles')
        if not isinstance(roles, list):
            raise ValueError('independent_observation_requires_manipulation_roles')
        for row in roles:
            if (not isinstance(row, dict) or row.get('actor') not in ids
                    or (row.get('object') is not None and row.get('object') not in ids)
                    or row.get('actor') == row.get('object')
                    or row.get('certainty') not in {'supported', 'uncertain'}
                    or row.get('contact_visibility') not in {'visible', 'occluded', 'uncertain'}
                    or not all(isinstance(row.get(key), str) and row[key].strip()
                               for key in ['id', 'role', 'visible_cues', 'contact_uncertainty'])
                    or (row.get('certainty') == 'supported' and row.get('object') is None)):
                raise ValueError('independent_observation_invalid_manipulation_role')
        hands = {row['id'] for row in entities if row['category'] == 'body_part'}
        if not hands <= {row['actor'] for row in roles}:
            raise ValueError('independent_observation_missing_body_part_role_assessment')
        all_ids += [row['id'] for row in roles]
    if len(set(all_ids)) != len(all_ids) or any(not key.startswith(frame_id + '_') for key in all_ids):
        raise ValueError('independent_observation_duplicate_or_unbound_id')
    if not isinstance(value.get('uncertainties'), list):
        raise ValueError('independent_observation_missing_uncertainties')
    if current_prompt_profile() == 'paper_v14':
        from orchestrator.spatial_grounding import validate_spatial_observation
        validate_spatial_observation(value)
    if current_prompt_profile() == 'paper_v15':
        from orchestrator.spatial_grounding import validate_spatial_observation
        from orchestrator.spatial_grounding_v15 import validate_reference_frames
        validate_reference_frames(value)
        validate_spatial_observation({**value, 'relations': [r for r in value['relations'] if r['reference_frame'] == 'image']})
    return value


def load_bound_observation(path: Path, image: str, frame_id: str, client: Any) -> dict:
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('binding', {}).get('image_sha256') != image_identity(image).get('sha256'):
        raise ValueError('independent_observation_image_binding_changed')
    prompt = render_prompt_text('paper', 'independent_observation', profile=current_prompt_profile(), frame_id=frame_id)
    expected = {'frame_id': frame_id, 'image_sha256': image_identity(image).get('sha256'),
                'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **service_binding(client)}
    if saved.get('binding') != expected:
        raise ValueError('independent_observation_request_binding_changed')
    observation = validate_observation(saved.get('observation'), frame_id)
    if digest(observation) != saved.get('observation_sha256') or parse_llm_json(saved['raw_response']) != observation:
        raise ValueError('independent_observation_content_changed')
    if (not saved.get('final_prompt', '').startswith(prompt)
            or hashlib.sha256(saved['final_prompt'].encode()).hexdigest() != saved.get('final_prompt_sha256')):
        raise ValueError('independent_observation_final_prompt_changed')
    return saved


@paper_prompt_stage('IndependentObservation')
def observe_frame(planner: Any, image: str, frame_id: str, cache_path: Path) -> dict:
    if cache_path.exists():
        return load_bound_observation(cache_path, image, frame_id, planner.vlm)
    prompt = render_prompt_text('paper', 'independent_observation', profile=current_prompt_profile(), frame_id=frame_id)
    binding = {'frame_id': frame_id, 'image_sha256': image_identity(image).get('sha256'),
               'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), **service_binding(planner.vlm)}
    if not binding['image_sha256']:
        raise ValueError('independent_observation_requires_original_image_file')
    note = ''
    for attempt in range(2):
        final_prompt = prompt + note
        raw = audited_vision(planner, [image], final_prompt, max_new_tokens=8000 if current_prompt_profile() in {'paper_v14', 'paper_v15'} else 6000)
        # Preserve even invalid model replies independently of the completed cache.
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        response_index = len(list(cache_path.parent.glob(cache_path.stem + '_response_*.json')))
        response_path = cache_path.with_name(cache_path.stem + f'_response_{response_index}.json')
        with response_path.open('x', encoding='utf-8') as f:
            json.dump({'binding': binding, 'final_prompt': final_prompt,
                       'final_prompt_sha256': hashlib.sha256(final_prompt.encode()).hexdigest(),
                       'raw_response': raw}, f, ensure_ascii=False, indent=2)
        try:
            observation = validate_observation(parse_llm_json(raw), frame_id)
        except ValueError as exc:
            if attempt:
                raise
            note = '\nThe previous response failed the output contract: ' + str(exc) + '. Return complete JSON; keep unresolved relations uncertain. Do not invent observations to repair format.'
            continue
        saved = {'binding': binding, 'source_image': str(image), 'observation': observation,
                 'observation_sha256': digest(observation), 'raw_response': raw,
                 'final_prompt': final_prompt, 'final_prompt_sha256': hashlib.sha256(final_prompt.encode()).hexdigest(),
                 'model_identity': dict(getattr(planner.vlm, 'last_attestation', {}) or {}),
                 'visual_acceptance': 'pending', 'training_approved': False}
        with cache_path.open('x', encoding='utf-8') as f:
            json.dump(saved, f, ensure_ascii=False, indent=2)
        return saved


def candidate_claims(item: dict) -> list[dict]:
    """Bind all displayed premises and open-answer sentences, not a chosen subset."""
    claims = []
    def add(value: Any, source: str, turn: str, kind: str):
        if isinstance(value, str) and value.strip():
            for text in re.split(r'(?<=[.!?])\s+', value.strip()):
                claims.append({'id': f'claim_{len(claims) + 1:03d}', 'turn': turn,
                               'source': source, 'kind': kind, 'text': text})
        elif isinstance(value, list):
            for index, child in enumerate(value):
                add(child, f'{source}[{index}]', turn, kind)
        elif isinstance(value, dict):
            for key, child in value.items():
                add(child, source + '.' + key, turn, kind)
    for index, turn in enumerate(item.get('turns') or []):
        name = str(turn.get('t') or f'turn_{index + 1}')
        if turn.get('role') in {'user', 'assistant'}:
            add(turn.get('text'), f'turns[{index}].text', name,
                'premise' if turn['role'] == 'user' else 'answer')
        elif not name.startswith(('select', 'gen')):
            add(turn.get('gt', {}).get('answer'), f'turns[{index}].gt.answer', name, 'answer')
    if not item.get('turns'):
        add(item.get('question') or item.get('instruction'), 'question', 'task', 'premise')
        if item.get('benchmark') != 'mmt_bench':
            add(item.get('gpt_answer') or item.get('answer'), 'answer', 'task', 'answer')
    return claims


def ancillary_claims(item: dict, *, decode_render_json: bool = False) -> list[dict]:
    """Explanations withheld during independent selection, audited afterward."""
    claims = []
    seen = {row['text'] for row in candidate_claims(item)}
    def add(value: Any, source: str, turn: str):
        if decode_render_json:
            from orchestrator.review_projection import text_leaves
            for path, text in text_leaves(value, source):
                append(text, path, turn)
            return
        append(value, source, turn)
    def append(value: Any, source: str, turn: str):
        if not isinstance(value, str) or not value.strip() or value.strip() in {'A', 'B', 'C', 'D'}:
            return
        for text in re.split(r'(?<=[.!?])\s+', value.strip()):
            if text in seen or re.fullmatch(r'target:F\d+', text):
                continue
            seen.add(text)
            claims.append({'id': f'ancillary_{len(claims) + 1:03d}', 'turn': turn,
                           'source': source, 'kind': 'answer', 'text': text})
    for index, turn in enumerate(item.get('turns') or []):
        name = str(turn.get('t') or f'turn_{index + 1}')
        add(turn.get('rationale'), f'turns[{index}].rationale', name)
        for j, message in enumerate(turn.get('render', {}).get('answer', {}).get('messages', [])):
            if message.get('type') == 'text':
                add(message.get('content'), f'turns[{index}].render.answer.messages[{j}]', name)
    for index, message in enumerate(item.get('render', {}).get('answer', {}).get('messages', [])):
        if message.get('type') == 'text':
            add(message.get('content'), f'render.answer.messages[{index}]', 'retained_answer_envelope')
    return claims


def neutral_evidence_bindings(source: dict) -> list[dict]:
    """Structural provenance only; excludes captions, conclusions and keys."""
    result = []
    for unit in source.get('evidence', {}).get('evidence_units', []):
        if not isinstance(unit, dict) or not unit.get('evidence_id'):
            continue
        result.append({key: unit[key] for key in ['evidence_id', 'frame_pos', 'pair_from_pos',
                       'pair_to_pos', 'pair_key'] if key in unit})
    if len({row['evidence_id'] for row in result}) != len(result):
        raise ValueError('duplicate_neutral_evidence_binding')
    return result


def structural_frame_proofs(claims: list[dict], item: dict, scope: dict) -> dict:
    """Prove a closed frame-order statement from its actual selection inputs."""
    proofs = {}
    for claim in claims:
        match = re.fullmatch(r'(F\d+) is the only supplied middle frame between (F\d+) and (F\d+)\.',
                             claim.get('text', ''))
        if not match or not claim.get('turn', '').startswith('select'):
            continue
        turns = [turn for turn in item.get('turns', []) if turn.get('t') == claim['turn']]
        if len(turns) != 1:
            continue
        supplied = turns[0].get('in', {})
        positions, images = supplied.get('frame_positions'), supplied.get('images')
        if (not isinstance(positions, list) or not isinstance(images, list)
                or len(positions) != len(images)
                or any(isinstance(pos, bool) or not isinstance(pos, int) for pos in positions)
                or len(set(positions)) != len(positions)):
            continue
        # An integer in the candidate alone is not a verified image binding.
        bound = {row.get('candidate_image', row.get('image')): row.get('frame_position')
                 for row in scope.get('review_image_bindings', [])}
        if not images or any(image not in bound or bound[image] != pos for image, pos in zip(images, positions)):
            continue
        middle, earlier, later = [int(frame[1:]) for frame in match.groups()]
        if not earlier < later or not {middle, earlier, later} <= set(positions):
            continue
        between = sorted(pos for pos in positions if earlier < pos < later)
        proofs[claim['id']] = {'kind': 'selection_frame_order', 'turn': claim['turn'],
            'supplied_frame_positions': list(positions), 'supplied_images': list(images),
            'earlier_frame_position': earlier, 'later_frame_position': later,
            'asserted_middle_frame_position': middle,
            'verified_status': 'supported' if between == [middle] else 'contradicted'}
    return proofs


def validate_claim_checks(result: dict, claims: list[dict], observations: list[dict],
                          *, structural_proofs: dict | None = None) -> None:
    checks = result.get('claim_checks')
    if not isinstance(checks, list) or [row.get('id') for row in checks if isinstance(row, dict)] != [r['id'] for r in claims]:
        raise ValueError('independent_review_missing_ordered_claim_checks')
    ids = {row['id'] for saved in observations for field in ['entities', 'relations', 'events', 'roles']
           for row in saved['observation'].get(field, [])}
    for check, claim in zip(checks, claims):
        refs = check.get('observation_refs')
        proof = (structural_proofs or {}).get(claim['id'])
        if proof:
            check['structural_proof'] = proof
            if check.get('status') in {'supported', 'contradicted'} and check['status'] != proof['verified_status']:
                raise ValueError('independent_review_structural_claim_mismatch:' + claim['id'])
        if (check.get('status') not in {'supported', 'contradicted', 'uncertain', 'not_applicable'}
                or not isinstance(check.get('scope_supported'), bool)
                or not str(check.get('reason', '')).strip() or not isinstance(refs, list)
                or any(ref not in ids for ref in refs)
                or (check['status'] in {'supported', 'contradicted'} and not refs and not proof)
                or (check['status'] == 'not_applicable' and claim['kind'] != 'premise')):
            raise ValueError('independent_review_invalid_claim_check')
    if result['decision'] == 'accept' and any(row['status'] in {'contradicted', 'uncertain'} or not row['scope_supported'] for row in checks):
        result.update(decision='hold', answer_comparison='independent_claim_conflict_or_uncertainty')
    if current_prompt_profile() in {'paper_v14', 'paper_v15'}:
        if current_prompt_profile() == 'paper_v15':
            from orchestrator.spatial_grounding_v15 import validate_spatial_grounding
        else:
            from orchestrator.spatial_grounding import validate_spatial_grounding
        validate_spatial_grounding(result, claims, observations)
