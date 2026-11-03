"""Freeze valid identities before asking for complete pair state changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from executor.tasks.prompt_loader import render_prompt_text
from orchestrator.independent_observation import digest, service_binding
from orchestrator.planner_helpers import parse_llm_json
from orchestrator.prompt_audit import audited_vision, image_identity, paper_prompt_stage
from orchestrator.visual_state import validate_entity_identity, validate_pair_observation


def _freeze(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _request(planner, images, prompt, path, tokens):
    binding = {'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
               'image_sha256': [image_identity(p)['sha256'] for p in images],
               **service_binding(planner.vlm)}
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['binding'] != binding or saved['final_prompt'] != prompt:
            raise ValueError('frozen_identity_or_change_request_changed')
    else:
        raw = audited_vision(planner, images, prompt, max_new_tokens=tokens)
        saved = {'binding': binding, 'raw_response': raw, 'final_prompt': prompt,
                 'model_identity': dict(getattr(planner.vlm, 'last_attestation', {}) or {})}
        _freeze(path, saved)
    return saved['raw_response']


@paper_prompt_stage('T3IdentityThenChanges')
def compare_bound_pair(planner, source: dict, target: dict, cache_root: Path) -> dict:
    left, right = source['visual_state'], target['visual_state']
    images = [source['img'], target['img']]
    base = render_prompt_text('paper', 'entity_identity',
        source_json=json.dumps(left, ensure_ascii=False), target_json=json.dumps(right, ensure_ascii=False))
    note = ''
    identity = None
    for attempt in range(3):
        raw = _request(planner, images, base + note, cache_root / f'identity_attempt_{attempt:03d}.json', 5500)
        try:
            candidate = parse_llm_json(raw)
            validate_entity_identity(candidate, left, right)
            if set(candidate) != {'source_frame', 'target_frame', 'matches', 'unmatched_source', 'unmatched_target', 'uncertainties'}:
                raise ValueError('identity_stage_contains_unrequested_change_or_missing_fields')
        except ValueError as exc:
            if attempt == 2:
                raise
            # Keep inventories and images in the base request; no recursive
            # injection of the full invalid response or guessed repaired IDs.
            note = '\nCONTRACT ERROR: ' + str(exc) + '. Use only the actual per-side IDs; unmatched grouping is not an add/remove event.'
        else:
            identity = candidate
            break
    base = render_prompt_text('paper', 'bound_pair_changes',
        source_json=json.dumps(left, ensure_ascii=False), target_json=json.dumps(right, ensure_ascii=False),
        identity_json=json.dumps(identity, ensure_ascii=False))
    note = ''
    attempts = []
    for attempt in range(3):
        raw = _request(planner, images, base + note, cache_root / f'changes_attempt_{attempt:03d}.json', 10000)
        check = validate_pair_observation(raw, left, right)
        attempts.append({'status': check['status'], 'validation_error': check.get('reason'), 'raw_response': raw})
        value = check.get('entity_comparison')
        frozen = {key: value.get(key) for key in identity} if isinstance(value, dict) else None
        if check['status'] in {'valid', 'no_change'} and frozen == identity:
            return {**check, 'attempts': attempts, 'identity_binding_sha256': digest(identity),
                    'identity_checkpoint': identity}
        error = check.get('reason') if check['status'] == 'invalid' else 'frozen_identity_changed'
        if attempt == 2:
            raise ValueError(error)
        note = '\nCONTRACT ERROR: ' + str(error) + '. Keep the frozen identity fields byte-for-byte as JSON values. Reinspect every listed entity; do not invent changes.'
    raise RuntimeError('unreachable_pair_comparison')
