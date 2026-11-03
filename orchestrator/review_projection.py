"""Lossless text references for compact, answer-blind review requests."""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from PIL import Image


def text_leaves(value: Any, source: str):
    """Decode only JSON containers; keep ordinary prose and invalid JSON intact."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            decoded = None
        if isinstance(decoded, (dict, list)):
            yield from text_leaves(decoded, source + '.decoded')
        elif value.strip():
            for sentence in re.split(r'(?<=[.!?])\s+', value.strip()):
                yield source, sentence
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from text_leaves(child, f'{source}[{index}]')
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from text_leaves(child, source + '.' + key)


def compact_observations(observations: list[dict]) -> list[dict]:
    # Full cache/service/raw evidence remains in the saved review scope.
    return [copy.deepcopy(saved['observation']) for saved in observations]


def image_dimensions(images: list[str]) -> list[dict]:
    result = []
    for index, path in enumerate(images):
        with Image.open(path) as image:
            width, height = image.size
        result.append({'review_image_number': index + 1, 'width': width, 'height': height})
    return result


def compact_review_input(item: dict, scope: dict, *, profile: str | None = None) -> dict:
    """Keep each sentence once and every occurrence's original round/source."""
    claims, by_text = [], {}

    def refs(value, source, turn, kind):
        result = []
        for path, sentence in text_leaves(value, source):
            if sentence not in by_text:
                claim = {'id': f'claim_{len(claims) + 1:03d}', 'text': sentence,
                         'kind': kind, 'occurrences': []}
                by_text[sentence] = claim
                claims.append(claim)
            claim = by_text[sentence]
            if kind == 'answer':
                claim['kind'] = 'answer'
            occurrence = {'turn': turn, 'source': path, 'kind': kind}
            if occurrence not in claim['occurrences']:
                claim['occurrences'].append(occurrence)
            result.append(claim['id'])
        return result

    def query_projection(query, source, turn):
        projected = {'images': copy.deepcopy(query.get('images', [])), 'messages': []}
        for index, message in enumerate(query.get('messages', [])):
            if message.get('type') == 'text':
                projected['messages'].append({'role': message.get('role'), 'type': 'text',
                    'claim_refs': refs(message.get('content'), f'{source}.messages[{index}].content',
                                       turn, 'answer' if message.get('role') == 'assistant' else 'premise')})
            else:
                projected['messages'].append(copy.deepcopy(message))
        return projected

    task = {key: copy.deepcopy(item[key]) for key in
            ['module', 'benchmark', 'capability', 'task_type', 'A', 'B', 'C', 'D'] if key in item}
    rounds = []
    for index, turn in enumerate(item.get('turns') or []):
        name = str(turn.get('t') or f'turn_{index + 1}')
        source = f'turns[{index}]'
        row = {key: copy.deepcopy(turn[key]) for key in ['t', 'mode', 'role', 'in', 'evidence_refs'] if key in turn}
        row.setdefault('t', name)
        if turn.get('text') is not None:
            row['text_claim_refs'] = refs(turn['text'], source + '.text', name,
                'premise' if turn.get('role') == 'user' else 'answer')
        query = turn.get('render', {}).get('query')
        if isinstance(query, dict):
            row['query'] = query_projection(query, source + '.render.query', name)
        gt = turn.get('gt') or {}
        if not name.startswith('select'):
            row['answer_claim_refs'] = refs(gt.get('answer'), source + '.gt.answer', name, 'answer')
        if 'image' in gt:
            row['reviewer_target_image'] = gt['image']
        # Retain nontext target configuration; keys and rationale stay hidden.
        extra_gt = {key: copy.deepcopy(value) for key, value in gt.items() if key not in {'answer', 'image'}}
        if extra_gt:
            row['target_configuration'] = extra_gt
        rounds.append(row)
    if not rounds:
        task['premise_claim_refs'] = refs(item.get('question') or item.get('instruction'), 'question', 'task', 'premise')
        if item.get('benchmark') != 'mmt_bench':
            task['answer_claim_refs'] = refs(item.get('gpt_answer') or item.get('answer'), 'answer', 'task', 'answer')
    elif isinstance(item.get('render', {}).get('query'), dict):
        # Retained package prose can differ from per-round queries; check it too.
        task['retained_query'] = query_projection(item['render']['query'], 'render.query', 'retained_query')
    projected_scope = {key: copy.deepcopy(value) for key, value in scope.items()
                       if key not in {'turn_inputs', 'independent_observations'}}
    projected_scope.setdefault('required_turn_checks', [row['t'] for row in rounds] or ['task'])
    projected_scope.setdefault('history_policy', 'Only previously shown solver images and answers; reviewer-only targets are not solver evidence')
    projected_scope['coordinate_conventions'] = {
        'origin': 'image top-left; x rightward, y downward',
        'observation_bbox_estimate': 'normalized 0..1000, approximate, not measured annotations',
        'candidate_coordinates': scope.get('candidate_coordinate_convention', 'unspecified; do not assume units'),
        'conversion': 'normalized_x = pixel_x / original_width * 1000; normalized_y = pixel_y / original_height * 1000',
        'precision': 'qualitative or approximate placement; use named parts and pixels, not equality of estimates'}
    from executor.tasks.prompt_loader import current_prompt_profile
    if (profile or current_prompt_profile()) in {'paper_v14', 'paper_v15'}:
        if (profile or current_prompt_profile()) == 'paper_v15':
            from orchestrator.spatial_grounding_v15 import spatial_signals
        else:
            from orchestrator.spatial_grounding import spatial_signals
        for claim in claims:
            claim['spatial_signals'] = spatial_signals(claim['text'])
    return {'adapter_version': 'compact_review_1', 'task': task, 'rounds': rounds,
            'claims': claims, 'scope': projected_scope,
            'observations': compact_observations(scope.get('independent_observations', []))}


def task_review_contract(item: dict) -> str:
    benchmark = item.get('benchmark', '')
    if benchmark == 'mmt_bench':
        return ('Single-image perception MCQ: verify the question premise and every option independently, including synonym/subtype overlap and independent spatial axes. '
                'Exactly one supported option is required. Unshown pouring/material transfer is not established by container tilt. The key is hidden.')
    if benchmark == 'llava_bench':
        return ('Single-image reasoning: require a cue-to-conclusion inference beyond direct description, grounded in this image and qualified common-sense physical knowledge. '
                'Reject unobserved transfer, future event, hidden stage or unsupported human purpose. Check every sentence; preserve the original object and question purpose.')
    if benchmark == 'dialog_mmdu_like':
        return ('Multi-image logic/dialogue allows comparison, connection and joint verification of directly visible facts across images. '
                'Inspect every user premise and assistant answer in ordered image/history scope; a separate why/how inference is unnecessary.')
    if benchmark == 'edit' or '.seed.' in benchmark:
        return ('Localized editing, or grounded edit-to-select based on its admitted pair: all source-target changes must fit one or two atomic local edits with stable invariants. '
                'Verify actual direction, generated target and independent option support.')
    if '.weave.' in benchmark:
        return ('Weave: Understand/Plan receive earlier AND later images. QA/Gen receive the earlier image and may recall the later image and answers already shown in that dialogue. '
                'Visible comparisons and recall are valid without a separate reasoning inference. No one/two-edit limit applies. Preserve the QA object and requested attribute. '
                'Select matches the generated target with independently checked options.')
    if '.c1_' in benchmark or '.c1.' in benchmark:
        return ('C1: Overall receives endpoints and may describe their supported net change, without claiming hidden detours/reversals/intermediate steps. '
                'Steps receives all frames and can describe their ordered changes. Each Gen uses its actual preceding state and history; each Select matches that generated target. '
                'Direct visual description is valid; no separate reasoning inference or one/two-edit limit applies.')
    if '.c2_' in benchmark or '.c2.' in benchmark:
        return ('C2: Explain/Gen construct a compatible real-middle target from endpoint constraints. The real middle is reviewer target evidence, not solver input. '
                'Judge endpoint compatibility and target match; do not demand unique prediction or claim direct observation of a hidden middle/trajectory. No one/two-edit limit applies.')
    if '.isg.' in benchmark:
        return ('ISG: shown context plus a grounded continuation instruction leads to a desired future state. Future targets are reviewer evidence. '
                'Instructions, edits, so_that and checks may specify a compatible desired state without predicting a unique future; because clauses asserting context must be visible in that context. '
                'Inspect observed-versus-desired wording in every sentence. No separate reasoning inference or one/two-edit limit applies. Select independently matches the desired generated target.')
    raise ValueError('Unknown task review contract: ' + benchmark)


def review_response_schema(item: dict, claims: list[dict], *, profile: str = 'paper_v12') -> dict:
    schema = {'visible_cues': ['concrete observed fact'], 'independent_answer': 'own concise supported assessment or uncertain',
        'unsupported_claims': [], 'premise_supported': True, 'answer_supported': True,
        'unique_answer': True, 'options_exclusive': True, 'task_type_supported': True,
        'scope_supported': True, 'decision': 'accept|reject|hold', 'reason': 'concrete findings',
        'claim_checks': [{'id': row['id'], 'status': 'supported|contradicted|uncertain|not_applicable',
            'scope_supported': True, 'observation_refs': ['actual observation id'], 'reason': 'pixel and every occurrence scope basis'} for row in claims]}
    option = {'status': 'supported|excluded|uncertain', 'reason': 'independent pixel basis'}
    if item.get('benchmark') == 'mmt_bench':
        schema['option_checks'] = {letter: dict(option) for letter in 'ABCD'}
        if profile in {'paper_v13', 'paper_v14', 'paper_v15'}:
            schema['independent_answer'] = 'exactly one independently selected A/B/C/D letter, or uncertain'
    elif item.get('benchmark') == 'llava_bench':
        schema['inference_check'] = {'conclusion': 'own supported conclusion', 'visible_cues': ['concrete cue'],
            'inference_link': 'cue-to-conclusion link', 'substantive_inference': True}
    else:
        schema['turn_checks'] = []
        turns = item.get('turns') or [{'t': 'task'}]
        for index, turn in enumerate(turns):
            name = str(turn.get('t') or f'turn_{index + 1}')
            row = {'turn': name, 'independent_answer': 'own supported open or target assessment',
                **dict.fromkeys(['premise_supported', 'answer_supported', 'scope_supported', 'target_supported', 'direction_supported'], True),
                'reason': 'concrete observed support or precise missing evidence'}
            if name.startswith('select'):
                row['independent_answer'] = 'one independently selected letter or uncertain'
                row['option_checks'] = {chr(65 + i): dict(option) for i in range(len(turn.get('in', {}).get('images', [])))}
            schema['turn_checks'].append(row)
    if profile in {'paper_v13', 'paper_v14', 'paper_v15'}:
        schema['scene_uncertainties'] = ['general unobserved scene detail, separate from any candidate error; empty if none']
    if profile in {'paper_v14', 'paper_v15'}:
        if profile == 'paper_v15':
            from orchestrator.spatial_grounding_v15 import spatial_schema
        else:
            from orchestrator.spatial_grounding import spatial_schema
        for check in schema['claim_checks']:
            check.update(spatial_schema())
    return schema


def compact_request_binding(item: dict, scope: dict, *, profile: str = 'paper_v12') -> dict:
    projection = compact_review_input(item, scope, profile=profile)
    return {'projection': projection, 'task_contract': task_review_contract(item),
            'response_schema': review_response_schema(item, projection['claims'], profile=profile)}


def separate_provenance_refs(result: dict, bindings: list[dict]) -> None:
    """Separate only known structural IDs; never invent pixel support or change a vote."""
    neutral = {row['evidence_id'] for row in bindings}
    normalizations = []
    for check in result.get('claim_checks', []):
        refs = check.get('observation_refs')
        if not isinstance(refs, list) or not any(ref in neutral for ref in refs):
            continue
        if 'provenance_refs' in check:
            raise ValueError('ambiguous_existing_provenance_refs')
        check['provenance_refs'] = [ref for ref in refs if ref in neutral]
        check['observation_refs'] = [ref for ref in refs if ref not in neutral]
        normalizations.append({'claim_id': check.get('id'), 'original_observation_refs': list(refs),
                               'separated_provenance_refs': check['provenance_refs']})
    if normalizations:
        result['response_normalizations'] = normalizations
