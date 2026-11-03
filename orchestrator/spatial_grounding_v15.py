"""Quote-bound spatial checks with explicit image versus scene reference frames."""
from __future__ import annotations

import re

from orchestrator.spatial_grounding import AXES, INVERSE, PARTS, _mention


RELATIONAL = re.compile(
    r'\b(?:to\s+the\s+)?(?:left|right)(?:\s+side)?\s+of\b|'
    r'\b(?:in\s+front\s+of|on\s+top\s+of|above|below|over|under|behind|inside|within|overlap\w*|beside|between)\b',
    re.I)


def spatial_signals(text: str) -> list[dict]:
    rows = []
    for match in RELATIONAL.finditer(text):
        if match.group().casefold() == 'between' and re.match(r'\s+F\d+\s+and\s+F\d+\b', text[match.end():]):
            continue
        rows.append({'span': [match.start(), match.end()], 'text': match.group()})
    return rows


def quote_span(row: dict, text: str) -> tuple[int, int] | None:
    """Resolve only exact quotes; ambiguous repetitions require an occurrence."""
    quote = row.get('text')
    if not isinstance(quote, str) or not quote:
        return None
    starts = [m.start() for m in re.finditer(re.escape(quote), text)]
    occurrence = row.get('occurrence')
    if occurrence is None:
        if len(starts) != 1:
            return None
        start = starts[0]
    elif type(occurrence) is int and 0 <= occurrence < len(starts):
        start = starts[occurrence]
    else:
        return None
    return start, start + len(quote)


def spatial_schema() -> dict:
    return {
        'spatial_clauses': [{
            'text': 'exact contiguous original clause; include object words or their antecedents',
            'occurrence': 'omit for unique quote; otherwise zero-based exact occurrence',
            'frame_id': 'actual Fxx', 'mode': 'observed|desired|question|hypothetical',
            'subject': 'actual frame-local O id', 'subject_mention': 'exact original words',
            'subject_part': '|'.join(sorted(PARTS)),
            'anchor': 'actual frame-local O id', 'anchor_mention': 'exact original words',
            'anchor_part': 'same part vocabulary',
            'reference_frame': 'image|scene', 'axis': 'horizontal|vertical|depth|contact',
            'relation': 'allowed relation on axis', 'polarity': 'positive|negative',
            'observation_refs': ['matching relation IDs for observed assertion'],
            'status': 'supported|contradicted|uncertain|not_applicable', 'scope_supported': True,
            'reason': 'pixel basis, language referent/part interpretation and occurrence scope',
        }],
        'nonrelational_spans': [{'text': 'exact original phrase',
            'reason': 'specific non-pair meaning, independently checked in pixels and history'}],
    }


def _bound_mention(value, entity, quote, mode):
    if _mention(value, entity, quote):
        return True
    # The unknown object of a question is solved independently, without a key.
    return (mode == 'question' and isinstance(value, str) and value in quote
            and re.fullmatch(r'(?:what|which)(?:\s+(?:object|item|utensil|vessel))?', value, re.I) is not None)


def _support(clause, relation):
    if (clause['reference_frame'] != relation.get('reference_frame')
            or clause['axis'] != relation['axis'] or relation['certainty'] != 'supported'):
        return False
    pair = (clause['subject'], clause['subject_part'], clause['anchor'], clause['anchor_part'])
    direct = (relation['subject'], relation['subject_part'], relation['anchor'], relation['anchor_part'])
    inverse = (direct[2], direct[3], direct[0], direct[1])
    observed = relation['relation'] if pair == direct else INVERSE.get(relation['relation']) if pair == inverse else None
    return observed == clause['relation'] and clause['polarity'] == relation.get('polarity', 'positive')


def _assesses(signal, clause):
    word = signal.casefold()
    if word in {'above', 'below', 'behind', 'inside', 'within', 'in front of', 'on top of'}:
        expected = {'above': ('vertical', 'above'), 'below': ('vertical', 'below'),
                    'behind': ('depth', 'behind'), 'inside': ('contact', 'inside'),
                    'within': ('contact', 'inside'), 'in front of': ('depth', 'in_front_of'),
                    'on top of': ('contact', 'on')}[word]
        return (clause.get('axis'), clause.get('relation')) == expected
    if word in {'over', 'under'}:
        return clause.get('axis') == 'vertical' and clause.get('relation') == ('above' if word == 'over' else 'below')
    if word.startswith('overlap'):
        return clause.get('reference_frame') == 'image' and clause.get('relation') == 'overlapping'
    if re.search(r'\b(left|right)\b.*\bof\b', word):
        return clause.get('axis') == 'horizontal' and clause.get('relation') == ('left_of' if 'left' in word else 'right_of')
    if word == 'beside':
        return clause.get('axis') == 'horizontal' and clause.get('relation') in {'left_of', 'right_of'}
    return True


def validate_spatial_grounding(result: dict, claims: list[dict], observations: list[dict]) -> None:
    frames = {saved['observation']['frame_id']: saved['observation'] for saved in observations}
    findings, bindings = [], []
    for claim, check in zip(claims, result['claim_checks']):
        text, problems = claim['text'], []
        clauses, exclusions = check.get('spatial_clauses'), check.get('nonrelational_spans')
        if not isinstance(clauses, list) or not isinstance(exclusions, list):
            problems.append('missing_typed_spatial_assessment')
            clauses, exclusions = [], []
        covered, valid = [], []
        for exclusion in exclusions:
            span = quote_span(exclusion, text) if isinstance(exclusion, dict) else None
            if span is None or not str(exclusion.get('reason', '')).strip() or spatial_signals(exclusion.get('text', '')):
                problems.append('invalid_nonrelational_quote')
            else:
                covered.append(span)
        for index, clause in enumerate(clauses):
            span = quote_span(clause, text) if isinstance(clause, dict) else None
            if span is None:
                problems.append('spatial_quote_missing_or_ambiguous')
                continue
            bindings.append({'claim_id': claim['id'], 'clause_index': index, 'span': list(span), 'text': clause['text']})
            axis, relation = clause.get('axis'), clause.get('relation')
            if (axis not in AXES or relation not in AXES[axis]
                    or clause.get('reference_frame') not in {'image', 'scene'}
                    or clause.get('subject_part') not in PARTS or clause.get('anchor_part') not in PARTS
                    or clause.get('polarity') not in {'positive', 'negative'}
                    or clause.get('mode') not in {'observed', 'desired', 'question', 'hypothetical'}):
                problems.append('invalid_spatial_tuple')
                continue
            frame = frames.get(clause.get('frame_id'), {})
            entities = {e['id']: e for e in frame.get('entities', [])}
            subject, anchor = entities.get(clause.get('subject')), entities.get(clause.get('anchor'))
            if (not subject or not anchor or subject == anchor
                    or not _bound_mention(clause.get('subject_mention'), subject, clause['text'], clause['mode'])
                    or not _bound_mention(clause.get('anchor_mention'), anchor, clause['text'], clause['mode'])):
                problems.append('spatial_referents_not_bound_to_original_quote')
                continue
            covered.append(span)
            valid.append((span, clause))
            if re.search(r'\b(?:not|never|neither|without|no\s+longer)\b|n.t\b', clause['text'], re.I) and clause['polarity'] != 'negative':
                problems.append('spatial_negation_lost')
            mode = clause['mode']
            if mode != 'observed':
                if mode == 'question' and '?' not in text:
                    problems.append('assertion_mislabeled_question')
                if mode == 'hypothetical' and not re.search(r'\b(if|could|would|hypothetical|compatible|possible)\b', text, re.I):
                    problems.append('hypothesis_lacks_conditional_context')
                if mode == 'desired' and not re.search(r'\b(move|place|keep|put|make|shift|position|generate|raise|lower|future|requested|should|next|target|desired|plan|continue|reach|verify)\w*\b', text, re.I):
                    problems.append('desire_lacks_instruction_or_future_context')
                if not clause.get('scope_supported') or clause.get('status') not in {'supported', 'not_applicable'}:
                    problems.append('nonobserved_spatial_scope_unresolved')
                continue
            refs = clause.get('observation_refs')
            relations = {r['id']: r for r in frame.get('relations', [])}
            if (not isinstance(refs, list) or any(ref not in relations for ref in refs)
                    or not any(_support(clause, relations[ref]) for ref in refs if ref in relations)
                    or clause.get('status') != 'supported' or not clause.get('scope_supported')):
                problems.append('spatial_evidence_pair_part_axis_reference_or_polarity_mismatch')
        for signal in spatial_signals(text):
            a, b = signal['span']
            if not any(start <= a and end >= b for start, end in covered):
                problems.append('uncovered_spatial_signal:' + signal['text'])
            if not any(start <= a and end >= b and _assesses(signal['text'], clause) for (start, end), clause in valid):
                problems.append('spatial_signal_relation_not_assessed:' + signal['text'])
        if problems:
            findings.append({'claim_id': claim['id'], 'text': text, 'problems': list(dict.fromkeys(problems))})
    result['spatial_quote_bindings'] = bindings
    result['spatial_grounding_findings'] = findings
    if findings and result['decision'] == 'accept':
        result.update(decision='hold', answer_comparison='typed_spatial_grounding_incomplete_or_conflicting')


def validate_reference_frames(value: dict) -> None:
    for relation in value['relations']:
        if (relation.get('reference_frame') not in {'image', 'scene'}
                or relation.get('polarity') not in {'positive', 'negative'}
                or relation['subject_part'] not in PARTS or relation['anchor_part'] not in PARTS):
            raise ValueError('spatial_observation_requires_explicit_reference_frame_and_polarity')
        if relation['reference_frame'] == 'image' and relation['axis'] == 'depth':
            raise ValueError('image_projection_does_not_establish_depth')
