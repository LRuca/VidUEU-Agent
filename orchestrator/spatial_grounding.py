"""Check typed spatial citations without treating observations as pixel truth."""
from __future__ import annotations

import re


AXES = {
    'horizontal': {'left_of', 'right_of', 'overlapping', 'uncertain'},
    'vertical': {'above', 'below', 'overlapping', 'uncertain'},
    'depth': {'in_front_of', 'behind', 'uncertain'},
    'contact': {'touching', 'inside', 'on', 'separate', 'uncertain'},
}
INVERSE = {'left_of': 'right_of', 'right_of': 'left_of', 'above': 'below', 'below': 'above',
           'in_front_of': 'behind', 'behind': 'in_front_of', 'overlapping': 'overlapping',
           'touching': 'touching', 'separate': 'separate'}
PARTS = {'whole', 'body', 'opening', 'rim', 'handle', 'bowl', 'tip', 'surface',
         'left_edge', 'right_edge', 'left_portion', 'right_portion', 'top_edge', 'bottom_edge'}
SIGNALS = re.compile(
    r'\b(?:to\s+the\s+)?(?:left|right)(?:\s+(?:side\s+)?of)?\b|'
    r'\b(?:above|below|over|under|overhead|upper|lower|upward|downward|leftward|rightward)\b|'
    r'\b(?:in\s+front\s+of|behind|inside|within|on\s+top\s+of|overlap\w*|beside|between)\b', re.I)


def spatial_signals(text: str) -> list[dict]:
    """A conservative lexical omission alarm, not a natural-language parser."""
    return [{'span': [m.start(), m.end()], 'text': m.group()} for m in SIGNALS.finditer(text)]


def _signal_relation(word: str) -> tuple[str, str] | None:
    word = word.casefold()
    direct = {'above': ('vertical', 'above'), 'over': ('vertical', 'above'),
              'below': ('vertical', 'below'), 'under': ('vertical', 'below'),
              'behind': ('depth', 'behind'), 'in front of': ('depth', 'in_front_of'),
              'inside': ('contact', 'inside'), 'within': ('contact', 'inside')}
    if word in direct:
        return direct[word]
    if re.search(r'\b(left|right)\b.*\bof\b', word):
        return ('horizontal', 'left_of' if 'left' in word else 'right_of')
    return None


def spatial_schema() -> dict:
    return {
        'spatial_clauses': [{
            'span': [0, 1], 'text': 'exact contiguous clause substring',
            'frame_id': 'actual Fxx', 'mode': 'observed|desired|question|hypothetical',
            'subject': 'actual frame-local O id', 'subject_mention': 'exact object words in clause',
            'subject_part': '|'.join(sorted(PARTS)),
            'anchor': 'actual frame-local O id', 'anchor_mention': 'exact reference object words in clause',
            'anchor_part': 'same part vocabulary', 'axis': 'horizontal|vertical|depth|contact',
            'relation': 'allowed relation on that axis', 'polarity': 'positive|negative',
            'observation_refs': ['actual matching relation id, not just an entity id'],
            'status': 'supported|contradicted|uncertain|not_applicable',
            'scope_supported': True, 'reason': 'pixels, exact pair/part/axis and occurrence scope',
        }],
        'nonrelational_spans': [{'span': [0, 1], 'text': 'exact directional word substring',
                                 'reason': 'object locator or absolute image motion/pose rather than a pair relation; still independently verify pixels and scope'}],
    }


def _span(row: dict, text: str) -> tuple[int, int] | None:
    span = row.get('span')
    if (not isinstance(span, list) or len(span) != 2
            or any(type(x) is not int for x in span) or not 0 <= span[0] < span[1] <= len(text)
            or row.get('text') != text[span[0]:span[1]]):
        return None
    return tuple(span)


def _mention(mention: str, entity: dict, clause: str) -> bool:
    if not isinstance(mention, str) or not mention.strip() or mention.casefold() not in clause.casefold():
        return False
    aliases = [entity.get('label', ''), *entity.get('aliases', [])]
    return any(re.search(r'(?<!\w)' + re.escape(alias.casefold()) + r'(?!\w)', mention.casefold())
               for alias in aliases if isinstance(alias, str) and alias)


def _relation_support(clause: dict, relation: dict) -> str | None:
    direct = (clause['subject'], clause['subject_part'], clause['anchor'], clause['anchor_part'])
    stated = (relation['subject'], relation['subject_part'], relation['anchor'], relation['anchor_part'])
    inverse = (clause['anchor'], clause['anchor_part'], clause['subject'], clause['subject_part'])
    if clause['axis'] != relation['axis'] or relation['certainty'] != 'supported':
        return None
    observed = relation['relation'] if direct == stated else INVERSE.get(relation['relation']) if inverse == stated else None
    if observed is None:
        return None
    # An unrelated relation is not a logical negation: overlap and left-offset
    # can coexist, as can horizontal left-of and vertical above.
    if clause['polarity'] == 'positive':
        return 'supported' if observed == clause['relation'] else None
    return 'contradicted' if observed == clause['relation'] else None


def validate_spatial_grounding(result: dict, claims: list[dict], observations: list[dict]) -> None:
    """Insufficient grounding holds an accept; never re-roll a semantic result.

    Desired states and interrogatives are not observations. Their semantic and
    round scope judgment still belongs to the independent reviewer. This guard
    checks known lexical omissions and typed references, not pixel correctness.
    """
    frames = {saved['observation']['frame_id']: saved['observation'] for saved in observations}
    findings = []
    for claim, check in zip(claims, result['claim_checks']):
        text = claim['text']
        clauses = check.get('spatial_clauses')
        exclusions = check.get('nonrelational_spans')
        local = []
        if not isinstance(clauses, list) or not isinstance(exclusions, list):
            local.append('missing_typed_spatial_assessment')
            clauses, exclusions = [], []
        spans = []
        for row in exclusions:
            span = _span(row, text) if isinstance(row, dict) else None
            if not span or not isinstance(row.get('reason'), str) or not row['reason'].strip():
                local.append('invalid_nonrelational_span')
            else:
                # A relational phrase cannot be dismissed as an object name.
                if re.search(r'\b(?:above|below|behind|inside|within|overlap\w*|of|between|beside)\b', row['text'], re.I):
                    local.append('relation_signal_dismissed_as_nonrelational')
                spans.append(span)
        for clause in clauses:
            if not isinstance(clause, dict) or not (span := _span(clause, text)):
                local.append('invalid_spatial_clause_span')
                continue
            spans.append(span)
            axis, rel = clause.get('axis'), clause.get('relation')
            if (axis not in AXES or rel not in AXES[axis]
                    or clause.get('subject_part') not in PARTS or clause.get('anchor_part') not in PARTS
                    or clause.get('polarity') not in {'positive', 'negative'}
                    or clause.get('mode') not in {'observed', 'desired', 'question', 'hypothetical'}):
                local.append('invalid_spatial_tuple')
                continue
            frame = frames.get(clause.get('frame_id'), {})
            entities = {e['id']: e for e in frame.get('entities', [])}
            subject, anchor = entities.get(clause.get('subject')), entities.get(clause.get('anchor'))
            if (not subject or not anchor or subject == anchor
                    or not _mention(clause.get('subject_mention'), subject, clause['text'])
                    or not _mention(clause.get('anchor_mention'), anchor, clause['text'])):
                local.append('spatial_subject_or_anchor_not_bound_to_exact_clause')
                continue
            # Explicit negation must survive decomposition. This catches the
            # fixed 'pot is not above tray' error without supplying its answer.
            if re.search(r'\b(?:not|never|neither|without)\b|n.t\b', clause['text'], re.I) and clause['polarity'] != 'negative':
                local.append('spatial_negation_lost')
            if clause['mode'] != 'observed':
                if clause['mode'] == 'question' and '?' not in text:
                    local.append('declarative_spatial_assertion_mislabeled_question')
                if clause['mode'] == 'hypothetical' and not re.search(r'\b(if|could|would|hypothetical|compatible|possible)\b', text, re.I):
                    local.append('spatial_hypothesis_lacks_conditional_context')
                if clause['mode'] == 'desired' and not re.search(r'\b(move|place|keep|put|make|shift|position|generate|raise|lower|future|requested|should|next|target|desired|plan|continue|reach|verify)\w*\b', text, re.I):
                    local.append('spatial_desire_lacks_instruction_or_future_context')
                if not clause.get('scope_supported') or clause.get('status') not in {'supported', 'not_applicable'}:
                    local.append('nonobserved_spatial_scope_unresolved')
                continue
            refs = clause.get('observation_refs')
            relations = {r['id']: r for r in frame.get('relations', [])}
            matched = [_relation_support(clause, relations[ref]) for ref in refs if ref in relations] if isinstance(refs, list) else []
            status = clause.get('status')
            if status != 'supported' or not clause.get('scope_supported') or 'supported' not in matched:
                local.append('spatial_relation_lacks_matching_subject_part_anchor_axis_polarity')
            if any(ref not in relations for ref in refs or []):
                local.append('spatial_citation_is_not_a_bound_relation')
        for signal in spatial_signals(text):
            if not any(a <= signal['span'][0] and b >= signal['span'][1] for a, b in spans):
                local.append('uncovered_spatial_signal:' + signal['text'])
            expected = _signal_relation(signal['text'])
            if expected and not any(isinstance(c, dict) and _span(c, text)
                    and c['span'][0] <= signal['span'][0] and c['span'][1] >= signal['span'][1]
                    and (c.get('axis'), c.get('relation')) == expected for c in clauses):
                local.append('spatial_signal_relation_not_assessed:' + signal['text'])
        if local:
            findings.append({'claim_id': claim['id'], 'text': text, 'problems': list(dict.fromkeys(local))})
    result['spatial_grounding_findings'] = findings
    if findings and result['decision'] == 'accept':
        result.update(decision='hold', answer_comparison='typed_spatial_grounding_incomplete_or_conflicting')


def validate_spatial_observation(value: dict) -> None:
    """Require canonical parts and generic container/support axis coverage."""
    for entity in value['entities']:
        aliases = entity.get('aliases')
        if (not isinstance(aliases, list) or not aliases
                or any(not isinstance(a, str) or not a.strip() for a in aliases)
                or not isinstance(entity.get('foreground'), bool)):
            raise ValueError('spatial_observation_requires_aliases_and_foreground')
    for row in value['relations']:
        if row['subject_part'] not in PARTS or row['anchor_part'] not in PARTS:
            raise ValueError('spatial_observation_requires_canonical_parts')
    foreground = [e for e in value['entities'] if e['foreground']]
    for subject in foreground:
        if subject['category'] != 'container':
            continue
        for anchor in foreground:
            if anchor['id'] == subject['id'] or anchor['category'] not in {'container', 'surface', 'fabric'}:
                continue
            # Every visible container body versus each foreground support/cloth
            # needs independent axes. Uncertain is an acceptable observation.
            for axis in ['horizontal', 'vertical']:
                if not any(r['axis'] == axis and (
                        (r['subject'] == subject['id'] and r['subject_part'] == 'body' and r['anchor'] == anchor['id'])
                        or (anchor['category'] == 'container' and r['subject'] == anchor['id']
                            and r['anchor'] == subject['id'] and r['subject_part'] == r['anchor_part'] == 'body'))
                           for r in value['relations']):
                    raise ValueError('spatial_observation_missing_container_support_axis:' + subject['id'] + ':' + anchor['id'] + ':' + axis)
