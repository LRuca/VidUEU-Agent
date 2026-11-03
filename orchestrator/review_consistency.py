"""Keep candidate errors distinct from general uncertainty in a scene."""
from __future__ import annotations


def validate_bound_review_errors(result: dict, claims: list[dict], *, mcq: bool) -> None:
    if not isinstance(result.get('scene_uncertainties'), list) or any(
            not isinstance(row, str) or not row.strip() for row in result['scene_uncertainties']):
        raise ValueError('review_requires_separate_scene_uncertainties')
    if mcq and result.get('independent_answer', '').strip() not in {'A', 'B', 'C', 'D', 'uncertain'}:
        raise ValueError('MCQ independent_answer must be one independently selected A/B/C/D letter or uncertain, not descriptive prose')
    by_id = {row['id']: row for row in claims}
    checks = {row.get('id'): row for row in result.get('claim_checks', []) if isinstance(row, dict)}
    seen = set()
    for row in result.get('unsupported_claims', []):
        if (not isinstance(row, dict) or set(row) != {'claim_id', 'text', 'reason'}
                or row['claim_id'] not in by_id or row['claim_id'] in seen
                or row['text'] != by_id[row['claim_id']]['text']
                or not isinstance(row['reason'], str) or not row['reason'].strip()):
            raise ValueError('unsupported_claim_requires_actual_claim_id_exact_original_sentence_and_reason; general unobserved facts belong in scene_uncertainties')
        seen.add(row['claim_id'])
        check = checks.get(row['claim_id']) or {}
        if check.get('status') not in {'contradicted', 'uncertain'} and check.get('scope_supported') is not False:
            raise ValueError('review_conflicting_supported_and_unsupported_claim:' + row['claim_id'])
    unresolved = {key for key, row in checks.items()
                  if row.get('status') in {'contradicted', 'uncertain'} or row.get('scope_supported') is False}
    if seen != unresolved:
        raise ValueError('review_error_list_does_not_cover_exactly_unresolved_claim_ids:' + ','.join(sorted(seen ^ unresolved)))
