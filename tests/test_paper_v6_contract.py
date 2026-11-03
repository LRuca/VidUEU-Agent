import json
from types import SimpleNamespace

import pytest

from executor.tasks.prompt_loader import prompt_profile_scope
from orchestrator.paper_validation import validate_single_image_item
from orchestrator.umm_task_producer import produce_task_samples, TASK_COMM_C1, StrictTaskLLM
from orchestrator.umm_task_producer import TASK_ISG, ISGCandidate


def test_c1_overall_author_cannot_see_middle_or_transitions_and_summary_is_optional():
    frames = [{'pos': p, 'img': f'f{p}.jpg', 'caption': text, 'cause': 'object_change'}
              for p, text in enumerate(['cup at left', 'SECRET_MIDDLE_DETOUR', 'cup at right'])]
    pairs = [{'pair_from_pos': p, 'pair_to_pos': p+1, 'transition_text': 'Edits(1): 1) move cup rightward'} for p in range(2)]
    units = [{'evidence_id': 'S_SEGEMENT_SUM', 'text': 'SECRET_SUMMARY_DETOUR'}]
    units += [{'evidence_id': f'F{p:02d}_{kind}', 'text': fr['caption']}
              for p, fr in enumerate(frames) for kind in ('CAP', 'STATE')]
    units += [{'evidence_id': f'P{p:02d}_TRANS', 'text': 'SECRET_ADJACENT_TRANSITION',
               'pair_from_pos': p, 'pair_to_pos': p+1} for p in range(2)]
    data = {'t3_out': {'task_profile': 'paper'}, 'frames_aligned': frames, 'pair_records': pairs, 'evidence_units': units}
    steps = {'question': 'Track the cup.', 'step_texts': ['cup middle', 'cup right'],
             'select': [{'question': 'Which state?', 'answer': f'target:F{p:02d}', 'rationale': 'Observed target position.'} for p in (1, 2)],
             'evidence_refs': ['F00_CAP', 'F01_CAP', 'F02_CAP', 'P00_TRANS', 'P01_TRANS']}
    overall = {'overall': 'The cup shifts from image-left to image-right.', 'evidence_refs': ['F00_CAP', 'F02_STATE']}
    prompts = []
    def chat(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(overall if len(prompts) == 2 else steps)
    with prompt_profile_scope('paper_v6'):
        result = produce_task_samples(data, llm=SimpleNamespace(chat=chat), selected_tasks=[TASK_COMM_C1], strict_generation=True)
    assert len(result['samples'][TASK_COMM_C1]) == 1, result['drops']
    assert len(prompts) == 2
    assert 'SECRET_MIDDLE' in prompts[0]
    assert not any(marker in prompts[1] for marker in ('SECRET_MIDDLE', 'SECRET_SUMMARY', 'SECRET_ADJACENT', 'F01_'))
    row = result['samples'][TASK_COMM_C1][0]
    assert row['turns']['overall'] == overall['overall']
    assert json.loads(row['meta']['raw_overall_response']) == overall
    assert json.loads(row['meta']['raw_generation_response']) == steps


def test_overall_reference_to_hidden_middle_is_rejected():
    llm = StrictTaskLLM(SimpleNamespace(chat=lambda *a, **k: json.dumps({
        'overall': 'Detour.', 'evidence_refs': ['F00_CAP', 'F01_CAP', 'F02_CAP']})), TASK_COMM_C1, None, None)
    with pytest.raises(ValueError, match='outside_endpoint_evidence'):
        llm.author_overall('bound endpoint prompt', [0, 2])


@pytest.mark.parametrize('options', [
    ['Above the pan', 'Below the pan', 'To the left of the pan', 'To the right of the pan'],
    ['Left of the chef', 'Right of the chef', 'Behind the chef', 'In front of the chef'],
])
def test_mixed_single_axis_spatial_options_are_withheld(options):
    item = dict(zip('ABCD', options))
    item.update(image_pos=0, question='Where is the object?', evidence_refs=['F00_CAP'], _answer='A')
    with prompt_profile_scope('paper_v6'), pytest.raises(ValueError, match='nonexclusive_spatial_axes'):
        validate_single_image_item(item, 'mmt_bench', {'F00_CAP'}, 0)


@pytest.mark.parametrize('change_task', [False, True])
def test_missing_context_citation_repair_preserves_original_content(change_task):
    units = {key: {'evidence_id': key, 'text': key} for key in ['S_SEGEMENT_SUM', 'F00_CAP', 'F01_CAP',
             'F02_CAP', 'F00_STATE', 'F01_STATE', 'F02_STATE', 'P00_TRANS', 'P01_TRANS']}
    ctx = SimpleNamespace(segment_evidence_id='S_SEGEMENT_SUM', frame_by_pos={p: {} for p in range(3)},
                          evidence_by_id=units, shot_summary_ids={}, chapter_summary_ids={})
    candidate = ISGCandidate(context_positions=[0, 1], future_positions=[2], pair_refs=['P00_TRANS', 'P01_TRANS'], review_refs=[], score=1)
    original = {'overall': 'Continue the cup movement.', 'edits': ['move cup'], 'because': ['continuous cup'],
                'so_that': ['cup right'], 'check': ['cup position'],
                'select': [{'question': 'Which target?', 'answer': 'target:F02', 'rationale': 'cup right'}],
                'evidence_refs': ['F01_CAP', 'F02_CAP', 'P00_TRANS', 'P01_TRANS']}
    corrected = json.loads(json.dumps(original)); corrected['evidence_refs'].append('F00_CAP')
    if change_task: corrected['overall'] = 'Different task.'
    prompts = []
    def chat(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps(original if len(prompts) == 1 else corrected)
    llm = StrictTaskLLM(SimpleNamespace(chat=chat), TASK_ISG, ctx, candidate)
    with prompt_profile_scope('paper_v6'):
        if change_task:
            with pytest.raises(ValueError, match='citation_repair_changed_task'): llm.chat('bound author prompt')
        else:
            assert json.loads(llm.chat('bound author prompt')) == corrected
    assert len(prompts) == 2 and 'F00_CAP' in prompts[1]
    assert len(llm.generation_attempts) == 2
