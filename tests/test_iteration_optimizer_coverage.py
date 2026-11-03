import pytest

from orchestrator.iteration.bagel_audit import summarize_training_audits
from orchestrator.iteration.taxonomy import TASK_TYPES


def episodes():
    return [{"id": task, "task_type": task, "turns": [{}, {}]} for task in TASK_TYPES]


def audit(turns):
    return {"real_optimizer_steps": 1, "steps": [{
        "gradient_l2": 1.0, "changed_parameters": ["lora_B"], "frozen_changed": [],
        "batches": [{"data_indexes": [{"data_indexes": [i, task, turn, "edit"]}
                                      for i, task in enumerate(TASK_TYPES) for turn in turns]}],
    }]}


def test_task_ids_alone_do_not_prove_complete_generation_and_selection_consumption():
    result = summarize_training_audits([audit([1])], episodes(), 1)
    assert result["missing_task_types"] == []
    assert result["complete_task_types"] == []
    assert set(result["missing_complete_task_types"]) == set(TASK_TYPES)


def test_complete_turns_across_successful_optimizer_updates_prove_full_task_coverage():
    result = summarize_training_audits([audit([1]), audit([2])], episodes(), 1)
    assert result["missing_complete_task_types"] == []
    assert set(result["complete_episode_ids"]) == set(TASK_TYPES)
    assert result["consumed_episode_turns"][TASK_TYPES[-1]] == [1, 2]


def test_unknown_turn_id_cannot_count_as_consumption():
    with pytest.raises(ValueError, match="unknown_episode_turn"):
        summarize_training_audits([audit([3])], episodes(), 1)
