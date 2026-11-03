import copy

import pytest

from orchestrator.iteration.distribution import hierarchical_counts, select_records
from orchestrator.iteration.feedback import METRIC_UNITS, canonical_metrics, fixture_feedback, validate_feedback
from orchestrator.iteration.policy import apply_reflection, default_policy, policy_diff
from orchestrator.iteration.schemas import validate_policy
from orchestrator.iteration.taxonomy import TASK_CAPABILITY, TASK_TYPES


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_invalid_policy_numbers_rejected_before_projection(bad):
    before = default_policy(1)
    broken = copy.deepcopy(before)
    broken["domain"][next(iter(broken["domain"]))] = bad
    with pytest.raises(ValueError):
        validate_policy(broken)
    with pytest.raises(ValueError):
        apply_reflection(before, {"next_target_distribution": {"capability": {"understanding": bad}}})


def test_paper_taxonomy_and_parent_probabilities_are_consistent():
    assert sum(v == "understanding" for v in TASK_CAPABILITY.values()) == 3
    assert sum(v == "editing" for v in TASK_CAPABILITY.values()) == 1
    assert sum(v == "unified" for v in TASK_CAPABILITY.values()) == 5
    for iteration in (1, 2):
        validate_policy(default_policy(iteration))
    broken = default_policy(1)
    broken["capability"].update(understanding=0.4, editing=0.35)
    with pytest.raises(ValueError, match="mismatch"):
        validate_policy(broken)


def test_capability_only_feedback_rebalances_children_with_bounds():
    before = default_policy(1)
    after = apply_reflection(before, {"next_target_distribution": {
        "capability": {"understanding": 0.35, "editing": 0.33, "unified": 0.32}}})
    validate_policy(after)
    assert max(abs(after["task_type"][t] - before["task_type"][t]) for t in TASK_TYPES) <= 0.150000001
    assert after["task_type"]["localized_editing"] == pytest.approx(0.33)


def test_policy_diff_does_not_invent_feedback_changes():
    before = default_policy(1)
    after = apply_reflection(before, {})
    assert after["iteration"] == 2
    assert policy_diff(before, after) == {}
    after["sampling_density"]["default"] += 0.1
    assert set(policy_diff(before, after)) == {"sampling_density"}


def test_integer_task_counts_preserve_parents_and_cover_nine_at_32():
    quota = hierarchical_counts(32, default_policy(1))
    assert quota["capability"] == {"understanding": 16, "editing": 8, "unified": 8}
    assert all(quota["task_type"][t] >= 1 for t in TASK_TYPES)
    for cap, count in quota["capability"].items():
        assert sum(n for t, n in quota["task_type"].items() if TASK_CAPABILITY[t] == cap) == count


def test_zero_cap_respects_target_and_never_backfills_missing_capability():
    records = [{"record_key": str(i), "task_type": "fine_grained_perception"} for i in range(20)]
    result = select_records(records, default_policy(1), target_train_samples=4, actual_sample_cap=0, seed=1)
    assert result["report"]["selection_requested_total"] == 4
    assert 0 < len(result["selected"]) <= 2
    assert result["report"]["selection_status"] == "shortage"
    assert result["report"]["shortage_at_actual_cap"]["capability"]["editing"] == 1
    assert result["report"]["shortage_at_actual_cap"]["capability"]["unified"] == 1


def test_selection_will_not_fill_an_absent_task_with_another_task():
    records = [{"record_key": str(i), "task_type": "single_image_reasoning"} for i in range(50)]
    result = select_records(records, default_policy(1), target_train_samples=10000, actual_sample_cap=32, seed=2)
    quota = result["report"]["selection_target_counts"]["task_type"]["single_image_reasoning"]
    assert len(result["selected"]) == quota
    assert result["report"]["selection_unfilled_count"] == 32 - quota


def test_metric_conversion_is_explicit_and_invalid_results_are_not_zeroed():
    values = {"MMDU": 0.297, "GEdit-Bench-En": 6.5, "WEAVEbench": 44.8}
    units = {"MMDU": "fraction", "GEdit-Bench-En": "score_0_10", "WEAVEbench": "percent"}
    assert canonical_metrics(values, units) == pytest.approx({"MMDU": 29.7, "GEdit-Bench-En": 6.5, "WEAVEbench": 0.448})
    for bad in (None, float("nan"), float("inf"), -1):
        feedback = fixture_feedback(1, {})
        feedback["feedback_metrics"]["MMDU"] = bad
        with pytest.raises(ValueError):
            validate_feedback(feedback)
    feedback = fixture_feedback(1, {})
    feedback["backend"] = "benchmark"
    feedback.pop("metric_units")
    with pytest.raises(ValueError):
        validate_feedback(feedback)
    feedback["metric_units"] = dict(METRIC_UNITS)
    validate_feedback(feedback)
