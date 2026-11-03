from orchestrator.iteration.distribution import largest_remainder_counts, select_records
from orchestrator.iteration.policy import default_policy


def test_largest_remainder_preserves_total():
    counts = largest_remainder_counts(10, {"a": 0.2, "b": 0.3, "c": 0.5})
    assert sum(counts.values()) == 10
    assert counts == {"a": 2, "b": 3, "c": 5}


def test_selection_records_shortage_and_unique_keys():
    records = [
        {"record_key": "a", "task_type": "fine_grained_perception", "video_type": "tutorial"},
        {"record_key": "b", "task_type": "localized_editing", "video_type": "sports"},
        {"record_key": "b", "task_type": "localized_editing", "video_type": "sports"},
    ]
    result = select_records(records, default_policy(1), target_train_samples=100, actual_sample_cap=32, seed=7)
    assert len(result["selected"]) == 2
    assert result["report"]["duplicate_record_keys"] == ["b"]
    assert result["report"]["target_train_samples"] == 100
    assert result["report"]["withheld_count"] == 0
    assert any(value > 0 for values in result["report"]["shortage_at_target"].values() for value in values.values())


def test_selection_audit_records_policy_weight_and_gate_c_counts():
    records = [
        {"record_key": "tutorial", "task_type": "fine_grained_perception",
         "policy_domain": "Technology&Tutorial", "video_type": "tutorial"},
        {"record_key": "gaming", "task_type": "fine_grained_perception",
         "policy_domain": "Gaming&ACG", "video_type": "gaming"},
    ]
    policy = default_policy(1)
    policy["video_type_weights"]["gaming"] = 4.0
    result = select_records(records, policy, target_train_samples=10, actual_sample_cap=1, seed=9)
    audit = result["report"]["selection_audit"]
    assert len(audit) == 1
    assert audit[0]["selected"] is True
    assert audit[0]["random_seed"] == 9
    assert result["report"]["gate_c"]["strategy"] == "proportional"
    assert "available_counts" in result["report"]


def test_unknown_domain_keeps_audit_status_and_is_reported():
    records = [{
        "record_key": "unknown-domain",
        "task_type": "fine_grained_perception",
        "video_type": "unknown",
        "domain_audit_status": "call_failed",
        "domain_unknown_reason": "metadata_audit:call_failed",
    }]
    result = select_records(records, default_policy(1), target_train_samples=10, actual_sample_cap=1, seed=4)
    report = result["report"]
    assert report["unknown"]["domain"] == 1
    assert report["domain_audit_status_counts"] == {"call_failed": 1}
    assert report["selection_audit"][0]["domain_audit_status"] == "call_failed"
