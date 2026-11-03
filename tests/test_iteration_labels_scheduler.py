import json

from orchestrator.iteration.controller import normalize_candidate
from orchestrator.iteration.distribution import select_records
from orchestrator.iteration.policy import default_policy
from orchestrator.iteration.scheduler import build_source_schedule
from orchestrator.iteration.scheduler import source_key
from orchestrator.iteration.labels import propagate_iteration_labels


def test_domain_metadata_propagates_from_global_perception_to_candidate():
    row = {
        "record_key": "r1",
        "task_type": "fine_grained_perception",
        "meta": {
            "global_perception": {
                "video_type": "tutorial",
                "policy_domain": "Technology&Tutorial",
                "domain_source": "qwen3.5_metadata_audit",
                "domain_confidence": 0.86,
            }
        },
    }
    result = normalize_candidate(row, 0)
    assert result["video_type"] == "tutorial"
    assert result["policy_domain"] == "Technology&Tutorial"
    assert result["domain_source"] == "qwen3.5_metadata_audit"
    assert result["meta"]["policy_domain"] == "Technology&Tutorial"
    assert result["global_perception"]["policy_domain"] == "Technology&Tutorial"


def test_generic_t4_video_type_is_replaced_by_canonical_t1_label():
    result = propagate_iteration_labels(
        {
            "record_key": "r-generic",
            "video_type": "generic",
            "task_type": "fine_grained_perception",
        },
        {"video_type": "tutorial", "policy_domain": "Technology&Tutorial",
         "domain_source": "qwen3.5_metadata_audit", "domain_confidence": 0.86},
    )
    assert result["video_type"] == "tutorial"
    assert result["policy_domain"] == "Technology&Tutorial"


def test_domain_audit_provenance_is_mirrored_and_generic_is_unknown():
    result = propagate_iteration_labels(
        {
            "record_key": "r-unknown",
            "video_type": "generic",
            "global_perception": {
                "domain_audit_status": "unknown",
                "domain_unknown_reason": "metadata_audit:unknown",
                "domain_audit_evidence": ["insufficient visual evidence"],
            },
            "task_type": "fine_grained_perception",
        },
        {
            "video_type": "unknown",
            "policy_domain": None,
            "domain_source": "qwen3.5_metadata_audit",
            "domain_confidence": 0.0,
            "domain_audit_status": "unknown",
            "domain_unknown_reason": "metadata_audit:unknown",
        },
    )
    assert result["video_type"] == "unknown"
    assert result["policy_domain"] is None
    assert result["domain_source"] == "qwen3.5_metadata_audit"
    assert result["domain_confidence"] == 0.0
    assert result["domain_audit_status"] == "unknown"
    assert result["domain_unknown_reason"] == "metadata_audit:unknown"
    assert result["global_perception"]["domain_audit_status"] == "unknown"
    assert result["meta"]["global_perception"]["domain_unknown_reason"] == "metadata_audit:unknown"


def test_legacy_unknown_candidate_gets_explicit_unresolved_provenance():
    result = normalize_candidate(
        {"record_key": "legacy", "video_type": "generic", "task_type": "fine_grained_perception"},
        0,
    )
    assert result["video_type"] == "unknown"
    assert result["policy_domain"] is None
    assert result["domain_source"] == "global_perception_unresolved"
    assert result["domain_confidence"] == 0.0
    assert result["domain_unknown_reason"]


def test_source_scheduler_uses_domain_and_video_type_weights():
    policy = default_policy(2)
    policy["video_type_weights"]["gaming"] = 4.0
    records = [
        normalize_candidate({
            "record_key": "tutorial",
            "video_path": "tutorial.mp4",
            "video_type": "tutorial",
            "policy_domain": "Technology&Tutorial",
            "task_type": "fine_grained_perception",
        }, 0),
        normalize_candidate({
            "record_key": "gaming",
            "video_path": "gaming.mp4",
            "video_type": "gaming",
            "policy_domain": "Gaming&ACG",
            "task_type": "fine_grained_perception",
        }, 1),
    ]
    schedule = build_source_schedule(records, policy, seed=17)
    by_type = {row["video_type"]: row for row in schedule}
    assert by_type["gaming"]["base_video_type_weight"] == 4.0
    assert by_type["gaming"]["sampling_weight"] > by_type["tutorial"]["sampling_weight"]
    assert all(row["random_seed"] == 17 for row in schedule)


def test_selection_audit_uses_the_same_source_key_as_schedule():
    policy = default_policy(1)
    records = [
        normalize_candidate({
            "record_key": "same-source-a",
            "segment_path": "temp\\segments\\seg_000.mp4",
            "task_type": "fine_grained_perception",
            "video_type": "tutorial",
            "policy_domain": "Technology&Tutorial",
        }, 0),
        normalize_candidate({
            "record_key": "same-source-b",
            "segment_path": "temp/segments/seg_000.mp4",
            "task_type": "single_image_reasoning",
            "video_type": "tutorial",
            "policy_domain": "Technology&Tutorial",
        }, 1),
    ]
    result = select_records(records, policy, target_train_samples=10, actual_sample_cap=1, seed=2)
    selected_key = result["report"]["selection_audit"][0]["source_key"]
    schedule_keys = {row["source_key"] for row in result["report"]["source_schedule"]}
    assert selected_key in schedule_keys
    assert selected_key == source_key(result["selected"][0])


def test_gate_c_reject_and_hold_never_enter_train_selection():
    policy = default_policy(1)
    records = [
        {"record_key": "accept", "task_type": "fine_grained_perception",
         "video_type": "tutorial", "meta": {"gate_c": {"decision": "accept"}}},
        {"record_key": "reject", "task_type": "fine_grained_perception",
         "video_type": "tutorial", "meta": {"gate_c": {"decision": "reject"}}},
        {"record_key": "hold", "task_type": "fine_grained_perception",
         "video_type": "tutorial", "meta": {"gate_c": {"decision": "hold"}}},
    ]
    result = select_records(records, policy, target_train_samples=10, actual_sample_cap=10, seed=5)
    assert [row["record_key"] for row in result["selected"]] == ["accept"]
    assert {row["record_key"] for row in result["withheld"]} == {"reject", "hold"}
    assert result["report"]["gate_c"]["ineligible_count"] == 2
    assert result["report"]["gate_c_status_counts"]["reject"] == 1


def test_gate_c_selection_records_actual_cap_quota_and_shortage():
    policy = default_policy(1)
    records = []
    # At cap=2, the policy allocates one understanding and one editing slot.
    # Supplying understanding rows alone must report shortage, not exceed it.
    for index, (domain, task) in enumerate((
        ("Technology&Tutorial", "fine_grained_perception"),
        ("Sports&Action", "localized_editing"),
        ("DailyLife&HumanActivities", "fine_grained_perception"),
    )):
        records.append({
            "record_key": f"{domain}-{index}",
            "task_type": task,
            "policy_domain": domain,
            "video_type": "tutorial",
            "meta": {"gate_c": {"decision": "accept"}},
        })
    result = select_records(records, policy, target_train_samples=10000, actual_sample_cap=2, seed=3)
    report = result["report"]
    assert sum(report["selection_target_counts"]["domain"].values()) == 2
    assert sum(report["selected_counts"]["domain"][key]["count"] for key in report["selected_counts"]["domain"]) == 2
    assert "shortage_at_actual_cap" in report

