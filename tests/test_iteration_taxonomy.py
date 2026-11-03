from orchestrator.iteration.taxonomy import (
    CAPABILITIES,
    TASK_TYPES,
    capability_for_task,
    labels_for_record,
    label_details_for_record,
)


def test_nine_task_taxonomy_is_complete_and_capability_mapped():
    assert len(TASK_TYPES) == 9
    assert {capability_for_task(task) for task in TASK_TYPES} == set(CAPABILITIES)
    assert all(capability_for_task(task) in CAPABILITIES for task in TASK_TYPES)


def test_unknown_domain_is_explicitly_unknown():
    labels = labels_for_record({"task_type": "fine_grained_perception", "video_type": "unlisted"})
    assert labels["capability"] == "understanding"
    assert labels["domain"] is None


def test_global_perception_domain_has_priority_over_video_type_fallback():
    details = label_details_for_record(
        {
            "task_type": "fine_grained_perception",
            "video_type": "sports",
            "global_perception": {"policy_domain": "Nature&Scenery", "domain_confidence": 0.82},
        }
    )
    assert details["domain"] == "Nature&Scenery"
    assert details["domain_source"] == "global_perception"
    assert details["domain_confidence"] == 0.82
