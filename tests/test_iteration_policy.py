from orchestrator.iteration.policy import apply_config_overlay, apply_reflection, default_policy, policy_diff


def test_default_distributions_are_normalized():
    policy = default_policy(1)
    for dimension in ("capability", "domain", "task_type"):
        assert abs(sum(policy[dimension].values()) - 1.0) < 1e-9


def test_reflection_delta_is_bounded_and_recordable():
    before = default_policy(1)
    after = apply_reflection(
        before,
        {"next_target_distribution": {"capability": {"understanding": 1.0, "editing": 0.0, "unified": 0.0}}},
        max_delta=0.05,
    )
    assert max(abs(after["capability"][key] - before["capability"][key]) for key in before["capability"]) <= 0.051
    assert policy_diff(before, after)


def test_overlay_reaches_global_keyframe_and_routing_layers_without_mutating_base():
    base = {"runtime": {}, "tasks": {"T2_frames": {}, "T4": {}}}
    policy = default_policy(1)
    output = apply_config_overlay(base, policy)
    assert "iteration_policy" not in base["runtime"]
    assert output["runtime"]["iteration_enabled"] is True
    assert output["tasks"]["T1"]["iteration_policy"] == policy
    assert output["tasks"]["T2_frames"]["iteration_policy"] == policy
    assert output["tasks"]["T4"]["iteration_policy"] == policy
