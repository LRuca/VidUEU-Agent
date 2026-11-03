"""Validation helpers for iteration manifests and feedback."""
from __future__ import annotations

import math
from typing import Any

from .taxonomy import CAPABILITIES, DOMAINS, TASK_TYPES, TASK_CAPABILITY


def finite_nonnegative(number: Any) -> bool:
    return (not isinstance(number, bool) and isinstance(number, (int, float))
            and math.isfinite(number) and number >= 0)


def _distribution(value: Any, labels: tuple[str, ...], name: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(labels):
        raise ValueError(f"{name}_labels_must_match_canonical_taxonomy")
    result = {}
    for label in labels:
        number = value[label]
        if not finite_nonnegative(number):
            raise ValueError(f"{name}_contains_invalid_value:{label}")
        result[label] = float(number)
    total = sum(result.values())
    if total <= 0 or abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name}_must_sum_to_one")
    return result


def validate_policy(policy: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, dict):
        raise ValueError("policy_must_be_object")
    _distribution(policy.get("capability"), CAPABILITIES, "capability_distribution")
    _distribution(policy.get("domain"), DOMAINS, "domain_distribution")
    _distribution(policy.get("task_type"), TASK_TYPES, "task_type_distribution")
    for capability in CAPABILITIES:
        child_total = sum(policy["task_type"][task] for task in TASK_TYPES
                          if TASK_CAPABILITY[task] == capability)
        if abs(child_total - policy["capability"][capability]) > 1e-6:
            raise ValueError(f"task_capability_distribution_mismatch:{capability}")
    for key in ("video_type_weights", "sampling_density", "keyframe_budget_multiplier", "routing_quota_bias"):
        value = policy.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"policy_{key}_must_be_object")
        for name, number in value.items():
            if not finite_nonnegative(number):
                raise ValueError(f"policy_{key}_invalid_value:{name}")
    score_weights = policy.get("keyframe_score_weights", {})
    if not isinstance(score_weights, dict) or any(
        not finite_nonnegative(value)
        for value in score_weights.values()
    ):
        raise ValueError("policy_keyframe_score_weights_invalid")
    return policy
