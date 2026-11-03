"""Paper target distributions and bounded policy updates."""
from __future__ import annotations

import copy
import math
from typing import Any

from .schemas import finite_nonnegative, validate_policy
from .taxonomy import CAPABILITIES, DOMAINS, TASK_TYPES, TASK_CAPABILITY

PAPER_CAPABILITY = (
    {"understanding": 0.50, "editing": 0.25, "unified": 0.25},
    {"understanding": 0.35, "editing": 0.33, "unified": 0.32},
)
PAPER_DOMAIN = (
    {"Technology&Tutorial": 0.221, "Sports&Action": 0.178, "DailyLife&HumanActivities": 0.282,
     "Nature&Scenery": 0.181, "Gaming&ACG": 0.138},
    {"Technology&Tutorial": 0.181, "Sports&Action": 0.219, "DailyLife&HumanActivities": 0.248,
     "Nature&Scenery": 0.161, "Gaming&ACG": 0.191},
)
PAPER_TASK = (
    {"fine_grained_perception": 0.283, "single_image_reasoning": 0.096, "multi_image_logic": 0.121,
     "localized_editing": 0.250, "grounded_edit_to_select": 0.046,
     "understand_recall_generate_select": 0.053, "stepwise_sequence": 0.048,
     "middle_step_gap": 0.051, "instructional_future_sequence": 0.052},
    {"fine_grained_perception": 0.224, "single_image_reasoning": 0.068, "multi_image_logic": 0.058,
     "localized_editing": 0.330, "grounded_edit_to_select": 0.090,
     "understand_recall_generate_select": 0.064, "stepwise_sequence": 0.061,
     "middle_step_gap": 0.048, "instructional_future_sequence": 0.057},
)


def _complete(source: dict[str, float], labels: tuple[str, ...]) -> dict[str, float]:
    return {label: float(source[label]) for label in labels}


def default_policy(iteration: int) -> dict[str, Any]:
    index = max(0, min(int(iteration) - 1, 1))
    policy = {
        "schema_version": 1,
        "iteration": int(iteration),
        "capability": _complete(PAPER_CAPABILITY[index], CAPABILITIES),
        "domain": _complete(PAPER_DOMAIN[index], DOMAINS),
        "task_type": _complete(PAPER_TASK[index], TASK_TYPES),
        "video_type_weights": {
            "tutorial": 1.0,
            "lecture": 1.0,
            "sports": 1.0,
            "gaming": 1.0,
            "vlog": 1.0,
            "drama": 1.0,
            "surveillance": 1.0,
        },
        "sampling_density": {"default": 1.0},
        "keyframe_budget_multiplier": {"default": 1.0},
        "keyframe_score_weights": {
            "image_quality": 1.0, "semantic_change": 1.0, "temporal_coverage": 1.0,
        },
        "routing_quota_bias": {},
        "gate_c_downsample": {"enabled": True, "strategy": "proportional"},
    }
    return validate_policy(policy)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _bounded_distribution(
    base: dict[str, float], proposed: dict[str, float], labels: tuple[str, ...], max_delta: float
) -> dict[str, float]:
    """Project a proposed distribution into per-label delta boxes."""
    if set(proposed) - set(labels) or any(not finite_nonnegative(v) for v in proposed.values()):
        raise ValueError("invalid_proposed_distribution")
    raw = {label: float(proposed.get(label, base[label])) for label in labels}
    total = sum(raw.values())
    if total <= 0:
        raw = {label: base[label] for label in labels}
        total = sum(raw.values())
    values = {label: raw[label] / total for label in labels}
    lower = {label: max(0.0, base[label] - max_delta) for label in labels}
    upper = {label: min(1.0, base[label] + max_delta) for label in labels}
    values = {label: max(lower[label], min(upper[label], values[label])) for label in labels}
    residual = 1.0 - sum(values.values())
    for _ in range(len(labels) * 2):
        if abs(residual) <= 1e-12:
            break
        room = {
            label: (upper[label] - values[label]) if residual > 0 else (values[label] - lower[label])
            for label in labels
        }
        available = sum(max(0.0, amount) for amount in room.values())
        if available <= 1e-12:
            break
        step = min(1.0, abs(residual) / available)
        for label in labels:
            amount = max(0.0, room[label]) * step
            values[label] += amount if residual > 0 else -amount
        residual = 1.0 - sum(values.values())
    return {label: values[label] / sum(values.values()) for label in labels}


def apply_reflection(base: dict[str, Any], reflection: dict[str, Any], *, max_delta: float = 0.15) -> dict[str, Any]:
    validate_policy(base)
    if not finite_nonnegative(max_delta) or max_delta > 1:
        raise ValueError("invalid_max_delta")
    candidate = copy.deepcopy(base)
    next_target = reflection.get("next_target_distribution") or {}
    for dimension, labels in (("capability", CAPABILITIES), ("domain", DOMAINS), ("task_type", TASK_TYPES)):
        update = next_target.get(dimension)
        if not isinstance(update, dict):
            continue
        candidate[dimension] = _bounded_distribution(
            candidate[dimension],
            update,
            labels,
            max_delta,
        )
    candidate = _merge(candidate, reflection.get("policy_updates") or {})
    # Reflection backends may place a distribution under policy_updates rather
    # than next_target_distribution.  Apply the same per-label bound after the
    # merge so malformed or over-aggressive updates cannot bypass the guard.
    for dimension, labels in (("capability", CAPABILITIES), ("domain", DOMAINS), ("task_type", TASK_TYPES)):
        if isinstance(candidate.get(dimension), dict):
            candidate[dimension] = _bounded_distribution(
                base[dimension], candidate[dimension], labels, max_delta
            )
    # Task probabilities are unconditional: their sums must equal the parent
    # capability. Project each group without breaking per-task delta bounds.
    for capability in CAPABILITIES:
        tasks = [task for task in TASK_TYPES if TASK_CAPABILITY[task] == capability]
        target = candidate["capability"][capability]
        lower = {task: max(0.0, base["task_type"][task] - max_delta) for task in tasks}
        upper = {task: min(1.0, base["task_type"][task] + max_delta) for task in tasks}
        if target < sum(lower.values()) - 1e-9 or target > sum(upper.values()) + 1e-9:
            raise ValueError(f"infeasible_task_capability_bounds:{capability}")
        low, high = -2.0, 2.0
        for _ in range(80):
            offset = (low + high) / 2
            total = sum(max(lower[t], min(upper[t], candidate["task_type"][t] + offset)) for t in tasks)
            if total < target:
                low = offset
            else:
                high = offset
        for task in tasks:
            candidate["task_type"][task] = max(lower[task], min(upper[task], candidate["task_type"][task] + (low + high) / 2))
    candidate["iteration"] = int(base.get("iteration", 0)) + 1
    return validate_policy(candidate)


def policy_diff(before: dict[str, Any], after: dict[str, Any], *, abs_tol: float = 1e-9) -> dict[str, Any]:
    def equal(left: Any, right: Any) -> bool:
        if isinstance(left, dict) and isinstance(right, dict):
            return set(left) == set(right) and all(equal(left[k], right[k]) for k in left)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
            return math.isclose(left, right, rel_tol=0.0, abs_tol=abs_tol)
        return left == right

    diff = {}
    for key in sorted(set(before) | set(after)):
        if key in {"iteration", "schema_version"}:
            continue
        if not equal(before.get(key), after.get(key)):
            diff[key] = {"before": before.get(key), "after": after.get(key)}
    return diff


def apply_config_overlay(config: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    """Attach policy to known stage configs without mutating the base config."""
    output = copy.deepcopy(config)
    runtime = output.setdefault("runtime", {})
    runtime["iteration_enabled"] = True
    runtime["iteration_policy"] = copy.deepcopy(policy)
    runtime["iteration_source_scheduler"] = {
        "video_type_weights": copy.deepcopy(policy.get("video_type_weights", {})),
        "domain_distribution": copy.deepcopy(policy.get("domain", {})),
    }
    tasks = output.setdefault("tasks", {})
    tasks.setdefault("T1", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T1_shot_profile", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T2_frames", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T2_gate_c", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T2_quota_llm", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T2_representative_selection", {})["iteration_policy"] = copy.deepcopy(policy)
    tasks.setdefault("T4", {})["iteration_policy"] = copy.deepcopy(policy)
    return output
