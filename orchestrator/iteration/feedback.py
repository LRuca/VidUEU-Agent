"""Feedback whitelist and deterministic dry-run feedback."""
from __future__ import annotations

from typing import Any
import copy

from .schemas import finite_nonnegative

FEEDBACK_BENCHMARKS = ("MMDU", "GEdit-Bench-En", "WEAVEbench")
HELD_OUT_BENCHMARKS = ("MMBench", "MMStar", "VSI-Bench")
METRIC_UNITS = {"MMDU": "percent", "GEdit-Bench-En": "score_0_10", "WEAVEbench": "fraction"}
METRIC_MAXIMA = {"MMDU": 100.0, "GEdit-Bench-En": 10.0, "WEAVEbench": 1.0}


def canonical_metrics(values: dict[str, Any], units: dict[str, str]) -> dict[str, float]:
    if set(values) != set(FEEDBACK_BENCHMARKS) or set(units) != set(FEEDBACK_BENCHMARKS):
        raise ValueError("metric_values_and_units_required")
    result = {}
    for name, value in values.items():
        if not finite_nonnegative(value):
            raise ValueError(f"invalid_metric_value:{name}")
        if units[name] == METRIC_UNITS[name]:
            converted = float(value)
        elif name == "MMDU" and units[name] == "fraction":
            converted = float(value) * 100
        elif name == "WEAVEbench" and units[name] == "percent":
            converted = float(value) / 100
        else:
            raise ValueError(f"unsupported_metric_unit:{name}:{units[name]}")
        if converted > METRIC_MAXIMA[name]:
            raise ValueError(f"metric_out_of_range:{name}")
        result[name] = converted
    return result


def fixture_feedback(iteration: int, distribution: dict[str, Any]) -> dict[str, Any]:
    values = (
        {"MMDU": 29.7, "GEdit-Bench-En": 6.541, "WEAVEbench": 0.448},
        {"MMDU": 31.4, "GEdit-Bench-En": 6.796, "WEAVEbench": 0.475},
    )
    metrics = values[min(max(iteration - 1, 0), len(values) - 1)]
    return {
        "iteration": iteration,
        "backend": "fixture",
        "feedback_metrics": dict(metrics),
        "metric_units": dict(METRIC_UNITS),
        "held_out_metrics": {name: None for name in HELD_OUT_BENCHMARKS},
        "distribution": distribution,
        "loss_curve_summary": {"available": False},
    }


def validate_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        raise ValueError("feedback_must_be_object")
    metrics = feedback.get("feedback_metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(FEEDBACK_BENCHMARKS):
        raise ValueError("feedback_metrics_must_match_control_whitelist")
    units = feedback.get("metric_units")
    if feedback.get("backend") == "benchmark" and units != METRIC_UNITS:
        raise ValueError("real_feedback_requires_canonical_metric_units")
    canonical_metrics(metrics, units if units is not None else METRIC_UNITS)
    held_out = feedback.get("held_out_metrics")
    if not isinstance(held_out, dict) or set(held_out) != set(HELD_OUT_BENCHMARKS):
        raise ValueError("held_out_metrics_must_match_whitelist")
    if set(metrics) & set(HELD_OUT_BENCHMARKS):
        raise ValueError("held_out_metric_entered_control_feedback")
    return feedback


def reflection_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    """Only aggregate control metrics/losses may cross into the data agent."""
    validate_feedback(feedback)
    result = {key: copy.deepcopy(feedback[key]) for key in ("iteration", "feedback_metrics", "metric_units")
              if key in feedback}
    losses = feedback.get("loss_curve_summary", {})
    if isinstance(losses, dict):
        allowed = {"available", "steps", "first_loss", "last_loss", "min_loss", "max_loss",
                   "ce_loss", "mse_loss", "total_loss", "learning_rate"}
        result["loss_curve_summary"] = {
            key: value for key, value in losses.items() if key in allowed
            and (isinstance(value, bool) if key == "available" else finite_nonnegative(value))
        }
        points = losses.get("points", [])
        if isinstance(points, list) and points:
            selected = sorted(set(round(i * (len(points) - 1) / min(15, len(points) - 1))
                                  for i in range(min(16, len(points))))) if len(points) > 1 else [0]
            curve = []
            for index in selected:
                point = points[index]
                if not isinstance(point, dict):
                    continue
                clean = {key: point[key] for key in ("step", "ce", "mse", "total")
                         if key in point and finite_nonnegative(point[key])}
                if set(clean) == {"step", "ce", "mse", "total"}:
                    curve.append(clean)
            result["loss_curve_summary"]["points"] = curve
    subcategories = feedback.get("subcategory_scores", {})
    allowed_parts = {
        "MMDU": {"Creativity", "Richness", "Visual Perception", "Logical Coherence", "Answer Accuracy",
                 "Image Relationship Understanding", "Overall Score"},
        "GEdit-Bench-En": {"G_SC", "G_PQ", "G_O"},
        "WEAVEbench": {"key_point", "visual_consistency", "image_quality", "accuracy", "Science", "Creation",
                       "Logic", "Game", "optics", "geography", "astronomy", "chemistry", "physics", "biology",
                       "edit", "recall", "fusion", "story", "spatial", "mathematics", "chess_game",
                       "visual_jigsaw", "minecraft", "maze"},
    }
    if isinstance(subcategories, dict):
        cleaned = {}
        for benchmark, names in allowed_parts.items():
            parts = subcategories.get(benchmark, {})
            maximum = {"MMDU": 100, "GEdit-Bench-En": 10, "WEAVEbench": 1}[benchmark]
            if isinstance(parts, dict):
                cleaned[benchmark] = {key: value for key, value in parts.items()
                                      if key in names and finite_nonnegative(value) and value <= maximum}
        if any(cleaned.values()):
            result["subcategory_scores"] = cleaned
    return result


def reflection_distribution(report: dict[str, Any]) -> dict[str, Any]:
    """Exclude source schedules, record IDs, and assets from aggregate feedback."""
    allowed = {"distributions", "target_distributions", "target_train_samples", "actual_samples",
               "selection_requested_total", "selection_unfilled_count", "selection_status",
               "selection_target_counts", "shortage_at_actual_cap", "unknown_domain_count"}
    return {key: copy.deepcopy(value) for key, value in report.items() if key in allowed}
