"""Deterministic quota allocation and distribution auditing."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any

from .schemas import validate_policy
from .taxonomy import CAPABILITIES, DOMAINS, TASK_TYPES, TASK_CAPABILITY, labels_for_record
from .scheduler import build_source_schedule, source_key


def largest_remainder_counts(total: int, proportions: dict[str, float]) -> dict[str, int]:
    if total < 0 or not proportions:
        raise ValueError("total_and_proportions_required")
    values = {key: float(value) * total for key, value in proportions.items()}
    counts = {key: int(value // 1) for key, value in values.items()}
    remaining = total - sum(counts.values())
    order = sorted(values, key=lambda key: (-(values[key] - counts[key]), key))
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _record_key(record: dict[str, Any]) -> str:
    value = record.get("record_key")
    if value:
        return str(value)
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def hierarchical_counts(total: int, policy: dict[str, Any]) -> dict[str, dict[str, int]]:
    capability = largest_remainder_counts(total, policy["capability"])
    tasks = {}
    for parent, count in capability.items():
        children = {t: policy["task_type"][t] for t in TASK_TYPES if TASK_CAPABILITY[t] == parent}
        mass = sum(children.values())
        proportions = {t: v / mass if mass else 1 / len(children) for t, v in children.items()}
        tasks.update(largest_remainder_counts(count, proportions))
    return {"capability": capability, "task_type": tasks,
            "domain": largest_remainder_counts(total, policy["domain"])}


def _gate_c_status(record: dict[str, Any]) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    gate = record.get("gate_c") if isinstance(record.get("gate_c"), dict) else meta.get("gate_c")
    if not isinstance(gate, dict):
        return "missing"
    decision = str(gate.get("decision") or gate.get("status") or "missing").strip().lower()
    return decision or "missing"


def _domain_audit_status(record: dict[str, Any]) -> str:
    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
    global_perception = record.get("global_perception")
    if not isinstance(global_perception, dict):
        global_perception = meta.get("global_perception") if isinstance(meta.get("global_perception"), dict) else {}
    value = (
        record.get("domain_audit_status")
        or meta.get("domain_audit_status")
        or global_perception.get("domain_audit_status")
    )
    return str(value or "not_audited").strip().lower()


def audit_distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"capability": Counter(), "domain": Counter(), "task_type": Counter()}
    unknown = Counter()
    domain_audit_statuses = Counter()
    for record in records:
        labels = labels_for_record(record)
        domain_audit_statuses[_domain_audit_status(record)] += 1
        for dimension in counts:
            value = labels[dimension]
            if value is None:
                unknown[dimension] += 1
            else:
                counts[dimension][value] += 1
    total = len(records)
    distributions = {}
    for dimension, labels in (("capability", CAPABILITIES), ("domain", DOMAINS), ("task_type", TASK_TYPES)):
        distributions[dimension] = {
            label: {"count": counts[dimension][label],
                    "ratio": counts[dimension][label] / total if total else 0.0}
            for label in labels
        }
    return {
        "total": total,
        "distributions": distributions,
        "unknown": dict(unknown),
        "domain_audit_status_counts": dict(domain_audit_statuses),
        "record_keys": [_record_key(record) for record in records],
    }


def select_records(records: list[dict[str, Any]], policy: dict[str, Any], *,
                   target_train_samples: int, actual_sample_cap: int, seed: int) -> dict[str, Any]:
    validate_policy(policy)
    if target_train_samples <= 0 or actual_sample_cap < 0:
        raise ValueError("invalid_iteration_sample_limits")
    unique = {}
    duplicates = []
    for record in records:
        key = _record_key(record)
        if key in unique:
            duplicates.append(key)
        else:
            unique[key] = record
    unique_records = list(unique.values())
    ineligible = [
        record for record in unique_records
        if _gate_c_status(record) in {"reject", "hold", "pending", "execution_pending", "ineligible"}
    ]
    pool = [record for record in unique_records if record not in ineligible]
    rng = random.Random(seed)
    rng.shuffle(pool)
    source_schedule = build_source_schedule(pool, policy, seed=seed)
    source_weights = {row["source_key"]: float(row.get("sampling_weight", 1.0) or 1.0)
                      for row in source_schedule}
    desired = hierarchical_counts(target_train_samples, policy)
    selection_total = min(target_train_samples, actual_sample_cap or target_train_samples)
    selection_quota = hierarchical_counts(selection_total, policy)
    selected: list[dict[str, Any]] = []
    remaining = pool[:]
    selection_audit: list[dict[str, Any]] = []
    while remaining and len(selected) < selection_total:
        current = audit_distribution(selected)["distributions"]

        def within_selection_quota(record: dict[str, Any]) -> bool:
            labels = labels_for_record(record)
            task = labels.get("task_type")
            if task is None or labels.get("capability") != TASK_CAPABILITY[task]:
                return False
            for dimension in ("capability", "task_type"):
                quotas = selection_quota[dimension]
                label = labels.get(dimension)
                if label is not None and current[dimension].get(label, {}).get("count", 0) >= quotas.get(label, 0):
                    return False
            return True

        quota_candidates = [record for record in remaining if within_selection_quota(record)]
        if not quota_candidates:
            break
        score_pool = quota_candidates

        def score(record: dict[str, Any]) -> tuple[float, float, str]:
            labels = labels_for_record(record)
            gain = 0.0
            for dimension in desired:
                label = labels[dimension]
                if label is not None:
                    need = selection_quota[dimension].get(label, 0) - current[dimension].get(label, {}).get("count", 0)
                    gain += need / max(selection_quota[dimension].get(label, 1), 1)
            domain = labels.get("domain")
            video_type = str(record.get("video_type") or (record.get("meta") or {}).get("video_type") or "")
            type_weight = float((policy.get("video_type_weights") or {}).get(video_type, 1.0) or 1.0)
            target_domain = float((policy.get("domain") or {}).get(domain, 0.0) or 0.0)
            realized_domain = current.get("domain", {}).get(domain, {}).get("ratio", 0.0) if domain else 0.0
            shortage_weight = target_domain / max(float(realized_domain), 1.0 / max(target_train_samples, 1)) if domain else 0.0
            quality = record.get("quality_score", (record.get("meta") or {}).get("quality_score", 1.0))
            try:
                quality_weight = max(0.0, float(quality))
            except (TypeError, ValueError):
                quality_weight = 1.0
            weighted_priority = type_weight * max(shortage_weight, 1.0) * max(quality_weight, 0.0)
            weighted_priority *= max(source_weights.get(source_key(record), 1.0), 0.0)
            return gain * weighted_priority, rng.random(), _record_key(record)

        chosen = max(score_pool, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        labels = labels_for_record(chosen)
        domain = labels.get("domain")
        current_after = audit_distribution(selected)["distributions"]
        video_type = str(chosen.get("video_type") or (chosen.get("meta") or {}).get("video_type") or "")
        type_weight = float((policy.get("video_type_weights") or {}).get(video_type, 1.0) or 1.0)
        target_ratio = float((policy.get("domain") or {}).get(domain, 0.0) or 0.0) if domain else 0.0
        realized_before = current.get("domain", {}).get(domain, {}).get("ratio", 0.0) if domain else 0.0
        realized_after = current_after.get("domain", {}).get(domain, {}).get("ratio", 0.0) if domain else 0.0
        selection_audit.append({
            "record_key": _record_key(chosen),
            # Keep this identical to the source schedule key.  The content
            # hash remains in the candidate provenance and is not a second
            # source identity.
            "source_key": source_key(chosen),
            "source_sha256": str(chosen.get("source_sha256") or ""),
            "video_type": video_type or None,
            "policy_domain": domain,
            "domain_audit_status": _domain_audit_status(chosen),
            "target_ratio": target_ratio,
            "realized_ratio_before": realized_before,
            "realized_ratio_after": realized_after,
            "base_video_type_weight": type_weight,
            "sampling_weight": type_weight * max(target_ratio / max(realized_before, 1.0 / max(target_train_samples, 1)), 1.0) if domain else type_weight,
            "selected": True,
            "selection_reason": "domain_shortage" if domain and target_ratio > realized_before else "largest_deficit",
            "selection_quota_count": selection_quota["domain"].get(domain, 0) if domain else None,
            "iteration": int(policy.get("iteration", 1)),
            "random_seed": seed,
            "source_sampling_weight": source_weights.get(source_key(chosen), 1.0),
        })
    selected_keys = {_record_key(row) for row in selected}
    report = audit_distribution(selected)
    target_ratios = {dimension: policy[dimension] for dimension in ("capability", "domain", "task_type")}
    shortages = {}
    for dimension in target_ratios:
        realized = report["distributions"][dimension]
        shortages[dimension] = {
            label: max(0, desired[dimension][label] - realized[label]["count"])
            for label in target_ratios[dimension]
        }
    cap_shortages = {}
    for dimension, quotas in selection_quota.items():
        realized = report["distributions"][dimension]
        cap_shortages[dimension] = {
            label: max(0, quotas[label] - realized[label]["count"])
            for label in quotas
        }
    report.update({"target_train_samples": target_train_samples,
                   "actual_sample_cap": actual_sample_cap,
                   "target_counts": desired,
                   "selection_target_counts": selection_quota,
                   "selection_requested_total": selection_total,
                   "selection_unfilled_count": selection_total - len(selected),
                   "selection_status": "complete" if len(selected) == selection_total else "shortage",
                   "quota_priority": ["capability", "task_type", "domain"],
                   "shortage_at_target": shortages,
                   "shortage_at_actual_cap": cap_shortages,
                   "duplicate_record_keys": duplicates,
                   "pool_size": len(pool),
                   "total_unique_count": len(unique_records),
                   "withheld_count": len(unique_records) - len(selected),
                   "gate_c_ineligible_count": len(ineligible),
                   "gate_c_status_counts": dict(Counter(_gate_c_status(row) for row in unique_records)),
                   "selection_seed": seed,
                   "selection_audit": selection_audit,
                   "source_schedule": build_source_schedule(pool, policy, selected=selected, seed=seed),
                   "available_counts": audit_distribution(pool)["distributions"],
                   "selected_counts": report["distributions"],
                   "withheld_counts": audit_distribution(
                       [row for row in unique_records if _record_key(row) not in selected_keys]
                   )["distributions"]})
    gate_cfg = policy.get("gate_c_downsample") if isinstance(policy.get("gate_c_downsample"), dict) else {}
    report["gate_c"] = {
        "enabled": bool(gate_cfg.get("enabled", True)),
        "strategy": str(gate_cfg.get("strategy", "proportional")),
        "eligible_count": len(pool),
        "selected_count": len(selected),
        "withheld_count": len(unique_records) - len(selected),
        "ineligible_count": len(ineligible),
    }
    return {"selected": selected,
            "withheld": [row for row in unique_records if _record_key(row) not in selected_keys],
            "report": report}
