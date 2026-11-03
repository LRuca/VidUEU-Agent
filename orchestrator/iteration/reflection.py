"""Fixture and optional DeepSeek reflection backends."""
from __future__ import annotations

import json
import hashlib
import re
import time
from typing import Any

from .policy import PAPER_CAPABILITY, PAPER_DOMAIN, PAPER_TASK, apply_reflection


def validate_reflection(policy: dict[str, Any], value: Any) -> dict[str, Any]:
    """Reject incomplete/schema-invalid responses inside the retry window."""
    if not isinstance(value, dict):
        raise ValueError("reflection_must_be_object")
    for key in ("next_target_distribution", "policy_updates"):
        if not isinstance(value.get(key), dict):
            raise ValueError(f"reflection_requires_object:{key}")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("reflection_reason_required")
    if value.get("fallback_to_previous_policy"):
        raise ValueError("reflection_fallback_not_success")
    targets = value["next_target_distribution"]
    if set(targets) - {"capability", "domain", "task_type"}:
        raise ValueError("reflection_unknown_distribution")
    if any(not isinstance(v, dict) for v in targets.values()):
        raise ValueError("reflection_distribution_must_be_object")
    controls = set(policy) - {"iteration", "schema_version"}
    if set(value["policy_updates"]) - controls:
        raise ValueError("reflection_unknown_control")
    return apply_reflection(policy, value)


def fixture_reflection(iteration: int, policy: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    """Produce a deterministic policy change for the two-round smoke path."""
    next_index = min(max(iteration, 0), 1)
    return {
        "backend": "fixture",
        "iteration": iteration,
        "reflection": {
            "weak_capabilities": ["editing"],
            "weak_domains": ["Gaming&ACG"],
            "weak_task_types": ["grounded_edit_to_select"],
            "observed_tradeoffs": [],
        },
        "next_target_distribution": {
            "capability": PAPER_CAPABILITY[next_index],
            "domain": PAPER_DOMAIN[next_index],
            "task_type": PAPER_TASK[next_index],
        },
        "policy_updates": {},
        "reason": "Fixture feedback moves the smoke run toward the next paper target distribution.",
    }


def deepseek_reflection(config: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Call the configured text LLM only when explicitly selected."""
    from executor.models.qwen_llm import QwenLLM

    prompt = (
        "You are the policy reflection controller for a video instruction-data pipeline.\n"
        "Return JSON only. Preserve the canonical labels in the payload. Do not change source data.\n"
        "Return next_target_distribution (object), policy_updates (object), reason (nonempty string), "
        "and optionally reflection (object). Empty update objects explicitly mean no change.\n"
        "Use only policy keys already present. Update capability, domain, and task_type distributions "
        "conservatively; task probabilities are unconditional and must match parent capabilities. "
        "Use reported scores, losses, distributions, deficits, and compressed history. "
        "Never infer missing benchmark results or copy paper iteration targets.\nPAYLOAD:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )
    model_config = ((config.get("models") or {}).get("qwen_llm") or config.get("qwen_llm"))
    if not isinstance(model_config, dict):
        raise ValueError("deepseek_reflection_qwen_llm_config_missing")
    model = QwenLLM(model_config)
    errors: list[str] = []
    raw = ""
    value: Any = None
    retry = config.get("reflection", {})
    attempts = retry.get("max_attempts", 3)
    delay = retry.get("retry_delay_s", 1.0)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("reflection_invalid_attempts")
    if isinstance(delay, bool) or not isinstance(delay, (float, int)) or not 0 <= delay <= 60:
        raise ValueError("reflection_invalid_retry_delay")
    for attempt in range(1, attempts + 1):
        try:
            raw = model.chat(prompt, max_new_tokens=2500)
            text = str(raw).strip()
            fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
            if fenced:
                text = fenced.group(1).strip()
            else:
                start, end = text.find("{"), text.rfind("}")
                if start >= 0 and end > start:
                    text = text[start : end + 1]
            value = json.loads(text)
            validate_reflection(payload["policy"], value)
            break
        except Exception as exc:
            errors.append(f"attempt_{attempt}:{type(exc).__name__}")
            value = None
            if attempt < attempts:
                time.sleep(min(60.0, delay * (2 ** min(attempt - 1, 6))))
    if not isinstance(value, dict):
        raise ValueError("deepseek_reflection_invalid_json:" + "|".join(errors))
    api = model_config.get("api") if isinstance(model_config.get("api"), dict) else model_config
    value["backend"] = "deepseek"
    value["raw_response"] = raw
    value["prompt_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    value["model_identity"] = {
        "provider": "deepseek",
        "model": api.get("model"),
        "base_url": api.get("base_url"),
        "attempts": len(errors) + 1 if value is not None else len(errors),
        "response_model": getattr(getattr(model, "_tls", None), "last_attestation", {}).get("response_model"),
    }
    if errors:
        value["retry_errors"] = errors
    return value


def next_policy(policy: dict[str, Any], reflection: dict[str, Any]) -> dict[str, Any]:
    return apply_reflection(policy, reflection)
