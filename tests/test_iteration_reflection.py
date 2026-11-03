import json
import sys
import types

from orchestrator.iteration.policy import default_policy
from orchestrator.iteration.reflection import deepseek_reflection


def test_deepseek_reflection_retries_invalid_response_then_accepts(monkeypatch):
    class FakeQwen:
        calls = 0

        def __init__(self, _config):
            self._tls = type("TLS", (), {"last_attestation": {"response_model": "deepseek-flash"}})()

        def chat(self, _prompt, max_new_tokens):
            assert max_new_tokens == 2500
            type(self).calls += 1
            if type(self).calls == 1:
                return "not-json"
            return json.dumps({
                "next_target_distribution": {},
                "policy_updates": {},
                "reason": "temporary response recovery",
            })

    fake_module = types.ModuleType("executor.models.qwen_llm")
    fake_module.QwenLLM = FakeQwen
    monkeypatch.setitem(sys.modules, "executor.models.qwen_llm", fake_module)
    policy = default_policy(1)
    result = deepseek_reflection(
        {
            "models": {"qwen_llm": {"api": {
                "model": "deepseek-v4-flash",
                "base_url": "https://example.invalid",
            }}},
            "reflection": {"max_attempts": 2, "retry_delay_s": 0},
        },
        {"policy": policy, "feedback_metrics": {"MMDU": 40.0}},
    )

    assert FakeQwen.calls == 2
    assert result["backend"] == "deepseek"
    assert result["retry_errors"] == ["attempt_1:JSONDecodeError"]
    assert result["model_identity"]["response_model"] == "deepseek-flash"
