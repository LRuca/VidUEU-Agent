import importlib.machinery
import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(importlib.machinery.PathFinder.find_spec("torch") is None,
                               reason="API wrapper imports PyTorch; validated in the A100 environment")


def make_client(tmp_path):
    from executor.models.qwen_llm import QwenLLM
    return QwenLLM({"mode": "api", "api": {"model": "deepseek-v4-flash", "base_url": "https://example.invalid",
                         "api_key": "unit-test-placeholder", "max_retries": 3, "max_api_calls": 1,
                         "call_log_path": str(tmp_path / "calls.jsonl")}})


def test_insufficient_balance_stops_without_retry(tmp_path):
    client = make_client(tmp_path)
    response = SimpleNamespace(status_code=402, headers={"Content-Type": "application/json"}, text="Insufficient Balance")
    calls = []
    client._get_session = lambda: SimpleNamespace(post=lambda **kwargs: calls.append(kwargs) or response)
    with pytest.raises(ValueError, match="http_402"):
        client.chat("test", max_new_tokens=10)
    assert len(calls) == 1
    audit = json.loads((tmp_path / "calls.jsonl").read_text())
    assert audit["http_status"] == 402
    assert audit["response_model"] is None


def test_response_model_usage_and_call_budget_are_recorded(tmp_path):
    client = make_client(tmp_path)
    data = {"id": "test-response", "model": "deepseek-v4-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "choices": [{"message": {"content": "valid answer"}}]}
    response = SimpleNamespace(status_code=200, headers={"Content-Type": "application/json"}, json=lambda: data)
    calls = []
    client._get_session = lambda: SimpleNamespace(post=lambda **kwargs: calls.append(kwargs) or response)
    assert client.chat("test", max_new_tokens=10) == "valid answer"
    with pytest.raises(ValueError, match="budget exhausted"):
        client.chat("test", max_new_tokens=10)
    assert len(calls) == 1
    audit = json.loads((tmp_path / "calls.jsonl").read_text())
    assert audit["requested_model"] == audit["response_model"] == "deepseek-v4-flash"
    assert audit["usage"] == data["usage"]
    assert "unit-test-placeholder" not in (tmp_path / "calls.jsonl").read_text()


def test_persistent_budget_stops_new_client_before_http(tmp_path):
    from executor.models.qwen_llm import QwenLLM
    config = {"mode": "api", "api": {"model": "deepseek-v4-flash", "base_url": "https://example.invalid",
              "api_key": "unit-test-placeholder", "max_retries": 3, "max_api_calls": 1,
              "budget_path": str(tmp_path / "budget.sqlite")}}
    response = SimpleNamespace(status_code=402, headers={}, text="Insufficient Balance")
    first = QwenLLM(config)
    calls = []
    first._get_session = lambda: SimpleNamespace(post=lambda **k: calls.append(k) or response)
    with pytest.raises(ValueError, match="http_402"):
        first.chat("first")
    second = QwenLLM(config)
    second._get_session = lambda: SimpleNamespace(post=lambda **k: pytest.fail("budget must stop HTTP"))
    with pytest.raises(ValueError, match="budget exhausted"):
        second.chat("second")
    assert len(calls) == 1 and second.api_budget.snapshot()["attempts_reserved"] == 1
