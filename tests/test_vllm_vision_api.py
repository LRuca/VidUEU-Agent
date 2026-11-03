from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from executor.models.vllm_vision_api import VLLMVisionApi


class _FakeOpenAI:
    def __init__(self, *args, **kwargs) -> None:
        self.max_retries = kwargs.get("max_retries")
        self.chat = type("ChatAPI", (), {"completions": type("Completions", (), {"create": lambda *a, **k: None})()})()

    def close(self) -> None:
        return None


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _SequencedCompletions:
    def __init__(self, events):
        self._events = list(events)
        self.calls = 0

    def create(self, **kwargs):
        event = self._events[self.calls]
        self.calls += 1
        if isinstance(event, BaseException):
            raise event
        return _FakeResponse(event)


class _FakeClient:
    def __init__(self, events):
        self.chat = type("ChatAPI", (), {"completions": _SequencedCompletions(events)})()

    def close(self) -> None:
        return None


class VLLMVisionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.gettempdir()) / "agentpack_test_tmp" / "vllm_vision_api"
        self.root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))

    def _make_image(self, name: str) -> Path:
        path = self.root / name
        Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path, format="PNG")
        return path

    def _build_api(self) -> VLLMVisionApi:
        with patch("executor.models.vllm_vision_api.OpenAI", _FakeOpenAI):
            return VLLMVisionApi({"base_url": "http://127.0.0.1:8000/v1", "model": "test-model"})

    def test_local_path_is_encoded_as_data_url(self) -> None:
        image_path = self._make_image("local.png")
        api = self._build_api()

        url = api._image_to_url(str(image_path))

        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_persistent_budget_stops_a_new_client_before_request(self) -> None:
        cfg = {"base_url": "http://example.invalid/v1", "model": "test-model",
               "budget_path": str(self.root / "budget.sqlite"), "max_api_calls": 1, "max_retries": 3}
        with patch("executor.models.vllm_vision_api.OpenAI", _FakeOpenAI):
            first = VLLMVisionApi(cfg)
            self.assertEqual(first.client.max_retries, 0)  # Hidden SDK retries cannot bypass the ledger.
            first.client = _FakeClient(["supported answer"])
            self.assertEqual(first.chat([], "first prompt"), "supported answer")
            second = VLLMVisionApi(cfg)
            second.client = _FakeClient(["should not be called"])
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                second.chat([], "second prompt")
            self.assertEqual(second.client.chat.completions.calls, 0)

    def test_file_uri_is_encoded_as_data_url(self) -> None:
        image_path = self._make_image("uri.png")
        api = self._build_api()

        url = api._image_to_url(image_path.resolve().as_uri())

        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_chat_retries_after_transient_exception(self) -> None:
        api = self._build_api()
        api.client = _FakeClient([RuntimeError("temporary"), "final answer"])

        with patch("executor.models.vllm_vision_api.time.sleep") as sleep_mock:
            text = api.chat([], "hello", max_new_tokens=32)

        self.assertEqual(text, "final answer")
        self.assertEqual(api.client.chat.completions.calls, 2)
        sleep_mock.assert_called_once()

    def test_chat_retries_after_empty_content(self) -> None:
        api = self._build_api()
        api.client = _FakeClient(["", "recovered answer"])

        with patch("executor.models.vllm_vision_api.time.sleep") as sleep_mock:
            text = api.chat([], "hello", max_new_tokens=32)

        self.assertEqual(text, "recovered answer")
        self.assertEqual(api.client.chat.completions.calls, 2)
        sleep_mock.assert_called_once()

    def test_openai_profile_sends_low_without_vllm_parameters(self) -> None:
        key_file = self.root / "private-test.key"
        key_file.write_text("unit-test-placeholder", encoding="utf-8")
        with patch("executor.models.vllm_vision_api.OpenAI", _FakeOpenAI):
            api = VLLMVisionApi({"base_url": "https://example.invalid/v1", "model": "gpt-6.1-sol",
                                "api_key_file": str(key_file), "wire_profile": "openai", "reasoning_effort": "low"})
        calls = []
        api.client.chat.completions.create = lambda **kwargs: calls.append(kwargs) or _FakeResponse("answer")
        self.assertEqual(api.chat([], "prompt", max_new_tokens=1024), "answer")
        self.assertEqual(calls[0]["reasoning_effort"], "low")
        self.assertEqual(calls[0]["max_completion_tokens"], 1024)
        for field in ["max_tokens", "temperature", "top_p", "presence_penalty", "extra_body"]:
            self.assertNotIn(field, calls[0])
        self.assertEqual(api.last_attestation["requested_model"], "gpt-6.1-sol")

    def test_api_errors_retry_with_persistent_budget_without_semantic_decisions(self) -> None:
        cfg = {"base_url": "https://example.invalid/v1", "model": "gpt-6.1-sol", "wire_profile": "openai",
               "retry_until_budget": True, "request_retry_attempts": 1,
               "budget_path": str(self.root / "retry.sqlite"), "max_api_calls": 3}
        with patch("executor.models.vllm_vision_api.OpenAI", _FakeOpenAI):
            api = VLLMVisionApi(cfg)
        api.client = _FakeClient([TimeoutError("temporary"), RuntimeError("temporary"), "supported answer"])
        with patch("executor.models.vllm_vision_api.time.sleep"):
            self.assertEqual(api.chat([], "prompt"), "supported answer")
        attempts = api.api_budget.snapshot()["attempts"]
        self.assertEqual([r["status"] for r in attempts], ["api_error", "api_error", "response"])
        self.assertTrue(all(r["result"]["semantic_decision"] is None for r in attempts[:2]))
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            api.chat([], "second prompt")
        self.assertEqual(api.client.chat.completions.calls, 3)

    def test_unlimited_api_retry_requires_persistent_budget(self) -> None:
        with patch("executor.models.vllm_vision_api.OpenAI", _FakeOpenAI):
            with self.assertRaisesRegex(ValueError, "persistent API budget"):
                VLLMVisionApi({"base_url": "https://example.invalid", "model": "test", "retry_until_budget": True})


if __name__ == "__main__":
    unittest.main()
