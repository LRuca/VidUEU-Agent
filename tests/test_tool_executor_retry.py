from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from executor.tasks.tool_executor import ToolExecutor


class _FakeRegistry:
    def enabled_tools(self):
        return {}

    def get(self, tool_name: str):
        return {
            "semanticizer": {
                "default_weight": 1.0,
            }
        }


class _FlakyAdapter:
    def __init__(self, first_result):
        self.calls = 0
        self.first_result = first_result

    def run_shot(self, ctx):
        self.calls += 1
        if self.calls == 1:
            if isinstance(self.first_result, BaseException):
                raise self.first_result
            return self.first_result
        return {
            "raw": {"ok": True},
            "semantic": {"frame_hints": []},
        }


def _build_executor() -> ToolExecutor:
    executor = ToolExecutor(
        {
            "runtime": {
                "tool_executor": {
                    "retry_attempts": 2,
                    "retry_backoff_s": 0.1,
                }
            }
        }
    )
    executor.registry = _FakeRegistry()
    return executor


def test_tool_executor_retries_after_exception() -> None:
    executor = _build_executor()
    adapter = _FlakyAdapter(RuntimeError("temporary tool failure"))
    executor.adapters = {"ocr": adapter}

    with patch("executor.tasks.tool_executor.time.sleep") as sleep_mock:
        result = executor.execute(
            shots=[SimpleNamespace(shot_id="shot_000")],
            candidates_by_shot={"shot_000": [{"shot_id": "shot_000", "frame_index": 1}]},
            tool_plan={"shot_plans": {"shot_000": {"tool_actions": [{"tool_name": "ocr"}]}}},
            video_path="/tmp/video.mp4",
            t2_workdir="/tmp/t2",
            model_manager=object(),
        )

    assert adapter.calls == 2
    assert result["raw_by_shot"]["shot_000"]["ocr"] == {"ok": True}
    sleep_mock.assert_called_once()


def test_tool_executor_retries_after_invalid_payload() -> None:
    executor = _build_executor()
    adapter = _FlakyAdapter(None)
    executor.adapters = {"ocr": adapter}

    with patch("executor.tasks.tool_executor.time.sleep") as sleep_mock:
        result = executor.execute(
            shots=[SimpleNamespace(shot_id="shot_000")],
            candidates_by_shot={"shot_000": [{"shot_id": "shot_000", "frame_index": 1}]},
            tool_plan={"shot_plans": {"shot_000": {"tool_actions": [{"tool_name": "ocr"}]}}},
            video_path="/tmp/video.mp4",
            t2_workdir="/tmp/t2",
            model_manager=object(),
        )

    assert adapter.calls == 2
    assert result["raw_by_shot"]["shot_000"]["ocr"] == {"ok": True}
    sleep_mock.assert_called_once()
