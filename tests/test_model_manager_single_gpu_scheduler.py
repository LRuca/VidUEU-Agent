from __future__ import annotations

import threading
import time
import unittest
from typing import Any, Dict

from executor.runtime.model_manager import ModelManager


class DummyOOMError(RuntimeError):
    pass


class FakeScheduledResource:
    def __init__(self, name: str, fail_once_with_oom: bool = False) -> None:
        self.name = name
        self.fail_once_with_oom = fail_once_with_oom
        self.calls = 0
        self.closed = False

    def run(self) -> str:
        self.calls += 1
        if self.fail_once_with_oom:
            self.fail_once_with_oom = False
            raise DummyOOMError(f"{self.name} oom")
        return f"{self.name}:{self.calls}"

    def close(self) -> None:
        self.closed = True


class TestModelManager(ModelManager):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.instantiate_counts: Dict[str, int] = {}
        self.fail_once_names: set[str] = set()
        self.fake_gpu_mem: tuple[int, int] = (16384, 24576)
        self.available_cuda_devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"]
        self.selected_devices: Dict[str, str | None] = {}
        self.instantiate_started_events: Dict[str, threading.Event] = {}
        self.instantiate_block_events: Dict[str, threading.Event] = {}
        super().__init__(config)

    def _instantiate_resource(self, name: str, device: str | None = None) -> Any:
        self.instantiate_counts[name] = self.instantiate_counts.get(name, 0) + 1
        started = self.instantiate_started_events.get(name)
        if started is not None:
            started.set()
        block = self.instantiate_block_events.get(name)
        if block is not None:
            block.wait(timeout=1.0)
        fail_once = name in self.fail_once_names
        self.fail_once_names.discard(name)
        resource = FakeScheduledResource(name, fail_once_with_oom=fail_once)
        device = device or self._select_device_for_resource(name)
        self.selected_devices[name] = device
        setattr(resource, "_agentpack_device", device)
        return resource

    def _gpu_mem_info_mb(self, device: str) -> tuple[int, int]:
        _ = device
        return self.fake_gpu_mem

    def _available_cuda_devices(self) -> list[str]:
        return list(self.available_cuda_devices)

    def _torch_oom_error_type(self) -> tuple[type[BaseException], ...]:
        return (DummyOOMError,)


class ModelManagerSingleGpuSchedulerTest(unittest.TestCase):
    def _make_manager(
        self,
        tool_runtimes: Dict[str, Dict[str, Any]] | None = None,
        scheduler_resources: list[str] | None = None,
    ) -> TestModelManager:
        scheduler = {
            "enabled": True,
            "device": "cuda:0",
            "reserved_margin_mb": 0,
            "poll_interval_s": 0.05,
            "global_wait_timeout_s": 2,
        }
        if scheduler_resources is not None:
            scheduler["resources"] = list(scheduler_resources)
        cfg = {
            "runtime": {
                "single_gpu_scheduler": scheduler,
                "tool_runtimes": tool_runtimes or {},
            },
            "models": {
                "omniparser": {
                    "device": "cuda:0",
                }
            },
        }
        return TestModelManager(cfg)

    def test_runtime_policy_override_and_registry_defaults(self) -> None:
        manager = self._make_manager(
            {
                "asr": {
                    "estimated_vram_mb": 1234,
                    "idle_unload_s": 7,
                }
            }
        )

        asr_rt = manager._resource_runtime["asr"]
        omni_spec = manager._specs["omniparser"]

        self.assertEqual(asr_rt.policy.estimated_vram_mb, 1234)
        self.assertEqual(asr_rt.policy.idle_unload_s, 7.0)
        self.assertTrue(omni_spec.gpu_heavy)
        self.assertGreater(manager._resource_runtime["omniparser"].policy.estimated_vram_mb, 0)

    def test_explicit_scheduler_resource_allowlist_is_preserved(self) -> None:
        manager = self._make_manager(
            {"yolo": {"estimated_vram_mb": 1024}},
            scheduler_resources=["asr", "raft"],
        )

        self.assertIn("raft", manager._scheduled_resource_names)
        self.assertNotIn("yolo", manager._scheduled_resource_names)
        self.assertTrue(manager._should_schedule_resource("raft"))
        self.assertFalse(manager._should_schedule_resource("yolo"))

    def test_unscheduled_yolo_honors_explicit_cpu_device(self) -> None:
        manager = self._make_manager(scheduler_resources=["asr", "raft"])
        manager.config["models"]["yolo_device"] = "cpu"

        self.assertFalse(manager._should_schedule_resource("yolo"))
        self.assertEqual(manager._select_device_for_resource("yolo"), "cpu")

    def test_acquire_reuses_loaded_instance_with_queueing(self) -> None:
        manager = self._make_manager(
            {
                "asr": {
                    "estimated_vram_mb": 1024,
                    "max_concurrency": 1,
                }
            }
        )
        release_first = threading.Event()
        second_finished = threading.Event()
        results: list[tuple[str, int]] = []

        def first_worker() -> None:
            inst = manager.acquire_resource("asr")
            results.append(("first", id(inst)))
            release_first.wait(timeout=1.0)
            manager.release_resource_slot("asr")

        def second_worker() -> None:
            inst = manager.acquire_resource("asr")
            results.append(("second", id(inst)))
            manager.release_resource_slot("asr")
            second_finished.set()

        t1 = threading.Thread(target=first_worker)
        t2 = threading.Thread(target=second_worker)
        t1.start()
        time.sleep(0.1)
        t2.start()
        time.sleep(0.15)
        self.assertFalse(second_finished.is_set())
        release_first.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        self.assertEqual(manager.instantiate_counts["asr"], 1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][1], results[1][1])

    def test_acquire_waits_for_idle_unload_then_loads_other_resource(self) -> None:
        manager = self._make_manager(
            {
                "asr": {
                    "estimated_vram_mb": 7000,
                    "idle_unload_s": 0,
                },
                "omniparser": {
                    "estimated_vram_mb": 5000,
                    "idle_unload_s": 0,
                },
            }
        )
        manager.fake_gpu_mem = (9000, 10000)
        asr = manager.acquire_resource("asr")
        self.assertIsNotNone(asr)

        acquired_omni = threading.Event()

        def omni_worker() -> None:
            inst = manager.acquire_resource("omniparser")
            self.assertEqual(inst.name, "omniparser")
            manager.release_resource_slot("omniparser")
            acquired_omni.set()

        t = threading.Thread(target=omni_worker)
        t.start()
        time.sleep(0.1)
        self.assertFalse(acquired_omni.is_set())
        manager.release_resource_slot("asr")
        t.join(timeout=1.5)

        self.assertTrue(acquired_omni.is_set())
        self.assertNotIn("asr", manager._instances)
        self.assertEqual(manager.instantiate_counts["omniparser"], 1)

    def test_run_with_resource_retries_once_after_oom(self) -> None:
        manager = self._make_manager(
            {
                "yolo": {
                    "estimated_vram_mb": 1024,
                    "idle_unload_s": 0,
                }
            }
        )
        manager.fail_once_names.add("yolo")

        result = manager.run_with_resource("yolo", lambda resource: resource.run())

        self.assertEqual(result, "yolo:1")
        self.assertEqual(manager.instantiate_counts["yolo"], 2)
        self.assertGreater(manager._resource_runtime["yolo"].policy.estimated_vram_mb, 1024)

    def test_snapshot_includes_runtime_state(self) -> None:
        manager = self._make_manager({"ocr": {"estimated_vram_mb": 2048}})
        inst = manager.acquire_resource("ocr")
        self.assertEqual(inst.name, "ocr")

        snapshot = manager.snapshot()

        self.assertIn("single_gpu_scheduler", snapshot)
        self.assertIn("resource_runtime", snapshot)
        self.assertTrue(snapshot["resource_runtime"]["ocr"]["loaded"])
        self.assertEqual(snapshot["resource_runtime"]["ocr"]["active_calls"], 1)
        self.assertEqual(snapshot["resource_runtime"]["ocr"]["estimated_vram_mb"], 2048)
        manager.release_resource_slot("ocr")
        manager.release("ocr")

    def test_scheduled_resources_use_single_gpu_scheduler_device(self) -> None:
        manager = self._make_manager({"asr": {"estimated_vram_mb": 2048}})
        manager._gpu_placement_policy["default_devices"] = ["cuda:3", "cuda:4"]
        manager._gpu_placement_policy["per_model"] = {
            "asr": {
                "devices": ["cuda:3", "cuda:4"],
            }
        }

        device = manager._select_device_for_resource("asr", fallback="cuda:4")
        inst = manager.acquire_resource("asr")

        self.assertEqual(device, "cuda:0")
        self.assertEqual(manager.selected_devices["asr"], "cuda:0")
        self.assertEqual(getattr(inst, "_agentpack_device", None), "cuda:0")
        manager.release_resource_slot("asr")

    def test_unavailable_scheduled_device_falls_back_to_visible_cuda(self) -> None:
        manager = self._make_manager({"asr": {"estimated_vram_mb": 2048}})
        manager.available_cuda_devices = ["cuda:0"]
        manager._scheduler_cfg["per_resource_devices"]["asr"] = "cuda:1"

        device = manager._select_device_for_resource("asr")
        inst = manager.acquire_resource("asr")

        self.assertEqual(device, "cuda:0")
        self.assertEqual(getattr(inst, "_agentpack_device", None), "cuda:0")
        manager.release_resource_slot("asr")

    def test_replicas_allow_multiple_tool_instances(self) -> None:
        manager = self._make_manager(
            {
                "yolo": {
                    "estimated_vram_mb": 1024,
                    "replicas": [
                        {"device": "cuda:0", "count": 1, "max_inflight": 1},
                        {"device": "cuda:1", "count": 1, "max_inflight": 1},
                    ],
                    "queue_timeout_s": 1,
                }
            }
        )
        first = manager.acquire_resource("yolo")
        second = manager.acquire_resource("yolo")

        self.assertIsNot(first, second)
        self.assertEqual(manager.instantiate_counts["yolo"], 2)
        self.assertEqual(getattr(first, "_agentpack_device", None), "cuda:0")
        self.assertEqual(getattr(second, "_agentpack_device", None), "cuda:1")
        snapshot = manager.snapshot()
        self.assertEqual(len(snapshot["resource_runtime"]["yolo"]["replicas"]), 2)
        self.assertEqual(snapshot["resource_runtime"]["yolo"]["active_calls"], 2)

        manager.release_resource_slot("yolo")
        manager.release_resource_slot("yolo")
        manager.release("yolo")

    def test_replica_loads_are_serialized_per_resource(self) -> None:
        manager = self._make_manager(
            {
                "yolo": {
                    "estimated_vram_mb": 1024,
                    "replicas": [
                        {"device": "cuda:0", "count": 1, "max_inflight": 1},
                        {"device": "cuda:1", "count": 1, "max_inflight": 1},
                    ],
                    "queue_timeout_s": 1,
                }
            }
        )
        first_load_started = threading.Event()
        release_first_load = threading.Event()
        manager.instantiate_started_events["yolo"] = first_load_started
        manager.instantiate_block_events["yolo"] = release_first_load
        acquired: list[Any] = []

        def acquire_yolo() -> None:
            inst = manager.acquire_resource("yolo")
            acquired.append(inst)

        t1 = threading.Thread(target=acquire_yolo)
        t2 = threading.Thread(target=acquire_yolo)
        t1.start()
        self.assertTrue(first_load_started.wait(timeout=1.0))
        t2.start()
        time.sleep(0.15)

        self.assertEqual(manager.instantiate_counts["yolo"], 1)
        release_first_load.set()
        t1.join(timeout=1.0)
        t2.join(timeout=1.0)

        self.assertEqual(manager.instantiate_counts["yolo"], 2)
        self.assertEqual(len(acquired), 2)
        manager.release_resource_slot("yolo")
        manager.release_resource_slot("yolo")
        manager.release("yolo")


if __name__ == "__main__":
    unittest.main()
