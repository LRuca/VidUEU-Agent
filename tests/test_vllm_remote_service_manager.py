from __future__ import annotations

import shutil
import signal
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

if "torch" not in sys.modules or not hasattr(sys.modules["torch"], "cuda"):
    import types

    fake_torch = types.ModuleType("torch")

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    fake_torch.cuda = _FakeCuda()
    sys.modules["torch"] = fake_torch

from executor.runtime.model_manager import ModelManager
from executor.runtime.vllm_service_manager import VLLMServiceManager, VLLMServiceRecord


class _FakeCombinedServiceManager:
    def __init__(self, *, remote_b_count: int = 8, remote_c_count: int = 0) -> None:
        self.enabled = True
        self._devices: list[str] = []
        self.ensure_calls: list[list[str]] = []
        self.remote_b_count = int(remote_b_count)
        self.remote_c_count = int(remote_c_count)

    def active_service_count(self) -> int:
        return len(self._devices)

    def active_devices(self) -> list[str]:
        return list(self._devices)

    def active_local_devices(self) -> list[str]:
        return [x for x in self._devices if not x.startswith("remote_")]

    def active_remote_devices(self) -> list[str]:
        return [x for x in self._devices if x.startswith("remote_")]

    def local_active_service_count(self) -> int:
        return len(self.active_local_devices())

    def remote_active_service_count(self) -> int:
        return len(self.active_remote_devices())

    def remote_service_target_count(self) -> int:
        return self.remote_b_count + self.remote_c_count

    def endpoint_for_device(self, device: str) -> str:
        if device.startswith("remote_b:"):
            idx = int(device.split(":", 1)[1])
            return f"http://127.0.0.1:{8008 + idx}/v1"
        if device.startswith("remote_c:"):
            idx = int(device.split(":", 1)[1])
            return f"http://127.0.0.1:{8024 + idx}/v1"
        idx = int(device.split(":", 1)[1])
        return f"http://127.0.0.1:{8000 + idx}/v1"

    def ensure_services(self, devices: list[str]) -> dict[str, str]:
        local_devices = list(devices)
        self.ensure_calls.append(local_devices)
        remote_devices = [f"remote_b:{idx}" for idx in range(self.remote_b_count)] + [
            f"remote_c:{idx}" for idx in range(self.remote_c_count)
        ]
        self._devices = local_devices + remote_devices
        return {device: self.endpoint_for_device(device) for device in self._devices}


class _FakeLocalServiceManager:
    def __init__(self, devices: list[str]) -> None:
        self.enabled = True
        self._devices = list(devices)

    def active_service_count(self) -> int:
        return len(self._devices)

    def active_devices(self) -> list[str]:
        return list(self._devices)

    def active_local_devices(self) -> list[str]:
        return list(self._devices)

    def active_remote_devices(self) -> list[str]:
        return []

    def local_active_service_count(self) -> int:
        return len(self._devices)

    def remote_active_service_count(self) -> int:
        return 0

    def remote_service_target_count(self) -> int:
        return 0

    def endpoint_for_device(self, device: str) -> str:
        idx = int(device.split(":", 1)[1])
        return f"http://127.0.0.1:{8000 + idx}/v1"

    def ensure_services(self, devices: list[str]) -> dict[str, str]:
        self._devices = list(devices)
        return {device: self.endpoint_for_device(device) for device in self._devices}


class _FakeVLMReplica:
    def __init__(self, device: str) -> None:
        self._agentpack_device = device

    def set_chat_logger(self, logger):
        return None

    def set_chat_context(self, context):
        return None


class _FakePopen:
    _next_pid = 7000

    def __init__(self, *args, **kwargs) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self._returncode = None

    def poll(self):
        return self._returncode

    def wait(self, timeout=None):
        self._returncode = 0
        return 0

    def terminate(self):
        self._returncode = 0

    def kill(self):
        self._returncode = -9


class VLLMRemoteServiceManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        VLLMServiceManager._shutdown_signal_in_progress = False

    def test_remote_clusters_require_explicit_enable_and_close_is_local(self) -> None:
        manager = VLLMServiceManager({"auto_start": True}, log_root=str(self._make_root("remote_opt_in")))
        self.assertFalse(manager.remote_b_enabled)
        self.assertFalse(manager.remote_c_enabled)
        self.assertEqual(manager.remote_service_target_count(), 0)
        with patch.object(manager, "_run_remote_b_script") as remote_b, patch.object(
            manager, "_run_remote_c_script"
        ) as remote_c:
            manager.close()
        remote_b.assert_not_called()
        remote_c.assert_not_called()

    def _make_root(self, name: str) -> Path:
        root = Path(tempfile.gettempdir()) / "agentpack_test_tmp" / str(name)
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        return root

    def test_model_manager_keeps_local_target_and_adds_remote_workers(self) -> None:
        cfg = {
            "models": {
                "qwen_vlm": {
                    "mode": "api",
                    "devices": [f"cuda:{idx}" for idx in range(8)],
                    "target_replicas": 6,
                    "auto_trim_by_free_vram": True,
                    "min_free_vram_gb_per_replica": 16,
                    "reserve_vram_gb": 4,
                    "api": {
                        "provider": "vllm_openai",
                        "service": {
                            "auto_start": False,
                            "startup_on_init": False,
                        },
                    },
                }
            },
            "runtime": {
                "max_gpu_resident_models": 8,
            },
        }
        manager = ModelManager(cfg)
        fake_service_manager = _FakeCombinedServiceManager()
        manager._vllm_service_manager = fake_service_manager
        manager._eligible_vllm_service_devices = lambda strict: [f"cuda:{idx}" for idx in range(6)]  # type: ignore[method-assign]

        self.assertEqual(manager._desired_local_vlm_pool_size(), 6)
        self.assertEqual(manager._desired_vlm_pool_size(), 14)

        endpoints = manager._ensure_vllm_service_pool()

        self.assertEqual(fake_service_manager.ensure_calls, [[f"cuda:{idx}" for idx in range(6)]])
        self.assertEqual(len(endpoints), 14)
        self.assertEqual(manager._desired_vlm_pool_size(), 14)

    def test_model_manager_counts_remote_c_workers_when_b_disabled(self) -> None:
        cfg = {
            "models": {
                "qwen_vlm": {
                    "mode": "api",
                    "devices": [f"cuda:{idx}" for idx in range(8)],
                    "target_replicas": 6,
                    "auto_trim_by_free_vram": True,
                    "min_free_vram_gb_per_replica": 16,
                    "reserve_vram_gb": 4,
                    "api": {
                        "provider": "vllm_openai",
                        "service": {
                            "auto_start": False,
                            "startup_on_init": False,
                        },
                    },
                }
            },
            "runtime": {
                "max_gpu_resident_models": 8,
            },
        }
        manager = ModelManager(cfg)
        fake_service_manager = _FakeCombinedServiceManager(remote_b_count=0, remote_c_count=4)
        manager._vllm_service_manager = fake_service_manager
        manager._eligible_vllm_service_devices = lambda strict: [f"cuda:{idx}" for idx in range(6)]  # type: ignore[method-assign]

        self.assertEqual(manager._desired_local_vlm_pool_size(), 6)
        self.assertEqual(manager._desired_vlm_pool_size(), 10)

        endpoints = manager._ensure_vllm_service_pool()

        self.assertEqual(fake_service_manager.ensure_calls, [[f"cuda:{idx}" for idx in range(6)]])
        self.assertEqual(len(endpoints), 10)
        self.assertIn("remote_c:3", endpoints)

    def test_model_manager_pins_each_video_thread_to_one_vlm_slot(self) -> None:
        cfg = {
            "models": {
                "qwen_vlm": {
                    "mode": "api",
                    "devices": ["cuda:0", "cuda:1"],
                    "target_replicas": 2,
                    "auto_trim_by_free_vram": False,
                    "api": {
                        "provider": "vllm_openai",
                        "service": {
                            "auto_start": False,
                            "startup_on_init": False,
                        },
                    },
                }
            },
            "runtime": {
                "max_gpu_resident_models": 8,
                "video_parallelism": {
                    "enabled": True,
                },
            },
        }
        manager = ModelManager(cfg)
        manager._vllm_service_manager = _FakeLocalServiceManager(["cuda:0", "cuda:1"])
        manager._eligible_vllm_service_devices = lambda strict: ["cuda:0", "cuda:1"]  # type: ignore[method-assign]
        manager._build_vlm_replica = lambda replica_idx: _FakeVLMReplica(["cuda:0", "cuda:1"][replica_idx])  # type: ignore[method-assign]
        manager.warm_vlm_pool()

        start_barrier = threading.Barrier(2)
        per_thread_devices: dict[str, list[str]] = {}
        errors: list[BaseException] = []

        def _worker(name: str) -> None:
            try:
                manager.acquire_video_vlm_slot(name)
                start_barrier.wait(timeout=5)
                results = manager.run_vlm_jobs(
                    [1, 2, 3],
                    worker_fn=lambda vlm, item: str(getattr(vlm, "_agentpack_device", "")),
                    max_workers=3,
                )
                per_thread_devices[name] = list(results)
            except BaseException as exc:  # pragma: no cover - test helper
                errors.append(exc)
            finally:
                manager.release_video_vlm_slot()

        threads = [
            threading.Thread(target=_worker, args=("video_a",), daemon=True),
            threading.Thread(target=_worker, args=("video_b",), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(errors)
        self.assertEqual(set(per_thread_devices.keys()), {"video_a", "video_b"})
        self.assertEqual(len(set(per_thread_devices["video_a"])), 1)
        self.assertEqual(len(set(per_thread_devices["video_b"])), 1)
        self.assertNotEqual(per_thread_devices["video_a"][0], per_thread_devices["video_b"][0])

    def test_remote_b_startup_and_shutdown_sequence(self) -> None:
        root = self._make_root("vllm_remote_manager")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 2,
                "wait_after_stop_s": 12,
                "wait_after_start_s": 0,
                "stop_all_script": "/root/weitong_lian/stop_all.sh",
                "start_vllm_8gpu_tmux_script": "/root/weitong_lian/start_vllm_8gpu_tmux.sh",
                "start_burn_all_script": "/root/weitong_lian/start_burn_all.sh",
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        remote_calls: list[str] = []

        def _record_remote(script_path: str, *, check: bool):
            remote_calls.append(script_path)
            return None

        with patch.object(manager, "_run_remote_b_script", side_effect=_record_remote), patch.object(
            manager,
            "_wait_until_ready",
            return_value=True,
        ), patch("executor.runtime.vllm_service_manager.subprocess.Popen", side_effect=_FakePopen), patch(
            "executor.runtime.vllm_service_manager.time.sleep"
        ) as sleep_mock:
            endpoints = manager.ensure_services([])
            self.assertEqual(set(endpoints.keys()), {"remote_b:0", "remote_b:1"})
            self.assertEqual(
                remote_calls[:2],
                [
                    "/root/weitong_lian/stop_all.sh",
                    "/root/weitong_lian/start_vllm_8gpu_tmux.sh",
                ],
            )

            manager.close()

        self.assertEqual(
            remote_calls[-2:],
            [
                "/root/weitong_lian/stop_all.sh",
                "/root/weitong_lian/start_burn_all.sh",
            ],
        )
        sleep_args = [tuple(call.args) for call in sleep_mock.call_args_list]
        self.assertIn((0.25,), sleep_args)
        self.assertIn((12.0,), sleep_args)

    def test_remote_c_startup_and_shutdown_sequence_without_burn(self) -> None:
        root = self._make_root("vllm_remote_manager_c")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": False,
            },
            "remote_c": {
                "enabled": True,
                "ssh_host_alias": "server-c",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8024,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8024,
                "service_count": 2,
                "wait_after_stop_s": 12,
                "wait_after_start_s": 0,
                "stop_all_script": "/root/weitonglian/stop_4_vllm_qwen35.sh",
                "start_4_vllm_qwen35_script": "/root/weitonglian/start_4_vllm_qwen35.sh",
                "start_burn_all_script": "",
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        remote_calls: list[str] = []

        def _record_remote(script_path: str, *, check: bool):
            remote_calls.append(script_path)
            return None

        with patch.object(manager, "_run_remote_c_script", side_effect=_record_remote), patch.object(
            manager,
            "_wait_until_ready",
            return_value=True,
        ), patch("executor.runtime.vllm_service_manager.subprocess.Popen", side_effect=_FakePopen), patch(
            "executor.runtime.vllm_service_manager.time.sleep"
        ) as sleep_mock:
            endpoints = manager.ensure_services([])
            self.assertEqual(set(endpoints.keys()), {"remote_c:0", "remote_c:1"})
            manager.close()

        self.assertEqual(
            remote_calls,
            [
                "/root/weitonglian/stop_4_vllm_qwen35.sh",
                "/root/weitonglian/start_4_vllm_qwen35.sh",
                "/root/weitonglian/stop_4_vllm_qwen35.sh",
            ],
        )
        sleep_args = [tuple(call.args) for call in sleep_mock.call_args_list]
        self.assertIn((0.25,), sleep_args)
        self.assertNotIn((12.0,), sleep_args)

    def test_remote_b_cluster_cleans_stale_orphan_wrappers_before_restart(self) -> None:
        root = self._make_root("vllm_remote_manager_cleanup_orphans")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "local_forward_base_port": 8008,
                "service_count": 2,
                "wait_after_stop_s": 0,
                "wait_after_start_s": 0,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        ps_stdout = "\n".join(
            [
                "123 1 /bin/bash -lc while true; do ssh -N -L 127.0.0.1:8008:127.0.0.1:8008 remote-b; done",
                "234 1 /bin/bash -lc while true; do ssh -N -L 127.0.0.1:8009:127.0.0.1:8009 remote-b; done",
                "345 2 /bin/bash -lc while true; do ssh -N -L 127.0.0.1:8010:127.0.0.1:8010 remote-b; done",
            ]
        )

        with patch("executor.runtime.vllm_service_manager.subprocess.run") as run_mock, patch(
            "executor.runtime.vllm_service_manager.os.killpg",
            create=True,
        ) as killpg_mock, patch.object(manager, "_cleanup_pid_path") as cleanup_mock, patch.object(
            manager, "_run_remote_b_script", return_value=None
        ):
            run_mock.return_value.returncode = 0
            run_mock.return_value.stdout = ps_stdout
            manager._ensure_remote_b_cluster()

        if sys.platform.startswith("win"):
            killpg_mock.assert_not_called()
            cleanup_mock.assert_not_called()
        else:
            self.assertEqual([call.args[0] for call in killpg_mock.call_args_list], [123, 234])
            self.assertEqual(cleanup_mock.call_count, 2)

    def test_remote_b_tunnel_wrapper_uses_parent_death_preexec(self) -> None:
        root = self._make_root("vllm_remote_manager_pdeathsig")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        with patch.object(manager, "_ensure_remote_b_cluster", return_value=None), patch.object(
            manager,
            "_wait_until_ready",
            side_effect=[False, True],
        ), patch(
            "executor.runtime.vllm_service_manager.subprocess.Popen",
            side_effect=_FakePopen,
        ) as popen_mock:
            manager._ensure_remote_b_service("remote_b:0")

        if sys.platform.startswith("win"):
            self.assertIsNone(popen_mock.call_args.kwargs.get("preexec_fn"))
        else:
            self.assertIsNotNone(popen_mock.call_args.kwargs.get("preexec_fn"))

    def test_remote_b_tunnel_launch_wraps_ssh_with_retry_loop(self) -> None:
        root = self._make_root("vllm_remote_manager_retry_loop")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
                "tunnel_retry_delay_s": 3,
                "ssh_connect_timeout_s": 7,
                "ssh_server_alive_interval_s": 11,
                "ssh_server_alive_count_max": 5,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        launch_command, use_shell = manager._build_ssh_tunnel_launch(
            device="remote_b:0",
            local_port=8008,
            remote_port=8008,
        )

        self.assertTrue(use_shell)
        self.assertEqual(launch_command[:2], ["/bin/bash", "-lc"])
        shell_script = launch_command[2]
        self.assertIn("while true; do", shell_script)
        self.assertIn("ssh", shell_script)
        self.assertIn("ExitOnForwardFailure=yes", shell_script)
        self.assertIn("ConnectTimeout=7", shell_script)
        self.assertIn("ServerAliveInterval=11", shell_script)
        self.assertIn("ServerAliveCountMax=5", shell_script)
        self.assertIn("retrying in 3.0s", shell_script)

    def test_record_alive_tolerates_remote_tunnel_reconnect_window(self) -> None:
        root = self._make_root("vllm_remote_manager_record_alive")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))
        record = VLLMServiceRecord(
            device="remote_b:0",
            port=8008,
            base_url="http://127.0.0.1:8008/v1",
            log_path="remote.log",
            process=_FakePopen(),
            owned=True,
            kind="remote_b_tunnel",
        )

        with patch.object(manager, "_wait_until_ready", return_value=False) as wait_mock:
            self.assertTrue(manager._record_alive(record))

        wait_mock.assert_not_called()

    def test_remote_b_ensure_service_waits_without_timeout(self) -> None:
        root = self._make_root("vllm_remote_manager_infinite_wait")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
                "startup_timeout_s": 240,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        with patch.object(manager, "_ensure_remote_b_cluster", return_value=None), patch.object(
            manager,
            "_wait_until_ready",
            side_effect=[False, True],
        ) as wait_mock, patch(
            "executor.runtime.vllm_service_manager.subprocess.Popen",
            side_effect=_FakePopen,
        ):
            endpoint = manager._ensure_remote_b_service("remote_b:0")

        self.assertEqual(endpoint, "http://127.0.0.1:8008/v1")
        self.assertEqual(wait_mock.call_args_list[0].kwargs["timeout_s"], 5.0)
        self.assertEqual(wait_mock.call_args_list[1].kwargs["timeout_s"], 0.0)

    def test_remote_b_reuses_existing_ready_tunnel(self) -> None:
        root = self._make_root("vllm_remote_manager_reuse_existing")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        with patch.object(manager, "_ensure_remote_b_cluster", return_value=None), patch.object(
            manager,
            "_wait_until_ready",
            return_value=True,
        ) as wait_mock, patch(
            "executor.runtime.vllm_service_manager.subprocess.Popen",
        ) as popen_mock:
            endpoint = manager._ensure_remote_b_service("remote_b:0")

        self.assertEqual(endpoint, "http://127.0.0.1:8008/v1")
        self.assertEqual(manager._services["remote_b:0"].process, None)
        self.assertEqual(wait_mock.call_args.kwargs["timeout_s"], 5.0)
        popen_mock.assert_not_called()

    def test_remote_b_unavailable_endpoint_is_skipped(self) -> None:
        root = self._make_root("vllm_remote_manager_skip")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 3,
                "wait_after_stop_s": 0,
                "wait_after_start_s": 0,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        with patch.object(manager, "_ensure_remote_b_cluster", return_value=None):
            def _wrapped(device: str) -> str:
                if device == "remote_b:1":
                    raise RuntimeError("remote endpoint not ready")
                return f"http://127.0.0.1:{8008 + int(device.split(':', 1)[1])}/v1"

            with patch.object(manager, "_ensure_remote_b_service", side_effect=_wrapped):
                endpoints = manager.ensure_services([])
                self.assertEqual(
                    endpoints,
                    {
                        "remote_b:0": "http://127.0.0.1:8008/v1",
                        "remote_b:2": "http://127.0.0.1:8010/v1",
                    },
                )
                self.assertIn("remote_b:1", manager._remote_b_unavailable_devices)
                self.assertEqual(manager._remote_b_device_names(), ["remote_b:0", "remote_b:2"])

    def test_remote_b_shutdown_runs_after_interrupt_during_startup(self) -> None:
        root = self._make_root("vllm_remote_manager_interrupt")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 2,
                "wait_after_stop_s": 12,
                "wait_after_start_s": 0,
                "stop_all_script": "/root/weitong_lian/stop_all.sh",
                "start_vllm_8gpu_tmux_script": "/root/weitong_lian/start_vllm_8gpu_tmux.sh",
                "start_burn_all_script": "/root/weitong_lian/start_burn_all.sh",
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        remote_calls: list[str] = []

        def _record_remote(script_path: str, *, check: bool):
            remote_calls.append(script_path)
            if script_path.endswith("start_vllm_8gpu_tmux.sh"):
                raise KeyboardInterrupt()
            return None

        with patch.object(manager, "_run_remote_b_script", side_effect=_record_remote), patch(
            "executor.runtime.vllm_service_manager.time.sleep"
        ) as sleep_mock:
            with self.assertRaises(KeyboardInterrupt):
                manager._ensure_remote_b_cluster()
            manager.close()

        self.assertEqual(
            remote_calls,
            [
                "/root/weitong_lian/stop_all.sh",
                "/root/weitong_lian/start_vllm_8gpu_tmux.sh",
                "/root/weitong_lian/stop_all.sh",
                "/root/weitong_lian/start_burn_all.sh",
            ],
        )
        sleep_args = [tuple(call.args) for call in sleep_mock.call_args_list]
        self.assertIn((0.25,), sleep_args)
        self.assertIn((12.0,), sleep_args)

    def test_signal_handler_resignals_after_cleanup(self) -> None:
        with patch.object(VLLMServiceManager, "_close_registered_managers") as close_mock, patch(
            "executor.runtime.vllm_service_manager.signal.signal"
        ) as signal_mock, patch("executor.runtime.vllm_service_manager.os.kill") as kill_mock:
            VLLMServiceManager._shutdown_signal_in_progress = False
            VLLMServiceManager._handle_shutdown_signal(int(signal.SIGINT), None)

        close_mock.assert_called_once()
        signal_mock.assert_called_once_with(int(signal.SIGINT), signal.SIG_DFL)
        kill_mock.assert_called_once_with(unittest.mock.ANY, int(signal.SIGINT))

    def test_signal_handler_duplicate_signal_does_not_abort_cleanup(self) -> None:
        with patch.object(VLLMServiceManager, "_close_registered_managers") as close_mock, patch(
            "executor.runtime.vllm_service_manager.signal.signal"
        ) as signal_mock, patch("executor.runtime.vllm_service_manager.os.kill") as kill_mock:
            VLLMServiceManager._shutdown_signal_in_progress = True
            VLLMServiceManager._handle_shutdown_signal(int(signal.SIGINT), None)

        close_mock.assert_not_called()
        signal_mock.assert_not_called()
        kill_mock.assert_not_called()

    def test_close_is_idempotent_for_remote_shutdown(self) -> None:
        root = self._make_root("vllm_remote_manager_idempotent")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
                "wait_after_stop_s": 0,
                "wait_after_start_s": 0,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))
        manager._remote_b_cleanup_required = True
        remote_calls: list[str] = []

        def _record_remote(script_path: str, *, check: bool):
            remote_calls.append(script_path)
            return None

        with patch.object(manager, "_run_remote_b_script", side_effect=_record_remote):
            manager.close()
            manager.close()

        self.assertEqual(
            remote_calls,
            [
                "/root/weitong_lian/stop_all.sh",
                "/root/weitong_lian/start_burn_all.sh",
            ],
        )

    def test_close_runs_remote_shutdown_even_if_local_terminate_fails(self) -> None:
        root = self._make_root("vllm_remote_manager_close_resilient")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
            "connect_host": "127.0.0.1",
            "remote_b": {
                "enabled": True,
                "ssh_host_alias": "remote-b",
                "remote_host": "127.0.0.1",
                "remote_base_port": 8008,
                "local_forward_host": "127.0.0.1",
                "local_forward_base_port": 8008,
                "service_count": 1,
                "wait_after_stop_s": 0,
                "wait_after_start_s": 0,
            },
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))
        manager._remote_b_cleanup_required = True
        manager._services = {
            "cuda:0": VLLMServiceRecord(
                device="cuda:0",
                port=8000,
                base_url="http://127.0.0.1:8000/v1",
                log_path="cuda.log",
                owned=True,
                kind="local",
            ),
            "remote_b:0": VLLMServiceRecord(
                device="remote_b:0",
                port=8008,
                base_url="http://127.0.0.1:8008/v1",
                log_path="remote.log",
                owned=True,
                kind="remote_b_tunnel",
            ),
        }
        remote_calls: list[str] = []

        def _record_remote(script_path: str, *, check: bool):
            remote_calls.append(script_path)
            return None

        with patch.object(manager, "_run_remote_b_script", side_effect=_record_remote), patch.object(
            manager,
            "_terminate_record",
            side_effect=[RuntimeError("local terminate failed"), None],
        ):
            manager.close()

        self.assertEqual(
            remote_calls,
            [
                "/root/weitong_lian/stop_all.sh",
                "/root/weitong_lian/start_burn_all.sh",
            ],
        )

    def test_ensure_services_skips_startup_during_shutdown(self) -> None:
        root = self._make_root("vllm_remote_manager_shutdown_gate")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        with patch.object(manager, "_ensure_single_service") as ensure_mock:
            VLLMServiceManager._shutdown_signal_in_progress = True
            endpoints = manager.ensure_services(["cuda:0"])

        self.assertEqual(endpoints, {})
        ensure_mock.assert_not_called()

    def test_wait_until_ready_aborts_when_shutdown_requested(self) -> None:
        root = self._make_root("vllm_remote_manager_wait_abort")

        cfg = {
            "auto_start": True,
            "startup_on_init": False,
            "shell_executable": "/bin/bash",
        }
        manager = VLLMServiceManager(cfg, log_root=str(root))

        VLLMServiceManager._shutdown_signal_in_progress = True
        with patch.object(manager, "_probe_models", return_value=False), patch(
            "executor.runtime.vllm_service_manager.time.sleep"
        ) as sleep_mock:
            ready = manager._wait_until_ready("http://127.0.0.1:8000/v1", timeout_s=5.0)

        self.assertFalse(ready)
        sleep_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
