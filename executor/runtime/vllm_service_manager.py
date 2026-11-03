from __future__ import annotations

import atexit
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


@dataclass
class VLLMServiceRecord:
    device: str
    port: int
    base_url: str
    log_path: str
    process: subprocess.Popen | None = None
    owned: bool = False
    pid_path: str = ""
    launcher_pid: int | None = None
    kind: str = "local"


class VLLMServiceManager:
    """Manage local vLLM services and optional remote-B SSH tunneled endpoints."""

    _signal_lock = threading.RLock()
    _registered_managers: "weakref.WeakSet[VLLMServiceManager]" = weakref.WeakSet()
    _signal_handlers_installed = False
    _atexit_installed = False
    _previous_signal_handlers: Dict[int, object] = {}
    _shutdown_signal_in_progress = False

    def __init__(self, cfg: Dict[str, object], log_root: str):
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("auto_start", False))
        self.shell_executable = str(cfg.get("shell_executable") or "/bin/bash").strip()
        self.activate_command = str(cfg.get("activate_command") or "").strip()
        self.workdir = str(cfg.get("workdir") or "").strip()
        self.model_path = str(cfg.get("model_path") or "").strip()
        self.served_model_name = str(cfg.get("served_model_name") or "").strip()
        self.bind_host = str(cfg.get("bind_host") or "0.0.0.0").strip()
        self.connect_host = str(cfg.get("connect_host") or "127.0.0.1").strip()
        self.base_port = int(cfg.get("base_port", 8000) or 8000)
        self.tensor_parallel_size = int(cfg.get("tensor_parallel_size", 1) or 1)
        self.reasoning_parser = str(cfg.get("reasoning_parser") or "").strip()
        self.allowed_local_media_path = str(cfg.get("allowed_local_media_path") or "").strip()
        self.startup_timeout_s = float(cfg.get("startup_timeout_s", 240.0) or 240.0)
        self.poll_interval_s = float(cfg.get("poll_interval_s", 2.0) or 2.0)
        self.reuse_existing = bool(cfg.get("reuse_existing", True))
        self.gpu_memory_utilization = float(cfg.get("gpu_memory_utilization", 0.9) or 0.9)
        self.adaptive_gpu_memory_utilization = bool(cfg.get("adaptive_gpu_memory_utilization", True))
        self.gpu_memory_utilization_margin_gb = float(cfg.get("gpu_memory_utilization_margin_gb", 1.0) or 0.0)
        self.extra_args = [str(x) for x in list(cfg.get("extra_args", []) or []) if str(x).strip()]
        self.log_dir = str(cfg.get("log_dir") or "") or os.path.join(str(log_root or ""), "vllm_services")
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        remote_b_cfg = dict(cfg.get("remote_b", {}) or {})
        self.remote_b_enabled = bool(remote_b_cfg.get("enabled", False))
        self.remote_b_ssh_host_alias = str(remote_b_cfg.get("ssh_host_alias") or "").strip()
        self.remote_b_ssh_host = str(remote_b_cfg.get("ssh_host") or "").strip()
        self.remote_b_ssh_user = str(remote_b_cfg.get("ssh_user") or "").strip()
        self.remote_b_ssh_port = int(remote_b_cfg.get("ssh_port", 0) or 0)
        self.remote_b_ssh_extra_args = [
            str(x) for x in list(remote_b_cfg.get("ssh_extra_args", []) or []) if str(x).strip()
        ]
        self.remote_b_remote_host = str(remote_b_cfg.get("remote_host") or "127.0.0.1").strip()
        self.remote_b_remote_base_port = int(remote_b_cfg.get("remote_base_port", 8008) or 8008)
        self.remote_b_local_forward_host = str(
            remote_b_cfg.get("local_forward_host") or self.connect_host or "127.0.0.1"
        ).strip()
        self.remote_b_local_forward_base_port = int(
            remote_b_cfg.get("local_forward_base_port", self.remote_b_remote_base_port)
            or self.remote_b_remote_base_port
        )
        self.remote_b_service_count = max(0, int(remote_b_cfg.get("service_count", 8) or 8))
        self.remote_b_wait_after_stop_s = float(remote_b_cfg.get("wait_after_stop_s", 12.0) or 0.0)
        self.remote_b_wait_after_start_s = float(remote_b_cfg.get("wait_after_start_s", 0.0) or 0.0)
        self.remote_b_startup_timeout_s = float(
            remote_b_cfg.get("startup_timeout_s", self.startup_timeout_s) or self.startup_timeout_s
        )
        self.remote_b_poll_interval_s = float(
            remote_b_cfg.get("poll_interval_s", self.poll_interval_s) or self.poll_interval_s
        )
        self.remote_b_tunnel_retry_delay_s = float(remote_b_cfg.get("tunnel_retry_delay_s", 2.0) or 0.0)
        self.remote_b_ssh_connect_timeout_s = float(remote_b_cfg.get("ssh_connect_timeout_s", 10.0) or 10.0)
        self.remote_b_ssh_server_alive_interval_s = int(
            remote_b_cfg.get("ssh_server_alive_interval_s", 15) or 0
        )
        self.remote_b_ssh_server_alive_count_max = int(
            remote_b_cfg.get("ssh_server_alive_count_max", 3) or 0
        )
        self.remote_b_stop_all_script = str(
            remote_b_cfg.get("stop_all_script") or "/root/weitong_lian/stop_all.sh"
        ).strip()
        self.remote_b_start_vllm_script = str(
            remote_b_cfg.get("start_vllm_8gpu_tmux_script") or "/root/weitong_lian/start_vllm_8gpu_tmux.sh"
        ).strip()
        self.remote_b_start_burn_all_script = str(
            remote_b_cfg.get("start_burn_all_script") or "/root/weitong_lian/start_burn_all.sh"
        ).strip()
        self.remote_b_heartbeat_enabled = bool(remote_b_cfg.get("heartbeat_enabled", False))
        self.remote_b_heartbeat_path = str(remote_b_cfg.get("heartbeat_path") or "").strip()
        self.remote_b_heartbeat_interval_s = float(remote_b_cfg.get("heartbeat_interval_s", 60.0) or 60.0)

        remote_c_cfg = dict(cfg.get("remote_c", {}) or {})
        self.remote_c_role = str(remote_c_cfg.get("role") or "vllm").strip().lower()
        self.remote_c_enabled = bool(remote_c_cfg.get("enabled", False)) and self.remote_c_role in {
            "",
            "vllm",
            "vlm",
            "split",
            "vllm_tools",
            "tools_vllm",
        }
        self.remote_c_ssh_host_alias = str(remote_c_cfg.get("ssh_host_alias") or "server-c").strip()
        self.remote_c_ssh_host = str(remote_c_cfg.get("ssh_host") or "").strip()
        self.remote_c_ssh_user = str(remote_c_cfg.get("ssh_user") or "").strip()
        self.remote_c_ssh_port = int(remote_c_cfg.get("ssh_port", 0) or 0)
        self.remote_c_ssh_extra_args = [
            str(x) for x in list(remote_c_cfg.get("ssh_extra_args", []) or []) if str(x).strip()
        ]
        self.remote_c_remote_host = str(remote_c_cfg.get("remote_host") or "127.0.0.1").strip()
        self.remote_c_remote_base_port = int(remote_c_cfg.get("remote_base_port", 8024) or 8024)
        self.remote_c_local_forward_host = str(
            remote_c_cfg.get("local_forward_host") or self.connect_host or "127.0.0.1"
        ).strip()
        self.remote_c_local_forward_base_port = int(
            remote_c_cfg.get("local_forward_base_port", self.remote_c_remote_base_port)
            or self.remote_c_remote_base_port
        )
        self.remote_c_service_count = max(0, int(remote_c_cfg.get("service_count", 4) or 4))
        self.remote_c_wait_after_stop_s = float(remote_c_cfg.get("wait_after_stop_s", 12.0) or 0.0)
        self.remote_c_wait_after_start_s = float(remote_c_cfg.get("wait_after_start_s", 0.0) or 0.0)
        self.remote_c_startup_timeout_s = float(
            remote_c_cfg.get("startup_timeout_s", self.startup_timeout_s) or self.startup_timeout_s
        )
        self.remote_c_poll_interval_s = float(
            remote_c_cfg.get("poll_interval_s", self.poll_interval_s) or self.poll_interval_s
        )
        self.remote_c_tunnel_retry_delay_s = float(remote_c_cfg.get("tunnel_retry_delay_s", 2.0) or 0.0)
        self.remote_c_ssh_connect_timeout_s = float(remote_c_cfg.get("ssh_connect_timeout_s", 10.0) or 10.0)
        self.remote_c_ssh_server_alive_interval_s = int(
            remote_c_cfg.get("ssh_server_alive_interval_s", 15) or 0
        )
        self.remote_c_ssh_server_alive_count_max = int(
            remote_c_cfg.get("ssh_server_alive_count_max", 3) or 0
        )
        self.remote_c_stop_all_script = str(
            remote_c_cfg.get("stop_all_script") or "/root/weitonglian/stop_4_vllm_qwen35.sh"
        ).strip()
        self.remote_c_start_vllm_script = str(
            remote_c_cfg.get("start_vllm_h200_3gpu_4x_qwen35_script")
            or remote_c_cfg.get("start_vllm_h200_script")
            or remote_c_cfg.get("start_vllm_3gpu_script")
            or remote_c_cfg.get("start_vllm_script")
            or remote_c_cfg.get("start_4_vllm_qwen35_script")
            or "/root/weitonglian/start_4_vllm_qwen35.sh"
        ).strip()
        self.remote_c_start_burn_all_script = str(
            remote_c_cfg.get("start_burn_all_script") or ""
        ).strip()
        self.remote_c_heartbeat_enabled = bool(remote_c_cfg.get("heartbeat_enabled", False))
        self.remote_c_heartbeat_path = str(remote_c_cfg.get("heartbeat_path") or "").strip()
        self.remote_c_heartbeat_interval_s = float(remote_c_cfg.get("heartbeat_interval_s", 60.0) or 60.0)

        self._services: Dict[str, VLLMServiceRecord] = {}
        self._close_lock = threading.RLock()
        self._close_started = False
        self._closed = False
        self._remote_heartbeat_stop_events: Dict[str, threading.Event] = {}
        self._remote_heartbeat_threads: Dict[str, threading.Thread] = {}
        self._remote_b_prepared = False
        self._remote_b_shutdown_done = False
        self._remote_b_cleanup_required = False
        self._remote_b_unavailable_devices: set[str] = set()
        self._remote_c_prepared = False
        self._remote_c_shutdown_done = False
        self._remote_c_cleanup_required = False
        self._remote_c_unavailable_devices: set[str] = set()
        self._register_runtime_hooks()

    @classmethod
    def shutdown_in_progress(cls) -> bool:
        return bool(cls._shutdown_signal_in_progress)

    def _shutdown_requested(self) -> bool:
        return bool(type(self)._shutdown_signal_in_progress or self._close_started or self._closed)

    def _remote_cluster_names(self) -> List[str]:
        names: List[str] = []
        if self.remote_b_enabled:
            names.append("remote_b")
        if self.remote_c_enabled:
            names.append("remote_c")
        return names

    def _remote_cluster_enabled(self, cluster: str) -> bool:
        return bool(getattr(self, f"{cluster}_enabled", False))

    def _remote_cluster_label(self, cluster: str) -> str:
        return f"[vLLMService][{cluster}]"

    def _remote_cluster_ssh_host_alias(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_ssh_host_alias", "") or "").strip()

    def _remote_cluster_ssh_destination(self, cluster: str) -> str:
        host = str(getattr(self, f"{cluster}_ssh_host", "") or "").strip()
        user = str(getattr(self, f"{cluster}_ssh_user", "") or "").strip()
        if host:
            return f"{user}@{host}" if user else host
        return self._remote_cluster_ssh_host_alias(cluster)

    def _remote_cluster_ssh_port(self, cluster: str) -> int:
        return max(0, int(getattr(self, f"{cluster}_ssh_port", 0) or 0))

    def _remote_cluster_ssh_extra_args(self, cluster: str) -> List[str]:
        raw = getattr(self, f"{cluster}_ssh_extra_args", []) or []
        return [str(x) for x in list(raw) if str(x).strip()]

    def _remote_cluster_ssh_base_command(self, cluster: str) -> List[str]:
        destination = self._remote_cluster_ssh_destination(cluster)
        if not destination:
            raise ValueError(f"{cluster} config requires ssh_host_alias or ssh_host")
        command = ["ssh"]
        command.extend(self._remote_cluster_ssh_extra_args(cluster))
        port = self._remote_cluster_ssh_port(cluster)
        if port > 0:
            command.extend(["-p", str(port)])
        command.append(destination)
        return command

    def _remote_cluster_remote_host(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_remote_host", "127.0.0.1") or "127.0.0.1").strip()

    def _remote_cluster_remote_base_port(self, cluster: str) -> int:
        return int(getattr(self, f"{cluster}_remote_base_port", 0) or 0)

    def _remote_cluster_local_forward_host(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_local_forward_host", self.connect_host) or self.connect_host).strip()

    def _remote_cluster_local_forward_base_port(self, cluster: str) -> int:
        return int(
            getattr(
                self,
                f"{cluster}_local_forward_base_port",
                self._remote_cluster_remote_base_port(cluster),
            )
            or self._remote_cluster_remote_base_port(cluster)
        )

    def _remote_cluster_service_count(self, cluster: str) -> int:
        return max(0, int(getattr(self, f"{cluster}_service_count", 0) or 0))

    def _remote_cluster_wait_after_stop_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_wait_after_stop_s", 0.0) or 0.0)

    def _remote_cluster_wait_after_start_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_wait_after_start_s", 0.0) or 0.0)

    def _remote_cluster_startup_timeout_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_startup_timeout_s", self.startup_timeout_s) or self.startup_timeout_s)

    def _remote_cluster_poll_interval_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_poll_interval_s", self.poll_interval_s) or self.poll_interval_s)

    def _remote_cluster_tunnel_retry_delay_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_tunnel_retry_delay_s", 2.0) or 0.0)

    def _remote_cluster_ssh_connect_timeout_s(self, cluster: str) -> float:
        return float(getattr(self, f"{cluster}_ssh_connect_timeout_s", 10.0) or 10.0)

    def _remote_cluster_ssh_server_alive_interval_s(self, cluster: str) -> int:
        return int(getattr(self, f"{cluster}_ssh_server_alive_interval_s", 15) or 0)

    def _remote_cluster_ssh_server_alive_count_max(self, cluster: str) -> int:
        return int(getattr(self, f"{cluster}_ssh_server_alive_count_max", 3) or 0)

    def _remote_cluster_stop_script(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_stop_all_script", "") or "").strip()

    def _remote_cluster_start_script(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_start_vllm_script", "") or "").strip()

    def _remote_cluster_burn_script(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_start_burn_all_script", "") or "").strip()

    def _remote_cluster_heartbeat_enabled(self, cluster: str) -> bool:
        return bool(getattr(self, f"{cluster}_heartbeat_enabled", False))

    def _remote_cluster_heartbeat_path(self, cluster: str) -> str:
        return str(getattr(self, f"{cluster}_heartbeat_path", "") or "").strip()

    def _remote_cluster_heartbeat_interval_s(self, cluster: str) -> float:
        return max(5.0, float(getattr(self, f"{cluster}_heartbeat_interval_s", 60.0) or 60.0))

    def _remote_cluster_prepared(self, cluster: str) -> bool:
        return bool(getattr(self, f"_{cluster}_prepared", False))

    def _set_remote_cluster_prepared(self, cluster: str, value: bool) -> None:
        setattr(self, f"_{cluster}_prepared", bool(value))

    def _remote_cluster_shutdown_done(self, cluster: str) -> bool:
        return bool(getattr(self, f"_{cluster}_shutdown_done", False))

    def _set_remote_cluster_shutdown_done(self, cluster: str, value: bool) -> None:
        setattr(self, f"_{cluster}_shutdown_done", bool(value))

    def _remote_cluster_cleanup_required(self, cluster: str) -> bool:
        return bool(getattr(self, f"_{cluster}_cleanup_required", False))

    def _set_remote_cluster_cleanup_required(self, cluster: str, value: bool) -> None:
        setattr(self, f"_{cluster}_cleanup_required", bool(value))

    def _remote_cluster_unavailable_devices(self, cluster: str) -> set[str]:
        value = getattr(self, f"_{cluster}_unavailable_devices", None)
        if value is None:
            value = set()
            setattr(self, f"_{cluster}_unavailable_devices", value)
        return value

    def can_spawn(self) -> bool:
        shell = str(self.shell_executable or "").strip()
        if not shell:
            return False
        if os.path.isabs(shell):
            return os.path.exists(shell)
        return shutil.which(shell) is not None

    def active_service_count(self) -> int:
        return int(len(self._services))

    def local_active_service_count(self) -> int:
        return int(len(self.active_local_devices()))

    def remote_active_service_count(self) -> int:
        return int(len(self.active_remote_devices()))

    def remote_service_target_count(self) -> int:
        return int(sum(self._remote_cluster_service_count(cluster) for cluster in self._remote_cluster_names()))

    def active_devices(self) -> List[str]:
        return [str(x) for x in self._services.keys()]

    def active_local_devices(self) -> List[str]:
        return [device for device in self.active_devices() if not self._is_remote_device(device)]

    def active_remote_devices(self) -> List[str]:
        return [device for device in self.active_devices() if self._is_remote_device(device)]

    def endpoint_for_device(self, device: str) -> str:
        record = self._services.get(str(device))
        return str(record.base_url) if record is not None else ""

    def ensure_services(self, devices: List[str]) -> Dict[str, str]:
        endpoints: Dict[str, str] = {}
        if not self.enabled or self._shutdown_requested():
            return endpoints

        targets: List[str] = []
        for device in list(devices or []):
            normalized = str(device or "").strip()
            if normalized and normalized not in targets:
                targets.append(normalized)
        for cluster in self._remote_cluster_names():
            for remote_device in self._remote_device_names(cluster):
                if remote_device not in targets:
                    targets.append(remote_device)

        print(f"[vLLMService] ensure target devices={targets}")
        for device in targets:
            if self._shutdown_requested():
                return endpoints
            try:
                endpoint = self._ensure_single_service(device)
            except Exception as exc:
                cluster = self._remote_cluster_for_device(device)
                if cluster is not None:
                    self._remote_cluster_unavailable_devices(cluster).add(str(device))
                    print(
                        f"{self._remote_cluster_label(cluster)} skip unavailable device={device}: {exc}"
                    )
                    continue
                raise
            if endpoint:
                endpoints[device] = endpoint
        return endpoints

    def close(self) -> None:
        with self._close_lock:
            if self._closed or self._close_started:
                return
            self._close_started = True
            try:
                records = list(self._services.values())
                self._services.clear()
                local_records = [record for record in records if str(record.kind or "local") == "local"]
                deferred_records = [record for record in records if str(record.kind or "local") != "local"]

                for record in local_records:
                    self._terminate_record_safely(record)

                for cluster in self._remote_cluster_names():
                    self._shutdown_remote_cluster(cluster)

                for record in deferred_records:
                    self._terminate_record_safely(record)
                self._closed = True
            finally:
                self._close_started = False

    def _register_runtime_hooks(self) -> None:
        cls = type(self)
        with cls._signal_lock:
            cls._registered_managers.add(self)
            if not cls._atexit_installed:
                atexit.register(cls._close_registered_managers)
                cls._atexit_installed = True
            if cls._signal_handlers_installed:
                return
            if threading.current_thread() is not threading.main_thread():
                return
            for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
                if sig is None:
                    continue
                try:
                    cls._previous_signal_handlers[int(sig)] = signal.getsignal(sig)
                    signal.signal(sig, cls._handle_shutdown_signal)
                except Exception:
                    continue
            cls._signal_handlers_installed = True

    @classmethod
    def _close_registered_managers(cls) -> None:
        with cls._signal_lock:
            managers = list(cls._registered_managers)
        for manager in managers:
            try:
                manager.close()
            except Exception:
                pass

    @classmethod
    def _handle_shutdown_signal(cls, signum: int, frame: object) -> None:
        with cls._signal_lock:
            if cls._shutdown_signal_in_progress:
                print(
                    f"[vLLMService] shutdown already in progress for signal={int(signum)}; "
                    "ignoring duplicate signal until cleanup completes."
                )
                return
            cls._shutdown_signal_in_progress = True
        cls._close_registered_managers()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        try:
            signal.signal(int(signum), signal.SIG_DFL)
        except Exception:
            pass
        try:
            os.kill(os.getpid(), int(signum))
        except Exception:
            raise SystemExit(128 + int(signum))
        return

    def _ensure_single_service(self, device: str) -> str:
        if self._shutdown_requested():
            return ""
        if self._is_remote_b_device(device):
            return self._ensure_remote_b_service(device)
        if self._is_remote_c_device(device):
            return self._ensure_remote_c_service(device)
        return self._ensure_local_service(device)

    def _ensure_local_service(self, device: str) -> str:
        with self._close_lock:
            if self._shutdown_requested():
                return ""
            existing = self._services.get(device)
            if existing is not None:
                if self._record_alive(existing):
                    return existing.base_url
                self._terminate_record(existing)
                self._services.pop(device, None)

            port = self._port_for_device(device)
            base_url = f"http://{self.connect_host}:{port}/v1"
            log_path = os.path.join(self.log_dir, f"vllm_{self._safe_device_name(device)}_{port}.log")
            pid_path = self._pid_path(device, port)
            managed_pid = self._load_managed_pid(pid_path)

            if self.reuse_existing and self._wait_until_ready(base_url, timeout_s=5.0):
                if managed_pid is not None:
                    print(
                        f"[vLLMService] adopt managed existing device={device} port={port} "
                        f"pid={managed_pid} base_url={base_url} log={log_path}"
                    )
                else:
                    print(
                        f"[vLLMService] reuse existing device={device} port={port} base_url={base_url} "
                        f"log={log_path}"
                    )
                record = VLLMServiceRecord(
                    device=device,
                    port=port,
                    base_url=base_url,
                    log_path=log_path,
                    process=None,
                    owned=managed_pid is not None,
                    pid_path=pid_path,
                    launcher_pid=managed_pid,
                    kind="local",
                )
                self._services[device] = record
                return base_url

            if managed_pid is not None:
                print(
                    f"[vLLMService] terminating stale managed service device={device} port={port} "
                    f"pid={managed_pid}"
                )
                self._terminate_record(
                    VLLMServiceRecord(
                        device=device,
                        port=port,
                        base_url=base_url,
                        log_path=log_path,
                        process=None,
                        owned=True,
                        pid_path=pid_path,
                        launcher_pid=managed_pid,
                        kind="local",
                    )
                )

            command = self._build_shell_command(device=device, port=port)
            print(
                f"[vLLMService] starting device={device} port={port} model={self.served_model_name} "
                f"log={log_path}"
            )
            if self._shutdown_requested():
                return ""
            log_fh = open(log_path, "a", encoding="utf-8")
            kwargs = {
                "cwd": self.workdir or None,
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
            if os.name != "nt":
                kwargs["start_new_session"] = True
            try:
                process = subprocess.Popen([self.shell_executable, "-lc", command], **kwargs)
            finally:
                log_fh.close()
            record = VLLMServiceRecord(
                device=device,
                port=port,
                base_url=base_url,
                log_path=log_path,
                process=process,
                owned=True,
                pid_path=pid_path,
                launcher_pid=int(process.pid),
                kind="local",
            )
            self._services[device] = record
            self._write_managed_pid(record)

            if not self._wait_until_ready(base_url, timeout_s=self.startup_timeout_s, process=process):
                if self._shutdown_requested():
                    self._terminate_record(record)
                    self._services.pop(device, None)
                    return ""
                print(
                    f"[vLLMService] startup failed device={device} port={port}; "
                    f"see log={log_path}"
                )
                self._terminate_record(record)
                self._services.pop(device, None)
                raise RuntimeError(
                    f"vLLM service failed to become ready for device={device} port={port}. "
                    f"See log: {log_path}"
                )

            print(f"[vLLMService] ready device={device} port={port} base_url={base_url}")
            return base_url

    def _ensure_remote_b_service(self, device: str) -> str:
        return self._ensure_remote_service("remote_b", device, ensure_cluster_fn=self._ensure_remote_b_cluster)

    def _ensure_remote_c_service(self, device: str) -> str:
        return self._ensure_remote_service("remote_c", device, ensure_cluster_fn=self._ensure_remote_c_cluster)

    def _ensure_remote_service(
        self,
        cluster: str,
        device: str,
        *,
        ensure_cluster_fn=None,
    ) -> str:
        with self._close_lock:
            if self._shutdown_requested():
                return ""
            existing = self._services.get(device)
            if existing is not None:
                if self._record_alive(existing):
                    return existing.base_url
                self._terminate_record(existing)
                self._services.pop(device, None)

            if callable(ensure_cluster_fn):
                ensure_cluster_fn()
            else:
                self._ensure_remote_cluster(cluster)

            remote_port = self._remote_remote_port_for_device(cluster, device)
            local_port = self._remote_local_port_for_device(cluster, device)
            base_url = f"http://{self._remote_cluster_local_forward_host(cluster)}:{local_port}/v1"
            log_path = os.path.join(self.log_dir, f"{cluster}_{self._safe_device_name(device)}_{local_port}.log")
            pid_path = self._pid_path(device, local_port)
            managed_pid = self._load_managed_pid(pid_path)
            label = self._remote_cluster_label(cluster)
            kind = f"{cluster}_tunnel"

            if self.reuse_existing and self._wait_until_ready(base_url, timeout_s=5.0):
                if managed_pid is not None:
                    print(
                        f"{label} adopt managed existing tunnel device={device} "
                        f"local_port={local_port} pid={managed_pid} base_url={base_url} log={log_path}"
                    )
                else:
                    print(
                        f"{label} reuse existing tunnel device={device} "
                        f"local_port={local_port} base_url={base_url} log={log_path}"
                    )
                record = VLLMServiceRecord(
                    device=device,
                    port=local_port,
                    base_url=base_url,
                    log_path=log_path,
                    process=None,
                    owned=managed_pid is not None,
                    pid_path=pid_path,
                    launcher_pid=managed_pid,
                    kind=kind,
                )
                self._services[device] = record
                return base_url

            if managed_pid is not None:
                print(
                    f"{label} terminating stale ssh tunnel device={device} "
                    f"local_port={local_port} pid={managed_pid}"
                )
                self._terminate_record(
                    VLLMServiceRecord(
                        device=device,
                        port=local_port,
                        base_url=base_url,
                        log_path=log_path,
                        process=None,
                        owned=True,
                        pid_path=pid_path,
                        launcher_pid=managed_pid,
                        kind=kind,
                    )
                )

            command, _use_shell = self._build_remote_ssh_tunnel_launch(
                cluster=cluster,
                device=device,
                local_port=local_port,
                remote_port=remote_port,
            )
            print(
                f"{label} starting tunnel device={device} local_port={local_port} "
                f"remote_port={remote_port} base_url={base_url} log={log_path}"
            )
            if self._shutdown_requested():
                return ""
            log_fh = open(log_path, "a", encoding="utf-8")
            kwargs = {
                "stdout": log_fh,
                "stderr": subprocess.STDOUT,
                "text": True,
            }
            if os.name != "nt":
                kwargs["start_new_session"] = True
                kwargs["preexec_fn"] = self._wrapper_preexec_fn()
            try:
                process = subprocess.Popen(command, **kwargs)
            finally:
                log_fh.close()

            record = VLLMServiceRecord(
                device=device,
                port=local_port,
                base_url=base_url,
                log_path=log_path,
                process=process,
                owned=True,
                pid_path=pid_path,
                launcher_pid=int(process.pid),
                kind=kind,
            )
            self._services[device] = record
            self._write_managed_pid(record)

            if not self._wait_until_ready(
                base_url,
                timeout_s=0.0,
                process=process,
                poll_interval_s=self._remote_cluster_poll_interval_s(cluster),
            ):
                if self._shutdown_requested():
                    self._terminate_record(record)
                    self._services.pop(device, None)
                    return ""
                print(
                    f"{label} startup failed device={device} local_port={local_port}; "
                    f"see log={log_path}"
                )
                self._terminate_record(record)
                self._services.pop(device, None)
                raise RuntimeError(
                    f"{cluster} vLLM service failed to become ready for device={device} "
                    f"remote_port={remote_port}. See log: {log_path}"
                )

            print(
                f"{label} ready device={device} local_port={local_port} "
                f"remote_port={remote_port} base_url={base_url}"
            )
            return base_url

    def _is_remote_b_device(self, device: str) -> bool:
        return self._is_remote_cluster_device("remote_b", device)

    def _is_remote_c_device(self, device: str) -> bool:
        return self._is_remote_cluster_device("remote_c", device)

    def _is_remote_device(self, device: str) -> bool:
        return self._remote_cluster_for_device(device) is not None

    def _is_remote_cluster_device(self, cluster: str, device: str) -> bool:
        return str(device or "").strip().lower().startswith(f"{cluster}:")

    def _remote_cluster_for_device(self, device: str) -> Optional[str]:
        text = str(device or "").strip().lower()
        if ":" not in text:
            return None
        prefix = text.split(":", 1)[0]
        if prefix in {"remote_b", "remote_c"}:
            return prefix
        return None

    def _remote_b_device_names(self) -> List[str]:
        return self._remote_device_names("remote_b")

    def _remote_c_device_names(self) -> List[str]:
        return self._remote_device_names("remote_c")

    def _remote_device_names(self, cluster: str) -> List[str]:
        unavailable = self._remote_cluster_unavailable_devices(cluster)
        return [
            f"{cluster}:{idx}"
            for idx in range(self._remote_cluster_service_count(cluster))
            if f"{cluster}:{idx}" not in unavailable
        ]

    def _remote_b_device_index(self, device: str) -> int:
        return self._remote_device_index(device)

    def _remote_c_device_index(self, device: str) -> int:
        return self._remote_device_index(device)

    def _remote_device_index(self, device: str) -> int:
        text = str(device or "").strip().lower()
        if ":" not in text:
            return 0
        try:
            return max(0, int(text.split(":", 1)[1]))
        except Exception:
            return 0

    def _remote_b_remote_port_for_device(self, device: str) -> int:
        return self._remote_remote_port_for_device("remote_b", device)

    def _remote_c_remote_port_for_device(self, device: str) -> int:
        return self._remote_remote_port_for_device("remote_c", device)

    def _remote_remote_port_for_device(self, cluster: str, device: str) -> int:
        return int(self._remote_cluster_remote_base_port(cluster) + self._remote_device_index(device))

    def _remote_b_local_port_for_device(self, device: str) -> int:
        return self._remote_local_port_for_device("remote_b", device)

    def _remote_c_local_port_for_device(self, device: str) -> int:
        return self._remote_local_port_for_device("remote_c", device)

    def _remote_local_port_for_device(self, cluster: str, device: str) -> int:
        return int(self._remote_cluster_local_forward_base_port(cluster) + self._remote_device_index(device))

    def _ensure_remote_b_cluster(self) -> None:
        self._ensure_remote_cluster("remote_b")

    def _ensure_remote_c_cluster(self) -> None:
        self._ensure_remote_cluster("remote_c")

    def _ensure_remote_cluster(self, cluster: str) -> None:
        with self._close_lock:
            if not self._remote_cluster_enabled(cluster) or self._shutdown_requested():
                return
            if self._remote_cluster_prepared(cluster) and not self._remote_cluster_shutdown_done(cluster):
                return

            self._set_remote_cluster_shutdown_done(cluster, False)
            self._set_remote_cluster_cleanup_required(cluster, True)
            self._remote_cluster_unavailable_devices(cluster).clear()
            self._cleanup_stale_remote_wrappers(cluster)

            print(
                f"{self._remote_cluster_label(cluster)} stop_all via ssh target="
                f"{self._remote_cluster_ssh_destination(cluster) or '<missing>'}"
            )
            self._dispatch_remote_script(cluster, self._remote_cluster_stop_script(cluster), check=False)
            if self._remote_cluster_wait_after_stop_s(cluster) > 0:
                print(
                    f"{self._remote_cluster_label(cluster)} waiting "
                    f"{self._remote_cluster_wait_after_stop_s(cluster):.0f}s after stop_all"
                )
                self._sleep_until_or_shutdown(self._remote_cluster_wait_after_stop_s(cluster))
            if self._shutdown_requested():
                return

            print(
                f"{self._remote_cluster_label(cluster)} start service script via ssh target="
                f"{self._remote_cluster_ssh_destination(cluster)}"
            )
            self._dispatch_remote_script(cluster, self._remote_cluster_start_script(cluster), check=True)
            if self._remote_cluster_wait_after_start_s(cluster) > 0:
                print(
                    f"{self._remote_cluster_label(cluster)} waiting "
                    f"{self._remote_cluster_wait_after_start_s(cluster):.0f}s after start script"
                )
                self._sleep_until_or_shutdown(self._remote_cluster_wait_after_start_s(cluster))
            if self._shutdown_requested():
                return

            self._set_remote_cluster_prepared(cluster, True)
            self._start_remote_cluster_heartbeat(cluster)

    def _shutdown_remote_b_cluster(self) -> None:
        self._shutdown_remote_cluster("remote_b")

    def _shutdown_remote_c_cluster(self) -> None:
        self._shutdown_remote_cluster("remote_c")

    def _shutdown_remote_cluster(self, cluster: str) -> None:
        with self._close_lock:
            if not self._remote_cluster_enabled(cluster):
                return
            if self._remote_cluster_shutdown_done(cluster) or not self._remote_cluster_cleanup_required(cluster):
                self._stop_remote_cluster_heartbeat(cluster)
                return

            self._stop_remote_cluster_heartbeat(cluster)
            print(
                f"{self._remote_cluster_label(cluster)} stop_all during shutdown via ssh target="
                f"{self._remote_cluster_ssh_destination(cluster)}"
            )
            try:
                self._dispatch_remote_script(cluster, self._remote_cluster_stop_script(cluster), check=False)
            except Exception as exc:
                print(f"{self._remote_cluster_label(cluster)} stop_all during shutdown failed: {exc}")
            burn_script = self._remote_cluster_burn_script(cluster)
            if burn_script and self._remote_cluster_wait_after_stop_s(cluster) > 0:
                print(
                    f"{self._remote_cluster_label(cluster)} waiting "
                    f"{self._remote_cluster_wait_after_stop_s(cluster):.0f}s before start_burn_all"
                )
                time.sleep(self._remote_cluster_wait_after_stop_s(cluster))
            if burn_script:
                print(
                    f"{self._remote_cluster_label(cluster)} start_burn_all during shutdown via ssh target="
                    f"{self._remote_cluster_ssh_destination(cluster)}"
                )
                try:
                    self._dispatch_remote_script(cluster, burn_script, check=False)
                except Exception as exc:
                    print(f"{self._remote_cluster_label(cluster)} start_burn_all during shutdown failed: {exc}")
            self._set_remote_cluster_shutdown_done(cluster, True)
            self._set_remote_cluster_cleanup_required(cluster, False)
            self._set_remote_cluster_prepared(cluster, False)
            self._remote_cluster_unavailable_devices(cluster).clear()

    def _start_remote_cluster_heartbeat(self, cluster: str) -> None:
        if not self._remote_cluster_heartbeat_enabled(cluster):
            return
        heartbeat_path = self._remote_cluster_heartbeat_path(cluster)
        if not heartbeat_path:
            print(f"{self._remote_cluster_label(cluster)} heartbeat enabled but heartbeat_path is empty")
            return
        thread = self._remote_heartbeat_threads.get(cluster)
        if thread is not None and thread.is_alive():
            return

        stop_event = threading.Event()
        self._remote_heartbeat_stop_events[cluster] = stop_event
        try:
            self._touch_remote_cluster_heartbeat(cluster, heartbeat_path)
        except Exception as exc:
            print(f"{self._remote_cluster_label(cluster)} initial heartbeat touch failed: {exc}")
        thread = threading.Thread(
            target=self._remote_cluster_heartbeat_loop,
            args=(cluster, heartbeat_path, stop_event),
            name=f"vllm-{cluster}-heartbeat",
            daemon=True,
        )
        self._remote_heartbeat_threads[cluster] = thread
        thread.start()

    def _stop_remote_cluster_heartbeat(self, cluster: str) -> None:
        stop_event = self._remote_heartbeat_stop_events.pop(cluster, None)
        if stop_event is not None:
            stop_event.set()
        thread = self._remote_heartbeat_threads.pop(cluster, None)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def _remote_cluster_heartbeat_loop(
        self,
        cluster: str,
        heartbeat_path: str,
        stop_event: threading.Event,
    ) -> None:
        interval_s = self._remote_cluster_heartbeat_interval_s(cluster)
        while not stop_event.wait(interval_s):
            if self._shutdown_requested():
                return
            try:
                self._touch_remote_cluster_heartbeat(cluster, heartbeat_path)
            except Exception as exc:
                print(f"{self._remote_cluster_label(cluster)} heartbeat touch failed: {exc}")

    def _touch_remote_cluster_heartbeat(self, cluster: str, heartbeat_path: str) -> None:
        quoted_path = shlex.quote(str(heartbeat_path or "").strip())
        remote_command = f"mkdir -p \"$(dirname {quoted_path})\" && date +%s > {quoted_path}"
        self._run_remote_command(cluster, remote_command, check=False)

    def _run_remote_b_command(self, remote_command: str, *, check: bool) -> subprocess.CompletedProcess:
        return self._run_remote_command("remote_b", remote_command, check=check)

    def _run_remote_c_command(self, remote_command: str, *, check: bool) -> subprocess.CompletedProcess:
        return self._run_remote_command("remote_c", remote_command, check=check)

    def _run_remote_command(self, cluster: str, remote_command: str, *, check: bool) -> subprocess.CompletedProcess:
        if shutil.which("ssh") is None:
            raise RuntimeError(f"{cluster} control requires ssh to be available on PATH")

        completed = subprocess.run(
            [*self._remote_cluster_ssh_base_command(cluster), "bash", "-lc", shlex.quote(remote_command)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if check and int(completed.returncode or 0) != 0:
            raise RuntimeError(
                f"{cluster} command failed with return code {completed.returncode}. "
                f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
            )
        return completed

    def _run_remote_b_script(self, script_path: str, *, check: bool) -> subprocess.CompletedProcess:
        return self._run_remote_script_impl("remote_b", script_path, check=check)

    def _run_remote_c_script(self, script_path: str, *, check: bool) -> subprocess.CompletedProcess:
        return self._run_remote_script_impl("remote_c", script_path, check=check)

    def _dispatch_remote_script(self, cluster: str, script_path: str, *, check: bool) -> subprocess.CompletedProcess:
        if cluster == "remote_b":
            return self._run_remote_b_script(script_path, check=check)
        if cluster == "remote_c":
            return self._run_remote_c_script(script_path, check=check)
        return self._run_remote_script_impl(cluster, script_path, check=check)

    def _run_remote_script_impl(self, cluster: str, script_path: str, *, check: bool) -> subprocess.CompletedProcess:
        quoted_script = shlex.quote(str(script_path or "").strip())
        remote_command = (
            f"set -euo pipefail; "
            f"test -f {quoted_script}; "
            f"chmod +x {quoted_script}; "
            f"{quoted_script}"
        )
        return self._run_remote_command(cluster, remote_command, check=check)

    def _build_ssh_tunnel_command(self, *, local_port: int, remote_port: int) -> List[str]:
        return self._build_remote_ssh_tunnel_command("remote_b", local_port=local_port, remote_port=remote_port)

    def _build_remote_c_ssh_tunnel_command(self, *, local_port: int, remote_port: int) -> List[str]:
        return self._build_remote_ssh_tunnel_command("remote_c", local_port=local_port, remote_port=remote_port)

    def _build_remote_ssh_tunnel_command(self, cluster: str, *, local_port: int, remote_port: int) -> List[str]:
        command = self._remote_cluster_ssh_base_command(cluster)
        ssh_destination = command.pop()
        command.extend([
            "-o",
            "ExitOnForwardFailure=yes",
        ])
        connect_timeout_s = max(1, int(self._remote_cluster_ssh_connect_timeout_s(cluster)))
        if connect_timeout_s > 0:
            command.extend(["-o", f"ConnectTimeout={connect_timeout_s}"])
        server_alive_interval_s = max(0, int(self._remote_cluster_ssh_server_alive_interval_s(cluster)))
        if server_alive_interval_s > 0:
            command.extend(["-o", f"ServerAliveInterval={server_alive_interval_s}"])
        server_alive_count_max = max(0, int(self._remote_cluster_ssh_server_alive_count_max(cluster)))
        if server_alive_count_max > 0:
            command.extend(["-o", f"ServerAliveCountMax={server_alive_count_max}"])
        command.extend(
            [
                "-o",
                "TCPKeepAlive=yes",
                "-N",
                "-L",
                (
                    f"{self._remote_cluster_local_forward_host(cluster)}:{int(local_port)}:"
                    f"{self._remote_cluster_remote_host(cluster)}:{int(remote_port)}"
                ),
                ssh_destination,
            ]
        )
        return command

    def _build_ssh_tunnel_launch(
        self,
        *,
        device: str,
        local_port: int,
        remote_port: int,
    ) -> tuple[List[str], bool]:
        return self._build_remote_ssh_tunnel_launch(
            cluster="remote_b",
            device=device,
            local_port=local_port,
            remote_port=remote_port,
        )

    def _build_remote_c_ssh_tunnel_launch(
        self,
        *,
        device: str,
        local_port: int,
        remote_port: int,
    ) -> tuple[List[str], bool]:
        return self._build_remote_ssh_tunnel_launch(
            cluster="remote_c",
            device=device,
            local_port=local_port,
            remote_port=remote_port,
        )

    def _build_remote_ssh_tunnel_launch(
        self,
        *,
        cluster: str,
        device: str,
        local_port: int,
        remote_port: int,
    ) -> tuple[List[str], bool]:
        ssh_command = self._build_remote_ssh_tunnel_command(cluster, local_port=local_port, remote_port=remote_port)
        shell = str(self.shell_executable or "").strip()
        if not shell:
            return ssh_command, False

        retry_delay_s = max(0.1, float(self._remote_cluster_tunnel_retry_delay_s(cluster) or 0.0))
        quoted_command = " ".join(shlex.quote(part) for part in ssh_command)
        retry_message = (
            f"{self._remote_cluster_label(cluster)} tunnel exited device={device} "
            f"local_port={int(local_port)} remote_port={int(remote_port)}; "
            f"retrying in {retry_delay_s:.1f}s"
        )
        loop_command = (
            "while true; do "
            f"{quoted_command}; "
            "exit_code=$?; "
            f"echo {shlex.quote(retry_message)} \"(exit=${{exit_code}})\"; "
            f"sleep {retry_delay_s:.1f}; "
            "done"
        )
        return [shell, "-lc", loop_command], True

    def _wrapper_preexec_fn(self) -> Optional[Callable[[], None]]:
        if os.name == "nt":
            return None

        parent_pid = os.getpid()

        def _preexec() -> None:
            try:
                import ctypes

                parent_now = os.getppid()
                if parent_now != parent_pid:
                    os._exit(1)
                libc = ctypes.CDLL(None)
                pr_set_pdeathsig = 1
                if libc.prctl(pr_set_pdeathsig, signal.SIGTERM, 0, 0, 0) != 0:
                    raise OSError("prctl(PR_SET_PDEATHSIG) failed")
                if os.getppid() != parent_pid:
                    os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass

        return _preexec

    def _cleanup_stale_remote_b_wrappers(self) -> None:
        self._cleanup_stale_remote_wrappers("remote_b")

    def _cleanup_stale_remote_c_wrappers(self) -> None:
        self._cleanup_stale_remote_wrappers("remote_c")

    def _cleanup_stale_remote_wrappers(self, cluster: str) -> None:
        if os.name == "nt":
            return
        ssh_alias = self._remote_cluster_ssh_destination(cluster)
        if not ssh_alias:
            return
        try:
            completed = subprocess.run(
                ["ps", "-eo", "pid=,ppid=,args="],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except Exception:
            return
        if int(completed.returncode or 0) != 0:
            return

        stale_pids: list[int] = []
        for raw_line in str(completed.stdout or "").splitlines():
            line = str(raw_line or "").strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            try:
                pid = int(parts[0])
                ppid = int(parts[1])
            except Exception:
                continue
            command = parts[2]
            if ppid != 1:
                continue
            if "while true; do ssh " not in command:
                continue
            if ssh_alias not in command:
                continue
            if "-N -L " not in command:
                continue
            stale_pids.append(pid)

        if not stale_pids:
            return

        for pid in stale_pids:
            try:
                os.killpg(pid, signal.SIGKILL)
                print(f"{self._remote_cluster_label(cluster)} cleaned stale orphan wrapper pid={pid}")
            except Exception as exc:
                print(f"{self._remote_cluster_label(cluster)} failed to clean stale orphan wrapper pid={pid}: {exc}")

        for idx in range(self._remote_cluster_service_count(cluster)):
            local_port = self._remote_cluster_local_forward_base_port(cluster) + idx
            self._cleanup_pid_path(self._pid_path(f"{cluster}:{idx}", local_port))

    def _port_for_device(self, device: str) -> int:
        gpu_idx = self._device_index(device)
        return int(self.base_port + max(0, gpu_idx))

    def _device_index(self, device: str) -> int:
        text = str(device or "").strip().lower()
        if ":" in text:
            try:
                return int(text.split(":", 1)[1])
            except Exception:
                return 0
        if text.isdigit():
            return int(text)
        return 0

    def _safe_device_name(self, device: str) -> str:
        return str(device or "").replace(":", "_").replace("/", "_")

    def _pid_path(self, device: str, port: int) -> str:
        return os.path.join(self.log_dir, f"vllm_{self._safe_device_name(device)}_{port}.pid.json")

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    def _cleanup_pid_path(self, pid_path: str) -> None:
        try:
            if pid_path and os.path.exists(pid_path):
                os.remove(pid_path)
        except Exception:
            pass

    def _load_managed_pid(self, pid_path: str) -> Optional[int]:
        if not pid_path or not os.path.exists(pid_path):
            return None
        try:
            payload = json.loads(Path(pid_path).read_text(encoding="utf-8"))
            pid = int(payload.get("launcher_pid", 0) or 0)
        except Exception:
            self._cleanup_pid_path(pid_path)
            return None
        if pid <= 0 or not self._pid_alive(pid):
            self._cleanup_pid_path(pid_path)
            return None
        return pid

    def _write_managed_pid(self, record: VLLMServiceRecord) -> None:
        pid_path = str(record.pid_path or "").strip()
        launcher_pid = int(record.launcher_pid or 0)
        if not pid_path or launcher_pid <= 0:
            return
        payload = {
            "device": str(record.device),
            "port": int(record.port),
            "base_url": str(record.base_url),
            "log_path": str(record.log_path),
            "kind": str(record.kind),
            "launcher_pid": launcher_pid,
        }
        try:
            Path(pid_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _query_gpu_memory_gb(self, device: str) -> Optional[tuple[float, float]]:
        gpu_idx = self._device_index(device)
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_idx),
                    "--query-gpu=memory.free,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            line = str(proc.stdout or "").strip().splitlines()[0]
            free_mib_str, total_mib_str = [x.strip() for x in line.split(",", 1)]
            free_gb = float(free_mib_str) / 1024.0
            total_gb = float(total_mib_str) / 1024.0
            if total_gb > 0:
                return free_gb, total_gb
        except Exception:
            pass

        try:
            import torch

            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_idx)
                gib = 1024 ** 3
                total_gb = float(total_bytes) / gib
                if total_gb > 0:
                    return float(free_bytes) / gib, total_gb
        except Exception:
            pass

        return None

    def _resolve_gpu_memory_utilization(self, device: str) -> tuple[float, Optional[tuple[float, float]]]:
        base = min(0.98, max(0.05, float(self.gpu_memory_utilization or 0.9)))
        if not self.adaptive_gpu_memory_utilization:
            return base, None

        snapshot = self._query_gpu_memory_gb(device)
        if snapshot is None:
            return base, None

        free_gb, total_gb = snapshot
        usable_gb = max(0.0, float(free_gb) - float(self.gpu_memory_utilization_margin_gb))
        adaptive = usable_gb / total_gb if total_gb > 0 else base
        resolved = min(base, adaptive)
        resolved = min(0.98, max(0.05, resolved))
        return resolved, snapshot

    def _normalized_activate_command(self) -> str:
        command = str(self.activate_command or "").strip()
        if not command:
            return ""

        lowered = command.lower()
        if "conda activate" not in lowered:
            return command
        if "conda.sh" in lowered or "conda run" in lowered:
            return command

        conda_init = 'source "$(conda info --base)/etc/profile.d/conda.sh"'
        return f"{conda_init} && {command}"

    def _build_shell_command(self, *, device: str, port: int) -> str:
        if not self.model_path:
            raise ValueError("vLLM service config requires model_path")
        if not self.served_model_name:
            raise ValueError("vLLM service config requires served_model_name")

        gpu_idx = self._device_index(device)
        gpu_mem_util, gpu_snapshot = self._resolve_gpu_memory_utilization(device)
        parts: List[str] = []
        activate_command = self._normalized_activate_command()
        if activate_command:
            parts.append(activate_command)
        if self.workdir:
            parts.append(f"cd {shlex.quote(self.workdir)}")

        serve_cmd = [
            f"CUDA_VISIBLE_DEVICES={gpu_idx}",
            "vllm",
            "serve",
            shlex.quote(self.model_path),
            "--served-model-name",
            shlex.quote(self.served_model_name),
            "--host",
            shlex.quote(self.bind_host),
            "--port",
            str(int(port)),
            "--tensor-parallel-size",
            str(int(self.tensor_parallel_size)),
            "--gpu-memory-utilization",
            f"{gpu_mem_util:.4f}",
        ]
        if self.reasoning_parser:
            serve_cmd.extend(["--reasoning-parser", shlex.quote(self.reasoning_parser)])
        if self.allowed_local_media_path:
            serve_cmd.extend(["--allowed-local-media-path", shlex.quote(self.allowed_local_media_path)])
        for arg in self.extra_args:
            serve_cmd.append(str(arg))

        parts.append(" ".join(serve_cmd))
        command = " && ".join(parts)
        if gpu_snapshot is not None:
            free_gb, total_gb = gpu_snapshot
            print(
                f"[vLLMService] device={device} free={free_gb:.2f}GiB total={total_gb:.2f}GiB "
                f"margin={self.gpu_memory_utilization_margin_gb:.2f}GiB "
                f"gpu_memory_utilization={gpu_mem_util:.4f}"
            )
        else:
            print(
                f"[vLLMService] device={device} gpu_memory_utilization={gpu_mem_util:.4f} "
                f"(gpu snapshot unavailable)"
            )
        return command

    def _record_alive(self, record: VLLMServiceRecord) -> bool:
        if record.process is None:
            return self._wait_until_ready(record.base_url, timeout_s=2.0)
        if str(record.kind or "").startswith("remote_") and str(record.kind or "").endswith("_tunnel"):
            return record.process.poll() is None
        return record.process.poll() is None and self._wait_until_ready(
            record.base_url,
            timeout_s=2.0,
            process=record.process,
        )

    def _wait_until_ready(
        self,
        base_url: str,
        *,
        timeout_s: float,
        process: subprocess.Popen | None = None,
        poll_interval_s: Optional[float] = None,
    ) -> bool:
        timeout_value = float(timeout_s)
        infinite_wait = timeout_value <= 0.0
        deadline = None if infinite_wait else (time.time() + max(0.1, timeout_value))
        announced = False
        interval = max(0.1, float(poll_interval_s or self.poll_interval_s))
        next_report_ts = time.time() + min(15.0, max(2.0, interval * 5.0))
        while True:
            if self._shutdown_requested():
                return False
            if process is not None and process.poll() is not None:
                return False
            if self._probe_models(base_url):
                return True
            now = time.time()
            if deadline is not None and now >= deadline:
                return False
            if process is not None and (not announced or now >= next_report_ts):
                remaining = -1.0 if deadline is None else max(0.0, deadline - now)
                remaining_text = "infinite" if deadline is None else f"{remaining:.0f}s"
                print(
                    f"[vLLMService] waiting for {base_url} to become ready "
                    f"(remaining={remaining_text})"
                )
                announced = True
                next_report_ts = now + 15.0
            time.sleep(interval)
        return False

    def _sleep_until_or_shutdown(self, duration_s: float) -> None:
        remaining = max(0.0, float(duration_s or 0.0))
        while remaining > 0.0:
            if self._shutdown_requested():
                return
            chunk = min(0.25, remaining)
            time.sleep(chunk)
            remaining = max(0.0, remaining - chunk)

    def _probe_models(self, base_url: str) -> bool:
        url = f"{base_url.rstrip('/')}/models"
        request = urllib.request.Request(url=url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                return int(getattr(response, "status", 200) or 0) == 200
        except urllib.error.HTTPError as exc:
            return int(getattr(exc, "code", 0) or 0) == 200
        except Exception:
            return False

    def _terminate_record_safely(self, record: VLLMServiceRecord) -> None:
        try:
            self._terminate_record(record)
        except Exception as exc:
            print(
                f"[vLLMService] terminate failed device={record.device} kind={record.kind}: {exc}"
            )

    def _terminate_record(self, record: VLLMServiceRecord) -> None:
        if not record.owned:
            return
        proc = record.process
        launcher_pid = int(record.launcher_pid or (proc.pid if proc is not None else 0) or 0)
        if proc is not None and proc.poll() is not None:
            self._cleanup_pid_path(str(record.pid_path or ""))
            return
        if proc is None and launcher_pid <= 0:
            self._cleanup_pid_path(str(record.pid_path or ""))
            return
        if proc is None and not self._pid_alive(launcher_pid):
            self._cleanup_pid_path(str(record.pid_path or ""))
            return
        try:
            if os.name != "nt":
                os.killpg(launcher_pid, signal.SIGTERM)
            else:
                if proc is not None:
                    proc.terminate()
                else:
                    os.kill(launcher_pid, signal.SIGTERM)
            if proc is not None:
                proc.wait(timeout=10.0)
        except Exception:
            try:
                if os.name != "nt":
                    os.killpg(launcher_pid, signal.SIGKILL)
                else:
                    if proc is not None:
                        proc.kill()
                    else:
                        os.kill(launcher_pid, signal.SIGKILL)
            except Exception:
                pass
        finally:
            self._cleanup_pid_path(str(record.pid_path or ""))
