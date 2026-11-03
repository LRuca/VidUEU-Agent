from __future__ import annotations

import copy
import json
import os
import posixpath
import re
import shlex
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np


_REMOTE_RUNNER = r"""
import argparse
import json
import sys
from pathlib import Path


def _to_jsonable(value):
    try:
        import numpy as _np
    except Exception:
        _np = None

    if _np is not None:
        if isinstance(value, _np.generic):
            return value.item()
        if isinstance(value, _np.ndarray):
            return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentpack-root", required=True)
    parser.add_argument("--resource", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _run():
    args = _build_parser().parse_args()
    sys.path.insert(0, args.agentpack_root)

    from executor.runtime.model_manager import ModelManager

    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    manager = ModelManager(config)
    try:
        with manager.use_resource(args.resource) as resource:
            method = getattr(resource, args.method)
            result = method(*list(payload.get("args", []) or []), **dict(payload.get("kwargs", {}) or {}))
        output = {"ok": True, "result": _to_jsonable(result)}
    except Exception as exc:
        output = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        try:
            manager.release(args.resource)
        except Exception:
            pass

    Path(args.output_json).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    _run()
"""


class RemoteToolInvocationError(RuntimeError):
    pass


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base.get(key) or {}), value)
        else:
            base[key] = copy.deepcopy(value)
    return base


class _RemoteGpuGuard:
    _guards: Dict[Tuple[str, int, str], "_RemoteGpuGuard"] = {}
    _guards_lock = threading.RLock()

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        cfg = dict(cfg or {})
        self.enabled = bool(cfg.get("enabled", False))
        self.max_inflight = max(1, int(cfg.get("max_inflight", 1) or 1))
        self.max_waiters = max(0, int(cfg.get("max_waiters", 0) or 0))
        self.queue_timeout_s = max(1.0, float(cfg.get("queue_timeout_s", 600.0) or 600.0))
        self.poll_interval_s = max(0.1, float(cfg.get("poll_interval_s", 2.0) or 2.0))
        min_free_mb = int(cfg.get("min_free_mb", 0) or 0)
        min_free_gb = float(cfg.get("min_free_gb", 0.0) or 0.0)
        self.min_free_mb = max(min_free_mb, int(min_free_gb * 1024.0))
        self.fail_open_on_probe_error = bool(cfg.get("fail_open_on_probe_error", False))
        self.active = 0
        self.waiters = 0
        self.last_probe_error = ""
        self.cv = threading.Condition(threading.RLock())

    @classmethod
    def get(cls, key: Tuple[str, int, str], cfg: Mapping[str, Any]) -> "_RemoteGpuGuard":
        with cls._guards_lock:
            guard = cls._guards.get(key)
            if guard is None:
                guard = cls(cfg)
                cls._guards[key] = guard
            return guard

    @contextmanager
    def acquire(self, client: "RemoteToolServerClient"):
        if not self.enabled:
            yield
            return
        deadline = time.time() + self.queue_timeout_s
        acquired = False
        with self.cv:
            self.waiters += 1
            try:
                while True:
                    if self.max_waiters > 0 and self.waiters > self.max_waiters:
                        raise TimeoutError(
                            f"remote GPU guard queue full for {client.resource_name}: "
                            f"waiters={self.waiters} max_waiters={self.max_waiters} "
                            f"active={self.active} device={client.remote_device}"
                        )

                    free_mb = -1
                    if self.active < self.max_inflight:
                        try:
                            free_mb = client.query_remote_gpu_free_mb()
                            self.last_probe_error = ""
                        except Exception as exc:
                            self.last_probe_error = f"{type(exc).__name__}: {exc}"
                            if self.fail_open_on_probe_error:
                                free_mb = self.min_free_mb

                        if self.min_free_mb <= 0 or free_mb >= self.min_free_mb:
                            self.active += 1
                            acquired = True
                            break

                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f"timed out waiting for remote GPU guard for {client.resource_name}: "
                            f"active={self.active} max_inflight={self.max_inflight} "
                            f"waiters={self.waiters} device={client.remote_device} "
                            f"free_mb={free_mb} min_free_mb={self.min_free_mb} "
                            f"last_probe_error={self.last_probe_error}"
                        )
                    self.cv.wait(timeout=min(self.poll_interval_s, remaining))
            finally:
                self.waiters = max(0, self.waiters - 1)
        try:
            yield
        finally:
            if acquired:
                with self.cv:
                    self.active = max(0, self.active - 1)
                    self.cv.notify_all()


class RemoteToolServerClient:
    def __init__(
        self,
        *,
        resource_name: str,
        ssh_host_alias: str,
        remote_agentpack_root: str,
        remote_request_root: str,
        remote_python_executable: str,
        remote_device: str,
        logical_device: str,
        base_config: Dict[str, Any],
        config_overrides: Dict[str, Any] | None = None,
        ssh_port: int = 0,
        ssh_extra_args: Sequence[str] | None = None,
        gpu_guard: Dict[str, Any] | None = None,
        preflight_script: str = "",
        preflight_every_request: bool = False,
        retry_attempts: int = 2,
        retry_backoff_s: float = 0.5,
        request_timeout_s: float = 900.0,
        ssh_connect_timeout_s: float = 15.0,
        cleanup_remote: bool = True,
    ) -> None:
        self.resource_name = str(resource_name or "").strip()
        self.ssh_host_alias = str(ssh_host_alias or "").strip()
        self.ssh_port = max(0, int(ssh_port or 0))
        self.ssh_extra_args = [str(x) for x in list(ssh_extra_args or []) if str(x).strip()]
        self.remote_agentpack_root = str(remote_agentpack_root or "").strip()
        self.remote_request_root = str(remote_request_root or "").strip()
        self.remote_python_executable = str(remote_python_executable or "").strip()
        self.remote_device = str(remote_device or "").strip()
        self.logical_device = str(logical_device or "").strip()
        self.base_config = copy.deepcopy(base_config or {})
        self.config_overrides = copy.deepcopy(config_overrides or {})
        self.retry_attempts = max(1, int(retry_attempts or 1))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s or 0.0))
        self.request_timeout_s = max(10.0, float(request_timeout_s or 0.0))
        self.ssh_connect_timeout_s = max(1.0, float(ssh_connect_timeout_s or 0.0))
        self.cleanup_remote = bool(cleanup_remote)
        self.preflight_script = str(preflight_script or "").strip()
        self.preflight_every_request = bool(preflight_every_request)
        self._preflight_done = False
        self._preflight_lock = threading.Lock()

        guard_key = (self.ssh_host_alias, self.ssh_port, self.remote_device)
        self._gpu_guard = _RemoteGpuGuard.get(guard_key, gpu_guard or {})

        if not self.ssh_host_alias:
            raise ValueError("Missing remote tool ssh_host_alias")
        if not self.remote_agentpack_root:
            raise ValueError("Missing remote tool remote_agentpack_root")
        if not self.remote_request_root:
            raise ValueError("Missing remote tool remote_request_root")
        if not self.remote_python_executable:
            raise ValueError("Missing remote tool remote_python_executable")

    def invoke(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                self._run_preflight_if_needed()
                with self._gpu_guard.acquire(self):
                    return self._invoke_once(method_name, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                print(
                    f"[RemoteToolServer][{self.resource_name}] attempt={attempt}/{self.retry_attempts} "
                    f"device={self.logical_device or self.remote_device or '<unknown>'} failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt >= self.retry_attempts:
                    break
                if self.retry_backoff_s > 0:
                    time.sleep(self.retry_backoff_s * (1.5 ** max(0, attempt - 1)))
        raise RemoteToolInvocationError(
            f"remote tool call failed for {self.resource_name}.{method_name}: {last_exc}"
        ) from last_exc

    def _run_preflight_if_needed(self) -> None:
        script = str(self.preflight_script or "").strip()
        if not script:
            return
        if self._preflight_done and not self.preflight_every_request:
            return
        with self._preflight_lock:
            if self._preflight_done and not self.preflight_every_request:
                return
            quoted = shlex.quote(script)
            self._ssh_bash(
                f"test -f {quoted} && chmod +x {quoted} && {quoted}",
                timeout_s=min(180.0, self.request_timeout_s),
            )
            self._preflight_done = True

    def _invoke_once(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        request_id = f"{self.resource_name}_{uuid.uuid4().hex[:12]}"
        remote_dir = posixpath.join(self.remote_request_root.rstrip("/"), request_id)
        remote_inputs_dir = posixpath.join(remote_dir, "inputs")
        remote_input_json = posixpath.join(remote_dir, "input.json")
        remote_output_json = posixpath.join(remote_dir, "output.json")
        remote_config_json = posixpath.join(remote_dir, "config.json")

        remote_path_to_local: Dict[str, str] = {}
        remote_name_to_local: Dict[str, str] = {}

        self._ssh_bash(
            f"mkdir -p {shlex.quote(remote_inputs_dir)}",
            timeout_s=min(60.0, self.request_timeout_s),
        )
        try:
            staged_args = self._stage_args(
                remote_inputs_dir,
                list(args),
                remote_path_to_local,
                remote_name_to_local,
            )
            staged_kwargs = self._stage_kwargs(
                remote_inputs_dir,
                dict(kwargs),
                remote_path_to_local,
                remote_name_to_local,
            )
            payload = {"args": staged_args, "kwargs": staged_kwargs}
            remote_config = self._build_remote_config()

            self._upload_json(remote_input_json, payload)
            self._upload_json(remote_config_json, remote_config)
            self._run_remote_runner(
                method_name=method_name,
                remote_config_json=remote_config_json,
                remote_input_json=remote_input_json,
                remote_output_json=remote_output_json,
            )
            response = self._download_json(remote_output_json)
        finally:
            if self.cleanup_remote:
                self._cleanup_remote_dir(remote_dir)

        if not bool(response.get("ok", False)):
            raise RemoteToolInvocationError(
                f"remote runner returned error_type={response.get('error_type', '')} "
                f"error={response.get('error', '')}"
            )
        result = response.get("result")
        return self._normalize_result(
            method_name=method_name,
            result=result,
            remote_path_to_local=remote_path_to_local,
            remote_name_to_local=remote_name_to_local,
        )

    def _build_remote_config(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.base_config or {})
        cfg = _deep_merge(cfg, self.config_overrides or {})
        runtime_cfg = cfg.setdefault("runtime", {})
        runtime_cfg["video_parallelism"] = {"enabled": False}
        runtime_cfg["tool_runtimes"] = {}
        runtime_cfg["local_tool_fallback"] = {"enabled": False}
        runtime_cfg["single_gpu_scheduler"] = {
            "enabled": True,
            "device": self.remote_device,
            "devices": [self.remote_device],
            "per_resource_devices": {self.resource_name: self.remote_device},
            "reserved_margin_mb": 0,
            "poll_interval_s": 0.2,
            "global_wait_timeout_s": max(60.0, self.request_timeout_s),
        }
        runtime_cfg.setdefault("remote_tool_server", {})
        if isinstance(runtime_cfg["remote_tool_server"], dict):
            server_c = dict(runtime_cfg["remote_tool_server"].get("server_c", {}) or {})
            server_c["enabled"] = False
            runtime_cfg["remote_tool_server"]["server_c"] = server_c

        models_cfg = cfg.setdefault("models", {})
        qwen_vlm = dict(models_cfg.get("qwen_vlm", {}) or {})
        qwen_vlm["mode"] = str(qwen_vlm.get("mode") or "api")
        api_cfg = dict(qwen_vlm.get("api", {}) or {})
        service_cfg = dict(api_cfg.get("service", {}) or {})
        service_cfg["auto_start"] = False
        service_cfg["startup_on_init"] = False
        for cluster_name in ("remote_b", "remote_c"):
            cluster_cfg = dict(service_cfg.get(cluster_name, {}) or {})
            cluster_cfg["enabled"] = False
            service_cfg[cluster_name] = cluster_cfg
        api_cfg["service"] = service_cfg
        qwen_vlm["api"] = api_cfg
        models_cfg["qwen_vlm"] = qwen_vlm

        self._apply_resource_device_overrides(cfg)
        return cfg

    def _apply_resource_device_overrides(self, cfg: Dict[str, Any]) -> None:
        resource = self.resource_name
        models_cfg = cfg.setdefault("models", {})
        tasks_cfg = cfg.setdefault("tasks", {})
        if resource == "embedder":
            task_cfg = dict(tasks_cfg.get("T2_image_embedding", {}) or {})
            task_cfg["device"] = self.remote_device
            tasks_cfg["T2_image_embedding"] = task_cfg
        elif resource == "asr":
            task_cfg = dict(tasks_cfg.get("T2_asr", {}) or {})
            task_cfg["device"] = self.remote_device
            tasks_cfg["T2_asr"] = task_cfg
        elif resource == "ocr":
            task_cfg = dict(tasks_cfg.get("T4_ocr", {}) or {})
            task_cfg["use_gpu"] = True
            task_cfg["device"] = self.remote_device
            tasks_cfg["T4_ocr"] = task_cfg
        elif resource == "yolo":
            models_cfg["yolo_device"] = self.remote_device
        elif resource == "raft":
            optical_flow_cfg = cfg.setdefault("optical_flow", {})
            raft_cfg = dict(optical_flow_cfg.get("raft", {}) or {})
            raft_cfg["device"] = self.remote_device
            optical_flow_cfg["raft"] = raft_cfg
        elif resource == "omniparser":
            omni_cfg = dict(models_cfg.get("omniparser", {}) or {})
            omni_cfg["device"] = self.remote_device
            models_cfg["omniparser"] = omni_cfg

    def _stage_args(
        self,
        remote_inputs_dir: str,
        args: List[Any],
        remote_path_to_local: Dict[str, str],
        remote_name_to_local: Dict[str, str],
    ) -> List[Any]:
        resource = self.resource_name
        if resource == "asr":
            if args:
                args[0] = self._stage_single_file(
                    remote_inputs_dir,
                    args[0],
                    prefix="audio",
                    remote_path_to_local=remote_path_to_local,
                    remote_name_to_local=remote_name_to_local,
                )
            return args
        if resource in {"ocr", "yolo", "raft", "omniparser", "embedder"}:
            if args and self._looks_like_path_list(args[0]):
                args[0] = self._stage_file_list(
                    remote_inputs_dir,
                    list(args[0]),
                    remote_path_to_local=remote_path_to_local,
                    remote_name_to_local=remote_name_to_local,
                )
            return args
        return args

    def _stage_kwargs(
        self,
        remote_inputs_dir: str,
        kwargs: Dict[str, Any],
        remote_path_to_local: Dict[str, str],
        remote_name_to_local: Dict[str, str],
    ) -> Dict[str, Any]:
        out = dict(kwargs)
        for key, value in list(out.items()):
            if self._looks_like_path_list(value):
                out[key] = self._stage_file_list(
                    remote_inputs_dir,
                    list(value),
                    remote_path_to_local=remote_path_to_local,
                    remote_name_to_local=remote_name_to_local,
                )
            elif self._looks_like_path(value):
                out[key] = self._stage_single_file(
                    remote_inputs_dir,
                    value,
                    prefix=str(key),
                    remote_path_to_local=remote_path_to_local,
                    remote_name_to_local=remote_name_to_local,
                )
        return out

    def _looks_like_path(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        s = str(value).strip()
        if not s:
            return False
        return os.path.exists(s)

    def _looks_like_path_list(self, value: Any) -> bool:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            return False
        seq = list(value)
        return bool(seq) and all(self._looks_like_path(x) for x in seq)

    def _stage_file_list(
        self,
        remote_inputs_dir: str,
        local_paths: List[str],
        *,
        remote_path_to_local: Dict[str, str],
        remote_name_to_local: Dict[str, str],
    ) -> List[str]:
        remote_paths: List[str] = []
        for idx, local_path in enumerate(local_paths):
            remote_path = self._stage_single_file(
                remote_inputs_dir,
                local_path,
                prefix=f"f{idx:03d}",
                remote_path_to_local=remote_path_to_local,
                remote_name_to_local=remote_name_to_local,
            )
            remote_paths.append(remote_path)
        return remote_paths

    def _stage_single_file(
        self,
        remote_inputs_dir: str,
        local_path: Any,
        *,
        prefix: str,
        remote_path_to_local: Dict[str, str],
        remote_name_to_local: Dict[str, str],
    ) -> str:
        local_path_str = str(local_path or "").strip()
        if not local_path_str:
            raise FileNotFoundError(f"missing local path for remote {self.resource_name} request")
        path_obj = Path(local_path_str)
        if not path_obj.exists():
            raise FileNotFoundError(f"local file not found: {local_path_str}")
        suffix = path_obj.suffix or ""
        safe_name = f"{prefix}__{path_obj.name}"
        remote_path = posixpath.join(remote_inputs_dir, safe_name)
        with path_obj.open("rb") as fh:
            self._upload_bytes(remote_path, fh.read())
        remote_path_to_local[remote_path] = str(path_obj)
        remote_name_to_local[safe_name] = str(path_obj)
        remote_name_to_local[path_obj.name] = str(path_obj)
        return remote_path

    def _normalize_result(
        self,
        *,
        method_name: str,
        result: Any,
        remote_path_to_local: Dict[str, str],
        remote_name_to_local: Dict[str, str],
    ) -> Any:
        resource = self.resource_name
        if resource == "asr" and method_name == "run":
            if isinstance(result, list) and len(result) >= 2:
                return str(result[0] or ""), list(result[1] or [])
            return "", []
        if resource in {"embedder"} and method_name in {"compute_embeddings", "compute_text_embeddings"}:
            return np.asarray(result or [], dtype=np.float32)
        if resource == "ocr" and isinstance(result, dict):
            remapped: Dict[str, Any] = {}
            for key, value in result.items():
                local_key = remote_path_to_local.get(str(key)) or remote_name_to_local.get(str(key)) or str(key)
                remapped[local_key] = value
            return remapped
        if resource in {"yolo", "omniparser"} and isinstance(result, dict):
            remapped = {}
            for key, value in result.items():
                local_key = remote_path_to_local.get(str(key)) or remote_name_to_local.get(str(key)) or str(key)
                remapped[local_key] = value
            return remapped
        return result

    def _upload_json(self, remote_path: str, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._upload_bytes(remote_path, data)

    def _download_json(self, remote_path: str) -> Dict[str, Any]:
        stdout = self._ssh_bash(
            f"cat {shlex.quote(remote_path)}",
            timeout_s=self.request_timeout_s,
        )
        return dict(json.loads(stdout.decode("utf-8")))

    def _upload_bytes(self, remote_path: str, data: bytes) -> None:
        self._ssh_bash(
            f"cat > {shlex.quote(remote_path)}",
            input_bytes=data,
            timeout_s=self.request_timeout_s,
        )

    def _cleanup_remote_dir(self, remote_dir: str) -> None:
        try:
            self._ssh_bash(
                f"rm -rf {shlex.quote(remote_dir)}",
                timeout_s=min(60.0, self.request_timeout_s),
                check=False,
            )
        except Exception:
            pass

    def _run_remote_runner(
        self,
        *,
        method_name: str,
        remote_config_json: str,
            remote_input_json: str,
            remote_output_json: str,
    ) -> None:
        env_prefix = self._remote_runner_env_prefix()
        command = " ".join(
            [
                *env_prefix,
                shlex.quote(self.remote_python_executable),
                "-",
                "--agentpack-root",
                shlex.quote(self.remote_agentpack_root),
                "--resource",
                shlex.quote(self.resource_name),
                "--method",
                shlex.quote(method_name),
                "--config-json",
                shlex.quote(remote_config_json),
                "--input-json",
                shlex.quote(remote_input_json),
                "--output-json",
                shlex.quote(remote_output_json),
            ]
        )
        self._ssh_bash(
            command,
            input_bytes=_REMOTE_RUNNER.encode("utf-8"),
            timeout_s=self.request_timeout_s,
        )

    def _remote_runner_env_prefix(self) -> List[str]:
        device = str(self.remote_device or "").strip().lower()
        if device == "cpu":
            return ["CUDA_VISIBLE_DEVICES="]
        if device == "cuda" or device.startswith("cuda:") or device == "gpu" or device.startswith("gpu:"):
            return [f"CUDA_VISIBLE_DEVICES={self._remote_cuda_device_index()}"]
        return []

    def query_remote_gpu_free_mb(self) -> int:
        idx = self._remote_cuda_device_index()
        stdout = self._ssh_bash(
            (
                f"nvidia-smi -i {idx} --query-gpu=memory.free "
                "--format=csv,noheader,nounits"
            ),
            timeout_s=min(30.0, self.request_timeout_s),
        )
        lines = stdout.decode("utf-8", errors="replace").splitlines()
        if not lines:
            raise RemoteToolInvocationError(f"empty nvidia-smi output for device={self.remote_device}")
        # The target host's login shell prints an ASCII banner before command
        # output. Do not assume the first line is the nvidia-smi value.
        for raw_line in lines:
            line = str(raw_line).strip()
            if not line:
                continue
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", line):
                return int(float(line))
        sample = "\n".join(lines[:5])
        raise RemoteToolInvocationError(
            f"invalid nvidia-smi output for device={self.remote_device}: {sample!r}"
        )

    def _remote_cuda_device_index(self) -> int:
        text = str(self.remote_device or "").strip().lower()
        if ":" in text:
            text = text.split(":", 1)[1]
        try:
            return max(0, int(text))
        except Exception:
            return 0

    def _ssh_bash(
        self,
        remote_command: str,
        *,
        input_bytes: bytes | None = None,
        timeout_s: float,
        check: bool = True,
    ) -> bytes:
        ssh_command = [
            "ssh",
            "-o",
            f"ConnectTimeout={max(1, int(self.ssh_connect_timeout_s))}",
            *self.ssh_extra_args,
        ]
        if self.ssh_port > 0:
            ssh_command.extend(["-p", str(self.ssh_port)])
        # OpenSSH sends the remote command through the user's shell. Quote the
        # bash -lc payload so compound commands are received as one argument.
        ssh_command.extend([self.ssh_host_alias, "bash", "--noprofile", "--norc", "-c", shlex.quote(remote_command)])
        proc = subprocess.run(
            ssh_command,
            input=input_bytes,
            capture_output=True,
            check=False,
            timeout=max(1.0, float(timeout_s or 0.0)),
        )
        if check and proc.returncode != 0:
            stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            stdout = (proc.stdout or b"").decode("utf-8", errors="replace").strip()
            raise RemoteToolInvocationError(
                f"ssh command failed rc={proc.returncode} alias={self.ssh_host_alias} "
                f"cmd={remote_command!r} stdout={stdout!r} stderr={stderr!r}"
            )
        return bytes(proc.stdout or b"")


class RemoteToolResourceProxy:
    def __init__(
        self,
        *,
        client: RemoteToolServerClient,
        fallback_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._fallback_factory = fallback_factory
        self._fallback_lock = threading.Lock()
        self._fallback_instance: Any | None = None

    def set_chat_logger(self, logger: Any) -> None:
        _ = logger

    def set_chat_context(self, context: Dict[str, Any] | None) -> None:
        _ = context

    def close(self) -> None:
        inst = self._fallback_instance
        self._fallback_instance = None
        if inst is None:
            return
        close_fn = getattr(inst, "close", None)
        if callable(close_fn):
            close_fn()

    def _fallback(self) -> Any | None:
        factory = self._fallback_factory
        if factory is None:
            return None
        if self._fallback_instance is not None:
            return self._fallback_instance
        with self._fallback_lock:
            if self._fallback_instance is None:
                self._fallback_instance = factory()
        return self._fallback_instance

    def _call_with_fallback(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._client.invoke(method_name, *args, **kwargs)
        except Exception as exc:
            fallback = self._fallback()
            if fallback is None:
                raise
            print(
                f"[RemoteToolServer][{self._client.resource_name}] warning: "
                f"remote call failed, falling back to local resource on error={type(exc).__name__}: {exc}"
            )
            return getattr(fallback, method_name)(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_fallback("run", *args, **kwargs)

    def compute_flow_for_frame_paths(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_fallback("compute_flow_for_frame_paths", *args, **kwargs)

    def compute_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_fallback("compute_embeddings", *args, **kwargs)

    def compute_text_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_with_fallback("compute_text_embeddings", *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return self._call_with_fallback(name, *args, **kwargs)

        return _wrapped
