from __future__ import annotations

import atexit
import gc
import copy
import importlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from executor.runtime.resource_registry import ResourceRegistry
from executor.runtime.remote_tool_client import RemoteToolServerClient, RemoteToolResourceProxy
from executor.runtime.vllm_service_manager import VLLMServiceManager

@dataclass
class ResourceSpec:
    name: str
    group: str
    scope: str
    factory_path: str
    persistent: bool = False
    warm_reuse: bool = False
    gpu_heavy: bool = False
    device_policy: Dict[str, Any] = field(default_factory=dict)
    config_binding: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimePolicy:
    estimated_vram_mb: int = 0
    max_concurrency: int = 1
    idle_unload_s: float = 60.0
    queue_timeout_s: float = 300.0
    load_timeout_s: float = 180.0
    unload_priority: int = 100
    max_waiters: int = 0


@dataclass
class ResourceRuntime:
    name: str
    policy: RuntimePolicy
    device: Optional[str] = None
    active_calls: int = 0
    waiters: int = 0
    loading: bool = False
    loaded: bool = False
    last_used_ts: float = 0.0
    last_loaded_ts: float = 0.0
    last_unloaded_ts: float = 0.0
    last_error: str = ""
    cv: threading.Condition | None = None


@dataclass
class ResourceReplica:
    name: str
    replica_id: str
    device: Optional[str]
    instance: Any
    max_inflight: int = 1
    active_calls: int = 0
    loaded: bool = True
    loading: bool = False
    last_used_ts: float = 0.0
    last_loaded_ts: float = 0.0
    fail_count: int = 0
    oom_count: int = 0


class SynchronizedResourceProxy:
    """Serialize method calls on a shared resource instance."""

    def __init__(self, target: Any, lock: threading.RLock):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_lock", lock)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, "_target"), name)
        if callable(attr):
            def _wrapped(*args, **kwargs):
                with object.__getattribute__(self, "_lock"):
                    return attr(*args, **kwargs)
            return _wrapped
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_target"), name, value)


class ModelManager:
    """
    Minimal lazy resource manager for T1/T2.

    It is intentionally conservative:
    - load on first use
    - optionally keep a small number of GPU-heavy models resident
    - release non-persistent resources on demand
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.resource_registry = ResourceRegistry()
        self._instances: Dict[str, Any] = {}
        self._specs: Dict[str, ResourceSpec] = {}
        self._last_used_ts: Dict[str, float] = {}
        self._load_counts: Dict[str, int] = {}
        self._eviction_counts: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._gpu_cv = threading.Condition(self._lock)
        self._vlm_pool_lock = threading.RLock()
        self._vlm_slot_cv = threading.Condition(self._vlm_pool_lock)
        self._vlm_busy_slots: Set[int] = set()
        self._vlm_pool: List[Any] = []
        self._vlm_rr_idx = 0
        self._chat_state = threading.local()
        self._vlm_thread_state = threading.local()
        self._vllm_service_manager: Optional[VLLMServiceManager] = None
        self._resource_method_locks: Dict[str, threading.RLock] = {}
        self._resource_replicas: Dict[str, List[ResourceReplica]] = {}
        self._resource_replica_state = threading.local()

        rt = (self.config.get("runtime", {}) or {}) if isinstance(self.config, dict) else {}
        self.max_gpu_resident_models = int(rt.get("max_gpu_resident_models", 2) or 2)
        video_parallel_cfg = (rt.get("video_parallelism", {}) or {}) if isinstance(rt.get("video_parallelism", {}), dict) else {}
        self._concurrent_video_enabled = bool(video_parallel_cfg.get("enabled", False))
        self._vlm_slots_per_video = max(1, int(video_parallel_cfg.get("vlm_slots_per_video", 1) or 1))
        self._scope_release_suspended = False
        self._manager_assign_enabled = bool(rt.get("manager_assign_devices", True))
        self._device_overrides = dict(rt.get("model_device_overrides", {}) or {})
        self._gpu_placement_policy = self._load_gpu_placement_policy()
        self._scheduler_cfg = self._load_single_gpu_scheduler_cfg()
        self._scheduled_resource_names = self._load_scheduled_resource_names()
        self._vlm_devices = self._discover_vlm_devices()
        self._vlm_policy = self._load_vlm_pool_policy()
        self._vlm_pool_size = max(0, int(self._vlm_policy.get("target_replicas", len(self._vlm_devices) or 0) or 0))
        self._init_vllm_service_manager()

        self._register_from_registry()
        self._resource_runtime: Dict[str, ResourceRuntime] = {}
        self._init_resource_runtime()
        atexit.register(self._shutdown_vllm_services)

    def _register(self, spec: ResourceSpec) -> None:
        self._specs[spec.name] = spec

    def _load_class(self, class_path: str) -> Any:
        module_name, symbol = str(class_path).split(":", 1)
        module = importlib.import_module(module_name)
        return getattr(module, symbol)

    def _init_resource_runtime(self) -> None:
        now = time.time()
        for name, spec in self._specs.items():
            policy = self._load_runtime_policy_for_resource(name, spec)
            self._resource_runtime[name] = ResourceRuntime(
                name=name,
                policy=policy,
                last_used_ts=now,
                last_unloaded_ts=now,
                cv=threading.Condition(self._lock),
            )

    def set_chat_logger(self, logger: Any) -> None:
        self._chat_state.logger = logger
        self._apply_chat_logging_to_loaded_instances()

    def set_chat_context(self, context: Dict[str, Any] | None) -> None:
        self._chat_state.context = dict(context or {})
        self._apply_chat_logging_to_loaded_instances()

    def clear_chat_logger(self) -> None:
        self._chat_state.logger = None
        self._apply_chat_logging_to_loaded_instances()

    def clear_chat_context(self) -> None:
        self._chat_state.context = {}
        self._apply_chat_logging_to_loaded_instances()

    def _current_chat_logger(self) -> Any:
        return getattr(self._chat_state, "logger", None)

    def _current_chat_context(self) -> Dict[str, Any]:
        return dict(getattr(self._chat_state, "context", {}) or {})

    def _apply_chat_logging(self, inst: Any) -> Any:
        if inst is None:
            return inst
        set_logger = getattr(inst, "set_chat_logger", None)
        if callable(set_logger):
            try:
                set_logger(self._current_chat_logger())
            except Exception:
                pass
        set_context = getattr(inst, "set_chat_context", None)
        if callable(set_context):
            try:
                set_context(self._current_chat_context())
            except Exception:
                pass
        return inst

    def _apply_chat_logging_to_loaded_instances(self) -> None:
        with self._lock:
            instances = list(self._instances.values())
            replicas = [
                replica.instance
                for pool in self._resource_replicas.values()
                for replica in pool
                if replica.instance is not None
            ]
        for inst in instances:
            self._apply_chat_logging(inst)
        for inst in replicas:
            self._apply_chat_logging(inst)
        with self._vlm_pool_lock:
            pool = list(self._vlm_pool)
        for inst in pool:
            self._apply_chat_logging(inst)

    def set_scope_release_suspended(self, suspended: bool) -> None:
        self._scope_release_suspended = bool(suspended)

    def _wrap_shared_resource(self, name: str, inst: Any) -> Any:
        if inst is None or name == "vlm" or not self._concurrent_video_enabled:
            return inst
        lock = self._resource_method_locks.setdefault(name, threading.RLock())
        return SynchronizedResourceProxy(inst, lock)

    def _lazy_import(self, module_name: str, symbol: str) -> Any:
        module = importlib.import_module(module_name)
        return getattr(module, symbol)

    def _build_raft_kwargs(self) -> Dict[str, Any]:
        optical_flow_cfg = (
            (self.config.get("optical_flow", {}) or {}) if isinstance(self.config, dict) else {}
        )
        raft_cfg = (optical_flow_cfg.get("raft", {}) or {}) if isinstance(optical_flow_cfg, dict) else {}
        models_cfg = (
            (self.config.get("models", {}) or {}) if isinstance(self.config, dict) else {}
        )

        repo_root = raft_cfg.get("repo_root") or optical_flow_cfg.get("repo_root")
        if not repo_root:
            repo_root = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "third_party",
                    "guru_video",
                    "third_party",
                    "raft",
                )
            )

        device = (
            raft_cfg.get("device")
            or optical_flow_cfg.get("device")
            or models_cfg.get("yolo_device")
            or "cuda:0"
        )

        return {
            "repo_root": repo_root,
            "weights": raft_cfg.get("weights", "models/raft-things.pth"),
            "device": device,
            "small": bool(raft_cfg.get("small", False)),
            "mixed_precision": bool(raft_cfg.get("mixed_precision", True)),
            "alternate_corr": bool(raft_cfg.get("alternate_corr", False)),
        }

    def _build_vlm(self) -> Any:
        return self._build_vlm_replica(0)

    def _runtime_cfg(self) -> Dict[str, Any]:
        return (self.config.get("runtime", {}) or {}) if isinstance(self.config, dict) else {}

    def _load_single_gpu_scheduler_cfg(self) -> Dict[str, Any]:
        cfg = dict((self._runtime_cfg().get("single_gpu_scheduler", {}) or {}))
        device = self._normalize_device_name(cfg.get("device")) or "cuda:0"
        devices = self._normalize_device_list(cfg.get("devices") or [])
        if not devices:
            devices = [device]
        raw_resources = cfg.get("resources", cfg.get("resource_names"))
        if isinstance(raw_resources, str):
            resources = [item.strip() for item in raw_resources.split(",") if item.strip()]
        elif isinstance(raw_resources, (list, tuple, set)):
            resources = [str(item).strip() for item in raw_resources if str(item).strip()]
        else:
            resources = None
        per_resource_devices = {}
        raw_per_resource_devices = cfg.get("per_resource_devices") or cfg.get("resource_devices") or {}
        if isinstance(raw_per_resource_devices, dict):
            per_resource_devices = {
                str(name): normalized
                for name, raw_device in raw_per_resource_devices.items()
                for normalized in [self._normalize_device_name(raw_device)]
                if normalized
            }
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "device": device,
            "devices": devices,
            "resources": resources,
            "per_resource_devices": per_resource_devices,
            "reserved_margin_mb": max(0, int(cfg.get("reserved_margin_mb", 2048) or 0)),
            "poll_interval_s": max(0.1, float(cfg.get("poll_interval_s", 1.0) or 1.0)),
            "global_wait_timeout_s": max(1.0, float(cfg.get("global_wait_timeout_s", 600.0) or 600.0)),
        }

    def _tool_runtime_cfg(self) -> Dict[str, Any]:
        runtime_cfg = self._runtime_cfg()
        return dict(runtime_cfg.get("tool_runtimes", {}) or {})

    def _load_scheduled_resource_names(self) -> Set[str]:
        cfg = self._scheduler_cfg if isinstance(self._scheduler_cfg, dict) else {}
        raw = cfg.get('resources') or cfg.get('resource_names')
        if raw is None:
            names = {'asr', 'ocr', 'yolo', 'raft', 'omniparser'}
            names.update(str(name) for name in self._tool_runtime_cfg().keys())
            return names
        if isinstance(raw, str):
            raw_items = [item.strip() for item in raw.split(',')]
        elif isinstance(raw, (list, tuple, set)):
            raw_items = list(raw)
        else:
            raw_items = []
        return {str(item).strip() for item in raw_items if str(item).strip()}

    def _remote_tool_server_cfg(self) -> Dict[str, Any]:
        runtime_cfg = self._runtime_cfg()
        remote_cfg = runtime_cfg.get("remote_tool_server", {}) if isinstance(runtime_cfg, dict) else {}
        if not isinstance(remote_cfg, dict):
            return {}
        return dict((remote_cfg.get("server_c", {}) or {}))

    def _remote_tool_server_enabled(self) -> bool:
        cfg = self._remote_tool_server_cfg()
        if not cfg:
            return False
        if not bool(cfg.get("enabled", False)):
            return False
        role = str(cfg.get("role", "tools") or "tools").strip().lower()
        return role == "tools"

    def _remote_tool_resource_cfg(self, name: str) -> Dict[str, Any]:
        cfg = self._remote_tool_server_cfg()
        resources = cfg.get("resources", {}) if isinstance(cfg, dict) else {}
        if not isinstance(resources, dict):
            return {}
        return dict((resources.get(str(name), {}) or {}))

    def _is_remote_tool_device(self, device: Any) -> bool:
        normalized = self._normalize_device_name(device) or ""
        return normalized.startswith("remote_c:")

    def _should_use_remote_tool_server(self, name: str, device: Any) -> bool:
        if not self._remote_tool_server_enabled():
            return False
        if not self._is_remote_tool_device(device):
            return False
        resource_cfg = self._remote_tool_resource_cfg(name)
        return bool(resource_cfg.get("enabled", False))

    def _local_tool_fallback_cfg(self) -> Dict[str, Any]:
        runtime_cfg = self._runtime_cfg()
        cfg = runtime_cfg.get("local_tool_fallback", {}) if isinstance(runtime_cfg, dict) else {}
        return dict(cfg or {}) if isinstance(cfg, dict) else {}

    def _local_tool_fallback_enabled(self) -> bool:
        cfg = self._local_tool_fallback_cfg()
        return bool(cfg.get("enabled", False))

    def _local_tool_fallback_device(self, name: str, fallback: Any = None) -> Optional[str]:
        cfg = self._local_tool_fallback_cfg()
        per_resource = cfg.get("per_resource_devices", {}) if isinstance(cfg, dict) else {}
        if isinstance(per_resource, dict):
            explicit = self._normalize_device_name(per_resource.get(str(name)))
            if explicit:
                return explicit
        devices = self._normalize_device_list(cfg.get("devices") or [])
        if devices:
            ordered = ["embedder", "asr", "ocr", "yolo", "raft", "omniparser"]
            try:
                idx = ordered.index(str(name))
            except ValueError:
                idx = 0
            return devices[idx % len(devices)]
        normalized_fallback = self._normalize_device_name(fallback)
        if normalized_fallback and not self._is_remote_tool_device(normalized_fallback):
            return normalized_fallback
        return None

    def _resource_replica_plan(self, name: str) -> List[Dict[str, Any]]:
        override = self._tool_runtime_cfg().get(name)
        if not isinstance(override, dict):
            return []
        raw_replicas = override.get("replicas")
        if raw_replicas is None:
            return []
        if isinstance(raw_replicas, dict):
            raw_items = [raw_replicas]
        elif isinstance(raw_replicas, (list, tuple)):
            raw_items = list(raw_replicas)
        else:
            return []

        plan: List[Dict[str, Any]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            device = self._available_or_fallback_device(
                self._normalize_device_name(item.get("device")) or self._scheduled_resource_device(name),
                resource_name=name,
            )
            count = max(1, int(item.get("count", 1) or 1))
            max_inflight = max(1, int(item.get("max_inflight", 1) or 1))
            for _ in range(count):
                plan.append({"device": device, "max_inflight": max_inflight})
        return plan

    def _uses_resource_replicas(self, name: str) -> bool:
        return bool(self._resource_replica_plan(name))

    def _load_runtime_policy_for_resource(self, name: str, spec: ResourceSpec) -> RuntimePolicy:
        merged = dict(spec.runtime or {})
        override = self._tool_runtime_cfg().get(name)
        if isinstance(override, dict):
            merged.update(override)
        queue_timeout_s = float(merged.get("queue_timeout_s", 300.0) or 0.0)
        if queue_timeout_s <= 0:
            queue_timeout_s = float(self._scheduler_cfg.get("global_wait_timeout_s", 600.0) or 600.0)
        return RuntimePolicy(
            estimated_vram_mb=max(0, int(merged.get("estimated_vram_mb", 0) or 0)),
            max_concurrency=max(1, int(merged.get("max_concurrency", 1) or 1)),
            idle_unload_s=max(0.0, float(merged.get("idle_unload_s", 60.0) or 0.0)),
            queue_timeout_s=queue_timeout_s,
            load_timeout_s=max(1.0, float(merged.get("load_timeout_s", 180.0) or 180.0)),
            unload_priority=max(0, int(merged.get("unload_priority", 100) or 0)),
            max_waiters=max(0, int(merged.get("max_waiters", 0) or 0)),
        )

    def _models_cfg(self) -> Dict[str, Any]:
        return (self.config.get("models", {}) or {}) if isinstance(self.config, dict) else {}

    def _tasks_cfg(self) -> Dict[str, Any]:
        return (self.config.get("tasks", {}) or {}) if isinstance(self.config, dict) else {}

    def _vlm_cfg(self) -> Dict[str, Any]:
        models_cfg = self._models_cfg()
        return (models_cfg.get("qwen_vlm", {}) or {}) if isinstance(models_cfg, dict) else {}

    def _vlm_api_cfg(self) -> Dict[str, Any]:
        vlm_cfg = self._vlm_cfg()
        return (vlm_cfg.get("api", {}) or {}) if isinstance(vlm_cfg, dict) else {}

    def _is_vllm_api_mode(self) -> bool:
        vlm_cfg = self._vlm_cfg()
        mode = str(vlm_cfg.get("mode", "") or "").strip().lower()
        if mode != "api":
            return False
        api_cfg = self._vlm_api_cfg()
        provider = str(api_cfg.get("provider") or api_cfg.get("backend") or "").strip().lower()
        return provider in ("vllm", "vllm_openai", "openai_compatible_vllm")

    def _init_vllm_service_manager(self) -> None:
        if not self._is_vllm_api_mode():
            self._vllm_service_manager = None
            return
        api_cfg = self._vlm_api_cfg()
        service_cfg = dict(api_cfg.get("service", {}) or {})
        if not service_cfg:
            self._vllm_service_manager = None
            return
        paths_cfg = (self.config.get("paths", {}) or {}) if isinstance(self.config, dict) else {}
        self._vllm_service_manager = VLLMServiceManager(service_cfg, log_root=str(paths_cfg.get("log_dir", "") or ""))
        eager_start = bool(service_cfg.get("startup_on_init", True))
        if eager_start and self._vllm_service_manager.can_spawn():
            self._ensure_vllm_service_pool()

    def _ensure_vllm_service_pool(self, target_size: Optional[int] = None) -> Dict[str, str]:
        manager = self._vllm_service_manager
        if manager is None or not manager.enabled:
            return {}
        desired_local = self._desired_local_vlm_pool_size()
        remote_target = int(getattr(manager, "remote_service_target_count", lambda: 0)() or 0)
        remote_active = int(getattr(manager, "remote_active_service_count", lambda: 0)() or 0)
        local_active = int(getattr(manager, "local_active_service_count", lambda: manager.active_service_count())() or 0)
        if local_active >= desired_local and remote_active >= remote_target:
            return {device: manager.endpoint_for_device(device) for device in manager.active_devices()}
        active_local_devices = list(getattr(manager, "active_local_devices", lambda: manager.active_devices())() or [])
        devices = list(dict.fromkeys(active_local_devices + self._select_vlm_devices_for_target(desired_local)))[:desired_local]
        return manager.ensure_services(devices)

    def _shutdown_vllm_services(self) -> None:
        manager = self._vllm_service_manager
        if manager is None:
            return
        try:
            manager.close()
        except Exception:
            pass

    def _discover_vlm_devices(self) -> List[Optional[str]]:
        models_cfg = self._models_cfg()
        vlm_cfg = (models_cfg.get("qwen_vlm", {}) or {}) if isinstance(models_cfg, dict) else {}

        runtime_cfg = self._runtime_cfg()
        placement_cfg = (runtime_cfg.get("gpu_placement", {}) or {}) if isinstance(runtime_cfg, dict) else {}
        per_model = (placement_cfg.get("per_model", {}) or {}) if isinstance(placement_cfg, dict) else {}
        vlm_place_cfg = (per_model.get("vlm", {}) or {}) if isinstance(per_model, dict) else {}

        configured = (
            vlm_cfg.get("devices")
            or vlm_cfg.get("device_pool")
            or vlm_cfg.get("cuda_devices")
            or vlm_place_cfg.get("devices")
            or placement_cfg.get("devices")
            or []
        )

        if configured and not isinstance(configured, (list, tuple)):
            configured = [configured]

        devices = [self._normalize_device_name(x) for x in configured if self._normalize_device_name(x)]
        if devices:
            return devices

        fallback = self._normalize_device_name(vlm_cfg.get("device"))
        if fallback:
            return [fallback]

        return self._available_cuda_devices()

    def _available_cuda_devices(self) -> List[str]:
        try:
            import torch
        except Exception:
            return []

        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return []
        try:
            if not cuda.is_available():
                return []
            device_count = int(cuda.device_count())
        except Exception:
            return []
        return [f"cuda:{idx}" for idx in range(device_count)]

    def _current_vlm_devices(self) -> List[str]:
        devices: List[str] = []

        with self._vlm_pool_lock:
            pool = list(self._vlm_pool)

        if pool:
            for inst in pool:
                device = self._extract_device(inst)
                if device:
                    devices.append(device)
        else:
            with self._lock:
                primary = self._instances.get("vlm")
            if primary is not None:
                device = self._extract_device(primary)
                if device:
                    devices.append(device)

        return devices
    
    def _select_next_vlm_device(self, replica_idx: int) -> Optional[str]:
        candidates = self._resource_device_candidates("vlm", self._vlm_devices)
        if not candidates:
            return self._select_device_for_resource("vlm")

        policy = self._resource_placement_policy("vlm")
        prefer_spread = bool(policy.get("prefer_spread", True))
        used_devices = set(self._current_vlm_devices()) if prefer_spread else set()

        # First try devices not already occupied by the VLM pool.
        device = self._select_device_for_resource(
            "vlm",
            fallback=candidates,
            exclude_devices=used_devices if used_devices else None,
        )
        if device:
            return device

        # If all candidates are occupied, fall back to rotating over the full candidate list.
        remaining = [d for d in candidates if d not in used_devices] or list(candidates)
        if not remaining:
            return None

        return remaining[replica_idx % len(remaining)]
    
    def _load_gpu_placement_policy(self) -> Dict[str, Any]:
        runtime_cfg = self._runtime_cfg()
        placement_cfg = (runtime_cfg.get("gpu_placement", {}) or {})
        default_devices = self._normalize_device_list(
            placement_cfg.get("devices") or self._available_cuda_devices()
        )
        return {
            "enabled": bool(placement_cfg.get("enabled", True)),
            "strategy": str(placement_cfg.get("strategy") or "most_free").strip().lower(),
            "default_devices": default_devices,
            "reserve_vram_gb": float(placement_cfg.get("reserve_vram_gb", 2.0) or 0.0),
            "min_free_vram_gb": float(placement_cfg.get("min_free_vram_gb", 0.0) or 0.0),
            "prefer_spread": bool(placement_cfg.get("prefer_spread", True)),
            "allow_fallback_to_cpu": bool(placement_cfg.get("allow_fallback_to_cpu", True)),
            "per_model": dict(placement_cfg.get("per_model", {}) or {}),
        }

    def _load_vlm_pool_policy(self) -> Dict[str, Any]:
        models_cfg = self._models_cfg()
        vlm_cfg = (models_cfg.get("qwen_vlm", {}) or {}) if isinstance(models_cfg, dict) else {}
        runtime_vlm_cfg = (self._runtime_cfg().get("vlm_pool", {}) or {})
        prefer_model_cfg = self._is_vllm_api_mode()

        def _pick(key: str, default: Any) -> Any:
            primary = vlm_cfg if prefer_model_cfg else runtime_vlm_cfg
            secondary = runtime_vlm_cfg if prefer_model_cfg else vlm_cfg
            if key in primary and primary.get(key) is not None:
                return primary.get(key)
            if key in secondary and secondary.get(key) is not None:
                return secondary.get(key)
            return default

        target_replicas = _pick("target_replicas", len(self._vlm_devices) or 1)
        return {
            "target_replicas": max(0, int(target_replicas or 0)),
            "auto_trim_by_free_vram": bool(_pick("auto_trim_by_free_vram", False)),
            "min_free_vram_gb_per_replica": float(_pick("min_free_vram_gb_per_replica", 0.0) or 0.0),
            "reserve_vram_gb": float(_pick("reserve_vram_gb", 0.0) or 0.0),
        }

    def _resource_device_candidates(self, name: str, fallback: Any = None) -> List[str]:
        override = self._device_overrides.get(name)
        if override is not None:
            if isinstance(override, (list, tuple)):
                candidates = self._normalize_device_list(override)
            else:
                single = self._normalize_device_name(override)
                candidates = [single] if single else []
            if candidates:
                return candidates

        if name == "vlm":
            configured = self._normalize_device_list(self._vlm_devices)
            if configured:
                return configured

        if isinstance(fallback, (list, tuple)):
            candidates = self._normalize_device_list(fallback)
        else:
            normalized = self._normalize_device_name(fallback)
            candidates = [normalized] if normalized else []
        if candidates:
            return candidates

        return list(self._gpu_placement_policy.get("default_devices", []))

    def _resource_placement_policy(self, name: str) -> Dict[str, Any]:
        base = dict(self._gpu_placement_policy)
        per_model = dict((base.get("per_model", {}) or {}).get(name, {}) or {})
        if per_model.get("devices") is not None:
            base["default_devices"] = self._normalize_device_list(per_model.get("devices") or [])
        for key in (
            "enabled",
            "strategy",
            "reserve_vram_gb",
            "min_free_vram_gb",
            "prefer_spread",
            "allow_fallback_to_cpu",
        ):
            if key in per_model:
                base[key] = per_model.get(key)
        expected_vram_gb = per_model.get("expected_vram_gb", 0.0)
        base["expected_vram_gb"] = float(expected_vram_gb or 0.0)
        return base

    def _normalize_device_name(self, device: Any) -> Optional[str]:
        if device is None:
            return None
        if isinstance(device, int):
            return f"cuda:{device}"
        s = str(device).strip()
        if not s:
            return None
        if s.isdigit():
            return f"cuda:{s}"
        return s

    def _normalize_device_list(self, devices: Sequence[Any]) -> List[str]:
        out: List[str] = []
        for device in devices or []:
            normalized = self._normalize_device_name(device)
            if normalized:
                out.append(normalized)
        return out

    def _available_or_fallback_device(self, device: Any, *, resource_name: str = "") -> Optional[str]:
        normalized = self._normalize_device_name(device)
        if not normalized or not normalized.startswith("cuda:"):
            return normalized
        available = self._available_cuda_devices()
        if normalized in available:
            return normalized
        fallback = available[0] if available else "cpu"
        print(
            f"[ModelManager] requested device {normalized} for {resource_name or '<unknown>'} "
            f"is not visible; using {fallback} instead"
        )
        return fallback

    def _resolve_device_override(self, name: str, fallback: Any = None) -> Optional[str]:
        if not self._manager_assign_enabled:
            return self._normalize_device_name(fallback)

        device = self._device_overrides.get(name, fallback)
        if isinstance(device, (list, tuple)):
            devices = self._normalize_device_list(device)
            return devices[0] if devices else None
        if device is None and name == "vlm" and self._vlm_devices:
            device = self._vlm_devices[0]
        return self._normalize_device_name(device)

    def _build_config_with_override(
        self,
        *,
        path: Sequence[str],
        value: Any,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.config)
        node: Dict[str, Any] = cfg
        for key in path[:-1]:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        node[path[-1]] = value
        return cfg

    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base or {})
        for key, value in dict(override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge_dicts(dict(merged.get(key) or {}), value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _extract_device(self, inst: Any) -> Optional[str]:
        device = getattr(inst, "_agentpack_device", None)
        if device is not None:
            return self._normalize_device_name(device)

        for attr in ("device", "_device"):
            value = getattr(inst, attr, None)
            normalized = self._normalize_device_name(value)
            if normalized:
                return normalized

        model = getattr(inst, "model", None)
        if model is not None:
            try:
                param = next(model.parameters())
                return self._normalize_device_name(param.device)
            except Exception:
                pass

        pipe = getattr(inst, "pipe", None)
        if pipe is not None:
            device = getattr(pipe, "device", None)
            normalized = self._normalize_device_name(device)
            if normalized:
                return normalized

        ocr = getattr(inst, "ocr", None)
        if ocr is not None:
            normalized = self._normalize_device_name(getattr(inst, "device", None))
            if normalized:
                return normalized

        return None

    def _cuda_device_index(self, device: Any) -> Optional[int]:
        normalized = self._normalize_device_name(device)
        if not normalized or not normalized.startswith("cuda"):
            return None
        if ":" not in normalized:
            return 0
        try:
            return int(normalized.split(":", 1)[1])
        except Exception:
            return None

    def _gpu_summary(self) -> Dict[str, Dict[str, float]]:
        try:
            import torch
        except Exception:
            return {}

        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return {}
        try:
            if not cuda.is_available():
                return {}
            device_count = int(cuda.device_count())
        except Exception:
            return {}

        summary: Dict[str, Dict[str, float]] = {}
        for idx in range(device_count):
            try:
                free, total = cuda.mem_get_info(idx)
                allocated = cuda.memory_allocated(idx)
                reserved = cuda.memory_reserved(idx)
            except Exception:
                continue
            gib = 1024 ** 3
            summary[f"cuda:{idx}"] = {
                "free_gb": free / gib,
                "total_gb": total / gib,
                "allocated_gb": allocated / gib,
                "reserved_gb": reserved / gib,
            }
        return summary

    def _gpu_mem_info_mb(self, device: str) -> tuple[int, int]:
        summary = self._gpu_summary().get(str(device), {})
        return (
            int(float(summary.get("free_gb", 0.0) or 0.0) * 1024.0),
            int(float(summary.get("total_gb", 0.0) or 0.0) * 1024.0),
        )

    def _should_schedule_resource(self, name: str) -> bool:
        spec = self._specs.get(str(name))
        if spec is None or not spec.gpu_heavy:
            return False
        return bool(self._scheduler_cfg.get('enabled', True)) and str(name) in self._scheduled_resource_names

    def _scheduled_resource_device(self, name: str) -> Optional[str]:
        if not self._should_schedule_resource(name):
            return None
        per_resource_devices = self._scheduler_cfg.get("per_resource_devices") or {}
        if isinstance(per_resource_devices, dict):
            explicit = self._available_or_fallback_device(
                per_resource_devices.get(str(name)),
                resource_name=str(name),
            )
            if explicit:
                return explicit

        devices = self._scheduler_cfg.get("devices") or []
        if isinstance(devices, (list, tuple)) and devices:
            normalized_devices = [
                device
                for raw_device in devices
                for device in [
                    self._available_or_fallback_device(raw_device, resource_name=str(name))
                ]
                if device
            ]
            if normalized_devices:
                scheduled_names = ["embedder", "asr", "ocr", "yolo", "raft", "omniparser"]
                try:
                    idx = scheduled_names.index(str(name))
                except ValueError:
                    idx = sum(ord(ch) for ch in str(name))
                return normalized_devices[idx % len(normalized_devices)]

        return (
            self._available_or_fallback_device(self._scheduler_cfg.get("device"), resource_name=str(name))
            or self._available_or_fallback_device("cuda:0", resource_name=str(name))
        )

    def _loaded_gpu_resources_on_device(self, device: str) -> List[str]:
        target = self._normalize_device_name(device)
        if not target:
            return []
        out: List[str] = []
        for name, rt in self._resource_runtime.items():
            if not self._should_schedule_resource(name):
                continue
            if not rt.loaded:
                continue
            if self._normalize_device_name(rt.device) == target:
                out.append(name)
        for name, pool in self._resource_replicas.items():
            for replica in pool:
                if not replica.loaded:
                    continue
                if self._normalize_device_name(replica.device) == target:
                    out.append(name)
        return out

    def _estimated_loaded_vram_mb(self, device: str) -> int:
        total = 0
        for name in self._loaded_gpu_resources_on_device(device):
            rt = self._resource_runtime.get(name)
            if rt is None:
                continue
            total += int(rt.policy.estimated_vram_mb or 0)
        return total

    def _can_admit_resource_load(self, name: str) -> bool:
        if not self._should_schedule_resource(name):
            return True
        rt = self._resource_runtime.get(name)
        if rt is None:
            return False
        device = self._scheduled_resource_device(name) or "cuda:0"
        if not device.startswith("cuda"):
            return True
        estimated_mb = int(rt.policy.estimated_vram_mb or 0)
        if estimated_mb <= 0:
            return True
        free_mb, total_mb = self._gpu_mem_info_mb(device)
        reserved_margin_mb = int(self._scheduler_cfg.get("reserved_margin_mb", 0) or 0)
        loaded_estimated_mb = self._estimated_loaded_vram_mb(device)
        free_ok = free_mb >= (estimated_mb + reserved_margin_mb)
        total_ok = (loaded_estimated_mb + estimated_mb) <= max(0, total_mb - reserved_margin_mb)
        return free_ok and total_ok

    def _wait_for_runtime(self, rt: ResourceRuntime, deadline: float) -> None:
        remaining = float(deadline - time.time())
        if remaining <= 0:
            context = self._current_chat_context()
            context_text = (
                f" stage={context.get('stage', '')}"
                f" video={context.get('video_path', '') or context.get('video_name', '')}"
                f" segment={context.get('segment_index', '')}"
                f" shot={context.get('shot_id', '')}"
            )
            raise TimeoutError(
                f"Timed out waiting for resource {rt.name}: "
                f"waiters={int(rt.waiters)} active_calls={int(rt.active_calls)} "
                f"max_concurrency={int(rt.policy.max_concurrency)} "
                f"device={rt.device or self._scheduled_resource_device(rt.name) or '<unset>'} "
                f"loaded={bool(rt.loaded)} loading={bool(rt.loading)} "
                f"queue_timeout_s={float(rt.policy.queue_timeout_s):.1f} "
                f"last_error={rt.last_error or ''}{context_text}"
            )
        timeout = min(remaining, float(self._scheduler_cfg.get("poll_interval_s", 1.0) or 1.0))
        if rt.cv is None:
            raise TimeoutError(f"Resource {rt.name} has no condition variable")
        rt.cv.wait(timeout=timeout)

    def _raise_if_queue_full_locked(self, rt: ResourceRuntime) -> None:
        max_waiters = int(rt.policy.max_waiters or 0)
        if max_waiters <= 0:
            return
        if int(rt.waiters or 0) <= max_waiters:
            return
        context = self._current_chat_context()
        context_text = (
            f" stage={context.get('stage', '')}"
            f" video={context.get('video_path', '') or context.get('video_name', '')}"
            f" segment={context.get('segment_index', '')}"
            f" shot={context.get('shot_id', '')}"
        )
        raise TimeoutError(
            f"Resource queue full for {rt.name}: "
            f"waiters={int(rt.waiters)} max_waiters={max_waiters} "
            f"active_calls={int(rt.active_calls)} "
            f"device={rt.device or self._scheduled_resource_device(rt.name) or '<unset>'}"
            f"{context_text}"
        )

    def _maybe_collect_idle_victims_locked(self) -> List[str]:
        now = time.time()
        victims: List[tuple[int, float, str]] = []
        for name, rt in self._resource_runtime.items():
            spec = self._specs.get(name)
            if spec is None or spec.persistent or not self._should_schedule_resource(name):
                continue
            if not rt.loaded or rt.loading or rt.active_calls > 0:
                continue
            idle_age = max(0.0, now - float(rt.last_used_ts or 0.0))
            if idle_age < float(rt.policy.idle_unload_s or 0.0):
                continue
            victims.append((int(rt.policy.unload_priority or 0), -idle_age, name))
        victims.sort()
        return [name for _, __, name in victims]

    def _maybe_unload_idle_resources(self) -> None:
        with self._lock:
            victims = self._maybe_collect_idle_victims_locked()
        for name in victims:
            self.release(name)

    def _collect_admission_victims_locked(
        self,
        incoming: str,
        device: Optional[str],
        required_mb: int,
    ) -> List[str]:
        target = self._normalize_device_name(device)
        if not target or not target.startswith('cuda') or required_mb <= 0:
            return []

        candidates: List[tuple[int, float, str]] = []
        for name, rt in self._resource_runtime.items():
            if name == incoming or not self._should_schedule_resource(name):
                continue
            spec = self._specs.get(name)
            if spec is None or spec.persistent:
                continue
            if not rt.loaded or rt.loading or rt.active_calls > 0:
                continue
            if self._normalize_device_name(rt.device) != target:
                continue
            candidates.append((int(rt.policy.unload_priority or 0), float(rt.last_used_ts or 0.0), name))

        candidates.sort()
        victims: List[str] = []
        reclaimed_mb = 0
        for _, __, name in candidates:
            victims.append(name)
            rt = self._resource_runtime.get(name)
            if rt is not None:
                reclaimed_mb += int(rt.policy.estimated_vram_mb or 0)
            if reclaimed_mb >= required_mb:
                break
        return victims

    def _admission_victims_for_resource_locked(self, name: str, device: Optional[str] = None) -> List[str]:
        rt = self._resource_runtime.get(name)
        if rt is None:
            return []
        target = self._normalize_device_name(device) or self._scheduled_resource_device(name) or 'cuda:0'
        if not target.startswith('cuda'):
            return []
        estimated_mb = int(rt.policy.estimated_vram_mb or 0)
        if estimated_mb <= 0:
            return []
        free_mb, _ = self._gpu_mem_info_mb(target)
        reserved_margin_mb = int(self._scheduler_cfg.get('reserved_margin_mb', 0) or 0)
        shortfall_mb = max(0, estimated_mb + reserved_margin_mb - free_mb)
        if shortfall_mb <= 0:
            shortfall_mb = estimated_mb
        return self._collect_admission_victims_locked(name, target, shortfall_mb)

    def _notify_resource_waiters_locked(self, name: str) -> None:
        rt = self._resource_runtime.get(name)
        if rt is not None and rt.cv is not None:
            rt.cv.notify_all()
        self._gpu_cv.notify_all()

    def _torch_oom_error_type(self) -> tuple[type[BaseException], ...]:
        try:
            import torch
        except Exception:
            return ()
        oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_type is None:
            return ()
        return (oom_type,)

    def _current_device_loads(self) -> Dict[str, int]:
        loads: Dict[str, int] = {}
        for item in self._resident_units():
            device = self._normalize_device_name(item.get("device"))
            if not device or not device.startswith("cuda"):
                continue
            loads[device] = loads.get(device, 0) + 1
        return loads

    def _select_device_for_resource(
        self,
        name: str,
        *,
        fallback: Any = None,
        exclude_devices: Optional[Set[str]] = None,
    ) -> Optional[str]:
        forced = getattr(self._resource_replica_state, "forced_device", None)
        if isinstance(forced, tuple) and forced[0] == name:
            forced_device = self._normalize_device_name(forced[1])
            if forced_device:
                return forced_device
        scheduled_device = self._scheduled_resource_device(name)
        if scheduled_device:
            return scheduled_device
        # Preserve an explicit CPU-only legacy model configuration when the
        # resource is outside the scheduler allowlist.  Without this fallback,
        # the generic GPU placement policy can silently move YOLO back to a
        # busy GPU even though models.yolo_device is set to ``cpu``.
        if fallback is None and str(name) == "yolo":
            fallback = self._models_cfg().get("yolo_device")
        if not self._manager_assign_enabled:
            return self._resolve_device_override(name, fallback)

        policy = self._resource_placement_policy(name)
        candidates = self._resource_device_candidates(name, fallback)
        if not bool(policy.get("enabled", True)):
            return candidates[0] if candidates else self._resolve_device_override(name, fallback)

        exclude = {d for d in (exclude_devices or set()) if d}
        gpu_summary = self._gpu_summary()
        loads = self._current_device_loads()
        reserve_gb = float(policy.get("reserve_vram_gb", 0.0) or 0.0)
        min_free_gb = float(policy.get("min_free_vram_gb", 0.0) or 0.0)
        expected_vram_gb = float(policy.get("expected_vram_gb", 0.0) or 0.0)
        prefer_spread = bool(policy.get("prefer_spread", True))

        scored: List[tuple[float, float, int, str]] = []
        for device in candidates:
            if not device or device in exclude:
                continue
            if device.startswith("cpu"):
                continue
            info = gpu_summary.get(device)
            if info is None:
                continue
            usable_gb = float(info.get("free_gb", 0.0)) - reserve_gb
            required_gb = min_free_gb + expected_vram_gb
            if usable_gb < required_gb:
                continue
            load_count = int(loads.get(device, 0))
            spread_bonus = -load_count if prefer_spread else 0
            scored.append((usable_gb, float(spread_bonus), -load_count, device))

        if scored:
            scored.sort(reverse=True)
            return scored[0][3]

        filtered_candidates = [
            d for d in candidates
            if d and d not in exclude
        ]

        if filtered_candidates:
            return filtered_candidates[0]

        if bool(policy.get("allow_fallback_to_cpu", True)):
            return "cpu"
        return None
    
    def _select_vlm_devices_for_target(self, target_size: int) -> List[str]:
        target_size = int(target_size or 0)
        if target_size <= 0:
            return []
        if self._is_vllm_api_mode():
            service_candidates = self._eligible_vllm_service_devices(strict=False)
            if service_candidates:
                return service_candidates[:target_size]
        candidates = self._resource_device_candidates("vlm", self._vlm_devices)
        if not candidates:
            selected = self._select_device_for_resource("vlm")
            return [selected] if selected else []

        selected: List[str] = []
        excluded: Set[str] = set()
        prefer_spread = bool(self._resource_placement_policy("vlm").get("prefer_spread", True))
        while len(selected) < target_size:
            device = self._select_device_for_resource(
                "vlm",
                fallback=candidates,
                exclude_devices=excluded if prefer_spread else None,
            )
            if not device:
                break
            selected.append(device)
            if prefer_spread:
                excluded.add(device)
                if len(excluded) >= len(set(candidates)):
                    excluded = set()
        return selected

    def _eligible_vllm_service_devices(self, *, strict: bool) -> List[str]:
        candidates = self._resource_device_candidates("vlm", self._vlm_devices)
        if not candidates:
            return []
        gpu_summary = self._gpu_summary()
        reserve_gb = float(self._vlm_policy.get("reserve_vram_gb", 0.0) or 0.0)
        min_free_gb = float(self._vlm_policy.get("min_free_vram_gb_per_replica", 0.0) or 0.0)

        scored: List[tuple[float, str]] = []
        fallback: List[str] = []
        for device in candidates:
            info = gpu_summary.get(device)
            if info is None:
                continue
            fallback.append(device)
            usable_gb = float(info.get("free_gb", 0.0)) - reserve_gb
            if usable_gb >= min_free_gb:
                scored.append((float(info.get("free_gb", 0.0)), device))

        if scored:
            scored.sort(reverse=True)
            return [device for _, device in scored]
        return [] if strict else fallback

    def _budget_available_for_new_units(self, incoming_units: int) -> int:
        return max(0, self.max_gpu_resident_models - self._current_gpu_heavy_count() - incoming_units)

    def _vlm_resident_count(self) -> int:
        with self._vlm_pool_lock:
            if self._vlm_pool:
                return len(self._vlm_pool)
        with self._lock:
            return 1 if self._instances.get("vlm") is not None else 0

    def _desired_local_vlm_pool_size(self) -> int:
        configured = max(0, int(self._vlm_pool_size or 0))
        if self._is_vllm_api_mode():
            budget_limited = configured
        else:
            budget_limited = min(configured, max(1, self.max_gpu_resident_models))
        requested = budget_limited

        if self._vlm_policy.get("auto_trim_by_free_vram"):
            allowed_devices = self._eligible_vllm_service_devices(strict=True)
            if allowed_devices:
                requested = min(requested, len(allowed_devices))
            elif configured > 0 and float(self._vlm_policy.get("min_free_vram_gb_per_replica", 0.0) or 0.0) > 0:
                requested = 1

        return max(0, requested)

    def _desired_vlm_pool_size(self) -> int:
        local_desired = self._desired_local_vlm_pool_size()
        if not self._is_vllm_api_mode():
            return local_desired

        service_manager = self._vllm_service_manager
        remote_target = int(getattr(service_manager, "remote_service_target_count", lambda: 0)() or 0) if service_manager is not None else 0
        total_desired = max(0, local_desired + max(0, remote_target))
        if service_manager is not None and service_manager.active_service_count() > 0:
            return max(0, min(total_desired, int(service_manager.active_service_count())))
        return total_desired

    def _resident_units(self) -> List[Dict[str, Any]]:
        units: List[Dict[str, Any]] = []
        with self._lock:
            instances = dict(self._instances)
            replica_pools = {name: list(pool) for name, pool in self._resource_replicas.items()}

        for name, inst in instances.items():
            spec = self._specs.get(name)
            if spec is None or not spec.gpu_heavy or inst is None:
                continue
            if name == "vlm":
                continue
            units.append(
                {
                    "unit": name,
                    "name": name,
                    "device": self._extract_device(inst),
                    "persistent": bool(spec.persistent),
                }
            )

        for name, pool in replica_pools.items():
            for replica in pool:
                if not replica.loaded or replica.instance is None:
                    continue
                spec = self._specs.get(name)
                if spec is None or not spec.gpu_heavy:
                    continue
                units.append(
                    {
                        "unit": replica.replica_id,
                        "name": name,
                        "device": replica.device,
                        "persistent": bool(spec.persistent),
                    }
                )

        with self._vlm_pool_lock:
            pool = list(self._vlm_pool)

        if pool:
            for idx, inst in enumerate(pool):
                units.append(
                    {
                        "unit": f"vlm#{idx}",
                        "name": "vlm",
                        "device": self._extract_device(inst),
                        "persistent": False,
                    }
                )
        elif instances.get("vlm") is not None:
            units.append(
                {
                    "unit": "vlm#0",
                    "name": "vlm",
                    "device": self._extract_device(instances["vlm"]),
                    "persistent": False,
                }
            )
        return units

    def _evict_one_gpu_model(self, incoming: Optional[str] = None) -> bool:
        victim = None
        victim_ts = None
        for name, inst in self._instances.items():
            if inst is None or name == incoming:
                continue
            spec = self._specs.get(name)
            if spec is None or not spec.gpu_heavy or spec.persistent:
                continue
            ts = self._last_used_ts.get(name, 0.0)
            if victim is None or ts < victim_ts:
                victim = name
                victim_ts = ts
        if victim is None:
            return False
        self._eviction_counts[victim] = self._eviction_counts.get(victim, 0) + 1
        self.release(victim)
        return True

    def _evict_one_idle_replica(self, incoming: Optional[str] = None) -> bool:
        victim: tuple[str, ResourceReplica] | None = None
        victim_ts: float | None = None
        for name, pool in self._resource_replicas.items():
            if name == incoming:
                continue
            spec = self._specs.get(name)
            if spec is None or spec.persistent:
                continue
            for replica in pool:
                if replica.active_calls > 0 or replica.loading:
                    continue
                ts = float(replica.last_used_ts or 0.0)
                if victim is None or ts < float(victim_ts or 0.0):
                    victim = (name, replica)
                    victim_ts = ts
        if victim is None:
            return False
        name, replica = victim
        pool = self._resource_replicas.get(name, [])
        if replica in pool:
            pool.remove(replica)
        self._eviction_counts[name] = self._eviction_counts.get(name, 0) + 1
        rt = self._resource_runtime.get(name)
        if rt is not None:
            rt.loaded = bool(pool)
            rt.last_unloaded_ts = time.time()
            self._notify_resource_waiters_locked(name)
        inst = replica.instance
        replica.instance = None
        replica.loaded = False
        if inst is not None:
            try:
                close_fn = getattr(inst, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            del inst
        return True

    def _build_vlm_replica(self, replica_idx: int) -> Any:
        models_cfg = self._models_cfg()
        vlm_cfg = copy.deepcopy((models_cfg.get("qwen_vlm", {}) or {}))

        service_manager = self._vllm_service_manager
        if self._is_vllm_api_mode() and service_manager is not None:
            self._ensure_vllm_service_pool()
        service_devices = service_manager.active_devices() if (self._is_vllm_api_mode() and service_manager is not None) else []
        device = service_devices[int(replica_idx) % len(service_devices)] if service_devices else self._select_next_vlm_device(replica_idx)
        api_cfg = dict((vlm_cfg.get("api", {}) or {}))
        if api_cfg:
            chosen_base_url = ""
            if self._is_vllm_api_mode() and service_manager is not None and device:
                chosen_base_url = str(service_manager.endpoint_for_device(device) or "").strip()
            if not chosen_base_url:
                base_url_by_device = dict(api_cfg.get("base_url_by_device", {}) or {})
                if device and str(device) in base_url_by_device:
                    chosen_base_url = str(base_url_by_device.get(str(device), "") or "").strip()
                if not chosen_base_url:
                    raw_base_urls = api_cfg.get("base_urls", []) or []
                    if not isinstance(raw_base_urls, (list, tuple)):
                        raw_base_urls = [raw_base_urls]
                    base_urls = [str(x).strip() for x in raw_base_urls if str(x).strip()]
                    if base_urls:
                        chosen_base_url = base_urls[int(replica_idx) % len(base_urls)]
            if chosen_base_url:
                api_cfg["base_url"] = chosen_base_url
            vlm_cfg["api"] = api_cfg

        print(
            f"[ModelManager][VLM] create replica={replica_idx} "
            f"used={self._current_vlm_devices()} chosen={device} "
            f"candidates={self._resource_device_candidates('vlm', self._vlm_devices)}"
        )

        vlm = self._lazy_import("executor.models.qwen_vlm", "QwenVLM")(
            vlm_cfg,
            device=device,
        )

        vlm_runtime_cfg = (self.config.get("qwen_vlm", {}) or {}) if isinstance(self.config, dict) else {}
        oom_cfg = (vlm_runtime_cfg.get("oom_retry", {}) or {}) if isinstance(vlm_runtime_cfg, dict) else {}

        def _as_bool(x: Any, default: bool = False) -> bool:
            if x is None:
                return default
            if isinstance(x, bool):
                return x
            s = str(x).strip().lower()
            return s not in ("0", "false", "no", "off", "")

        vlm.mem_debug = _as_bool(vlm_runtime_cfg.get("mem_debug", False), False)
        vlm.empty_cache_on_return = _as_bool(vlm_runtime_cfg.get("empty_cache_on_return", False), False)
        vlm.gc_on_return = _as_bool(vlm_runtime_cfg.get("gc_on_return", True), True)

        cap = int(vlm_runtime_cfg.get("max_new_tokens_cap", 0) or 0)
        vlm.max_new_tokens_cap = cap if cap > 0 else None

        vlm.oom_retry_enable = _as_bool(oom_cfg.get("enable", True), True)
        vlm.oom_retry_sleep_s = float(oom_cfg.get("sleep_s", 0.2) or 0.2)

        max_sides = oom_cfg.get("max_sides", [1024, 768, 512]) or [1024, 768, 512]
        token_scales = oom_cfg.get("token_scales", [1.0, 0.75, 0.5]) or [1.0, 0.75, 0.5]
        vlm.oom_retry_max_sides = [int(v) for v in max_sides if int(v) > 0]
        vlm.oom_retry_token_scales = [float(v) for v in token_scales if float(v) > 0]

        setattr(vlm, "_agentpack_replica_idx", int(replica_idx))
        setattr(vlm, "_agentpack_device", device)
        if api_cfg.get("base_url"):
            setattr(vlm, "_agentpack_api_base_url", str(api_cfg.get("base_url")))
        return self._apply_chat_logging(vlm)

    def _ensure_vlm_pool_size(self, target_size: int) -> None:
        if self._is_vllm_api_mode():
            self._ensure_vllm_service_pool(target_size)
            manager = self._vllm_service_manager
            if manager is not None and manager.active_service_count() > 0:
                target_size = min(max(0, int(target_size or 0)), int(manager.active_service_count()))
        requested_size = max(0, int(target_size or 0))
        desired_size = self._desired_vlm_pool_size()
        if self._is_vllm_api_mode():
            manager = self._vllm_service_manager
            active_count = 0
            if manager is not None:
                active_count = max(
                    int(getattr(manager, "active_service_count", lambda: 0)() or 0),
                    len(list(getattr(manager, "active_devices", lambda: [])() or [])),
                )
            if active_count > 0:
                desired_size = max(desired_size, min(requested_size, active_count))
        target_size = min(requested_size, desired_size)
        with self._vlm_pool_lock:
            if target_size <= 0:
                self._touch("vlm")
                self._vlm_slot_cv.notify_all()
                return
            if not self._vlm_pool:
                with self._lock:
                    primary = self._instances.get("vlm")
                if primary is not None:
                    self._vlm_pool.append(primary)
            while len(self._vlm_pool) < target_size:
                if not self._is_vllm_api_mode():
                    while self._current_gpu_heavy_count() >= self.max_gpu_resident_models:
                        if not self._evict_one_gpu_model(incoming="vlm"):
                            break
                    if self._current_gpu_heavy_count() >= self.max_gpu_resident_models:
                        break
                replica_idx = len(self._vlm_pool)
                replica = self._build_vlm_replica(replica_idx)
                self._vlm_pool.append(replica)
                self._load_counts["vlm"] = self._load_counts.get("vlm", 0) + 1
                if replica_idx == 0:
                    with self._lock:
                        self._instances.setdefault("vlm", replica)
            self._touch("vlm")
            self._vlm_slot_cv.notify_all()

    def warm_vlm_pool(self) -> None:
        self._ensure_vlm_pool_size(self.get_vlm_pool_size())

    def acquire_video_vlm_slot(self, video_path: str = "") -> None:
        if not self._concurrent_video_enabled:
            return
        current = self._current_video_vlm_slot_indices()
        if current:
            return
        requested_slots = max(1, int(self._vlm_slots_per_video or 1))
        requested_pool_size = max(requested_slots, int(self.get_vlm_pool_size() or 1))
        self._ensure_vlm_pool_size(requested_pool_size)
        with self._vlm_pool_lock:
            pool_size = len(self._vlm_pool)
        if pool_size <= 0:
            raise RuntimeError("No VLM replicas are configured for video parallelism")
        target_slots = min(requested_slots, pool_size)
        slot_indices, vlms = self._checkout_vlm_slots(target_slots)
        devices = [str(getattr(vlm, "_agentpack_device", "") or "") for vlm in vlms]
        self._vlm_thread_state.slot_idx = int(slot_indices[0]) if slot_indices else None
        self._vlm_thread_state.slot_indices = [int(idx) for idx in slot_indices]
        self._vlm_thread_state.video_path = str(video_path or "")
        self._vlm_thread_state.device = devices[0] if devices else ""
        self._vlm_thread_state.devices = devices
        print(
            f"[ModelManager][VLM] bind video={self._vlm_thread_state.video_path or '<unknown>'} "
            f"slots={slot_indices} devices={devices or ['<unknown>']}"
        )

    def release_video_vlm_slot(self) -> None:
        slot_indices = self._current_video_vlm_slot_indices()
        if not slot_indices:
            return
        devices = list(getattr(self._vlm_thread_state, "devices", []) or [])
        device = str(getattr(self._vlm_thread_state, "device", "") or "")
        video_path = str(getattr(self._vlm_thread_state, "video_path", "") or "")
        self._vlm_thread_state.slot_idx = None
        self._vlm_thread_state.slot_indices = []
        self._vlm_thread_state.device = ""
        self._vlm_thread_state.devices = []
        self._vlm_thread_state.video_path = ""
        for slot_idx in slot_indices:
            self._checkin_vlm_slot(int(slot_idx))
        print(
            f"[ModelManager][VLM] release video={video_path or '<unknown>'} "
            f"slots={slot_indices} devices={devices or ([device] if device else ['<unknown>'])}"
        )

    def _current_video_vlm_slot_indices(self) -> List[int]:
        raw_slots = getattr(self._vlm_thread_state, "slot_indices", None)
        if raw_slots:
            out: List[int] = []
            for raw in list(raw_slots or []):
                try:
                    out.append(int(raw))
                except Exception:
                    continue
            if out:
                return out
        slot_idx = getattr(self._vlm_thread_state, "slot_idx", None)
        if slot_idx is None:
            return []
        try:
            return [int(slot_idx)]
        except Exception:
            return []

    def _current_video_vlm_slot_idx(self) -> Optional[int]:
        slots = self._current_video_vlm_slot_indices()
        return int(slots[0]) if slots else None

    def _checkout_vlm_slot(self) -> tuple[int, Any]:
        slots, vlms = self._checkout_vlm_slots(1)
        return slots[0], vlms[0]

    def _checkout_vlm_slots(self, count: int) -> tuple[List[int], List[Any]]:
        target = max(1, int(count or 1))
        with self._vlm_slot_cv:
            while True:
                available = [
                    (idx, vlm)
                    for idx, vlm in enumerate(self._vlm_pool)
                    if idx not in self._vlm_busy_slots
                ]
                if len(available) >= target:
                    chosen = available[:target]
                    slot_indices = [int(idx) for idx, _ in chosen]
                    for idx in slot_indices:
                        self._vlm_busy_slots.add(idx)
                    self._touch("vlm")
                    return slot_indices, [vlm for _, vlm in chosen]
                self._vlm_slot_cv.wait()

    def _checkin_vlm_slot(self, slot_idx: int) -> None:
        with self._vlm_slot_cv:
            self._vlm_busy_slots.discard(int(slot_idx))
            self._vlm_slot_cv.notify()

    def get_vlm_pool_size(self) -> int:
        return self._desired_vlm_pool_size()

    def run_vlm_jobs(
        self,
        jobs: Sequence[Any],
        worker_fn: Callable[[Any, Any], Any],
        max_workers: Optional[int] = None,
    ) -> List[Any]:
        items = list(jobs or [])
        if not items:
            return []
        pool_size = self.get_vlm_pool_size()
        if pool_size <= 0:
            raise RuntimeError("No VLM replicas are available for queued jobs")
        affinity_slot_indices = self._current_video_vlm_slot_indices()
        worker_count = max(1, min(int(max_workers or pool_size), pool_size, len(items)))
        if affinity_slot_indices:
            worker_count = max(
                1,
                min(
                    int(max_workers or len(affinity_slot_indices)),
                    len(affinity_slot_indices),
                    len(items),
                ),
            )
        chat_logger = self._current_chat_logger()
        chat_context = self._current_chat_context()
        stage = str(chat_context.get("stage") or "unknown")
        segment_index = chat_context.get("segment_index", None)
        segment_text = f" segment={segment_index}" if segment_index is not None else ""
        print(
            f"[ModelManager][VLM] batch start stage={stage}{segment_text} "
            f"jobs={len(items)} workers={worker_count}"
        )
        if affinity_slot_indices:
            self._ensure_vlm_pool_size(max(1, max(affinity_slot_indices) + 1))
        else:
            self._ensure_vlm_pool_size(worker_count)

        def _run_item(item_idx: int, item: Any) -> tuple[int, Any]:
            slot_idx, vlm = self._checkout_vlm_slot()
            try:
                set_logger = getattr(vlm, "set_chat_logger", None)
                if callable(set_logger):
                    set_logger(chat_logger)
                set_context = getattr(vlm, "set_chat_context", None)
                if callable(set_context):
                    set_context(chat_context)
                return item_idx, worker_fn(vlm, item)
            finally:
                self._checkin_vlm_slot(slot_idx)

        if affinity_slot_indices:
            with self._vlm_pool_lock:
                missing = [idx for idx in affinity_slot_indices if idx >= len(self._vlm_pool)]
                if missing:
                    raise RuntimeError(
                        f"Bound VLM slots {missing} are unavailable for current video thread"
                    )
                bound_vlms = [self._vlm_pool[idx] for idx in affinity_slot_indices[:worker_count]]
                self._touch("vlm")

            def _run_bound_chunk(worker_idx: int, chunk: List[tuple[int, Any]]) -> List[tuple[int, Any]]:
                vlm = bound_vlms[worker_idx]
                set_logger = getattr(vlm, "set_chat_logger", None)
                if callable(set_logger):
                    set_logger(chat_logger)
                set_context = getattr(vlm, "set_chat_context", None)
                if callable(set_context):
                    set_context(chat_context)
                out: List[tuple[int, Any]] = []
                for item_idx, item in chunk:
                    out.append((item_idx, worker_fn(vlm, item)))
                return out

            chunks: List[List[tuple[int, Any]]] = [[] for _ in range(worker_count)]
            for idx, item in enumerate(items):
                chunks[idx % worker_count].append((idx, item))
            results = [None] * len(items)
            if worker_count <= 1:
                for item_idx, value in _run_bound_chunk(0, chunks[0]):
                    results[item_idx] = value
            else:
                with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vlm-bound") as executor:
                    futures = [executor.submit(_run_bound_chunk, idx, chunk) for idx, chunk in enumerate(chunks) if chunk]
                    for future in futures:
                        for item_idx, value in future.result():
                            results[item_idx] = value
            print(
                f"[ModelManager][VLM] batch done stage={stage}{segment_text} "
                f"jobs={len(items)} workers={worker_count} bound_slots={affinity_slot_indices[:worker_count]}"
            )
            return results

        if worker_count <= 1:
            results = [None] * len(items)
            for idx, item in enumerate(items):
                item_idx, value = _run_item(idx, item)
                results[item_idx] = value
            print(
                f"[ModelManager][VLM] batch done stage={stage}{segment_text} "
                f"jobs={len(items)} workers=1"
            )
            return results

        results: List[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="vlm-pool") as executor:
            futures = [executor.submit(_run_item, idx, item) for idx, item in enumerate(items)]
            for future in futures:
                item_idx, value = future.result()
                results[item_idx] = value
        print(
            f"[ModelManager][VLM] batch done stage={stage}{segment_text} "
            f"jobs={len(items)} workers={worker_count}"
        )
        return results

    def _register_from_registry(self) -> None:
        for name, spec in self.resource_registry.enabled_resources().items():
            self._register(
                ResourceSpec(
                    name=name,
                    group=str(spec.get("group", "model") or "model"),
                    scope=str(spec.get("scope", "segment") or "segment"),
                    factory_path=str(((spec.get("factory", {}) or {}).get("class_path")) or ""),
                    persistent=bool(spec.get("persistent", False)),
                    warm_reuse=bool(spec.get("warm_reuse", False)),
                    gpu_heavy=bool(spec.get("gpu_heavy", False)),
                    device_policy=dict(spec.get("device_policy", {}) or {}),
                    config_binding=dict(spec.get("config_binding", {}) or {}),
                    runtime=dict(spec.get("runtime", {}) or {}),
                )
            )

    def _tag_instance_device(self, inst: Any, device: Optional[str]) -> Any:
        if device:
            setattr(inst, "_agentpack_device", device)
        return inst

    def _build_llm(self) -> Any:
        device = self._select_device_for_resource("llm")
        use_auto_device_map = (self.config.get("models", {}) or {}).get(
            "qwen_llm_auto_device_map", True
        )
        if device and device.startswith("cuda:"):
            use_auto_device_map = False
        inst = self._lazy_import("executor.models.qwen_llm", "QwenLLM")(
            (self.config.get("models", {}) or {})["qwen_llm"],
            device=device or "cuda",
            use_auto_device_map=use_auto_device_map,
            load_in_4bit=(self.config.get("models", {}) or {}).get(
                "qwen_llm_load_in_4bit", True
            ),
            load_in_8bit=(self.config.get("models", {}) or {}).get(
                "qwen_llm_load_in_8bit", False
            ),
        )
        return self._tag_instance_device(inst, device)

    def _build_local_embedder_for_device(self, device: Optional[str]) -> Any:
        inst = self._lazy_import(
            "executor.tasks.image_embedding_diversity", "ImageEmbeddingDiversity"
        )(
            self._build_config_with_override(
                path=("tasks", "T2_image_embedding", "device"),
                value=device,
            )
        )
        return self._tag_instance_device(inst, device)

    def _build_local_asr_for_device(self, device: Optional[str]) -> Any:
        inst = self._lazy_import(
            "executor.tasks.whisper_asr", "WhisperASR"
        )(
            self._build_config_with_override(
                path=("tasks", "T2_asr", "device"),
                value=device,
            )
        )
        return self._tag_instance_device(inst, device)

    def _build_local_ocr_for_device(self, device: Optional[str]) -> Any:
        inst = self._lazy_import(
            "executor.tasks.paddle_ocr", "PaddleOCRExecutor"
        )(
            self._build_config_with_override(
                path=("tasks", "T4_ocr", "device"),
                value=device,
            )
        )
        return self._tag_instance_device(inst, device)

    def _build_local_yolo_for_device(self, device: Optional[str]) -> Any:
        inst = self._lazy_import(
            "executor.tasks.yolov8_detect", "YoloDetector"
        )(
            self._build_config_with_override(
                path=("models", "yolo_device"),
                value=device,
            )
        )
        return self._tag_instance_device(inst, device)

    def _build_local_raft_for_device(self, device: Optional[str]) -> Any:
        raft_kwargs = self._build_raft_kwargs()
        inst = self._lazy_import(
            "executor.tasks.optical_flow_raft", "RAFTOpticalFlow"
        )(
            **{
                **raft_kwargs,
                "device": device,
            }
        )
        return self._tag_instance_device(inst, device)

    def _build_local_omniparser_for_device(self, device: Optional[str]) -> Any:
        inst = self._lazy_import(
            "executor.tasks.omniparser", "OmniParserExecutor"
        )(
            self._build_config_with_override(
                path=("models", "omniparser", "device"),
                value=device,
            )
        )
        return self._tag_instance_device(inst, device)

    def _build_remote_tool_resource(
        self,
        name: str,
        *,
        device: Optional[str],
        fallback_device: Optional[str],
        local_builder: Callable[[Optional[str]], Any],
    ) -> Any:
        server_cfg = self._remote_tool_server_cfg()
        resource_cfg = self._remote_tool_resource_cfg(name)
        logical_device = self._normalize_device_name(device) or ""
        remote_device = (
            self._normalize_device_name(resource_cfg.get("remote_device"))
            or self._normalize_device_name(resource_cfg.get("device"))
            or "cuda:0"
        )
        config_overrides = self._deep_merge_dicts(
            dict(server_cfg.get("config_overrides", {}) or {}),
            dict(resource_cfg.get("config_overrides", {}) or {}),
        )
        gpu_guard_cfg = self._deep_merge_dicts(
            dict(server_cfg.get("gpu_guard", {}) or {}),
            dict(resource_cfg.get("gpu_guard", {}) or {}),
        )
        remote_python = str(
            resource_cfg.get("python_executable")
            or server_cfg.get("python_executable")
            or ""
        ).strip()
        fallback_factory: Callable[[], Any] | None = None
        if self._local_tool_fallback_enabled():
            local_device = self._local_tool_fallback_device(name, fallback=fallback_device)
            if local_device:
                fallback_factory = lambda local_device=local_device: local_builder(local_device)
        proxy = RemoteToolResourceProxy(
            client=RemoteToolServerClient(
                resource_name=name,
                ssh_host_alias=str(server_cfg.get("ssh_host_alias") or "server-c").strip(),
                remote_agentpack_root=str(server_cfg.get("remote_agentpack_root") or "").strip(),
                remote_request_root=str(server_cfg.get("remote_request_root") or "").strip(),
                remote_python_executable=remote_python,
                remote_device=remote_device,
                logical_device=logical_device or remote_device,
                base_config=self.config,
                config_overrides=config_overrides,
                ssh_port=int(server_cfg.get("ssh_port", 0) or 0),
                ssh_extra_args=list(server_cfg.get("ssh_extra_args", []) or []),
                gpu_guard=gpu_guard_cfg,
                preflight_script=str(
                    resource_cfg.get("preflight_script") or server_cfg.get("preflight_script") or ""
                ).strip(),
                preflight_every_request=bool(
                    resource_cfg.get("preflight_every_request", server_cfg.get("preflight_every_request", False))
                ),
                retry_attempts=max(1, int(server_cfg.get("retry_attempts", 2) or 1)),
                retry_backoff_s=float(server_cfg.get("retry_backoff_s", 0.5) or 0.0),
                request_timeout_s=float(server_cfg.get("request_timeout_s", 1200.0) or 1200.0),
                ssh_connect_timeout_s=float(server_cfg.get("ssh_connect_timeout_s", 15.0) or 15.0),
                cleanup_remote=bool(server_cfg.get("cleanup_remote", True)),
            ),
            fallback_factory=fallback_factory,
        )
        return self._tag_instance_device(proxy, logical_device or remote_device)

    def _build_embedder(self) -> Any:
        fallback = ((self._tasks_cfg().get("T2_image_embedding", {}) or {}).get("device"))
        device = self._select_device_for_resource("embedder", fallback=fallback)
        if self._should_use_remote_tool_server("embedder", device):
            return self._build_remote_tool_resource(
                "embedder",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_embedder_for_device,
            )
        return self._build_local_embedder_for_device(device)

    def _build_asr(self) -> Any:
        fallback = ((self._tasks_cfg().get("T2_asr", {}) or {}).get("device"))
        device = self._select_device_for_resource("asr", fallback=fallback)
        if self._should_use_remote_tool_server("asr", device):
            return self._build_remote_tool_resource(
                "asr",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_asr_for_device,
            )
        return self._build_local_asr_for_device(device)

    def _build_ocr(self) -> Any:
        fallback = ((self._tasks_cfg().get("T4_ocr", {}) or {}).get("device"))
        device = self._select_device_for_resource("ocr", fallback=fallback)
        if self._should_use_remote_tool_server("ocr", device):
            return self._build_remote_tool_resource(
                "ocr",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_ocr_for_device,
            )
        return self._build_local_ocr_for_device(device)

    def _build_yolo(self) -> Any:
        fallback = self._models_cfg().get("yolo_device")
        device = self._select_device_for_resource("yolo", fallback=fallback)
        if self._should_use_remote_tool_server("yolo", device):
            return self._build_remote_tool_resource(
                "yolo",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_yolo_for_device,
            )
        return self._build_local_yolo_for_device(device)

    def _build_raft(self) -> Any:
        raft_kwargs = self._build_raft_kwargs()
        fallback = raft_kwargs.get("device")
        device = self._select_device_for_resource("raft", fallback=fallback)
        if self._should_use_remote_tool_server("raft", device):
            return self._build_remote_tool_resource(
                "raft",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_raft_for_device,
            )
        return self._build_local_raft_for_device(device)

    def _build_omniparser(self) -> Any:
        fallback = (((self._models_cfg().get("omniparser", {}) or {}).get("device")))
        device = self._select_device_for_resource("omniparser", fallback=fallback)
        if self._should_use_remote_tool_server("omniparser", device):
            return self._build_remote_tool_resource(
                "omniparser",
                device=device,
                fallback_device=fallback,
                local_builder=self._build_local_omniparser_for_device,
            )
        return self._build_local_omniparser_for_device(device)

    def _touch(self, name: str) -> None:
        self._last_used_ts[name] = time.time()

    def _current_gpu_heavy_count(self) -> int:
        return len(self._resident_units())

    def _evict_if_needed(self, incoming: Optional[str] = None) -> None:
        while self._current_gpu_heavy_count() >= self.max_gpu_resident_models:
            if not self._evict_one_gpu_model(incoming=incoming):
                if not self._evict_one_idle_replica(incoming=incoming):
                    break

    def _instantiate_resource(self, name: str, device: Optional[str] = None) -> Any:
        spec = self._specs[name]
        builder_cls = self._load_class(spec.factory_path)
        builder = builder_cls()
        resolved_device = self._normalize_device_name(device) or self._select_device_for_resource(name)
        old_override = self._device_overrides.get(name)
        old_forced = getattr(self._resource_replica_state, "forced_device", None)
        if resolved_device:
            self._device_overrides[name] = resolved_device
            self._resource_replica_state.forced_device = (name, resolved_device)
        try:
            return self._wrap_shared_resource(
                name,
                self._apply_chat_logging(
                    builder.build(
                    manager=self,
                    resource_name=name,
                    resource_spec={
                        "name": spec.name,
                        "group": spec.group,
                        "scope": spec.scope,
                        "persistent": spec.persistent,
                        "warm_reuse": spec.warm_reuse,
                        "gpu_heavy": spec.gpu_heavy,
                        "device_policy": dict(spec.device_policy),
                        "config_binding": dict(spec.config_binding),
                        "runtime": dict(spec.runtime),
                    },
                    config=self.config,
                    resolved_device=resolved_device,
                    )
                )
            )
        finally:
            self._resource_replica_state.forced_device = old_forced
            if old_override is None:
                self._device_overrides.pop(name, None)
            else:
                self._device_overrides[name] = old_override

    def _legacy_get_resource(self, name: str) -> Any:
        with self._lock:
            if name not in self._instances:
                spec = self._specs[name]
                if spec.gpu_heavy:
                    self._evict_if_needed(incoming=name)
                inst = self._instantiate_resource(name)
                self._instances[name] = inst
                self._load_counts[name] = self._load_counts.get(name, 0) + 1
                rt = self._resource_runtime.get(name)
                if rt is not None:
                    now = time.time()
                    rt.loaded = True
                    rt.device = self._extract_device(inst)
                    rt.last_loaded_ts = now
                    rt.last_used_ts = now
            self._touch(name)
            return self._instances[name]

    def _next_replica_spec_locked(self, name: str) -> Dict[str, Any] | None:
        plan = self._resource_replica_plan(name)
        idx = len(self._resource_replicas.get(name, []))
        if idx >= len(plan):
            return None
        return dict(plan[idx])

    def _can_admit_replica_load(self, name: str, device: Optional[str]) -> bool:
        rt = self._resource_runtime.get(name)
        if rt is None:
            return False
        target = self._normalize_device_name(device) or self._scheduled_resource_device(name) or "cuda:0"
        if not target.startswith("cuda"):
            return True
        estimated_mb = int(rt.policy.estimated_vram_mb or 0)
        if estimated_mb <= 0:
            return True
        free_mb, total_mb = self._gpu_mem_info_mb(target)
        reserved_margin_mb = int(self._scheduler_cfg.get("reserved_margin_mb", 0) or 0)
        loaded_estimated_mb = self._estimated_loaded_vram_mb(target)
        free_ok = free_mb >= (estimated_mb + reserved_margin_mb)
        total_ok = (loaded_estimated_mb + estimated_mb) <= max(0, total_mb - reserved_margin_mb)
        return free_ok and total_ok

    def _load_replica_placeholder(
        self,
        name: str,
        placeholder: ResourceReplica,
        device: Optional[str],
    ) -> None:
        rt = self._resource_runtime[name]
        load_started_ts = time.time()
        try:
            inst = self._instantiate_resource(name, device=device)
        except Exception as exc:
            with self._lock:
                pool = self._resource_replicas.get(name, [])
                if placeholder in pool:
                    pool.remove(placeholder)
                rt.loading = any(replica.loading for replica in pool)
                rt.last_error = repr(exc)
                self._notify_resource_waiters_locked(name)
            raise

        load_elapsed = time.time() - load_started_ts
        with self._lock:
            placeholder.instance = inst
            placeholder.loaded = True
            placeholder.loading = False
            placeholder.active_calls = 0
            placeholder.last_loaded_ts = time.time()
            placeholder.last_used_ts = placeholder.last_loaded_ts
            placeholder.device = self._extract_device(inst) or device
            rt.loaded = True
            rt.loading = any(replica.loading for replica in self._resource_replicas.get(name, []))
            rt.last_loaded_ts = placeholder.last_loaded_ts
            rt.last_used_ts = placeholder.last_used_ts
            rt.device = placeholder.device
            if load_elapsed > float(rt.policy.load_timeout_s or 0.0):
                rt.last_error = f"load_timeout_exceeded:{load_elapsed:.2f}s"
            else:
                rt.last_error = ""
            self._load_counts[name] = self._load_counts.get(name, 0) + 1
            self._touch(name)
            self._notify_resource_waiters_locked(name)

    def _warm_replica_pool(self, name: str, deadline: float) -> None:
        rt = self._resource_runtime[name]
        while True:
            placeholder: ResourceReplica | None = None
            device: Optional[str] = None
            with self._lock:
                pool = self._resource_replicas.setdefault(name, [])
                if any(replica.loading for replica in pool):
                    self._wait_for_runtime(rt, deadline)
                    continue
                next_spec = self._next_replica_spec_locked(name)
                if next_spec is None:
                    return
                if not self._can_admit_replica_load(name, next_spec.get("device")):
                    return
                replica_idx = len(pool)
                replica_id = f"{name}#{replica_idx}"
                device = self._normalize_device_name(next_spec.get("device"))
                max_inflight = max(1, int(next_spec.get("max_inflight", 1) or 1))
                placeholder = ResourceReplica(
                    name=name,
                    replica_id=replica_id,
                    device=device,
                    instance=None,
                    max_inflight=max_inflight,
                    active_calls=0,
                    loaded=False,
                    loading=True,
                    last_used_ts=time.time(),
                    last_loaded_ts=0.0,
                )
                pool.append(placeholder)
                rt.loading = True
                rt.last_error = ""
            self._load_replica_placeholder(name, placeholder, device)

    def _acquire_replica_resource(self, name: str) -> Any:
        rt = self._resource_runtime[name]
        deadline = time.time() + float(rt.policy.queue_timeout_s or self._scheduler_cfg.get("global_wait_timeout_s", 600.0))
        victims_to_release: List[str] = []
        placeholder: ResourceReplica | None = None
        device: Optional[str] = None
        with self._lock:
            rt.waiters += 1
            try:
                while True:
                    pool = self._resource_replicas.setdefault(name, [])
                    idle = [
                        replica for replica in pool
                        if replica.loaded and not replica.loading and replica.active_calls < replica.max_inflight
                    ]
                    if idle:
                        idle.sort(key=lambda replica: (replica.active_calls, replica.last_used_ts))
                        replica = idle[0]
                        replica.active_calls += 1
                        replica.last_used_ts = time.time()
                        rt.active_calls += 1
                        rt.loaded = True
                        rt.device = replica.device
                        rt.last_used_ts = replica.last_used_ts
                        self._resource_replica_state.current = (name, replica.replica_id)
                        self._touch(name)
                        return replica.instance

                    if any(replica.loading for replica in pool):
                        self._raise_if_queue_full_locked(rt)
                        self._wait_for_runtime(rt, deadline)
                        continue

                    next_spec = self._next_replica_spec_locked(name)
                    if (
                        next_spec is not None
                        and self._can_admit_replica_load(name, next_spec.get("device"))
                    ):
                        replica_idx = len(pool)
                        replica_id = f"{name}#{replica_idx}"
                        device = self._normalize_device_name(next_spec.get("device"))
                        max_inflight = max(1, int(next_spec.get("max_inflight", 1) or 1))
                        placeholder = ResourceReplica(
                            name=name,
                            replica_id=replica_id,
                            device=device,
                            instance=None,
                            max_inflight=max_inflight,
                            active_calls=0,
                            loaded=False,
                            loading=True,
                            last_used_ts=time.time(),
                            last_loaded_ts=0.0,
                        )
                        pool.append(placeholder)
                        rt.loading = True
                        rt.last_error = ""
                        break

                    victims = self._admission_victims_for_resource_locked(
                        name,
                        self._normalize_device_name(next_spec.get('device')) if next_spec is not None else None,
                    )
                    if not victims:
                        victims = self._maybe_collect_idle_victims_locked()
                    if victims:
                        victims_to_release = victims
                        break
                    self._raise_if_queue_full_locked(rt)
                    self._wait_for_runtime(rt, deadline)
            finally:
                rt.waiters -= 1

        if victims_to_release:
            for victim in victims_to_release:
                self.release(victim)
            return self._acquire_replica_resource(name)
        if placeholder is None:
            return self._acquire_replica_resource(name)

        self._load_replica_placeholder(name, placeholder, device)
        return self._acquire_replica_resource(name)

    def acquire_resource(self, name: str) -> Any:
        if name not in self._specs:
            raise KeyError(f"Unknown resource: {name}")
        if name == "vlm" or not self._should_schedule_resource(name):
            return self._legacy_get_resource(name)
        if self._uses_resource_replicas(name):
            return self._acquire_replica_resource(name)

        rt = self._resource_runtime[name]
        with self._lock:
            deadline = time.time() + float(rt.policy.queue_timeout_s or self._scheduler_cfg.get("global_wait_timeout_s", 600.0))
            rt.waiters += 1
            try:
                while True:
                    if rt.loaded and name in self._instances and rt.active_calls < rt.policy.max_concurrency:
                        inst = self._instances[name]
                        rt.active_calls += 1
                        rt.last_used_ts = time.time()
                        self._touch(name)
                        return inst

                    if rt.loading:
                        self._raise_if_queue_full_locked(rt)
                        self._wait_for_runtime(rt, deadline)
                        continue

                    if not rt.loaded and self._can_admit_resource_load(name):
                        rt.loading = True
                        rt.last_error = ""
                        break

                    victims = self._admission_victims_for_resource_locked(name)
                    if not victims:
                        victims = self._maybe_collect_idle_victims_locked()
                    if victims:
                        pass
                    else:
                        self._raise_if_queue_full_locked(rt)
                        self._wait_for_runtime(rt, deadline)
                        continue
                    # Release happens outside the lock.
                    break
            finally:
                rt.waiters -= 1

        victims = []
        with self._lock:
            if not rt.loading and not (rt.loaded and rt.active_calls < rt.policy.max_concurrency):
                victims = self._admission_victims_for_resource_locked(name)
                if not victims:
                    victims = self._maybe_collect_idle_victims_locked()
        if victims:
            for victim in victims:
                self.release(victim)
            return self.acquire_resource(name)

        load_started_ts = time.time()
        try:
            inst = self._instantiate_resource(name)
        except Exception as exc:
            with self._lock:
                rt.loading = False
                rt.last_error = repr(exc)
                self._notify_resource_waiters_locked(name)
            raise

        load_elapsed = time.time() - load_started_ts
        with self._lock:
            self._instances[name] = inst
            rt.loaded = True
            rt.loading = False
            rt.active_calls = 1
            rt.last_loaded_ts = time.time()
            rt.last_used_ts = rt.last_loaded_ts
            rt.device = self._extract_device(inst) or self._scheduled_resource_device(name)
            if load_elapsed > float(rt.policy.load_timeout_s or 0.0):
                rt.last_error = f"load_timeout_exceeded:{load_elapsed:.2f}s"
            else:
                rt.last_error = ""
            self._load_counts[name] = self._load_counts.get(name, 0) + 1
            self._touch(name)
            self._notify_resource_waiters_locked(name)
            return inst

    def release_resource_slot(self, name: str) -> None:
        if name not in self._resource_runtime:
            return
        victims: List[str] = []
        with self._lock:
            rt = self._resource_runtime[name]
            if self._uses_resource_replicas(name):
                current = getattr(self._resource_replica_state, "current", None)
                replica_id = current[1] if isinstance(current, tuple) and current[0] == name else ""
                pool = self._resource_replicas.get(name, [])
                target = next((replica for replica in pool if replica.replica_id == replica_id), None)
                if target is None:
                    active = [replica for replica in pool if replica.active_calls > 0]
                    target = active[0] if active else None
                if target is not None and target.active_calls > 0:
                    target.active_calls -= 1
                    target.last_used_ts = time.time()
                if isinstance(current, tuple) and current[0] == name:
                    self._resource_replica_state.current = None
                if rt.active_calls > 0:
                    rt.active_calls -= 1
                now = time.time()
                rt.last_used_ts = now
                self._last_used_ts[name] = now
                rt.loaded = bool(pool)
                rt.loading = any(replica.loading for replica in pool)
                self._notify_resource_waiters_locked(name)
                return
            if rt.active_calls > 0:
                rt.active_calls -= 1
            now = time.time()
            rt.last_used_ts = now
            self._last_used_ts[name] = now
            self._notify_resource_waiters_locked(name)
            victims = self._maybe_collect_idle_victims_locked()
        for victim in victims:
            self.release(victim)

    @contextmanager
    def use_resource(self, name: str):
        inst = self.acquire_resource(name)
        try:
            yield inst
        finally:
            if self._should_schedule_resource(name):
                self.release_resource_slot(name)

    def _handle_resource_oom(self, name: str) -> None:
        try:
            current = getattr(self._resource_replica_state, "current", None)
            if isinstance(current, tuple) and current[0] == name:
                with self._lock:
                    for replica in self._resource_replicas.get(name, []):
                        if replica.replica_id == current[1]:
                            replica.oom_count += 1
                            break
            self.release(name)
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            with self._lock:
                rt = self._resource_runtime.get(name)
                if rt is not None:
                    rt.last_error = "oom"
                    rt.policy.estimated_vram_mb = int(rt.policy.estimated_vram_mb * 1.15) + 256
                    self._notify_resource_waiters_locked(name)

    def run_with_resource(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        oom_types = self._torch_oom_error_type()
        attempts = 2 if self._should_schedule_resource(name) and oom_types else 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                with self.use_resource(name) as inst:
                    return fn(inst, *args, **kwargs)
            except Exception as exc:
                if oom_types and isinstance(exc, oom_types) and self._should_schedule_resource(name):
                    self._handle_resource_oom(name)
                    last_exc = exc
                    if attempt + 1 < attempts:
                        continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"run_with_resource({name}) failed without returning")

    def get_resource(self, name: str) -> Any:
        if name not in self._specs:
            raise KeyError(f"Unknown resource: {name}")
        inst = self._legacy_get_resource(name)
        if name == "vlm":
            vlm_mode = str(((self._models_cfg().get("qwen_vlm", {}) or {}).get("mode", "") or "")).strip().lower()
            target_size = self.get_vlm_pool_size() if vlm_mode == "api" else 1
            self._ensure_vlm_pool_size(target_size)
            with self._vlm_pool_lock:
                if self._vlm_pool:
                    slot = int(self._vlm_rr_idx) % len(self._vlm_pool)
                    self._vlm_rr_idx = int(self._vlm_rr_idx + 1) % max(1, len(self._vlm_pool))
                    return self._apply_chat_logging(self._vlm_pool[slot])
        return inst

    def release(self, name: str) -> None:
        with self._lock:
            inst = self._instances.pop(name, None)
            replicas = self._resource_replicas.pop(name, [])
            rt = self._resource_runtime.get(name)
            if rt is not None:
                rt.loaded = False
                rt.loading = False
                rt.active_calls = 0
                rt.device = None
                rt.last_unloaded_ts = time.time()
                rt.last_used_ts = rt.last_unloaded_ts
                self._notify_resource_waiters_locked(name)
        if name == "vlm":
            self._release_vlm_pool(exclude=inst)
        for replica in replicas:
            replica_inst = replica.instance
            replica.instance = None
            replica.loaded = False
            if replica_inst is None:
                continue
            try:
                close_fn = getattr(replica_inst, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
            del replica_inst
        if inst is None:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return
        try:
            close_fn = getattr(inst, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        del inst
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _release_vlm_pool(self, exclude: Any = None) -> None:
        with self._vlm_pool_lock:
            pool = self._vlm_pool
            self._vlm_pool = []
            self._vlm_rr_idx = 0
            self._vlm_busy_slots.clear()
            self._vlm_slot_cv.notify_all()
        for inst in pool:
            if inst is exclude:
                continue
            try:
                close_fn = getattr(inst, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:
                pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def release_many(self, names: List[str]) -> None:
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            self.release(name)

    def release_scope(self, scope: str) -> None:
        if scope == "all_non_persistent":
            names = [name for name, spec in self._specs.items() if not spec.persistent]
        else:
            names = [name for name, spec in self._specs.items() if spec.scope == scope]
        self.release_many(names)

    def on_segment_end(self) -> None:
        if self._scope_release_suspended:
            return
        self.release_scope("segment")

    def on_video_end(self) -> None:
        if self._scope_release_suspended:
            return
        self.release_scope("segment")
        self.release_scope("video")
        self.release_scope("all_non_persistent")

    def close_all(self) -> None:
        with self._lock:
            names = list(dict.fromkeys(list(self._instances.keys()) + list(self._resource_replicas.keys())))
        self.release_many(names)
        self._shutdown_vllm_services()

    def maybe_release_cold_models(self) -> None:
        self._maybe_unload_idle_resources()
        for name, spec in list(self._specs.items()):
            if spec.persistent:
                continue
            if (name in self._instances or name in self._resource_replicas) and not spec.warm_reuse:
                self.release(name)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            loaded = sorted(self._instances.keys())
            last_used_age_sec = {
                name: max(0.0, now - self._last_used_ts.get(name, now))
                for name in loaded
            }
            resident_units = self._resident_units()
            gpu_summary = self._gpu_summary()
            resource_runtime = {}
            for name, rt in self._resource_runtime.items():
                replicas = [
                    {
                        "replica_id": replica.replica_id,
                        "device": replica.device,
                        "loaded": bool(replica.loaded),
                        "loading": bool(replica.loading),
                        "active_calls": int(replica.active_calls),
                        "max_inflight": int(replica.max_inflight),
                        "last_loaded_ts": float(replica.last_loaded_ts or 0.0),
                        "idle_age_sec": max(0.0, now - float(replica.last_used_ts or now)),
                        "fail_count": int(replica.fail_count),
                        "oom_count": int(replica.oom_count),
                    }
                    for replica in self._resource_replicas.get(name, [])
                ]
                resource_runtime[name] = {
                    "loaded": bool(rt.loaded),
                    "loading": bool(rt.loading),
                    "active_calls": int(rt.active_calls),
                    "waiters": int(rt.waiters),
                    "device": rt.device,
                    "estimated_vram_mb": int(rt.policy.estimated_vram_mb),
                    "max_concurrency": int(rt.policy.max_concurrency),
                    "idle_unload_s": float(rt.policy.idle_unload_s),
                    "queue_timeout_s": float(rt.policy.queue_timeout_s),
                    "load_timeout_s": float(rt.policy.load_timeout_s),
                    "unload_priority": int(rt.policy.unload_priority),
                    "last_error": str(rt.last_error or ""),
                    "idle_age_sec": max(0.0, now - float(rt.last_used_ts or now)),
                    "last_loaded_ts": float(rt.last_loaded_ts or 0.0),
                    "last_unloaded_ts": float(rt.last_unloaded_ts or 0.0),
                    "replicas": replicas,
                }
            return {
                "loaded": loaded,
                "max_gpu_resident_models": self.max_gpu_resident_models,
                "current_gpu_resident_models": len(resident_units),
                "load_counts": dict(self._load_counts),
                "eviction_counts": dict(self._eviction_counts),
                "last_used_age_sec": last_used_age_sec,
                "resident_units": resident_units,
                "device_assignments": {item["unit"]: item.get("device") for item in resident_units},
                "estimated_loaded_vram_mb": self._estimated_loaded_vram_mb(
                    str(self._scheduler_cfg.get("device", "cuda:0") or "cuda:0")
                ),
                "single_gpu_scheduler": dict(self._scheduler_cfg),
                "scheduled_resource_names": sorted(self._scheduled_resource_names),
                "resource_runtime": resource_runtime,
                "gpu_summary": gpu_summary,
                "vlm_pool": {
                    "configured_size": self._vlm_pool_size,
                    "configured_local_size": self._desired_local_vlm_pool_size(),
                    "desired_size": self._desired_vlm_pool_size(),
                    "active_size": self._vlm_resident_count(),
                    "slots_per_video": int(self._vlm_slots_per_video),
                    "devices": list(self._resource_device_candidates("vlm", self._vlm_devices)),
                    "policy": dict(self._vlm_policy),
                    "service_devices": list(self._vllm_service_manager.active_devices()) if self._vllm_service_manager is not None else [],
                    "local_service_devices": list(self._vllm_service_manager.active_local_devices()) if self._vllm_service_manager is not None else [],
                    "remote_service_devices": list(self._vllm_service_manager.active_remote_devices()) if self._vllm_service_manager is not None else [],
                },
                "gpu_placement": {
                    "manager_assign_devices": self._manager_assign_enabled,
                    "policy": {
                        k: v
                        for k, v in self._gpu_placement_policy.items()
                        if k != "per_model"
                    },
                    "per_model": dict(self._gpu_placement_policy.get("per_model", {})),
                },
            }

    def _build_shot_detector(self) -> Any:
        return self._lazy_import("executor.tasks.shot_detector", "ShotDetector")(self.config)
