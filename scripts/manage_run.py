#!/usr/bin/env python3
"""Linux run manager: exclusive run, append-only logs, identity-checked stop."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidueu_config import load_config, redact

PROXIES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY")


def write_json(path, value):
    temporary = path.with_suffix(".next")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("run_state_symlink")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


@contextmanager
def lock(path):
    if path.is_symlink():
        raise ValueError("run_lock_symlink")
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": fields[19],
                "argv": Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")[:-1]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(saved):
    current = identity(saved.get("pid", 0)) if saved else None
    return bool(current and current == saved)


def owned_child(state):
    saved = state.get("child")
    if not same_process(saved) or saved["argv"] != state.get("command"):
        return False
    try:
        entries = Path(f"/proc/{saved['pid']}/environ").read_bytes().split(b"\0")
        return (f"VIDUEU_RUN_TOKEN={state['token']}".encode() in entries
                and os.getpgid(saved["pid"]) == saved["pid"])
    except (OSError, KeyError):
        return False


def state_path(run_dir, kind):
    return run_dir / f".{kind}.state.json"


def read_state(run_dir, kind):
    path = state_path(run_dir, kind)
    if path.is_symlink():
        raise ValueError("run_state_symlink")
    return json.loads(path.read_text()) if path.exists() else {}


def status(run_dir, kind):
    state = read_state(run_dir, kind)
    alive = same_process(state.get("worker"))
    result = {"kind": kind, "run_dir": str(run_dir), "alive": alive,
            "status": state.get("status", "not_started") if alive or state.get("status") in {"completed", "failed", "stopped"} else "not_running",
            "pid": (state.get("child") or {}).get("pid"), "log": state.get("log"),
            "returncode": state.get("returncode"), "tmux_attach": state.get("tmux_attach")}
    if kind == "pipeline":
        from vidueu.progress import inspect_progress
        result["progress"] = inspect_progress(run_dir)
    return result


def child_environment(env, mode, config_path=None):
    result = dict(env)
    if mode != "inherit":
        for key in PROXIES:
            result.pop(key, None)
    if mode == "explicit":
        config = load_config(config_path)
        url = config.get("_vidueu", {}).get("proxy_url")
        if not url:
            raise ValueError("explicit_proxy_missing")
        for key in PROXIES:
            result[key] = url
        result.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
        result.setdefault("no_proxy", result["NO_PROXY"])
    return result


def worker(run_dir, kind, token):
    with lock(run_dir / f".{kind}.runtime.lock"):
        state = read_state(run_dir, kind)
        if state.get("token") != token or state.get("status") != "starting":
            raise ValueError("worker_reservation_changed")
        state["worker"] = identity(os.getpid())
        state["status"] = "starting"
        child = None
        stopping = False
        stop_at = None

        def stop_signal(_signum, _frame):
            nonlocal stopping, stop_at
            stopping = True
            stop_at = time.monotonic()
            if child is not None and same_process(state.get("child")):
                os.killpg(child.pid, signal.SIGTERM)

        signal.signal(signal.SIGTERM, stop_signal)
        signal.signal(signal.SIGINT, stop_signal)
        write_json(state_path(run_dir, kind), state)
        env = child_environment(os.environ, state.get("proxy_mode", "direct"), state.get("config_file"))
        env.update(state["environment"])
        env["VIDUEU_RUN_TOKEN"] = token
        try:
            with Path(state["log"]).open("ab", buffering=0) as log_stream:
                if not stopping:
                    child = subprocess.Popen(state["command"], cwd=ROOT, env=env,
                                             stdin=subprocess.DEVNULL, stdout=log_stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    state["child"] = identity(child.pid)
                    state["status"] = "running"
                    write_json(state_path(run_dir, kind), state)
                    # A stop may arrive between Popen and recording its identity.
                    if stopping and same_process(state.get("child")):
                        os.killpg(child.pid, signal.SIGTERM)
                    while child.poll() is None:
                        if stopping and stop_at is not None and time.monotonic() - stop_at > 30:
                            if same_process(state.get("child")):
                                os.killpg(child.pid, signal.SIGKILL)
                        time.sleep(0.1)
                    result = child.wait()
                else:
                    result = 0
            state.update(status="stopped" if stopping else "completed" if result == 0 else "failed",
                         returncode=result, finished_at=time.time())
        except Exception:
            if child is not None and same_process(state.get("child")):
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)
            state.update(status="failed", returncode=1, finished_at=time.time())
            raise
        finally:
            write_json(state_path(run_dir, kind), state)
    return 0


def build_command(a):
    if a.gpus and not re.fullmatch(r"[0-9]+(,[0-9]+)*", a.gpus):
        raise ValueError("gpus_must_be_explicit_integer_indices")
    if a.gpus and len(set(a.gpus.split(","))) != len(a.gpus.split(",")):
        raise ValueError("duplicate_gpu_index")
    env = {"CUDA_VISIBLE_DEVICES": a.gpus or "", "VIDUEU_OUTPUT_ROOT": str(a.run_dir)}
    if a.input:
        env["VIDUEU_INPUT"] = str(Path(a.input).expanduser().resolve())
    # Keep a configured venv path lexical. Venv ``bin/python`` is commonly a
    # symlink to the base interpreter; resolving it here makes deployment
    # diagnostics claim the system interpreter and can bypass a managed venv.
    python = str(Path(a.python).expanduser().absolute()) if "/" in a.python else a.python
    if a.kind == "pipeline":
        config_path = Path(a.config).resolve()
        overrides = {"output": str(a.run_dir)}
        if a.input:
            overrides["input"] = env["VIDUEU_INPUT"]
        config = load_config(config_path, overrides=overrides)
        if config.get("runtime", {}).get("video_run_mode", "skip_completed") != "skip_completed":
            raise ValueError("managed_launch_disallows_force_overwrite")
        if config.get("models", {}).get("qwen_vlm", {}).get("api", {}).get("service", {}).get("auto_start", False):
            raise ValueError("start_vllm_separately_so_ownership_is_explicit")
        command = [python, "-u", str(ROOT / "run_pipeline.py"), "--config", str(config_path),
                   "--output-root", str(a.run_dir), "--dashboard" if a.dashboard else "--no-dashboard"]
        if a.input:
            command += ["--input", env["VIDUEU_INPUT"]]
        if a.source_name:
            command += ["--source-name", a.source_name]
        if a.dashboard:
            command += ["--dashboard-port", str(a.port or 8766)]
        if not a.dry_run:
            from scripts.doctor import inspect
            report = inspect(config, input_path=env.get("VIDUEU_INPUT"), output_root=a.run_dir,
                             assets=True, endpoint=True, media=True)
            if not report["ok"]:
                print(json.dumps(report, indent=2))
                raise ValueError("preflight_failed")
        audit = {"config_file": str(config_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                 "resolved_config_redacted": redact(config)}
    else:
        if not a.gpus or not a.model_path:
            raise ValueError("vllm_requires_explicit_gpus_and_model_path")
        model = Path(a.model_path).expanduser().resolve()
        if not a.dry_run and not (model / "config.json").is_file():
            raise ValueError("vllm_model_missing")
        port = a.port or 8110
        command = [python, "-m", "vllm.entrypoints.openai.api_server", "--host", "127.0.0.1",
                   "--port", str(port), "--model", str(model), "--served-model-name", a.served_model_name,
                   "--tensor-parallel-size", str(len(a.gpus.split(","))),
                   "--gpu-memory-utilization", str(a.gpu_memory_utilization)]
        if a.reasoning_parser:
            command += ["--reasoning-parser", a.reasoning_parser]
        if a.media_root:
            command += ["--allowed-local-media-path", str(Path(a.media_root).resolve())]
        if a.max_model_len is not None:
            command += ["--max-model-len", str(a.max_model_len)]
        if a.max_num_seqs is not None:
            command += ["--max-num-seqs", str(a.max_num_seqs)]
        if a.enforce_eager:
            command.append("--enforce-eager")
        audit = {"model": str(model), "port": port, "served_model_name": a.served_model_name}
    return command, env, audit


def start(a):
    command, env, audit = build_command(a)
    plan = {"command": command, "environment": env, "run_dir": str(a.run_dir), "kind": a.kind,
            "proxy_mode": getattr(a, "proxy_mode", "direct")}
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    a.run_dir.mkdir(parents=True, exist_ok=True)
    with lock(a.run_dir / f".{a.kind}.launch.lock"):
        previous = read_state(a.run_dir, a.kind)
        if same_process(previous.get("child")) and not same_process(previous.get("worker")):
            raise ValueError("orphan_child_still_active_use_explicit_stop")
        if same_process(previous.get("worker")) or (previous.get("status") == "starting" and time.time() - previous.get("created_at", 0) < 30):
            raise ValueError("run_already_active")
        # The runtime lock catches a stale/corrupted state pointing at no worker.
        with lock(a.run_dir / f".{a.kind}.runtime.lock"):
            pass
        token = uuid.uuid4().hex
        log = a.run_dir / f"{a.kind}.{time.time_ns()}.log"
        state = {**plan, **audit, "token": token, "status": "starting", "created_at": time.time(), "log": str(log)}
        cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--kind", a.kind,
               "--run-dir", str(a.run_dir), "--token", token]
        worker_env = child_environment(os.environ, getattr(a, "proxy_mode", "direct"), audit.get("config_file"))
        worker_env.update(env)
        if a.tmux:
            # A dedicated server inherits THIS launch environment, not stale
            # credentials/device variables from an unrelated tmux server.
            socket = "vidueu-" + hashlib.sha256((str(a.run_dir) + a.kind).encode()).hexdigest()[:12]
            session = a.kind
            state["tmux_attach"] = shlex.join(["tmux", "-L", socket, "attach", "-t", session])
            write_json(state_path(a.run_dir, a.kind), state)
            try:
                subprocess.run(["tmux", "-L", socket, "new-session", "-d", "-s", session,
                                shlex.join(cmd)], cwd=ROOT, env=worker_env, check=True)
            except Exception:
                state.update(status="failed", returncode=1)
                write_json(state_path(a.run_dir, a.kind), state)
                raise
        else:
            write_json(state_path(a.run_dir, a.kind), state)
            with log.open("ab") as stream:
                subprocess.Popen(cmd, cwd=ROOT, env=worker_env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = read_state(a.run_dir, a.kind)
            if state.get("worker") or state.get("status") != "starting":
                break
            time.sleep(0.1)
    report = status(a.run_dir, a.kind)
    print(json.dumps(report, indent=2))
    if report["status"] not in {"starting", "running", "completed"}:
        return 1
    if a.kind == "vllm" and report["alive"]:
        from scripts.doctor import probe_endpoint
        deadline = time.monotonic() + a.ready_timeout
        while time.monotonic() < deadline and status(a.run_dir, a.kind)["alive"]:
            try:
                probe_endpoint({"base_url": f"http://127.0.0.1:{a.port or 8110}/v1", "model": a.served_model_name})
                print(json.dumps({"ready": True, "port": a.port or 8110}))
                return 0
            except Exception:
                time.sleep(2)
        print(json.dumps({"ready": False, "action": "inspect log; stop this owned service with start_vllm.sh stop"}))
        return 1
    return 0


def stop(run_dir, kind):
    with lock(run_dir / f".{kind}.launch.lock"):
        state = read_state(run_dir, kind)
        saved = state.get("worker")
        if same_process(saved):
            required = {"worker", str(Path(__file__).resolve()), str(run_dir), kind, state["token"]}
            if not required.issubset(set(saved["argv"])):
                raise ValueError("worker_identity_mismatch")
            os.kill(saved["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 35
            while same_process(saved) and time.monotonic() < deadline:
                time.sleep(0.1)
            if same_process(saved):
                raise ValueError("worker_still_stopping_no_unverified_kill")
        elif state.get("status") == "starting":
            state["status"] = "stopped"
            write_json(state_path(run_dir, kind), state)
        elif same_process(state.get("child")):
            if not owned_child(state):
                raise ValueError("orphan_child_identity_unverified")
            child = state["child"]
            os.killpg(child["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 30
            while same_process(child) and time.monotonic() < deadline:
                time.sleep(0.1)
            if same_process(child):
                if not owned_child(state):
                    raise ValueError("orphan_child_identity_changed")
                os.killpg(child["pid"], signal.SIGKILL)
            state.update(status="stopped", returncode=None, finished_at=time.time())
            write_json(state_path(run_dir, kind), state)
    print(json.dumps(status(run_dir, kind), indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("start", "status", "stop", "worker"), nargs="?", default="start")
    p.add_argument("--kind", choices=("pipeline", "vllm"), default="pipeline")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--input")
    p.add_argument("--source-name")
    p.add_argument("--gpus", default="")
    p.add_argument("--proxy-mode", choices=("direct", "inherit", "explicit"), default="direct")
    p.add_argument("--tmux", action="store_true")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--port", type=int)
    p.add_argument("--model-path")
    p.add_argument("--served-model-name", default="Qwen3.5-9B-local")
    p.add_argument("--reasoning-parser", default="qwen3", help="Pass an empty string to disable")
    p.add_argument("--media-root")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, help="Explicit vLLM context limit for available GPU memory")
    p.add_argument("--max-num-seqs", type=int, help="Explicit vLLM concurrent sequence limit")
    p.add_argument("--enforce-eager", action="store_true", help="Disable vLLM CUDA graphs for a small smoke run")
    p.add_argument("--ready-timeout", type=int, default=600)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--token", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.run_dir = a.run_dir.expanduser().absolute()
    if a.run_dir.resolve() != a.run_dir:
        p.error("Run directory must not contain symlinks or '..'")
    if a.port is not None and not 1024 <= a.port <= 65535:
        p.error("Port must be between 1024 and 65535")
    if not 0 < a.gpu_memory_utilization < 1:
        p.error("GPU memory utilization must be between 0 and 1")
    if any(value is not None and value < 1 for value in (a.max_model_len, a.max_num_seqs)):
        p.error("vLLM context and sequence limits must be positive")
    if a.action == "status":
        print(json.dumps(status(a.run_dir, a.kind), indent=2))
        return 0
    if a.action == "stop":
        if not a.run_dir.is_dir():
            print(json.dumps({"status": "not_started"}))
            return 0
        return stop(a.run_dir, a.kind)
    if a.action == "worker":
        return worker(a.run_dir, a.kind, a.token)
    return start(a)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Run manager failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
