#!/usr/bin/env python3
"""Offline preflight by default; network and GPU probes require explicit flags."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import ssl
import sqlite3
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vidueu_config import load_config
from vidueu.diagnostics import diagnostic

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}


def probe_endpoint(api, *, proxy_mode=None, proxy_url=None):
    base = str(api.get("base_url", "")).rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("invalid_vlm_base_url")
    if base.endswith("/chat/completions"):
        base = base[:-len("/chat/completions")]
    url = base + ("/models" if base.endswith("/v1") else "/v1/models")
    key = api.get("api_key") or os.environ.get(api.get("api_key_env", ""), "")
    if not key and api.get("api_key_file"):
        key_path = Path(str(api["api_key_file"]))
        if key_path.is_symlink() or not key_path.is_file():
            raise ValueError("api_key_file_missing")
        key = key_path.read_text(encoding="utf-8").strip()
    # Local vLLM commonly permits unauthenticated /models. Let the configured
    # endpoint decide whether authentication is required, just like the client.
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    # Probe the vision server only. Never send a billable generation request.
    request = urllib.request.Request(url, headers=headers)
    proxy = str(proxy_url or api.get("proxy_url") or "").strip()
    if proxy_mode == "direct":
        proxy_handler = urllib.request.ProxyHandler({})
    elif proxy_mode == "explicit" or (proxy_mode is None and proxy):
        if not proxy:
            raise ValueError("explicit_proxy_missing")
        proxy_handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    else:
        # Keep the process proxy environment available for remote providers.
        # Local loopback endpoints remain direct when NO_PROXY is set.
        proxy_handler = urllib.request.ProxyHandler()
    context = ssl.create_default_context()
    if (context.cert_store_stats().get("x509_ca", 0) == 0
            and not os.environ.get("SSL_CERT_FILE") and not os.environ.get("SSL_CERT_DIR")):
        import certifi
        context.load_verify_locations(cafile=certifi.where())
    opener = urllib.request.build_opener(proxy_handler, urllib.request.HTTPSHandler(context=context))
    with opener.open(request, timeout=10) as response:
        body = json.load(response)
    models = {item.get("id") for item in body.get("data", [])}
    if api.get("model") not in models:
        raise ValueError("vlm_served_model_mismatch")
    return {"model": api["model"], "http": 200}


def inspect(config, *, input_path=None, output_root=None, assets=False,
            endpoint=False, gpu=False, media=False, lightweight=False):
    checks = []

    def check(name, action):
        try:
            result = action()
            checks.append({"name": name, "ok": True, "detail": result})
        except Exception as exc:
            # Do not emit provider bodies, credentials, or resolved config values.
            checks.append({"name": name, "ok": False, **diagnostic(exc, check=name)})

    def require(value, message):
        if not value:
            raise ValueError(message)
        return "ok"

    models = config.get("models", {})
    runtime = config.get("runtime", {})
    tasks = config.get("tasks", {})
    check("resume_mode", lambda: require(runtime.get("video_run_mode", "skip_completed") == "skip_completed", "overwrite_disallowed"))
    check("external_vlm", lambda: require(not models.get("qwen_vlm", {}).get("api", {}).get("service", {}).get("auto_start", False), "use_start_vllm_separately"))
    def device_mapping():
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask is None:
            return "driver visibility inherited; use --check-gpu to probe"
        visible = [value.strip() for value in mask.split(",") if value.strip()]
        def walk(value, key=""):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, k)
            elif isinstance(value, list):
                for v in value:
                    walk(v, key)
            elif isinstance(value, str) and key in {"device", "devices", "yolo_device"} and value.startswith("cuda"):
                ordinal = int(value.split(":", 1)[1]) if ":" in value else 0
                require(0 <= ordinal < len(visible), "device_outside_visible_mask")
        walk(config)
        return {"mask": mask, "logical_device_count": len(visible),
                "logical_to_physical": {f"cuda:{i}": device for i, device in enumerate(visible)}}
    check("device_mapping", device_mapping)
    for command in ("ffmpeg", "ffprobe"):
        check(command, lambda command=command: require(shutil.which(command), "missing_system_command"))
    packages = () if lightweight else ("torch", "torchvision", "transformers", "numpy", "PyYAML", "opencv-python-headless", "ruptures", "scenedetect", "ultralytics", "paddleocr")
    for name in packages:
        check("package:" + name, lambda name=name: importlib.metadata.version(name))
    from executor.tasks.prompt_loader import configured_prompt_profile, load_prompt_text
    check("prompt_profile", lambda: load_prompt_text("paper", "routing", profile=configured_prompt_profile(config)) and "ok")
    chosen_input = input_path or runtime.get("input_path")
    check("input_exists", lambda: require(chosen_input and Path(chosen_input).exists(), "input_missing"))
    videos = []
    if chosen_input and Path(chosen_input).exists():
        source = Path(chosen_input)
        videos = [source] if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES else []
        if source.is_dir():
            from itertools import islice
            videos = list(islice((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES), 1))
    check("input_video", lambda: require(videos, "no_video_inputs"))
    if media and videos:
        def decode_probe():
            value = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height", "-of", "json", str(videos[0])], timeout=20)
            streams = json.loads(value)["streams"]
            require(streams and streams[0].get("width", 0) > 0, "invalid_video")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(videos[0]), "-frames:v", "1", "-f", "null", "-"], check=True, timeout=20, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "first frame decoded"
        check("input_decode", decode_probe)
    destination = Path(output_root or runtime.get("output_root", "outputs/run")).absolute()
    def output_probe():
        parent = destination
        while not parent.exists():
            parent = parent.parent
        require(parent.is_dir() and os.access(parent, os.W_OK), "output_not_writable")
        free = shutil.disk_usage(parent).free
        require(free >= int(runtime.get("min_free_disk_gb", 5)) * 1024**3, "low_disk")
        return {"free_gib": round(free / 1024**3, 1)}
    check("output_space", output_probe)
    budget = models.get("qwen_llm", {}).get("api", {})
    if budget.get("budget_path"):
        def budget_probe():
            ledger = Path(budget["budget_path"])
            limit = budget.get("max_api_calls")
            if not ledger.exists():
                return {"limit": limit, "used": 0, "remaining": limit, "ledger": "not created"}
            with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
                stored = db.execute("SELECT max_calls FROM budget WHERE singleton=1").fetchone()[0]
                used = db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
            require(stored == limit, "api_budget_limit_changed")
            require(used < stored, "api_budget_exhausted")
            return {"limit": stored, "used": used, "remaining": stored - used}
        check("api_attempt_budget", budget_probe)
    for name in (() if lightweight else ("qwen_llm", "qwen_vlm")):
        api = models.get(name, {}).get("api", {})
        def api_probe(api=api, name=name):
            require(api.get("base_url") and api.get("model"), "api_model_or_url_missing")
            require(not urllib.parse.urlsplit(api["base_url"]).username, "url_contains_credentials")
            if name == "qwen_llm":
                paths = api.get("api_key_files") or ([api["api_key_file"]] if api.get("api_key_file") else [])
                envs = api.get("api_key_envs") or ([api["api_key_env"]] if api.get("api_key_env") else [])
                valid_file = any(Path(path).is_file() and Path(path).stat().st_size for path in paths)
                require(valid_file or any(os.environ.get(e) for e in envs) or bool(api.get("api_key")), "api_credentials_missing")
            return "configured (no API request)"
        check(name, api_probe)
    if assets:
        def exists_file(path):
            return require(path and Path(path).is_file() and Path(path).stat().st_size > 0, "asset_file_missing")
        def model_dir(path):
            require(path and Path(path).is_dir(), "model_directory_missing")
            root = Path(path)
            require((root / "config.json").is_file(), "model_config_missing")
            indices = list(root.glob("*.safetensors.index.json"))
            if indices:
                from scripts.download_models import safe_path
                for index in indices:
                    for shard in set(json.loads(index.read_text())["weight_map"].values()):
                        exists_file(safe_path(root, shard))
            else:
                require(any(root.glob("*.safetensors")) or any(root.glob("pytorch_model*.bin")), "model_weights_missing")
            return "local files present; use download_models --verify for byte verification"
        for key in ("whisper_model", "image_embedding_model"):
            if key == "whisper_model" and not tasks.get("T2_asr", {}).get("enabled", True):
                continue
            check(key, lambda key=key: model_dir(models.get(key)))
        enabled_tools = set(runtime.get("enabled_tools", ("asr", "ocr", "yolo", "omniparser")))
        if "yolo" in enabled_tools:
            check("yolo_weights", lambda: exists_file(models.get("yolo")))
        for key in (("paddleocr_det", "paddleocr_rec") if "ocr" in enabled_tools else ()):
            def ocr_probe(key=key):
                directory = models.get(key)
                require(directory, "ocr_path_missing")
                exists_file(Path(directory) / "inference.json")
                return exists_file(Path(directory) / "inference.pdiparams")
            check(key, ocr_probe)
        raft = config.get("optical_flow", {}).get("raft", {})
        if config.get("optical_flow", {}).get("backend") == "raft":
            check("raft_code", lambda: exists_file(Path(raft.get("repo_root", "")) / "core/raft.py"))
            check("raft_weights", lambda: exists_file(raft.get("weights")))
        omni = models.get("omniparser", {})
        if omni and "omniparser" in enabled_tools:
            check("omni_code", lambda: exists_file(Path(omni.get("repo_root", "")) / "util/utils.py"))
            check("omni_python", lambda: require(shutil.which(omni.get("python_executable", "")), "omni_python_missing"))
            check("omni_detection", lambda: exists_file(Path(omni.get("weights_dir", "")) / "icon_detect/model.pt"))
            check("omni_caption", lambda: model_dir(omni.get("caption_model_path")))
            for key, filename in (("caption_processor_path", "tokenizer.json"), ("caption_model_code_path", "modeling_florence2.py")):
                check("omni_" + key, lambda key=key, filename=filename: exists_file(Path(omni.get(key, "")) / filename))
    if endpoint:
        network = config.get("_vidueu", {})
        check("vlm_models_endpoint", lambda: probe_endpoint(
            models.get("qwen_vlm", {}).get("api", {}),
            proxy_mode=network.get("proxy_mode"), proxy_url=network.get("proxy_url")))
    if gpu:
        def gpu_probe():
            import torch
            count = torch.cuda.device_count()
            require(count > 0, "no_visible_cuda_device")
            return {"visible": os.environ.get("CUDA_VISIBLE_DEVICES", "driver default"), "count": count}
        check("cuda", gpu_probe)
        if "ocr" in set(runtime.get("enabled_tools", ("asr", "ocr", "yolo", "omniparser"))):
            def ocr_backend_probe():
                from executor.tasks.ocr_runtime import paddle_device_info
                cfg = tasks.get("T4_ocr", {})
                return paddle_device_info(cfg.get("device") or ("cuda:0" if cfg.get("use_gpu") else "cpu"))
            check("ocr_backend", ocr_backend_probe)
    return {"ok": all(c["ok"] for c in checks), "mode": "lightweight" if lightweight else "pipeline", "checks": checks}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/paper.yaml")
    p.add_argument("--input")
    p.add_argument("--output-root")
    p.add_argument("--check-assets", action="store_true")
    p.add_argument("--check-endpoint", action="store_true")
    p.add_argument("--check-gpu", action="store_true")
    p.add_argument("--check-media", action="store_true")
    a = p.parse_args(argv)
    try:
        config = load_config(a.config, overrides={k: str(Path(v).resolve()) for k, v in (("input", a.input), ("output", a.output_root)) if v})
    except Exception as exc:
        # The loader reports variable names only; don't print arbitrary YAML text.
        print(json.dumps({"ok": False, **diagnostic(exc)}))
        return 1
    report = inspect(config, input_path=a.input, output_root=a.output_root, assets=a.check_assets,
                     endpoint=a.check_endpoint, gpu=a.check_gpu, media=a.check_media)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
