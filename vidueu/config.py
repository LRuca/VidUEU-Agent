"""Compile a small public configuration to the existing pipeline contract."""
from __future__ import annotations

import copy
import os
import re
import sys
from urllib.parse import urlsplit
from importlib.resources import files
from pathlib import Path

import yaml

CAPABILITIES = ("understanding", "editing", "unified")
TOOLS = ("ocr", "asr", "yolo", "omniparser")
FIELDS = {"version", "preset", "input", "output", "model_root", "python", "tool_python",
          "source_name", "capabilities", "tools", "hardware", "llm", "vlm", "budget",
          "network", "overrides"}
API_FIELDS = {"base_url", "model", "api_key_env", "api_key_file", "timeout_s", "max_inflight",
              "max_retries", "thinking", "enable_thinking", "wire_profile", "reasoning_effort"}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def require_object(value, field, allowed=None):
    if not isinstance(value, dict):
        raise ValueError(f"config_field_must_be_object:{field}")
    if allowed is not None and set(value) - set(allowed):
        raise ValueError(f"unknown_config_field:{field}.{sorted(set(value) - set(allowed))[0]}")
    return value


def local_path(value, base):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("config_path_must_be_nonempty_string")
    path = Path(value).expanduser()
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def executable_path(value, base):
    """Make an executable absolute without collapsing a venv's python symlink."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("config_executable_must_be_nonempty_string")
    path = Path(value).expanduser()
    return str((base / path).absolute() if not path.is_absolute() else path.absolute())


def small_config(raw):
    return "version" in raw or "preset" in raw


def compile_config(raw, *, path, environ, overrides=None):
    from vidueu_config import ConfigLoader, expand_environment

    require_object(raw, "root", FIELDS)
    if type(raw.get("version", 1)) is not int or raw.get("version", 1) != 1:
        raise ValueError("unsupported_config_version:expected_1")
    if raw.get("preset", "paper") != "paper":
        raise ValueError("unknown_preset:expected_paper")
    # Explicit CLI values replace unresolved placeholders before expansion.
    raw = merge(raw, overrides or {})
    raw = expand_environment(raw, environ)
    base = Path(path).resolve().parent
    configured_python = raw.get("python")
    input_path = local_path(raw.get("input", "./videos"), base)
    output = local_path(raw.get("output", "./outputs/demo"), base)
    if Path(input_path) == Path(output) or Path(input_path) in Path(output).parents:
        raise ValueError("output_inside_input")
    model_root = Path(local_path(raw.get("model_root", "./model"), base))
    hardware = require_object(raw.get("hardware", {}), "hardware", {"profile", "gpus"})
    if hardware.get("profile", "external_vlm_cpu_tools") != "external_vlm_cpu_tools":
        raise ValueError("unknown_hardware_profile:external_vlm_cpu_tools")
    gpus = hardware.get("gpus", [])
    if not isinstance(gpus, list) or any(type(v) is not int or v < 0 for v in gpus) or len(gpus) != len(set(gpus)):
        raise ValueError("invalid_gpu_list:hardware.gpus")
    capabilities = raw.get("capabilities", list(CAPABILITIES))
    tools = raw.get("tools", list(TOOLS))
    for value, allowed, field in ((capabilities, CAPABILITIES, "capabilities"), (tools, TOOLS, "tools")):
        if not isinstance(value, list) or any(not isinstance(v, str) or v not in allowed for v in value) or len(value) != len(set(value)):
            raise ValueError(f"invalid_config_choices:{field}")
    if not capabilities:
        raise ValueError("empty_capabilities")
    source = raw.get("source_name", "")
    if not isinstance(source, str) or (source and (not re.fullmatch(r"[A-Za-z0-9_.-]+", source) or source in {".", ".."})):
        raise ValueError("invalid_source_name:use_a_single_folder_name")
    budget = require_object(raw.get("budget", {}), "budget", {"max_api_attempts"})
    attempts = budget.get("max_api_attempts", 100)
    if type(attempts) is not int or attempts < 1:
        raise ValueError("invalid_api_budget:positive_integer_required")
    network = require_object(raw.get("network", {}), "network", {"proxy_mode", "proxy_url"})
    mode = network.get("proxy_mode", "direct")
    if mode not in {"direct", "inherit", "explicit"} or (mode == "explicit" and not network.get("proxy_url")):
        raise ValueError("invalid_proxy_mode:direct_inherit_or_explicit")
    definitions = yaml.safe_load(files("manifests").joinpath("models.yaml").read_text(encoding="utf-8"))["resources"]
    targets = {entry["id"]: str(model_root / entry["target"]) for entry in definitions}
    values = {
        "VIDUEU_BLIP_MODEL": targets["blip2"], "VIDUEU_WHISPER_MODEL": targets["whisper"],
        "VIDUEU_CLIP_MODEL": targets["clip"], "VIDUEU_OCR_DET": targets["ocr_det"],
        "VIDUEU_OCR_REC": targets["ocr_rec"], "VIDUEU_YOLO_WEIGHTS": str(Path(targets["yolo"]) / "yolov8m.pt"),
        "VIDUEU_RAFT_ROOT": targets["raft_code"], "VIDUEU_OMNIPARSER_ROOT": targets["omni_code"],
        "VIDUEU_FLORENCE_ROOT": targets["florence"], "VIDUEU_QWEN_MODEL": targets["qwen"],
        "VIDUEU_TOOL_PYTHON": executable_path(raw.get("tool_python", "./environments/omniparser/bin/python"), base),
        "VIDUEU_LLM_MODEL": "deepseek-flash", "DEEPSEEK_API_KEY_FILE": "",
        "VIDUEU_VLM_BASE_URL": "http://127.0.0.1:8110/v1",
    }
    values.update(environ)
    values.update(VIDUEU_INPUT=input_path, VIDUEU_OUTPUT_ROOT=output, VIDUEU_WORK_ROOT=output)
    config = yaml.load(files("manifests").joinpath("pipeline_defaults.yaml").read_text(encoding="utf-8"), Loader=ConfigLoader)
    for public, internal in (("llm", "qwen_llm"), ("vlm", "qwen_vlm")):
        api = require_object(raw.get(public, {}), public, API_FIELDS)
        if "wire_profile" in api and api["wire_profile"] not in {"openai", "vllm"}:
            raise ValueError("invalid_api_wire_profile:" + public)
        for key in ("base_url", "model", "api_key_env", "api_key_file"):
            if key in api and (not isinstance(api[key], str) or not api[key].strip()):
                raise ValueError("invalid_api_field:" + public + "." + key)
        if "base_url" in api:
            url = urlsplit(api["base_url"])
            if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
                raise ValueError("invalid_api_base_url:" + public)
        if api.get("api_key_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api["api_key_env"]):
            raise ValueError("invalid_api_key_environment_name")
        config["models"][internal]["api"].update(api)
        target = config["models"][internal]["api"]
        if api.get("api_key_env"):
            target.pop("api_key", None)
        # Reasoning-capable providers must not inherit DeepSeek's disabled
        # thinking field, which some gateways interpret as reasoning=none.
        if api.get("reasoning_effort") and "thinking" not in api and "enable_thinking" not in api:
            target["thinking"] = None
        if public == "llm" and "api_key_file" not in api:
            target.pop("api_key_file", None)
            target["api_key_env"] = api.get("api_key_env", "DEEPSEEK_API_KEY")
        if api.get("api_key_file"):
            target["api_key_file"] = local_path(api["api_key_file"], base)
            target.pop("api_key", None)
    config["models"]["qwen_llm"]["api"]["max_api_calls"] = attempts
    configured_proxy = network.get("proxy_url", "") if mode == "explicit" else ""
    config["models"]["qwen_llm"]["api"]["proxy_url"] = configured_proxy
    config["models"]["qwen_vlm"]["api"]["proxy_url"] = configured_proxy
    config["runtime"]["enabled_tools"] = tools
    config["runtime"]["capabilities"] = capabilities
    config["tasks"]["T2_asr"].update(enabled="asr" in tools, language=None)
    if "omniparser" not in tools:
        config["models"].pop("omniparser", None)
    config = merge(config, require_object(raw.get("overrides", {}), "overrides"))
    config = expand_environment(config, values)
    config["_vidueu"] = {"config_path": str(Path(path).resolve()), "model_root": str(model_root),
                         "python": (executable_path(configured_python, base)
                                    if isinstance(configured_python, str) and any(c in configured_python for c in ("/", "\\"))
                                    else configured_python or sys.executable),
                         "source_name": source, "gpus": ",".join(map(str, gpus)), "proxy_mode": mode,
                         "proxy_url": network.get("proxy_url", "") if mode == "explicit" else ""}
    resolve_internal_paths(config, base)
    return config


def resolve_internal_paths(config, base):
    scalar_paths = {"input_path", "output_root", "work_root", "source_root", "temp_dir", "output_dir", "log_dir", "debug_dir",
                    "semantic_dir", "dataset_output", "output_file", "whisper_model", "image_embedding_model", "blip2_model",
                    "paddleocr_det", "paddleocr_rec", "yolo", "repo_root", "weights_dir", "model_path", "weights",
                    "caption_model_path", "caption_processor_path", "caption_model_code_path", "python_executable",
                    "api_key_file", "budget_path", "call_log_path", "prompt_audit_path", "allowed_local_media_path"}
    def walk(value):
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key == "python_executable" and isinstance(child, str) and child:
                    value[key] = executable_path(child, base)
                elif key in scalar_paths and isinstance(child, str) and child:
                    value[key] = local_path(child, base)
                elif key == "api_key_files" and isinstance(child, list):
                    value[key] = [local_path(v, base) for v in child]
                else:
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(config)


def required_assets(config):
    tools = set(config.get("runtime", {}).get("enabled_tools", TOOLS))
    result = ["clip", "blip2"]
    if config.get("optical_flow", {}).get("backend") == "raft":
        result += ["raft_code", "raft_weights"]
    for tool, ids in (("asr", ["whisper"]), ("ocr", ["ocr_det", "ocr_rec"]), ("yolo", ["yolo"]),
                      ("omniparser", ["omni_code", "omni_weights", "florence"])):
        if tool in tools:
            result += ids
    return result


def template(preset="external-vlm"):
    if preset != "external-vlm":
        raise ValueError("unknown_init_preset:external-vlm")
    return {"version": 1, "preset": "paper", "input": "./videos", "output": "./outputs/demo", "model_root": "./model",
            "capabilities": list(CAPABILITIES), "tools": list(TOOLS),
            "hardware": {"profile": "external_vlm_cpu_tools", "gpus": []},
            "llm": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash", "api_key_env": "DEEPSEEK_API_KEY"},
            "vlm": {"base_url": "http://127.0.0.1:8110/v1", "model": "Qwen3.5-9B-local"},
            "budget": {"max_api_attempts": 100}, "network": {"proxy_mode": "direct"}}
