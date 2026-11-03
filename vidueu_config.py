"""Lightweight, shared YAML configuration loading. No model imports or IO to APIs."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
SECRET_FIELDS = {"api_key", "api_keys", "authorization", "password", "token", "proxy_url"}


class ConfigLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate_config_key:{key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


ConfigLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def expand_environment(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    env = os.environ if environ is None else environ
    if isinstance(value, str):
        missing = sorted({a or b for a, b in ENV_PATTERN.findall(value) if (a or b) not in env})
        if missing:
            raise ValueError("unexpanded_environment_variables:" + ",".join(missing))
        # Expand parsed scalars, so a key containing ': ' or '#' stays a scalar.
        return ENV_PATTERN.sub(lambda m: env[m.group(1) or m.group(2)], value)
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("config_keys_must_be_strings")
        return {k: expand_environment(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_environment(v, env) for v in value]
    return value


def read_environment(path: str | Path, environ: Mapping[str, str] | None = None) -> dict:
    """Read adjacent dotenv as data; never execute shell code or alter os.environ."""
    env = dict(os.environ if environ is None else environ)
    dotenv = Path(path).resolve().parent / ".env"
    if not dotenv.is_file():
        return env
    values = {}
    for number, line in enumerate(dotenv.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name.strip()):
            raise ValueError(f"invalid_dotenv_line:{number}")
        value = value.strip()
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"invalid_dotenv_quote:{number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if "`" in value or "$(" in value:
            raise ValueError(f"shell_substitution_disallowed_in_dotenv:{number}")
        values[name.strip()] = value
    # Process environment wins. Expand dotenv references with the same strict rules.
    resolved = dict(env)
    active = set()

    def resolve(name):
        if name in resolved:
            return resolved[name]
        if name in active:
            raise ValueError("dotenv_reference_cycle")
        if name not in values:
            raise ValueError("unexpanded_environment_variables:" + name)
        active.add(name)
        result = ENV_PATTERN.sub(lambda m: resolve(m.group(1) or m.group(2)), values[name])
        active.remove(name)
        resolved[name] = result
        return result

    for name in values:
        resolve(name)
    return resolved


def load_config(path: str | Path, *, environ: Mapping[str, str] | None = None,
                overrides: Mapping[str, Any] | None = None) -> dict:
    value = yaml.load(Path(path).read_text(encoding="utf-8-sig"), Loader=ConfigLoader)
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("config_must_be_object")
    env = read_environment(path, environ)
    from vidueu.config import compile_config, merge, small_config
    if small_config(value):
        config = compile_config(value, path=path, environ=env, overrides=overrides)
    else:
        mapped = {}
        for key, replacement in (overrides or {}).items():
            if key in {"input", "output"}:
                env["VIDUEU_INPUT" if key == "input" else "VIDUEU_OUTPUT_ROOT"] = str(replacement)
                mapped.setdefault("runtime", {})["input_path" if key == "input" else "output_root"] = str(replacement)
        config = expand_environment(merge(value, mapped), env)
    for section in ("models", "runtime", "tasks", "paths", "loop"):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"config_section_must_be_object:{section}")
    for name in ("qwen_llm", "qwen_vlm"):
        api = config.get("models", {}).get(name, {}).get("api", {})
        if api.get("api_key_env") and env.get(api["api_key_env"]):
            api["api_key"] = env[api["api_key_env"]]
    return config


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if str(k).lower() in SECRET_FIELDS else redact(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value
