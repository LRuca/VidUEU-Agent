from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from build_umm_bundles import build_bundle_from_stage_set, build_manifest, write_json, write_jsonl, _find_stage_sets
from orchestrator.umm_task_producer import produce_task_samples


# If you really want to hardcode a key for local experiments, place it here.
# Leaving this empty keeps the script in deterministic fallback mode unless
# you pass --deepseek-api-key at runtime.
DEEPSEEK_API_KEY = ""
DEEPSEEK_BASE_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"


TASK_TYPES = (
    "unify.seed.l3.gts",
    "unify.weave.uie_select_recall",
    "unify.comm.task4.c1_steps",
    "unify.comm.task4.c2_fill_gap",
    "unify.isg.i2.seq_select",
)


class DeepSeekLLM:
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = 0.2,
        timeout_s: float = 120.0,
        thinking: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "",
    ) -> None:
        if not str(api_key or "").strip():
            raise ValueError("DeepSeekLLM requires a non-empty api_key.")
        self.api_key = str(api_key).strip()
        self.model = str(model).strip() or DEEPSEEK_MODEL
        self.base_url = str(base_url).strip() or DEEPSEEK_BASE_URL
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.thinking = dict(thinking) if isinstance(thinking, dict) else None
        self.reasoning_effort = str(reasoning_effort or "").strip()

    def chat(self, prompt: str, max_new_tokens: int = 1200) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": str(prompt)}],
            "temperature": self.temperature,
            "max_tokens": int(max_new_tokens),
            "stream": False,
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        context = ssl.create_default_context()
        with urllib.request.urlopen(request, timeout=self.timeout_s, context=context) as response:
            raw = response.read().decode("utf-8")
        data = json.loads(raw)
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"DeepSeek response missing choices: {raw[:500]}")
        message = choices[0].get("message", {})
        content = message.get("content", "") if isinstance(message, dict) else ""
        return str(content or "")


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _split_keys_blob(v: str) -> List[str]:
    if not v:
        return []
    parts: List[str] = []
    text = v.replace("\r", "\n").replace(",", "\n")
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts.append(line.split()[0].strip())
    return [x for x in parts if x]


def _dedupe_keep_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_config_path(config_path: Path, maybe_path: str) -> Path:
    raw = Path(str(maybe_path).strip())
    if raw.is_absolute():
        return raw
    return (config_path.parent / raw).resolve()


def _normalize_chat_completions_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return DEEPSEEK_BASE_URL
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _normalize_thinking(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        kind = str(value.get("type") or "").strip().lower()
        if kind in {"enabled", "disabled"}:
            return {"type": kind}
        return dict(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return {"type": "enabled"}
    if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return {"type": "disabled"}
    return None


def _collect_api_keys_from_conf(api_conf: Dict[str, Any], config_path: Path) -> List[str]:
    keys: List[str] = []

    inline_keys = api_conf.get("api_keys")
    if isinstance(inline_keys, list):
        keys.extend([str(x).strip() for x in inline_keys if str(x).strip()])
    else:
        one_key = str(api_conf.get("api_key") or "").strip()
        if one_key:
            keys.append(one_key)

    api_key_files = api_conf.get("api_key_files")
    api_key_file = api_conf.get("api_key_file")
    key_file_list: List[str] = []
    if isinstance(api_key_files, list) and api_key_files:
        key_file_list = [str(x).strip() for x in api_key_files if str(x).strip()]
    elif api_key_file:
        key_file_list = [str(api_key_file).strip()]
    for item in key_file_list:
        path = _resolve_config_path(config_path, item)
        if not path.exists():
            raise FileNotFoundError(f"API key file not found: {path}")
        keys.extend(_split_keys_blob(path.read_text(encoding="utf-8")))

    api_key_envs = api_conf.get("api_key_envs")
    api_key_env = api_conf.get("api_key_env")
    env_names: List[str] = []
    if isinstance(api_key_envs, list) and api_key_envs:
        env_names = [str(x).strip() for x in api_key_envs if str(x).strip()]
    elif api_key_env:
        env_names = [x.strip() for x in str(api_key_env).split(",") if x.strip()]
    for env_name in env_names:
        env_value = os.getenv(env_name, "")
        if env_value:
            keys.extend(_split_keys_blob(env_value))

    return _dedupe_keep_order(keys)


def _resolve_deepseek_settings(
    *,
    config: Dict[str, Any],
    config_path: Optional[Path],
    cli_key: str,
    cli_model: str,
    cli_base_url: str,
) -> Dict[str, Any]:
    models = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
    deepseek_conf = models.get("deepseek_replay", {}) if isinstance(models.get("deepseek_replay"), dict) else {}
    deepseek_api = deepseek_conf.get("api", {}) if isinstance(deepseek_conf.get("api"), dict) else {}
    qwen_conf = models.get("qwen_llm", {}) if isinstance(models.get("qwen_llm"), dict) else {}
    qwen_api = qwen_conf.get("api", {}) if isinstance(qwen_conf.get("api"), dict) else {}
    chosen_api = deepseek_api or qwen_api

    keys: List[str] = []
    if config_path is not None and chosen_api:
        keys = _collect_api_keys_from_conf(chosen_api, config_path)

    api_key = str(cli_key or "").strip() or (keys[0] if keys else "") or str(DEEPSEEK_API_KEY or "").strip()
    model = str(cli_model or "").strip() or str(chosen_api.get("model") or "").strip() or DEEPSEEK_MODEL
    configured_url = str(cli_base_url or "").strip() or str(chosen_api.get("base_url") or "").strip() or DEEPSEEK_BASE_URL
    base_url = _normalize_chat_completions_url(configured_url)
    thinking = _normalize_thinking(chosen_api.get("thinking"))
    if thinking is None:
        thinking = _normalize_thinking(
            chosen_api.get("enable_thinking", chosen_api.get("thinking_enabled"))
        )
    return {
        "api_key": api_key,
        "model": model,
        "base_url": base_url,
        "thinking": thinking or {},
        "reasoning_effort": str(chosen_api.get("reasoning_effort") or "").strip(),
    }


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return _make_jsonable(dict(value.__dict__))
    return str(value)


def _bundle_sort_key(bundle: Dict[str, Any]) -> tuple[str, int]:
    segment = bundle.get("segment", {}) if isinstance(bundle.get("segment"), dict) else {}
    return (
        str(segment.get("video_path", "") or ""),
        int(segment.get("index", 0) or 0),
    )


def assemble_bundles(input_root: Path) -> List[Dict[str, Any]]:
    stage_sets = _find_stage_sets(input_root)
    bundles = [build_bundle_from_stage_set(stage_set, root=input_root) for stage_set in stage_sets]
    bundles.sort(key=_bundle_sort_key)
    return bundles


def replay_bundles(
    bundles: Iterable[Dict[str, Any]],
    *,
    llm: Any = None,
    comm_c1_k: int = 3,
    isg_k: int = 2,
    llm_retry_attempts: int = 1,
) -> Dict[str, Any]:
    replay_rows: List[Dict[str, Any]] = []
    flat_samples: List[Dict[str, Any]] = []
    summary_by_task = {task_type: {"segments_with_sample": 0, "samples": 0, "drops": 0} for task_type in TASK_TYPES}

    for bundle in bundles:
        result = produce_task_samples(
            bundle,
            llm=llm,
            comm_c1_k=comm_c1_k,
            isg_k=isg_k,
            llm_retry_attempts=llm_retry_attempts,
        )
        segment = bundle.get("segment", {}) if isinstance(bundle.get("segment"), dict) else {}
        samples = result.get("samples", {}) if isinstance(result.get("samples"), dict) else {}
        drops = result.get("drops", {}) if isinstance(result.get("drops"), dict) else {}

        row = {
            "segment": {
                "index": int(segment.get("index", 0) or 0),
                "video_id": str(segment.get("video_id", "") or ""),
                "video_path": str(segment.get("video_path", "") or ""),
                "segment_path": str(segment.get("segment_path", "") or ""),
                "debug_paths": _make_jsonable(segment.get("debug_paths", {}) or {}),
            },
            "samples": {},
            "drops": _make_jsonable(drops),
        }

        for task_type in TASK_TYPES:
            task_samples = list(samples.get(task_type) or [])
            row["samples"][task_type] = _make_jsonable(task_samples)
            if task_samples:
                summary_by_task[task_type]["segments_with_sample"] += 1
                summary_by_task[task_type]["samples"] += len(task_samples)
            task_drops = list(drops.get(task_type) or [])
            summary_by_task[task_type]["drops"] += len(task_drops)

            for sample in task_samples:
                flat_samples.append(
                    {
                        "task_type": task_type,
                        "segment_index": int(segment.get("index", 0) or 0),
                        "video_id": str(segment.get("video_id", "") or ""),
                        "video_path": str(segment.get("video_path", "") or ""),
                        "segment_path": str(segment.get("segment_path", "") or ""),
                        "sample": _make_jsonable(sample),
                    }
                )

        replay_rows.append(row)

    summary = {
        "segment_count": len(replay_rows),
        "task_summary": summary_by_task,
    }
    return {"replay_rows": replay_rows, "flat_samples": flat_samples, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end UMM replay: stage dumps -> segment bundles -> replay samples -> summary outputs."
    )
    parser.add_argument("--input-root", required=True, help="Run root or parent directory containing logs/debug stage json files.")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for all generated outputs. Defaults to <input-root>/umm_replay",
    )
    parser.add_argument("--comm-c1-k", type=int, default=3, help="Step count for comm.task4.c1_steps replay.")
    parser.add_argument("--isg-k", type=int, default=2, help="Future step count for isg.i2.seq_select replay.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional config yaml. In deepseek mode the script can read api_key_file/api_key_env/api_key from config.",
    )
    parser.add_argument(
        "--llm-mode",
        choices=("deterministic", "deepseek"),
        default="deterministic",
        help="Use deterministic fallback or call DeepSeek for grounded text generation.",
    )
    parser.add_argument(
        "--deepseek-api-key",
        default="",
        help="Override the hardcoded DEEPSEEK_API_KEY at the top of this script.",
    )
    parser.add_argument(
        "--deepseek-model",
        default=DEEPSEEK_MODEL,
        help="DeepSeek model name for --llm-mode deepseek.",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default=DEEPSEEK_BASE_URL,
        help="DeepSeek-compatible chat completions endpoint for --llm-mode deepseek.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for DeepSeek generation.",
    )
    parser.add_argument(
        "--llm-retry-attempts",
        type=int,
        default=1,
        help="How many generation attempts to allow per task when using LLM-backed replay.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_root / "umm_replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = assemble_bundles(input_root)
    manifest = build_manifest(bundles, input_root=input_root)
    config_path = Path(args.config).resolve() if args.config else None
    config = _load_config(config_path) if config_path is not None else {}
    llm = None
    if args.llm_mode == "deepseek":
        settings = _resolve_deepseek_settings(
            config=config,
            config_path=config_path,
            cli_key=str(args.deepseek_api_key),
            cli_model=str(args.deepseek_model),
            cli_base_url=str(args.deepseek_base_url),
        )
        if not settings["api_key"]:
            raise ValueError(
                "DeepSeek mode requires an API key. Provide --deepseek-api-key, or set api_key/api_key_file/api_key_env "
                "under models.deepseek_replay.api or models.qwen_llm.api in the config."
            )
        llm = DeepSeekLLM(
            api_key=settings["api_key"],
            model=settings["model"],
            base_url=settings["base_url"],
            temperature=float(args.temperature),
            thinking=settings.get("thinking") or None,
            reasoning_effort=str(settings.get("reasoning_effort") or ""),
        )
    replay_result = replay_bundles(
        bundles,
        llm=llm,
        comm_c1_k=int(args.comm_c1_k),
        isg_k=int(args.isg_k),
        llm_retry_attempts=max(1, int(args.llm_retry_attempts)),
    )

    bundles_out = output_dir / "segment_bundles.jsonl"
    manifest_out = output_dir / "run_manifest.json"
    replay_out = output_dir / "replay_results.jsonl"
    flat_samples_out = output_dir / "replay_samples.jsonl"
    summary_out = output_dir / "replay_summary.json"

    write_jsonl(bundles_out, bundles)
    write_json(manifest_out, manifest)
    write_jsonl(replay_out, replay_result["replay_rows"])
    write_jsonl(flat_samples_out, replay_result["flat_samples"])
    write_json(summary_out, replay_result["summary"])

    print(f"Wrote {len(bundles)} bundles to: {bundles_out}")
    print(f"Wrote replay rows to: {replay_out}")
    print(f"Wrote flat replay samples to: {flat_samples_out}")
    print(f"Wrote replay summary to: {summary_out}")
    print(f"Replay mode: {args.llm_mode}")
    if args.llm_mode == "deepseek":
        print(f"DeepSeek model: {llm.model}")
        print(f"DeepSeek base_url: {llm.base_url}")


if __name__ == "__main__":
    main()
