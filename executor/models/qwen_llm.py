# executor/models/qwen_llm.py
import os
import json
import time
import random
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from executor.models.api_budget import ApiBudget

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

try:
    import requests
    from requests.adapters import HTTPAdapter
except Exception:
    requests = None
    HTTPAdapter = None


ConfigLike = Union[str, Dict[str, Any]]


def _split_keys_blob(v: str) -> List[str]:
    """
    将一个字符串中的 key 拆成列表。
    支持：
      - 每行一个 key
      - 逗号分隔
      - 空白分隔
      - 忽略空行
      - 忽略 # 注释行
    """
    if not v:
        return []

    parts: List[str] = []
    text = v.replace("\r", "\n").replace(",", "\n")
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 允许行尾带注释/多余内容，只取第一个 token
        parts.append(line.split()[0].strip())
    return [x for x in parts if x]


def _dedupe_keep_order(xs: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _coerce_optional_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    text = str(v).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled", "enable"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled", "disable"}:
        return False
    return None


class QwenLLM:
    """
    纯文本 Qwen LLM 封装，支持：
    - 本地 HF 模型加载
    - 远程 OpenAI-compatible API（多 key / 轮询 / cooldown / 线程安全）

    外部接口保持不变：
        chat(prompt: str, max_new_tokens: int = 512) -> str
    """

    def __init__(
        self,
        model_dir: ConfigLike,
        device: str = "cuda",
        *,
        use_auto_device_map: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
    ):
        self.raw_config: ConfigLike = model_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.mode: str = "local"  # "local" or "api"
        self._chat_state = threading.local()

        local_model_dir: Optional[str] = None
        api_conf: Dict[str, Any] = {}

        if isinstance(model_dir, str):
            # 兼容旧配置：直接给本地模型目录
            self.mode = "local"
            local_model_dir = model_dir

        elif isinstance(model_dir, dict):
            mode = str(model_dir.get("mode") or "local").strip().lower()
            if mode == "api":
                self.mode = "api"
                # 支持嵌套 api: {...} 或平铺
                api_conf = model_dir.get("api") or model_dir
            else:
                self.mode = "local"
                local_model_dir = (
                    model_dir.get("local_model_dir")
                    or model_dir.get("model_dir")
                    or model_dir.get("path")
                )
                if not local_model_dir:
                    raise ValueError(
                        "[QwenLLM] mode='local' 但未提供 local_model_dir/model_dir/path"
                    )
        else:
            raise TypeError(
                f"[QwenLLM] model_dir 期望为 str 或 dict，收到: {type(model_dir)}"
            )

        if self.mode == "api":
            self._init_api_runtime(api_conf)
        else:
            self._init_local_model(
                local_model_dir,
                use_auto_device_map=use_auto_device_map,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )

    # ------------------------------------------------------------------
    # 本地 HF 模型初始化
    # ------------------------------------------------------------------
    def _init_local_model(
        self,
        model_dir: str,
        *,
        use_auto_device_map: bool,
        load_in_4bit: bool,
        load_in_8bit: bool,
    ) -> None:
        self.model_dir = model_dir

        print(f"[QwenLLM] 使用本地 LLM 模型目录: {model_dir}")
        print(f"    device = {self.device}")
        print(f"    use_auto_device_map = {use_auto_device_map}")
        print(f"    load_in_4bit = {load_in_4bit}, load_in_8bit = {load_in_8bit}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        model_kwargs: Dict[str, Any] = dict(
            trust_remote_code=True,
        )

        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = torch_dtype

        if use_auto_device_map and self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_dir,
            **model_kwargs,
        )

        if not model_kwargs.get("device_map"):
            self.model.to(self.device)

        self.model.eval()

    # ------------------------------------------------------------------
    # API runtime 初始化
    # ------------------------------------------------------------------
    def _init_api_runtime(self, api_conf: Dict[str, Any]) -> None:
        if requests is None or HTTPAdapter is None:
            raise RuntimeError(
                "[QwenLLM] API 模式需要 requests，请先安装 requests"
            )

        self.api_model: str = str(api_conf.get("model") or api_conf.get("api_model") or "").strip()
        if not self.api_model:
            raise ValueError(
                "[QwenLLM] API 模式需要提供 model，例如 'gpt-4.1' 或你的远程模型名"
            )

        self.base_url: str = str(
            api_conf.get("base_url") or api_conf.get("api_base") or ""
        ).rstrip("/")
        if not self.base_url:
            raise ValueError("[QwenLLM] API 模式需要提供 base_url")

        self.timeout_s: float = float(api_conf.get("timeout_s") or 120.0)
        self.pause_interval_s: float = float(api_conf.get("pause_interval_s") or 0.5)

        # max_retries = 0 表示无限重试
        self.max_retries: int = int(api_conf.get("max_retries", 0) or 0)
        self.max_inflight: int = int(api_conf.get("max_inflight", 0) or 0)

        self.pool_connections: int = int(api_conf.get("pool_connections", 64) or 64)
        self.pool_maxsize: int = int(api_conf.get("pool_maxsize", 64) or 64)

        self.proxy_url: str = str(api_conf.get("proxy_url") or "").strip()
        self.verify_ssl: bool = bool(api_conf.get("verify_ssl", True))

        self.reload_key_file: bool = bool(api_conf.get("reload_key_file", False))

        self.cooldown_base_s_429: float = float(api_conf.get("cooldown_base_s_429", 2.0) or 2.0)
        self.cooldown_base_s_other: float = float(api_conf.get("cooldown_base_s_other", 0.5) or 0.5)
        self.cooldown_max_s: float = float(api_conf.get("cooldown_max_s", 30.0) or 30.0)
        self.slow_key_sec: float = float(api_conf.get("slow_key_sec", 45.0) or 45.0)

        self.http_referer: str = str(api_conf.get("http_referer") or "").strip()
        self.x_title: str = str(api_conf.get("x_title") or "").strip()
        self.temperature: float = float(api_conf.get("temperature", 0.0) or 0.0)
        self.stream: bool = bool(api_conf.get("stream", False))
        self.reasoning_effort: str = str(api_conf.get("reasoning_effort") or "").strip()
        self.extra_body: Dict[str, Any] = dict(api_conf.get("extra_body", {}) or {})

        thinking = api_conf.get("thinking")
        if isinstance(thinking, dict):
            self.thinking: Optional[Dict[str, Any]] = dict(thinking)
        else:
            enabled = _coerce_optional_bool(
                api_conf.get("enable_thinking", api_conf.get("thinking_enabled"))
            )
            if enabled is None:
                self.thinking = None
            else:
                self.thinking = {"type": "enabled" if enabled else "disabled"}

        # 1) inline key(s)
        inline_keys = api_conf.get("api_keys")
        if isinstance(inline_keys, list):
            self.inline_api_keys: List[str] = [str(x).strip() for x in inline_keys if str(x).strip()]
        else:
            one_key = str(api_conf.get("api_key") or "").strip()
            self.inline_api_keys = [one_key] if one_key else []

        # 2) key 文件
        api_key_file = api_conf.get("api_key_file")
        api_key_files = api_conf.get("api_key_files")
        if isinstance(api_key_files, list) and api_key_files:
            self.api_key_files: List[str] = [str(x).strip() for x in api_key_files if str(x).strip()]
        elif api_key_file:
            self.api_key_files = [str(api_key_file).strip()]
        else:
            self.api_key_files = []

        # 3) env
        api_key_envs = api_conf.get("api_key_envs")
        api_key_env = api_conf.get("api_key_env")
        if isinstance(api_key_envs, list) and api_key_envs:
            self.api_key_envs: List[str] = [str(x).strip() for x in api_key_envs if str(x).strip()]
        elif api_key_env:
            self.api_key_envs = [x.strip() for x in str(api_key_env).split(",") if x.strip()]
        else:
            # 不再强绑 OPENAI_API_KEY；只有你显式写了才读取
            self.api_key_envs = []

        self.api_call_log = str(api_conf.get("call_log_path", "") or "")
        self.max_api_calls = int(api_conf.get("max_api_calls", 0) or 0)
        budget_path = str(api_conf.get("budget_path", "") or "")
        self.api_budget = ApiBudget(budget_path, self.max_api_calls) if budget_path else None
        self._api_call_count = 0
        self._attestation_lock = threading.Lock()
        self._endpoint = f"{self.base_url}/chat/completions"

        # key 调度状态
        self._rr_lock = threading.Lock()
        self._rr_idx = 0

        self._cooldown_lock = threading.Lock()
        self._cooldown_until: Dict[str, float] = {}

        # key 缓存 / 热更新
        self._keys_lock = threading.RLock()
        self._api_keys: List[str] = []
        self._api_keys_sig: Optional[tuple] = None

        # 每线程 session
        self._tls = threading.local()
        self._sessions_lock = threading.Lock()
        self._all_sessions: List[Any] = []

        # 并发限流
        self._if_sem = threading.Semaphore(self.max_inflight) if self.max_inflight > 0 else None

        # API 模式下置空本地模型对象，兼容外部访问
        self.model_dir = f"api::{self.api_model}"
        self.tokenizer = None
        self.model = None

        self._refresh_api_keys(force=True)

        print("[QwenLLM] 使用远程 API LLM：")
        print(f"    base_url        = {self.base_url}")
        print(f"    model           = {self.api_model}")
        print(f"    key_files       = {self.api_key_files}")
        print(f"    key_envs        = {self.api_key_envs}")
        print(f"    loaded_keys     = {len(self._api_keys)}")
        print(f"    reload_key_file = {self.reload_key_file}")
        print(f"    max_inflight    = {self.max_inflight if self.max_inflight > 0 else 'disabled'}")
        print(f"    timeout_s       = {self.timeout_s}")
        print(f"    pool_conn/max   = {self.pool_connections}/{self.pool_maxsize}")
        if self.thinking is not None:
            print(f"    thinking        = {self.thinking}")
        if self.reasoning_effort:
            print(f"    reasoning_effort= {self.reasoning_effort}")

    # ------------------------------------------------------------------
    # API: key 文件与缓存
    # ------------------------------------------------------------------
    def _build_key_file_sig(self) -> tuple:
        sig = []
        for p in self.api_key_files:
            path = Path(p)
            if not path.exists():
                sig.append((str(path), None, None))
            else:
                st = path.stat()
                sig.append((str(path), st.st_mtime_ns, st.st_size))
        return tuple(sig)

    def _refresh_api_keys(self, force: bool = False) -> List[str]:
        with self._keys_lock:
            sig = self._build_key_file_sig()

            if not force:
                if not self.reload_key_file and self._api_keys:
                    return list(self._api_keys)
                if self.reload_key_file and self._api_keys and sig == self._api_keys_sig:
                    return list(self._api_keys)

            try:
                keys: List[str] = []

                # inline
                keys.extend(self.inline_api_keys)

                # file
                for p in self.api_key_files:
                    path = Path(p)
                    if not path.exists():
                        raise RuntimeError(f"[QwenLLM] api_key_file 不存在: {p}")
                    text = path.read_text(encoding="utf-8")
                    keys.extend(_split_keys_blob(text))

                # env
                for env_name in self.api_key_envs:
                    v = os.getenv(env_name, "")
                    if v:
                        keys.extend(_split_keys_blob(v))

                keys = _dedupe_keep_order(keys)
                if not keys:
                    raise RuntimeError(
                        "[QwenLLM] API 模式未加载到任何 key；请检查 api_key_file/api_key_files/api_key_env(s)/api_keys"
                    )

                self._api_keys = keys
                self._api_keys_sig = sig
                return list(self._api_keys)

            except Exception:
                # 非 force 热更新失败时，保留旧 key 池，避免运行时因文件瞬时异常直接中断
                if not force and self._api_keys:
                    return list(self._api_keys)
                raise

    # ------------------------------------------------------------------
    # API: session / proxies
    # ------------------------------------------------------------------
    def _get_session(self):
        sess = getattr(self._tls, "session", None)
        if sess is not None:
            return sess

        sess = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
            max_retries=0,
        )
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)

        self._tls.session = sess
        with self._sessions_lock:
            self._all_sessions.append(sess)
        return sess

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        if self.proxy_url:
            return {"http": self.proxy_url, "https": self.proxy_url}
        return None

    # ------------------------------------------------------------------
    # API: key 选择 / cooldown
    # ------------------------------------------------------------------
    def _rr_pick(self, items: Sequence[str]) -> str:
        if not items:
            raise RuntimeError("[QwenLLM] no api keys available")
        with self._rr_lock:
            idx = self._rr_idx % len(items)
            self._rr_idx += 1
        return items[idx]

    def _pick_api_key(self, keys: List[str]) -> str:
        if not keys:
            raise RuntimeError("[QwenLLM] no api keys available")

        now = time.time()
        with self._cooldown_lock:
            avail = [k for k in keys if self._cooldown_until.get(k, 0.0) <= now]
            if avail:
                return self._rr_pick(avail)

            k_min = min(keys, key=lambda k: self._cooldown_until.get(k, 0.0))
            wake = self._cooldown_until.get(k_min, now)

        sleep_s = max(0.05, wake - now)
        time.sleep(min(1.0, sleep_s))
        return k_min

    def _mark_key_cooldown(self, api_key: str, attempt: int, *, is_429: bool) -> None:
        base = self.cooldown_base_s_429 if is_429 else self.cooldown_base_s_other
        cd = min(self.cooldown_max_s, base * (1.8 ** max(0, attempt - 1)))
        until = time.time() + cd

        with self._cooldown_lock:
            prev = self._cooldown_until.get(api_key, 0.0)
            if until > prev:
                self._cooldown_until[api_key] = until

    def _maybe_cooldown_slow_key(self, api_key: str, t0: float) -> None:
        dt = time.time() - float(t0)
        if self.slow_key_sec > 0 and dt > self.slow_key_sec:
            self._mark_key_cooldown(api_key, attempt=1, is_429=False)
            print(f"[QwenLLM][API][SLOW_KEY] dt={dt:.1f}s -> cooldown")

    def _pause_and_retry(self, attempt: int, reason: str) -> None:
        base = float(self.pause_interval_s) if isinstance(self.pause_interval_s, (int, float)) else 0.5
        base = max(0.1, base)

        sleep_s = min(2.0, base * (1.2 ** max(0, attempt - 1)))
        sleep_s *= 1.0 + random.uniform(-0.15, 0.15)
        sleep_s = max(0.05, sleep_s)

        print(f"⚠️ [QwenLLM][API] attempt={attempt} failed: {reason}\n    -> sleep {sleep_s:.2f}s then retry")
        time.sleep(sleep_s)

    # ------------------------------------------------------------------
    # API: payload / response parse
    # ------------------------------------------------------------------
    def _build_api_headers(self, api_key: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title
        return headers

    def _build_api_payload(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        try:
            mn = int(max_new_tokens)
        except Exception:
            mn = 512

        payload: Dict[str, Any] = {
            "model": self.api_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(1, mn),
            "temperature": self.temperature,
            "stream": self.stream,
        }
        if self.extra_body:
            payload.update(self.extra_body)
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    @staticmethod
    def _extract_text_from_api_response(resp: Any) -> str:
        """
        将 OpenAI-compatible 返回统一抽取成 string。
        """
        if isinstance(resp, str):
            return resp

        if isinstance(resp, dict):
            # chat.completions
            try:
                content = resp["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                parts.append(str(c.get("text", "")))
                            elif "text" in c:
                                parts.append(str(c.get("text", "")))
                    return "\n".join(parts)
                return str(content)
            except Exception:
                pass

            # old completions
            try:
                txt = resp["choices"][0]["text"]
                if isinstance(txt, str):
                    return txt
            except Exception:
                pass

            for k in ("text", "output_text", "content"):
                v = resp.get(k)
                if isinstance(v, str):
                    return v

            return str(resp)

        return str(resp)

    def _single_api_request(self, prompt: str, max_new_tokens: int, api_key: str) -> str:
        sess = self._get_session()
        payload = self._build_api_payload(prompt, max_new_tokens=max_new_tokens)
        headers = self._build_api_headers(api_key)
        proxies = self._get_proxies()

        t0 = time.time()
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        with self._attestation_lock:
            if self.max_api_calls and self._api_call_count >= self.max_api_calls:
                raise ValueError("DeepSeek API call budget exhausted")
            budget_id = self.api_budget.reserve(prompt_sha256, self.api_model) if self.api_budget else None
            self._api_call_count += 1
        try:
            r = sess.post(
                url=self._endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
                proxies=proxies,
                verify=self.verify_ssl,
            )
        except requests.RequestException as e:
            if budget_id:
                self.api_budget.finish(budget_id, "request_failed", {"error_type": type(e).__name__})
            raise RuntimeError(f"request_exception: {repr(e)}") from e

        ct = (r.headers.get("Content-Type") or "").lower()
        if r.status_code != 200:
            if budget_id:
                self.api_budget.finish(budget_id, "http_error", {"http_status": r.status_code})
            if self.api_call_log:
                with self._attestation_lock:
                    Path(self.api_call_log).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.api_call_log, "a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"requested_model": self.api_model, "response_model": None,
                            "http_status": r.status_code, "usage": {}, "elapsed_s": time.time() - t0,
                             "context": dict(getattr(self._chat_state, "context", {}) or {}),
                             "prompt_sha256": prompt_sha256, "budget_attempt_id": budget_id}) + "\n")
            if r.status_code in {400, 401, 402, 403, 404}:
                raise ValueError(f"Non-retryable API error http_{r.status_code}: {r.text[:1200]}")
            raise RuntimeError(f"http_{r.status_code} ct={ct}: {r.text[:1200]}")

        try:
            data = r.json()
        except Exception as e:
            if budget_id:
                self.api_budget.finish(budget_id, "invalid_response", {"http_status": r.status_code})
            raise RuntimeError(f"invalid_json ct={ct}: {e}; raw={r.text[:1200]}") from e

        attestation = {"requested_model": self.api_model, "response_model": data.get("model"),
                       "http_status": r.status_code,
                       "response_id": data.get("id"), "usage": data.get("usage", {}),
                       "elapsed_s": time.time() - t0,
                        "context": dict(getattr(self._chat_state, "context", {}) or {}),
                        "prompt_sha256": prompt_sha256, "budget_attempt_id": budget_id}
        if budget_id:
            self.api_budget.finish(budget_id, "response", attestation)
        self._tls.last_attestation = attestation
        if self.api_call_log:
            with self._attestation_lock:
                Path(self.api_call_log).parent.mkdir(parents=True, exist_ok=True)
                with open(self.api_call_log, "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(attestation, ensure_ascii=False) + "\n")
        text = self._extract_text_from_api_response(data).strip()
        if not text:
            raise RuntimeError(f"empty_content ct={ct}: resp={str(data)[:1200]}")

        self._maybe_cooldown_slow_key(api_key, t0)
        return text

    # ------------------------------------------------------------------
    # 统一 chat 接口
    # ------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, prompt: str, max_new_tokens: int = 512) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"QwenLLM.chat(prompt) 期望 str 类型，但收到: {type(prompt)}")

        request = {
            "prompt": prompt,
            "max_new_tokens": int(max_new_tokens),
        }
        try:
            if self.mode == "api":
                out = self._chat_via_api(prompt, max_new_tokens=max_new_tokens)
            else:
                out = self._chat_via_local_model(prompt, max_new_tokens=max_new_tokens)
            self._emit_chat_record(request=request, response=out)
            return out
        except Exception as e:
            self._emit_chat_record(request=request, error=e)
            raise

    def set_chat_logger(self, logger: Any) -> None:
        self._chat_state.logger = logger

    def set_chat_context(self, context: Dict[str, Any] | None) -> None:
        self._chat_state.context = dict(context or {})

    def _emit_chat_record(
        self,
        *,
        request: Dict[str, Any],
        response: Any = None,
        error: BaseException | None = None,
    ) -> None:
        logger = getattr(self._chat_state, "logger", None)
        if logger is None:
            return
        try:
            logger.log_interaction(
                model_kind="llm",
                model_name=self._current_model_name(),
                mode=self.mode,
                context=dict(getattr(self._chat_state, "context", {}) or {}),
                request=request,
                response=response,
                error=error,
            )
        except Exception:
            pass

    def _current_model_name(self) -> str:
        if self.mode == "api":
            return str(getattr(self, "api_model", "") or "api")
        return str(getattr(self, "model_dir", "") or "local")

    # ----------------- 本地模型推理路径 -----------------
    def _chat_via_local_model(self, prompt: str, max_new_tokens: int) -> str:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("[QwenLLM] 本地模式下 tokenizer/model 未初始化")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )

        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)  # type: ignore[attr-defined]
        else:
            inputs = inputs.to(self.device)

        generated_ids = self.model.generate(  # type: ignore[call-arg]
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        input_len = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0, input_len:]
        out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return out.strip()

    # ----------------- API 推理路径 -----------------
    def _chat_via_api(self, prompt: str, max_new_tokens: int) -> str:
        if self._if_sem is not None:
            self._if_sem.acquire()

        try:
            attempt = 0
            while True:
                attempt += 1
                api_key = ""

                try:
                    keys = self._refresh_api_keys(force=False)
                    api_key = self._pick_api_key(keys)
                    return self._single_api_request(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        api_key=api_key,
                    )

                except ValueError:
                    raise
                except BaseException as e:
                    msg = f"{type(e).__name__}: {e}"
                    is_429 = (
                        (" 429" in msg)
                        or ("http_429" in msg.lower())
                        or ("rate limit" in msg.lower())
                    )

                    if api_key:
                        self._mark_key_cooldown(api_key, attempt, is_429=is_429)

                    # max_retries = 0 => infinite
                    if self.max_retries > 0 and attempt >= self.max_retries:
                        raise RuntimeError(
                            f"[QwenLLM][API] failed after {attempt} retries, last_error={repr(e)}"
                        ) from e

                    self._pause_and_retry(attempt, msg)

        finally:
            if self._if_sem is not None:
                self._if_sem.release()

    # ------------------------------------------------------------------
    # 资源释放
    # ------------------------------------------------------------------
    def close(self) -> None:
        # 关闭 API 模式下创建的 session
        try:
            with self._sessions_lock:
                sessions = list(self._all_sessions)
                self._all_sessions = []
            for sess in sessions:
                try:
                    sess.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 本地模型释放显存（可选）
        try:
            if getattr(self, "model", None) is not None:
                del self.model
                self.model = None
        except Exception:
            pass

        try:
            if getattr(self, "tokenizer", None) is not None:
                del self.tokenizer
                self.tokenizer = None
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
