# executor/models/api_llm.py
# 远程 LLM 封装：兼容 self.llm.chat(prompt) 用法

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


class RemoteLLM:
    """
    通过 OpenAI 兼容接口调用远程 LLM 的简单封装。

    设计目标：
    - 和原来的 QwenLLM 一样，暴露一个 chat(prompt: str) -> str 方法；
    - 内部使用 chat.completions.create；
    - 不做多轮对话记忆，由上层自己拼 prompt。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gemini-2.5-flash-nothinking",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ):
        """
        参数：
        - api_key: 平台的 API Key。建议从环境变量里读，而不是写死。
        - base_url: OpenAI 兼容的服务地址，比如 "https://www.chataiapi.com/v1"
        - model: 模型名称
        - temperature / max_tokens: 默认采样参数
        - system_prompt: 可选的全局 system 指令，常规场景可以留空，调用方直接把 prompt 拼好
        """
        self.api_key = api_key or os.getenv("REMOTE_LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("REMOTE_LLM_BASE_URL", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

        if not self.api_key:
            raise ValueError("RemoteLLM: api_key 未设置，请传入 api_key 或设置 REMOTE_LLM_API_KEY 环境变量")
        if not self.base_url:
            raise ValueError("RemoteLLM: base_url 未设置，请传入 base_url 或设置 REMOTE_LLM_BASE_URL 环境变量")

        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        print("🧠 [RemoteLLM] 使用远程 LLM：")
        print(f"    base_url   = {self.base_url}")
        print(f"    model      = {self.model}")
        print(f"    temp       = {self.temperature}")
        print(f"    max_tokens = {self.max_tokens}")

    def chat(self, prompt: str) -> str:
        """
        与原来的 QwenLLM.chat 接口保持一致：
        - 输入：单个字符串 prompt
        - 输出：模型生成的文本（string）
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        messages.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        # 统一返回字符串
        return resp.choices[0].message.content or ""
