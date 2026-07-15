"""
脚本作用：
封装 GPT-5.5 的 OpenAI-compatible Chat Completions 调用，供 service supervisor 使用。

执行逻辑：
1. 从 VAPI 环境变量读取 GPT-5.5 的 API Key 和 Base URL。
2. 将项目内别名 `gpt5.5` 归一化为官方模型名 `gpt-5.5`。
3. 调用 Chat Completions，失败最多重试 3 次，超时或失败后进入下一次重试。

运行示例：
    python run/multi_agent.py --service_model_name gpt5.5
"""
import os
import random
import time
from openai import OpenAI


GPT55_MODEL_NAME = "gpt-5.5"
GPT55_API_KEY = os.environ.get("VAPI_API_KEY")
GPT55_API_BASE_URL = os.environ.get("VAPI_API_URL")
GPT55_MAX_TOKENS = 32768
GPT55_TEMPERATURE = 0
GPT55_ENABLE_THINKING = False
GPT55_REASONING_EFFORT = os.environ.get("TOOL_PLANNER_REASONING_EFFORT", "medium")
GPT55_TIMEOUT = float(os.environ.get("GPT55_TIMEOUT", "120"))


class GPT55API:
    """GPT-5.5 API client class"""

    def __init__(self, api_key=None, api_base_url=None):
        """
        Initialize GPT-5.5 client

        Args:
            api_key: GPT-5.5 API key (falls back to VAPI_API_KEY env var)
            api_base_url: GPT-5.5 API base URL (falls back to VAPI_API_URL env var)
        """
        self.api_key = api_key or GPT55_API_KEY
        if not self.api_key:
            raise ValueError("GPT-5.5 API key not provided. Set VAPI_API_KEY environment variable or pass api_key argument.")
        self.api_base_url = api_base_url or GPT55_API_BASE_URL
        client_kwargs = {"api_key": self.api_key, "timeout": GPT55_TIMEOUT}
        if self.api_base_url:
            client_kwargs["base_url"] = self.api_base_url
        self.client = OpenAI(**client_kwargs)
        self.default_model = GPT55_MODEL_NAME

    def _normalize_model_name(self, model):
        """Normalize model alias, input: model name -> output: official model name"""
        model = str(model or self.default_model).strip()
        if model.lower() in {"gpt5.5", "gpt-5.5"}:
            return GPT55_MODEL_NAME
        return model

    def chat(self, messages, model=None, enable_thinking=None, max_retries=3):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            enable_thinking: Whether to send reasoning_effort
            max_retries: Maximum retry attempts

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        model = self._normalize_model_name(model)
        if enable_thinking is None:
            enable_thinking = GPT55_ENABLE_THINKING

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": GPT55_MAX_TOKENS,
            "temperature": GPT55_TEMPERATURE,
            "timeout": GPT55_TIMEOUT,
        }
        if enable_thinking and GPT55_REASONING_EFFORT:
            kwargs["reasoning_effort"] = GPT55_REASONING_EFFORT

        base_delay = 10
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(**kwargs)
                content = completion.choices[0].message.content
                if content is None:
                    raise ValueError("GPT-5.5 API returned empty content")

                input_tokens = 0
                output_tokens = 0
                if hasattr(completion, "usage") and completion.usage:
                    input_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(completion.usage, "completion_tokens", 0) or 0
                return content, input_tokens, output_tokens
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"[LLM Retry] Service(gpt-5.5) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[LLM Error] Service(gpt-5.5) failed after {max_retries} attempts: {str(e)}")
                    return f"Error: {str(e)}", 0, 0
