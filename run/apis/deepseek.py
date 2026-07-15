"""
DeepSeek API module
"""
import os
from openai import OpenAI
from config.service_agent_config import SERVICE_MAX_TOKENS, SERVICE_TEMPERATURE


class DeepSeekAPI:
    """DeepSeek API client class"""

    def __init__(self, api_key=None):
        """
        Initialize DeepSeek client

        Args:
            api_key: DeepSeek API key (falls back to DEEPSEEK_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DeepSeek API key not provided. Set DEEPSEEK_API_KEY environment variable or pass api_key argument.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com")
        )
        self.default_model = "deepseek-v4-pro"

    def chat(self, messages, model=None, thinking_type="enabled", reasoning_effort="high"):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            thinking_type: DeepSeek thinking mode type
            reasoning_effort: DeepSeek reasoning effort level

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if model is None:
            model = self.default_model

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": SERVICE_MAX_TOKENS,
            "temperature": SERVICE_TEMPERATURE,
            "extra_body": {
                "thinking": {"type": thinking_type}
            }
        }
        if reasoning_effort is not None:
            kwargs["extra_body"]["reasoning_effort"] = reasoning_effort

        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("DeepSeek API returned empty content")

        input_tokens = 0
        output_tokens = 0
        if hasattr(completion, "usage") and completion.usage:
            input_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(completion.usage, "completion_tokens", 0) or 0

        return content, input_tokens, output_tokens
