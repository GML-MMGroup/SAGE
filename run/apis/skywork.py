"""
Skywork API module
"""
import os
from openai import OpenAI
from config.service_agent_config import SERVICE_TEMPERATURE


class SkyworkAPI:
    """Skywork SkyClaw API client class"""

    def __init__(self, api_key=None):
        """
        Initialize Skywork client

        Args:
            api_key: SkyClaw API key (falls back to SKYCLAW_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("SKYCLAW_API_KEY", "")
        if not self.api_key:
            raise ValueError("SkyClaw API key not provided. Set SKYCLAW_API_KEY environment variable or pass api_key argument.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("SKYCLAW_API_URL", "https://api.apifree.ai/agent/v1")
        )
        self.default_model = "skywork-ai/skyclaw-v1"

    def chat(self, messages, model=None, enable_thinking=False):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            enable_thinking: Whether to enable thinking mode, default is False

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if model is None:
            model = self.default_model

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": 65536,
            "temperature": SERVICE_TEMPERATURE,
            "top_p": 0.95,
            "extra_body": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        }

        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content
        if content is None:
            raise ValueError("SkyClaw API returned empty content")

        input_tokens = 0
        output_tokens = 0
        if hasattr(completion, 'usage') and completion.usage:
            input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

        return content, input_tokens, output_tokens
