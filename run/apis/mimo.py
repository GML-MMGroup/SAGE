"""
Xiaomi MiMo API module
"""
import os
from openai import OpenAI

class MimoAPI:
    """Xiaomi MiMo API client class"""

    def __init__(self, api_key=None):
        """
        Initialize Xiaomi MiMo client

        Args:
            api_key: Xiaomi MiMo API key (falls back to MIMO_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("MIMO_API_KEY", "")
        if not self.api_key:
            raise ValueError("MiMo API key not provided. Set MIMO_API_KEY environment variable or pass api_key argument.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("MIMO_API_BASE_URL", "https://api.xiaomimimo.com/v1")
        )
        self.default_model = "mimo-v2-omni"
        self.default_system_prompt = "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."

    def chat(self, messages, model=None, max_tokens=1024, enable_thinking=False):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            max_tokens: Max tokens
            enable_thinking: Whether to enable thinking mode, default is False

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if model is None:
            model = self.default_model

        kwargs = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "extra_body": {"thinking": {"type": "enabled"} if enable_thinking else {"type": "disabled"}}
        }

        completion = self.client.chat.completions.create(**kwargs)
        content = completion.choices[0].message.content

        # Extract token information
        input_tokens = 0
        output_tokens = 0
        if hasattr(completion, 'usage') and completion.usage:
            input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

        return content, input_tokens, output_tokens
