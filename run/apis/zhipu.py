"""
ZhipuAI API module
"""
from zai import ZhipuAiClient
import os


class ZhipuAPI:
    """ZhipuAI API client class"""

    def __init__(self, api_key=None):
        """
        Initialize ZhipuAI client

        Args:
            api_key: ZhipuAI API key (falls back to ZHIPU_API_KEY env var)
        """
        api_key = api_key or os.environ.get("ZHIPU_API_KEY", "")
        if not api_key:
            raise ValueError("ZhipuAI API key not provided. Set ZHIPU_API_KEY environment variable or pass api_key argument.")
        self.client = ZhipuAiClient(api_key=api_key)
        self.default_model = "glm-5v-turbo"

    def chat(self, messages, model=None, thinking_enabled=False):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            thinking_enabled: Whether to enable thinking mode

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if model is None:
            model = self.default_model

        kwargs = {
            "model": model,
            "messages": messages,
            "thinking": {"type": "enabled"} if thinking_enabled else {"type": "disabled"}
        }

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        # Extract token information
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

        return content, input_tokens, output_tokens
