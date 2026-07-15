"""
Kimi API module
"""
import os
from openai import OpenAI

class KimiAPI:
    """Kimi API client class"""

    def __init__(self, api_key=None):
        """
        Initialize Kimi client

        Args:
            api_key: Kimi API key (falls back to KIMI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("KIMI_API_KEY", "")
        if not self.api_key:
            raise ValueError("Kimi API key not provided. Set KIMI_API_KEY environment variable or pass api_key argument.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("KIMI_API_URL", "https://api.moonshot.cn/v1")
        )
        self.default_model = "Pro/moonshotai/Kimi-K2.6"
        self.default_system_prompt = "You are Kimi."
        # Kimi's thinking mode is implemented by selecting different models: kimi-k2-thinking is the thinking model

    def chat(self, messages, model=None, enable_thinking=False):
        """
        General chat interface

        Args:
            messages: Chat message list
            model: Model name
            enable_thinking: Whether to enable thinking mode, uses kimi-k2-thinking model when enabled

        Returns:
            (response_text, input_tokens, output_tokens)
        """
        if model is None:
            # Kimi's thinking mode is implemented by switching models
            model = "kimi-k2-thinking" if enable_thinking else self.default_model

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        content = completion.choices[0].message.content

        # Extract token information
        input_tokens = 0
        output_tokens = 0
        if hasattr(completion, 'usage') and completion.usage:
            input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

        return content, input_tokens, output_tokens
