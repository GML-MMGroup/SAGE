"""
Doubao API module
"""
import os
from openai import OpenAI

class DoubaoAPI:
    """Doubao API client class"""

    def __init__(self, api_key=None):
        """
        Initialize Doubao client

        Args:
            api_key: Doubao API key (falls back to API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("API_KEY", "")
        if not self.api_key:
            raise ValueError("Doubao API key not provided. Set API_KEY environment variable or pass api_key argument.")
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.environ.get("LLM_API_BASE_URL", "https://api.example.com/v1")
        )
        self.default_model = "doubao-seed-2-0-pro-260215"

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
            "extra_body": {"enable_thinking": enable_thinking}
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
