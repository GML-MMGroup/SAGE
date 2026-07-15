"""
User Agent Configuration Interface
===================================

This file configures the simulated user model for the competition.
Participants should modify this file to use their preferred model for user simulation.

Default: Qwen3.5-397B-A17B

Environment Variables Required:
- API_KEY: Your API key for the user simulation model
- LLM_API_BASE_URL: Base URL for the API endpoint (optional)
"""

import os
from openai import OpenAI

# ==============================================================================
# USER SIMULATION MODEL CONFIGURATION
# ==============================================================================

# Model name for user simulation
# Default: Qwen3.5-397B-A17B
# You can change this to any model you prefer

# USER_MODEL_NAME = os.environ.get("USER_MODEL_NAME", "Qwen/Qwen3.5-122B-A10B")
USER_MODEL_NAME ="Qwen/Qwen3.5-397B-A17B"
# USER_MODEL_NAME = "deepseek-v4-flash"

# API Key for user simulation model
# This should be set as an environment variable: export API_KEY="your-api-key"

USER_API_KEY=os.environ.get("SILICON_API_KEY")
# USER_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# USER_API_KEY = os.environ.get("SKYCLAW_API_KEY")

# Base URL for the API endpoint
# Default: https://api.example.com/v1/chat/completions
# Participants should set this according to their model deployment

USER_API_BASE_URL = os.environ.get("SILICON_API_URL")
# USER_API_BASE_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.siliconflow.cn/v1")
# USER_API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.apifree.ai/agent/v1")


# Maximum tokens for user model responses
USER_MAX_TOKENS = 8192

# Temperature for user model (0.0 - 2.0)
USER_TEMPERATURE = 0

# Whether to enable thinking mode (if supported by the model)
USER_ENABLE_THINKING = False


# ==============================================================================
# USER RESPONSE CORRECTOR MODEL CONFIGURATION
# ==============================================================================

# Defaults reuse the simulated user model. Change these when the corrector should
# use a different model, API endpoint, or decoding configuration.
USER_CORRECTOR_MODEL_NAME = "gpt-5.5"
USER_CORRECTOR_API_KEY = os.environ.get("VAPI_API_KEY")
USER_CORRECTOR_API_BASE_URL = os.environ.get("VAPI_API_URL")
USER_CORRECTOR_MAX_TOKENS = 4096
USER_CORRECTOR_TEMPERATURE = 0
USER_CORRECTOR_ENABLE_THINKING = False


USER_CORRECTOR_SYSTEM_PROMPT = '''
# Role: User Response Corrector

You are an expert in correcting simulated user responses in a multi-turn dialogue to ensure they align with the user's persona and instructions.
'''


# ==============================================================================
# API CALL FUNCTION
# ==============================================================================

def call_user_model(messages, max_retries=3, enable_thinking=None):
    """
    Call the user simulation model with the given messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_retries: Maximum number of retry attempts
        enable_thinking: Whether to enable thinking mode (None = use default)

    Returns:
        tuple: (response_text, input_tokens, output_tokens)
    """
    import time
    import random
    import requests

    if enable_thinking is None:
        enable_thinking = USER_ENABLE_THINKING

    BASE_DELAY = 10
    last_error = None

    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=USER_API_KEY,
                base_url=USER_API_BASE_URL
            )
            kwargs = {
                "model": USER_MODEL_NAME,
                "messages": messages,
                "extra_body": {"enable_thinking": enable_thinking},
                "temperature": USER_TEMPERATURE
            }
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content

            # Extract token information
            input_tokens = 0
            output_tokens = 0
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

            return content, input_tokens, output_tokens


        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[User Model Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[User Model Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def call_user_corrector_model(messages, max_retries=3, enable_thinking=None):
    """
    Call the user response corrector model with the given messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_retries: Maximum number of retry attempts
        enable_thinking: Whether to enable thinking mode (None = use default)

    Returns:
        tuple: (response_text, input_tokens, output_tokens)
    """
    import time
    import random
    import requests

    if enable_thinking is None:
        enable_thinking = USER_CORRECTOR_ENABLE_THINKING

    BASE_DELAY = 10
    last_error = None

    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=USER_CORRECTOR_API_KEY,
                base_url=USER_CORRECTOR_API_BASE_URL
            )
            kwargs = {
                "model": USER_CORRECTOR_MODEL_NAME,
                "messages": messages,
                "extra_body": {"enable_thinking": enable_thinking},
                "temperature": USER_CORRECTOR_TEMPERATURE,
                "max_tokens": USER_CORRECTOR_MAX_TOKENS
            }
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content

            input_tokens = 0
            output_tokens = 0
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0

            return content, input_tokens, output_tokens

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (BASE_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[User Corrector Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[User Corrector Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_config():
    """
    Validate the user agent configuration.

    Returns:
        tuple: (is_valid, error_message)
    """
    if not USER_API_KEY:
        return False, "API_KEY environment variable is not set. Please set it before running."

    if not USER_API_BASE_URL:
        return False, "LLM_API_BASE_URL is not configured."

    if not USER_CORRECTOR_API_KEY:
        return False, "USER_CORRECTOR_API_KEY is not configured."

    if not USER_CORRECTOR_API_BASE_URL:
        return False, "USER_CORRECTOR_API_BASE_URL is not configured."

    return True, None


if __name__ == "__main__":
    # Test configuration
    is_valid, error_msg = validate_config()
    if is_valid:
        print(f"User Agent Configuration:")
        print(f"  Model: {USER_MODEL_NAME}")
        print(f"  API Base URL: {USER_API_BASE_URL}")
        print(f"  Max Tokens: {USER_MAX_TOKENS}")
        print(f"  Temperature: {USER_TEMPERATURE}")
        print(f"  Thinking Mode: {USER_ENABLE_THINKING}")
        print(f"User Corrector Configuration:")
        print(f"  Model: {USER_CORRECTOR_MODEL_NAME}")
        print(f"  API Base URL: {USER_CORRECTOR_API_BASE_URL}")
        print(f"  Max Tokens: {USER_CORRECTOR_MAX_TOKENS}")
        print(f"  Temperature: {USER_CORRECTOR_TEMPERATURE}")
        print(f"  Thinking Mode: {USER_CORRECTOR_ENABLE_THINKING}")
    else:
        print(f"Configuration Error: {error_msg}")
