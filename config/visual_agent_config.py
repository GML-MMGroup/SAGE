"""
Visual Agent Configuration Interface
====================================

This file configures the model used by the visual recognition agent.
The visual agent identifies products from visual context and catalog candidates.

Environment Variables Required:
- VISUAL_AGENT_API_KEY: API key for the visual agent model (falls back to SKYCLAW_API_KEY/SERVICE_API_KEY/API_KEY)
- VISUAL_AGENT_API_BASE_URL: Base URL for the API endpoint (falls back to SKYCLAW_API_URL/SERVICE_API_BASE_URL)
"""
from openai import OpenAI
import os
from config.service_agent_config import (
    SERVICE_MODEL_NAME,
    SERVICE_API_KEY,
    SERVICE_API_BASE_URL,
    SERVICE_MAX_TOKENS,
    SERVICE_TEMPERATURE,
    SERVICE_ENABLE_THINKING,
)

# ==============================================================================
# VISUAL AGENT MODEL CONFIGURATION
# ===============================================================================

# Model name for visual recognition agent.
# VISUAL_AGENT_MODEL_NAME = "qwen3.7-plus"  # You can change this to your desired model name or set it via environment variable
VISUAL_AGENT_MODEL_NAME = "gpt-5.5"

# API Key for visual recognition agent model.
VISUAL_AGENT_API_KEY = os.environ.get("VAPI_API_KEY", os.environ.get("SKYCLAW_API_KEY", SERVICE_API_KEY))
# VISUAL_AGENT_API_KEY = os.environ.get("QWEN_API_KEY")  # Support Qwen-specific env var for backward compatibility

# Base URL for the API endpoint.
VISUAL_AGENT_API_BASE_URL = os.environ.get("VAPI_API_URL", os.environ.get("SKYCLAW_API_URL", SERVICE_API_BASE_URL))
# VISUAL_AGENT_API_BASE_URL = os.environ.get("QWEN_API_URL")  # Support Qwen-specific env var for backward compatibility

# Maximum tokens for visual agent responses.
VISUAL_AGENT_MAX_TOKENS = 8192

# Temperature for visual agent.
VISUAL_AGENT_TEMPERATURE = 0
VISUAL_AGENT_SEED = 66
# GPT/OpenAI Chat Completions reasoning effort. Use "none" to disable reasoning.
VISUAL_AGENT_REASONING_EFFORT = os.environ.get("VISUAL_AGENT_REASONING_EFFORT", "").strip() or None


# Kept only for backward compatibility; OpenAI/GPT thinking is controlled by reasoning_effort.
# VISUAL_AGENT_ENABLE_THINKING = True
VISUAL_AGENT_ENABLE_THINKING = False


# GPT/OpenAI image detail level for image_base64 frame inputs.
# Official values: "auto", "low", or "high".
VISUAL_IMAGE_DETAIL ="high"

# System prompt for visual recognition agent.
VISUAL_AGENT_SYSTEM_PROMPT = "You are a visual agent. Ignore non-visual questions. Only answer visual questions by identifying the required visual target, whether it is a physical object/entity or a textual representation such as a label, sign, menu entry, price tag, or on-screen text."


# ==============================================================================
# API CALL FUNCTION
# ===============================================================================

def call_visual_agent_model(messages, max_retries=3, enable_thinking=None):
    """Call visual model, input: messages/retries/unused thinking flag -> output: response text and token counts"""
    import time
    import random

    model_lower = str(VISUAL_AGENT_MODEL_NAME or "").strip().lower()
    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=VISUAL_AGENT_API_KEY, base_url=VISUAL_AGENT_API_BASE_URL)
            kwargs = {
                "model": VISUAL_AGENT_MODEL_NAME,
                "messages": messages,
                "max_tokens": VISUAL_AGENT_MAX_TOKENS,
                "seed": VISUAL_AGENT_SEED,
            }
            if model_lower.startswith("gpt") or "gpt-5" in model_lower:
                if VISUAL_AGENT_REASONING_EFFORT:
                    kwargs["reasoning_effort"] = VISUAL_AGENT_REASONING_EFFORT
                if VISUAL_AGENT_REASONING_EFFORT == "none":
                    kwargs["temperature"] = VISUAL_AGENT_TEMPERATURE
            else:
                kwargs["temperature"] = VISUAL_AGENT_TEMPERATURE
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            input_tokens = 0
            output_tokens = 0
            if hasattr(completion, "usage") and completion.usage:
                input_tokens = getattr(completion.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(completion.usage, "completion_tokens", 0) or 0
            return content, input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[Visual Agent Model Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[Visual Agent Model Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


# ==============================================================================
# CONFIGURATION VALIDATION
# ===============================================================================

def validate_config():
    """Validate the visual agent configuration, input: none -> output: (is_valid, error_message)"""
    if not VISUAL_AGENT_API_KEY:
        return False, "VISUAL_AGENT_API_KEY (or SKYCLAW_API_KEY/SERVICE_API_KEY/API_KEY) environment variable is not set."
    if not VISUAL_AGENT_API_BASE_URL:
        return False, "VISUAL_AGENT_API_BASE_URL is not configured."
    if VISUAL_IMAGE_DETAIL not in {"high", "auto", "low"}:
        return False, "VISUAL_IMAGE_DETAIL must be one of: high, auto, low."
    if VISUAL_AGENT_REASONING_EFFORT not in {None, "none", "minimal", "low", "medium", "high", "xhigh"}:
        return False, "VISUAL_AGENT_REASONING_EFFORT must be one of: none, minimal, low, medium, high, xhigh."
    return True, None


def print_config():
    """Print current visual agent configuration."""
    print("Visual Agent Configuration:")
    print(f"  Model: {VISUAL_AGENT_MODEL_NAME}")
    print(f"  API Base URL: {VISUAL_AGENT_API_BASE_URL}")
    print(f"  Max Tokens: {VISUAL_AGENT_MAX_TOKENS}")
    if VISUAL_AGENT_REASONING_EFFORT == "none":
        print(f"  Temperature: {VISUAL_AGENT_TEMPERATURE}")
    else:
        print("  Temperature: omitted for GPT reasoning")
    print(f"  Seed: {VISUAL_AGENT_SEED}")
    print(f"  Reasoning Effort: {VISUAL_AGENT_REASONING_EFFORT or 'default'}")
    print(f"  Legacy Thinking Flag: {VISUAL_AGENT_ENABLE_THINKING}")
    print(f"  Image Detail: {VISUAL_IMAGE_DETAIL}")
    print(f"  System Prompt: {VISUAL_AGENT_SYSTEM_PROMPT}")


if __name__ == "__main__":
    is_valid, error_msg = validate_config()
    if is_valid:
        print_config()
    else:
        print(f"Configuration Error: {error_msg}")
