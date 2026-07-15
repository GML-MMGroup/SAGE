"""
Boxed Frame Reviewer Configuration Interface
============================================

Configures the Gemini model that reviews and corrects one full boxed frame at
a time before target crops are generated.
"""
from openai import OpenAI
import os


# ==============================================================================
# FRAMES REVIEW MODEL CONFIGURATION
# ==============================================================================

FRAMES_REVIEW_MODEL_NAME = "gpt-5.6-sol"
FRAMES_REVIEW_API_KEY = os.environ.get("VAPI_API_KEY", "")
FRAMES_REVIEW_API_BASE_URL = os.environ.get("VAPI_API_URL", "")
FRAMES_REVIEW_MAX_TOKENS = 4096
FRAMES_REVIEW_TEMPERATURE = 0
FRAMES_REVIEW_SEED = 66
FRAMES_REVIEW_REASONING_EFFORT = None
FRAMES_REVIEW_ENABLE_THINKING = None
FRAMES_REVIEW_MAX_RETRIES = 3
FRAMES_REVIEW_DEBUG_STDOUT = os.environ.get("FRAMES_REVIEW_DEBUG_STDOUT", "").strip() == "1"
FRAMES_REVIEW_SYSTEM_PROMPT = (
    "You are a boxed-frame review and correction agent. Review the single full boxed image against "
    "the supplied visual target task. Decide whether the green Target box is correct, can be corrected "
    "unambiguously, or must be rejected. Use only visible evidence and return JSON only."
)


def _build_openai_client(api_key, api_base_url):
    """Build VAPI client, input: key/base URL -> output: OpenAI-compatible client."""
    return OpenAI(api_key=api_key, base_url=api_base_url)


def _extract_usage_tokens(usage):
    """Extract usage tokens, input: usage object -> output: input/output tuple."""
    if not usage:
        return 0, 0
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return input_tokens or 0, output_tokens or 0


def call_frames_review_model(messages, max_retries=None):
    """Call Gemini boxed reviewer, input: messages/retries -> output: response and token counts."""
    import random
    import time

    if max_retries is None:
        max_retries = FRAMES_REVIEW_MAX_RETRIES
    if not FRAMES_REVIEW_API_KEY or not FRAMES_REVIEW_API_BASE_URL:
        return "Error: VAPI_API_KEY or VAPI_API_URL is not configured", 0, 0

    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = _build_openai_client(FRAMES_REVIEW_API_KEY, FRAMES_REVIEW_API_BASE_URL)
            kwargs = {
                "model": FRAMES_REVIEW_MODEL_NAME,
                "messages": messages,
                "max_tokens": FRAMES_REVIEW_MAX_TOKENS,
                "temperature": FRAMES_REVIEW_TEMPERATURE,
                "seed": FRAMES_REVIEW_SEED,
            }
            model_lower = str(FRAMES_REVIEW_MODEL_NAME).lower()
            if (model_lower.startswith("gpt") or "gpt-5" in model_lower) and FRAMES_REVIEW_REASONING_EFFORT:
                kwargs["reasoning_effort"] = FRAMES_REVIEW_REASONING_EFFORT
            elif "qwen" in model_lower and FRAMES_REVIEW_ENABLE_THINKING is not None:
                kwargs["extra_body"] = {"enable_thinking": FRAMES_REVIEW_ENABLE_THINKING}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            input_tokens, output_tokens = _extract_usage_tokens(getattr(completion, "usage", None))
            return content, input_tokens, output_tokens
        except Exception as exc:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(
                    f"[Frames Review Model Retry] Attempt {attempt + 1}/{max_retries} failed: "
                    f"{str(exc)}. Retrying in {wait_time:.2f}s..."
                )
                time.sleep(wait_time)
            else:
                print(f"[Frames Review Model Error] Failed after {max_retries} attempts: {str(exc)}")
                return f"Error: {str(exc)}", 0, 0


def print_config():
    """Print current frames reviewer configuration."""
    print("Frames Review Configuration:")
    print(f"  Model: {FRAMES_REVIEW_MODEL_NAME}")
    print(f"  API Base URL: {FRAMES_REVIEW_API_BASE_URL}")
    print(f"  Max Tokens: {FRAMES_REVIEW_MAX_TOKENS}")
    print(f"  Temperature: {FRAMES_REVIEW_TEMPERATURE}")
    print(f"  Seed: {FRAMES_REVIEW_SEED}")
    print(f"  Reasoning Effort: {FRAMES_REVIEW_REASONING_EFFORT or 'default'}")
    print(f"  Thinking Enabled: {FRAMES_REVIEW_ENABLE_THINKING}")
    print(f"  Max Retries: {FRAMES_REVIEW_MAX_RETRIES}")
    print(f"  Debug Stdout: {FRAMES_REVIEW_DEBUG_STDOUT}")
    print(f"  System Prompt: {FRAMES_REVIEW_SYSTEM_PROMPT}")


if __name__ == "__main__":
    print_config()
