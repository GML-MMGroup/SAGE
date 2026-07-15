"""
Frame Selecter Configuration Interface
======================================

This file configures the model used by the image_base64 frame selecter.
The frame selecter chooses task-relevant sampled frames before visual recognition.
"""
from openai import OpenAI
import os

from config.service_agent_config import SERVICE_API_KEY, SERVICE_API_BASE_URL


# ==============================================================================
# FRAME SELECTER MODEL CONFIGURATION
# ==============================================================================

FRAME_SELECTER_MODEL_NAME = "gpt-5.5"
FRAME_SELECTER_API_KEY = os.environ.get("VAPI_API_KEY", os.environ.get("SKYCLAW_API_KEY", SERVICE_API_KEY))
FRAME_SELECTER_API_BASE_URL = os.environ.get("VAPI_API_URL", os.environ.get("SKYCLAW_API_URL", SERVICE_API_BASE_URL))
FRAME_SELECTER_MAX_TOKENS = 32768
FRAME_SELECTER_TEMPERATURE = 0
FRAME_SELECTER_SEED = 66
FRAME_SELECTER_REASONING_EFFORT = os.environ.get("FRAME_SELECTER_REASONING_EFFORT", "").strip() or None
FRAME_SELECTER_ENABLE_THINKING = None
FRAME_SELECTER_FRAME_INTERVAL_SECONDS = 1
FRAME_SELECTER_MAX_SELECTED_FRAMES = 6
FRAME_SELECTER_SYSTEM_PROMPT = """
You are a frame selection agent. Select the frame(s) most relevant to the visual recognition task and determine whether the task's final visual target cardinality is single or multiple.

## Chronological Action Order Rule
- When the task requires identifying a target or selecting frames based on an action process, use the frame time information to understand the order in which the action occurs in the video. First reason along the timeline about how the action starts, changes, and completes, then select the most appropriate frame(s) that best capture the relevant action or target.

## Visibility Rule
- If a relevant frame does not clearly show the target or the needed visual information is occluded, return both that relevant frame and an earlier or later frame where it is more visible.
- If a frame shows only one form of the target, either its textual representation or the physical object itself, return that frame together with earlier or later frame(s) that show both the textual representation and the physical object for the same target.

## Primary And Auxiliary Frame Rule

中文：
对于每个视觉目标，优先选择一张主帧作为目标锚点；只有在需要补充无遮挡、标签、文字或实物信息时，才额外选择一张辅助帧，且辅助帧必须展示同一个实体。

English:
For each visual target, first select one primary frame as the target anchor; select one additional auxiliary frame only when complementary evidence such as an unobstructed view, label, text, or physical appearance is needed, and the auxiliary frame must show the same entity.

## Preference Rule
- Select the frame showing the item that best matches the target description; if multiple frames show that same item, choose the clearest one, but never switch to a different item just because it is easier to recognize.
- If one frame clearly best matches a target and the target is not occluded and is fully visible, return only that frame for that target.
""".strip()
FRAME_SELECTER_DEBUG_STDOUT = os.environ.get("FRAME_SELECTER_DEBUG_STDOUT", "").strip() == "1"


# ==============================================================================
# API CALL FUNCTION
# ==============================================================================

def _build_openai_client(api_key, api_base_url):
    """Build OpenAI client, input: key/base_url -> output: OpenAI client"""
    if api_base_url:
        return OpenAI(api_key=api_key, base_url=api_base_url)
    return OpenAI(api_key=api_key)


def _extract_usage_tokens(usage):
    """Extract token counts, input: usage object -> output: input/output token tuple"""
    if not usage:
        return 0, 0
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    if output_tokens is None:
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return input_tokens or 0, output_tokens or 0


def _call_frame_selecter_chat_model(messages, model_name, api_key, api_base_url, max_tokens, temperature, enable_thinking, reasoning_effort, max_retries=3):
    """Call frame selecter model, input: messages/model config -> output: response text and token counts"""
    import time
    import random

    model_lower = str(model_name or "").strip().lower()
    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = _build_openai_client(api_key, api_base_url)
            kwargs = {
                "model": model_name.strip(),
                "messages": messages,
                "max_tokens": max_tokens,
                "seed": FRAME_SELECTER_SEED,
            }
            if model_lower.startswith("gpt") or "gpt-5" in model_lower:
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                if reasoning_effort == "none":
                    kwargs["temperature"] = temperature
            else:
                kwargs["temperature"] = temperature
                if "qwen" in model_lower and enable_thinking is not None:
                    kwargs["extra_body"] = {"enable_thinking": enable_thinking}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            input_tokens, output_tokens = _extract_usage_tokens(getattr(completion, "usage", None))
            return content, input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[Frame Selecter Model Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[Frame Selecter Model Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def call_frame_selecter_model(messages, max_retries=3, enable_thinking=None):
    """Call frame selecter model, input: messages/retries/thinking -> output: response text and token counts"""
    if enable_thinking is None:
        enable_thinking = FRAME_SELECTER_ENABLE_THINKING
    return _call_frame_selecter_chat_model(
        messages, FRAME_SELECTER_MODEL_NAME, FRAME_SELECTER_API_KEY, FRAME_SELECTER_API_BASE_URL,
        FRAME_SELECTER_MAX_TOKENS, FRAME_SELECTER_TEMPERATURE, enable_thinking,
        FRAME_SELECTER_REASONING_EFFORT, max_retries
    )


def print_config():
    """Print current frame selecter configuration."""
    print("Frame Selecter Configuration:")
    print(f"  Model: {FRAME_SELECTER_MODEL_NAME}")
    print(f"  API Base URL: {FRAME_SELECTER_API_BASE_URL}")
    print(f"  Max Tokens: {FRAME_SELECTER_MAX_TOKENS}")
    if FRAME_SELECTER_REASONING_EFFORT == "none":
        print(f"  Temperature: {FRAME_SELECTER_TEMPERATURE}")
    else:
        print("  Temperature: omitted for GPT reasoning")
    print(f"  Seed: {FRAME_SELECTER_SEED}")
    print(f"  Reasoning Effort: {FRAME_SELECTER_REASONING_EFFORT or 'default'}")
    print(f"  Qwen Thinking Mode: {FRAME_SELECTER_ENABLE_THINKING}")
    print(f"  Frame Interval Seconds: {FRAME_SELECTER_FRAME_INTERVAL_SECONDS}")
    print(f"  Max Selected Frames: {FRAME_SELECTER_MAX_SELECTED_FRAMES}")
    print(f"  Debug Stdout: {FRAME_SELECTER_DEBUG_STDOUT}")
    print(f"  System Prompt: {FRAME_SELECTER_SYSTEM_PROMPT}")


if __name__ == "__main__":
    print_config()
