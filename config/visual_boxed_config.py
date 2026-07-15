"""
Visual Boxed Configuration Interface
====================================

This file configures the model used by the generic image_base64 visual target
grounding layer before visual recognition.
"""
from openai import OpenAI
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.service_agent_config import SERVICE_API_KEY, SERVICE_API_BASE_URL


# ==============================================================================
# VISUAL BOXED MODEL CONFIGURATION
# ==============================================================================

VISUAL_BOXED_MODEL_NAMES = [
    item.strip()
    for item in os.environ.get("VISUAL_BOXED_MODEL_NAMES", "gpt-5.6-sol,qwen-vl-max").split(",")
    if item.strip()
]
VISUAL_BOXED_API_KEY = os.environ.get("VISUAL_BOXED_API_KEY", os.environ.get("VAPI_API_KEY", os.environ.get("SKYCLAW_API_KEY", SERVICE_API_KEY)))
VISUAL_BOXED_API_BASE_URL = os.environ.get("VISUAL_BOXED_API_URL", os.environ.get("VAPI_API_URL", os.environ.get("SKYCLAW_API_URL", SERVICE_API_BASE_URL)))
VISUAL_BOXED_MAX_TOKENS = 8192
VISUAL_BOXED_TEMPERATURE = 0
VISUAL_BOXED_SEED = 66
VISUAL_BOXED_REASONING_EFFORT = os.environ.get("VISUAL_BOXED_REASONING_EFFORT", "").strip() or None
VISUAL_BOXED_ENABLE_THINKING = None
VISUAL_BOXED_MAX_RETRIES = 3
VISUAL_BOXED_SCENARIO_PREFIXES = ("retail", "order", "restaurant", "kitchen")
VISUAL_BOXED_SYSTEM_PROMPT = """
You are a visual target grounding agent. Locate and verify physical products, ingredients, recipes, dishes, menu entries, order entries, cooking-scene targets, and service-scene visual targets from selected frames using only visible evidence and the task wording.

## BBox Rule
- Each bbox must tightly cover the requested target and any associated identifying text required by these rules.
- A bbox must contain exactly one target object and must not include any part of any other object, including a neighboring object's body, cap, label, packaging, or associated text. If the target cannot be isolated without including part of another object, set `certainty` to `uncertain`, and the candidate must not be drawn.

## Label And Text Rule
- If the target appears both as a visible physical entity and as corresponding text or label, return one bbox that contains the matched pair: the target entity and its corresponding readable text/label region. The text or label must clearly correspond to that exact entity, not to a nearby or unrelated target.
- If there is no visible physical entity and the target appears only as text or a label, return the corresponding text item or label item itself.

## Pointing Rule
- For a pointing task, the only valid pointed-at target is the single object directly indicated by the fingertip in its forward direction; do not select an object to the left or right of that direction, or any nearby object merely because it is close to the hand or fingertip, and if the fingertip does not unambiguously indicate exactly one object, set `certainty` to `uncertain` and do not draw a box.

## Binary Certainty Rule
- Set `certainty` to `confident` only when the bbox is an unambiguous final requested target, its coordinates are reliable, the same physical or textual entity satisfies every required action, appearance, position, temporal, and association constraint, and there is no reasonable competing target.
- Set `certainty` to `uncertain` when the match is partial, weak, occluded, based only on proximity, has multiple plausible interpretations, has unreliable box boundaries, or does not visibly satisfy every task constraint.
- An uncertain candidate may retain its best plausible bbox and a concise ambiguity explanation in `desc`, but it will not be drawn. If no plausible target can be localized, return an empty targets list.
""".strip()
VISUAL_BOXED_DEBUG_STDOUT = os.environ.get("VISUAL_BOXED_DEBUG_STDOUT", "").strip() == "1"


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


def call_visual_boxed_model(messages, model_name=None, max_retries=None, enable_thinking=None):
    """Call visual boxed model, input: messages/config overrides -> output: response text and token counts"""
    import random
    import time

    if not model_name:
        model_name = VISUAL_BOXED_MODEL_NAMES[0] if VISUAL_BOXED_MODEL_NAMES else "gpt-5.5"
    if max_retries is None:
        max_retries = VISUAL_BOXED_MAX_RETRIES
    if enable_thinking is None:
        enable_thinking = VISUAL_BOXED_ENABLE_THINKING

    model_lower = str(model_name or "").strip().lower()
    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = _build_openai_client(VISUAL_BOXED_API_KEY, VISUAL_BOXED_API_BASE_URL)
            kwargs = {
                "model": str(model_name).strip(),
                "messages": messages,
                "max_tokens": VISUAL_BOXED_MAX_TOKENS,
                "seed": VISUAL_BOXED_SEED,
            }
            if model_lower.startswith("gpt") or "gpt-5" in model_lower:
                if VISUAL_BOXED_REASONING_EFFORT:
                    kwargs["reasoning_effort"] = VISUAL_BOXED_REASONING_EFFORT
                if VISUAL_BOXED_REASONING_EFFORT == "none":
                    kwargs["temperature"] = VISUAL_BOXED_TEMPERATURE
            else:
                kwargs["temperature"] = VISUAL_BOXED_TEMPERATURE
                if "qwen" in model_lower and enable_thinking is not None:
                    kwargs["extra_body"] = {"enable_thinking": enable_thinking}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            input_tokens, output_tokens = _extract_usage_tokens(getattr(completion, "usage", None))
            return content, input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[Visual Boxed Model Retry] {model_name} attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[Visual Boxed Model Error] {model_name} failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def print_config():
    """Print current visual boxed configuration."""
    print("Visual Boxed Configuration:")
    print(f"  Models: {', '.join(VISUAL_BOXED_MODEL_NAMES)}")
    print(f"  API Base URL: {VISUAL_BOXED_API_BASE_URL}")
    print(f"  Max Tokens: {VISUAL_BOXED_MAX_TOKENS}")
    if VISUAL_BOXED_REASONING_EFFORT == "none":
        print(f"  GPT Temperature: {VISUAL_BOXED_TEMPERATURE}")
    else:
        print(f"  GPT Temperature: omitted for {VISUAL_BOXED_REASONING_EFFORT or 'server-default'} reasoning")
    print(f"  Non-GPT Temperature: {VISUAL_BOXED_TEMPERATURE}")
    print(f"  Seed: {VISUAL_BOXED_SEED}")
    print(f"  Reasoning Effort: {VISUAL_BOXED_REASONING_EFFORT or 'default'}")
    print(f"  Qwen Thinking Mode: {VISUAL_BOXED_ENABLE_THINKING}")
    print(f"  Max Retries: {VISUAL_BOXED_MAX_RETRIES}")
    print(f"  Scenario Prefixes: {VISUAL_BOXED_SCENARIO_PREFIXES}")
    print(f"  Debug Stdout: {VISUAL_BOXED_DEBUG_STDOUT}")
    print(f"  System Prompt: {VISUAL_BOXED_SYSTEM_PROMPT}")


if __name__ == "__main__":
    print_config()
