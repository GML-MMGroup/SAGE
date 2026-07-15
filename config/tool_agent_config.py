"""
Tool Agent Configuration Interface
==================================

This file configures the model used by the tool-calling agent.
The tool agent performs plan_and_solve before executing database tools.

Environment Variables Required:
- TOOL_AGENT_API_KEY: API key for the tool agent model (falls back to SERVICE_API_KEY/API_KEY)
- TOOL_AGENT_API_BASE_URL: Base URL for the API endpoint (falls back to SERVICE_API_BASE_URL)
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
# TOOL AGENT MODEL CONFIGURATION
# ===============================================================================

# Model name for tool agent.
TOOL_AGENT_MODEL_NAME = "gpt-5.5"

# API Key for tool agent model.
TOOL_AGENT_API_KEY = os.environ.get("VAPI_API_KEY", SERVICE_API_KEY)

# Base URL for the API endpoint.
TOOL_AGENT_API_BASE_URL = os.environ.get("VAPI_API_URL", SERVICE_API_BASE_URL)

# Maximum tokens for tool agent responses.
TOOL_AGENT_MAX_TOKENS = 32768
# Temperature for tool agent.
TOOL_AGENT_TEMPERATURE = 0

# Whether to enable thinking mode if supported by the model.
TOOL_AGENT_ENABLE_THINKING = "true"

# Planner defaults to the current tool agent configuration.
TOOL_PLANNER_MODEL_NAME = "gpt-5.5"
TOOL_PLANNER_API_KEY = os.environ.get("VAPI_API_KEY")
TOOL_PLANNER_API_BASE_URL = os.environ.get("VAPI_API_URL")
TOOL_PLANNER_MAX_TOKENS = TOOL_AGENT_MAX_TOKENS
TOOL_PLANNER_TEMPERATURE = TOOL_AGENT_TEMPERATURE
TOOL_PLANNER_ENABLE_THINKING = True
TOOL_PLANNER_REASONING_EFFORT = os.environ.get("TOOL_PLANNER_REASONING_EFFORT", "medium")

# Executor defaults to the current tool agent configuration.
TOOL_EXECUTOR_MODEL_NAME = "gpt-5.5"
TOOL_EXECUTOR_API_KEY = os.environ.get("VAPI_API_KEY")
TOOL_EXECUTOR_API_BASE_URL = os.environ.get("VAPI_API_URL")
TOOL_EXECUTOR_MAX_TOKENS = TOOL_AGENT_MAX_TOKENS
TOOL_EXECUTOR_TEMPERATURE = TOOL_AGENT_TEMPERATURE
TOOL_EXECUTOR_ENABLE_THINKING = False

# Reporter defaults to the current tool agent configuration.
TOOL_REPORTER_MODEL_NAME ="gpt-5.5"
TOOL_REPORTER_API_KEY =  os.environ.get("VAPI_API_KEY")
TOOL_REPORTER_API_BASE_URL = os.environ.get("VAPI_API_URL")
TOOL_REPORTER_MAX_TOKENS =  TOOL_AGENT_MAX_TOKENS
TOOL_REPORTER_TEMPERATURE = TOOL_AGENT_TEMPERATURE
TOOL_REPORTER_ENABLE_THINKING = False


# ==============================================================================
# API CALL FUNCTION
# ===============================================================================

def _build_openai_client(api_key, api_base_url):
    """Build OpenAI client, input: key/base_url -> output: OpenAI client"""
    if api_base_url:
        return OpenAI(api_key=api_key, base_url=api_base_url)
    return OpenAI(api_key=api_key)


def _is_deepseek_config(model_name, api_base_url):
    """Check DeepSeek config, input: model/base url -> output: bool"""
    text = f"{model_name or ''} {api_base_url or ''}".lower()
    return "deepseek" in text


def _extract_response_text(response):
    """Extract Responses API text, input: response object -> output: text"""
    text = getattr(response, "output_text", "")
    if text:
        return text

    lines = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") == "output_text":
                lines.append(getattr(content, "text", ""))
    return "\n".join(line for line in lines if line)


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


def _call_configured_tool_model(messages, model_name, api_key, api_base_url, max_tokens, temperature, enable_thinking, log_name, max_retries=3):
    """Call a configured tool-side model, input: model config/messages -> output: response text and token counts"""
    import time
    import random

    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = _build_openai_client(api_key, api_base_url)
            kwargs = {
                "model": model_name.strip(),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if enable_thinking is not None:
                kwargs["extra_body"] = {"enable_thinking": enable_thinking}
            completion = client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            input_tokens, output_tokens = _extract_usage_tokens(getattr(completion, "usage", None))
            return content, input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[{log_name} Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[{log_name} Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_openai_responses_model(messages, model_name, api_key, api_base_url, max_tokens, reasoning_effort, log_name, max_retries=3):
    """Call OpenAI Responses API, input: model config/messages -> output: response text and token counts"""
    import time
    import random

    base_delay = 10
    for attempt in range(max_retries):
        try:
            client = _build_openai_client(api_key, api_base_url)
            kwargs = {
                "model": model_name.strip(),
                "input": messages,
                "max_output_tokens": max_tokens,
            }
            if reasoning_effort:
                kwargs["reasoning"] = {"effort": reasoning_effort}
            response = client.responses.create(**kwargs)
            input_tokens, output_tokens = _extract_usage_tokens(getattr(response, "usage", None))
            return _extract_response_text(response), input_tokens, output_tokens
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[{log_name} Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[{log_name} Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def call_tool_agent_model(messages, max_retries=3, enable_thinking=None):
    """Call the tool agent model, input: messages/retries/thinking -> output: response text and token counts"""
    if enable_thinking is None:
        enable_thinking = TOOL_AGENT_ENABLE_THINKING
    return _call_configured_tool_model(
        messages, TOOL_AGENT_MODEL_NAME, TOOL_AGENT_API_KEY, TOOL_AGENT_API_BASE_URL,
        TOOL_AGENT_MAX_TOKENS, TOOL_AGENT_TEMPERATURE, enable_thinking,
        "Tool Agent Model", max_retries
    )


def call_tool_planner_model(messages, max_retries=3, enable_thinking=None):
    """Call the tool planner model, input: messages/retries/thinking -> output: response text and token counts"""
    if enable_thinking is None:
        enable_thinking = TOOL_PLANNER_ENABLE_THINKING
    if _is_deepseek_config(TOOL_PLANNER_MODEL_NAME, TOOL_PLANNER_API_BASE_URL):
        return _call_configured_tool_model(
            messages, TOOL_PLANNER_MODEL_NAME, TOOL_PLANNER_API_KEY, TOOL_PLANNER_API_BASE_URL,
            TOOL_PLANNER_MAX_TOKENS, TOOL_PLANNER_TEMPERATURE, None,
            "Tool Planner Model", max_retries
        )
    reasoning_effort = TOOL_PLANNER_REASONING_EFFORT if enable_thinking else None
    return _call_openai_responses_model(
        messages, TOOL_PLANNER_MODEL_NAME, TOOL_PLANNER_API_KEY, TOOL_PLANNER_API_BASE_URL,
        TOOL_PLANNER_MAX_TOKENS, reasoning_effort,
        "Tool Planner Model", max_retries
    )


def call_tool_executor_model(messages, max_retries=3, enable_thinking=None):
    """Call the tool executor model, input: messages/retries/thinking -> output: response text and token counts"""
    if enable_thinking is None:
        enable_thinking = TOOL_EXECUTOR_ENABLE_THINKING
    return _call_configured_tool_model(
        messages, TOOL_EXECUTOR_MODEL_NAME, TOOL_EXECUTOR_API_KEY, TOOL_EXECUTOR_API_BASE_URL,
        TOOL_EXECUTOR_MAX_TOKENS, TOOL_EXECUTOR_TEMPERATURE, enable_thinking,
        "Tool Executor Model", max_retries
    )


def call_tool_reporter_model(messages, max_retries=3, enable_thinking=None):
    """Call the tool reporter model, input: messages/retries/thinking -> output: response text and token counts"""
    if enable_thinking is None:
        enable_thinking = TOOL_REPORTER_ENABLE_THINKING
    return _call_configured_tool_model(
        messages, TOOL_REPORTER_MODEL_NAME, TOOL_REPORTER_API_KEY, TOOL_REPORTER_API_BASE_URL,
        TOOL_REPORTER_MAX_TOKENS, TOOL_REPORTER_TEMPERATURE, enable_thinking,
        "Tool Reporter Model", max_retries
    )


# ==============================================================================
# CONFIGURATION VALIDATION
# ===============================================================================

def validate_config():
    """Validate the tool agent configuration, input: none -> output: (is_valid, error_message)"""
    if not TOOL_AGENT_API_KEY:
        return False, "TOOL_AGENT_API_KEY (or SERVICE_API_KEY/API_KEY) environment variable is not set."
    if not TOOL_AGENT_API_BASE_URL:
        return False, "TOOL_AGENT_API_BASE_URL is not configured."
    return True, None


def print_config():
    """Print current tool agent configuration."""
    print("Tool Agent Configuration:")
    print(f"  Model: {TOOL_AGENT_MODEL_NAME}")
    print(f"  API Base URL: {TOOL_AGENT_API_BASE_URL}")
    print(f"  Max Tokens: {TOOL_AGENT_MAX_TOKENS}")
    print(f"  Temperature: {TOOL_AGENT_TEMPERATURE}")
    print(f"  Thinking Mode: {TOOL_AGENT_ENABLE_THINKING}")
    print(f"  Reporter Model: {TOOL_REPORTER_MODEL_NAME}")
    print(f"  Reporter Max Tokens: {TOOL_REPORTER_MAX_TOKENS}")
    print(f"  Reporter Temperature: {TOOL_REPORTER_TEMPERATURE}")
    print(f"  Reporter Thinking Mode: {TOOL_REPORTER_ENABLE_THINKING}")
    print(f"  Planner Model: {TOOL_PLANNER_MODEL_NAME}")
    print(f"  Planner Reasoning Effort: {TOOL_PLANNER_REASONING_EFFORT}")


if __name__ == "__main__":
    is_valid, error_msg = validate_config()
    if is_valid:
        print_config()
    else:
        print(f"Configuration Error: {error_msg}")
