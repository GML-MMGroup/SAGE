"""
Unified API module - integrates all LLM API calls

This module now uses configuration from config/ directory:
- config/user_agent_config.py: User simulation model configuration
- config/service_agent_config.py: Service agent model configuration
"""
import os
import sys
import time
import random
import requests
from openai import OpenAI

# Add project root to path for config imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import configuration modules
from config.user_agent_config import (
    call_user_model as user_call_llm
)
from config.service_agent_config import (
    SERVICE_MODEL_NAME,
    call_service_model as service_call_llm
)

# Import model APIs
from .zhipu import ZhipuAPI
from .qwen import QwenAPI
from .mimo import MimoAPI
from .kimi import KimiAPI
from .doubao import DoubaoAPI
from .qwen3_5 import Qwen3_5_API
from .skywork import SkyworkAPI
from .deepseek import DeepSeekAPI
from .gpt55 import GPT55API

def call_llm(messages, agent_type="service", service_model_name="qwen3-vl-225b", enable_thinking=False):
    """
    Unified LLM call interface

    Args:
        messages: Message list
        agent_type: "service" or "user"
        service_model_name: Service model name
        enable_thinking: Whether to enable thinking mode

    Returns:
        (response_text, input_tokens, output_tokens)
    """
    MAX_RETRIES = 3
    BASE_DELAY = 10

    if agent_type == "user":
        # User model only uses Qwen3.5-397B-A17B
        return _call_user_model(messages, MAX_RETRIES, BASE_DELAY, enable_thinking)
    else:
        # Service model
        return _call_service_model(messages, service_model_name, MAX_RETRIES, BASE_DELAY, enable_thinking)


def _call_user_model(messages, max_retries, base_delay, enable_thinking):
    """Call user model - uses config from user_agent_config.py"""
    # Use the config from user_agent_config.py
    return user_call_llm(messages, max_retries, enable_thinking)


def _call_service_model(messages, model_name, max_retries, base_delay, enable_thinking):
    """Call service model - uses config from service_agent_config.py by default"""
    # If using the default configured model, use service_agent_config
    if model_name == SERVICE_MODEL_NAME or model_name == "Qwen3.5-397B-A17B":
        return service_call_llm(messages, max_retries, enable_thinking)

    # Select API based on model type
    if model_name in ["zhipu", "glm-5v-turbo"]:
        return _call_zhipu_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["qwen", "qwen3.6-plus"]:
        return _call_qwen_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["mimo", "mimo-v2-omni"]:
        return _call_mimo_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["kimi", "kimi-k2.5","Pro/moonshotai/Kimi-K2.6","Pro/moonshotai/Kimi-K2.5"]:
        return _call_kimi_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["doubao", "doubao-seed-2-0-pro-260215"]:
        return _call_doubao_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["gemini-3.1-pro-preview","gemini-3.5-flash","gemini-3.1-flash-lite"]:
        return _call_gemini_api(messages, max_retries, base_delay)
    elif model_name == "qwen3-vl-225b":
        return _call_qwen3_vl_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name in ["Qwen3.5-397B-A17B","Qwen/Qwen3.5-397B-A17B"]:
        return _call_qwen3_5_api(messages, max_retries, base_delay, enable_thinking)
    elif model_name == "deepseek-v4-pro":
        return _call_deepseek_api(messages, model_name, thinking_type="disabled", reasoning_effort=None)
    elif model_name == "deepseek-v4-flash":
        return _call_deepseek_api(messages, model_name)
    elif str(model_name).strip().lower() in ["gpt-5.5", "gpt5.5"]:
        return _call_gpt55_api(messages, model_name, enable_thinking)
    elif model_name in ["skywork-ai/skyclaw-v1", "skyclaw-v1"]:
        return _call_skywork_api(messages, max_retries, base_delay, enable_thinking)
    else:
        # Other models use generic API
        return _call_generic_api(messages, model_name, max_retries, base_delay, enable_thinking)


def _call_zhipu_api(messages, max_retries, base_delay, enable_thinking):
    """Call Zhipu API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = ZhipuAPI()
            result, input_tokens, output_tokens = api.chat(messages, thinking_enabled=enable_thinking)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(zhipu) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(zhipu) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_qwen_api(messages, max_retries, base_delay, enable_thinking):
    """Call Qwen API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = QwenAPI()
            result, input_tokens, output_tokens = api.chat(messages, enable_thinking=enable_thinking)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(qwen) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(qwen) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0

def _call_qwen3_5_api(messages, max_retries, base_delay, enable_thinking):
    """Call Qwen API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = Qwen3_5_API()
            result, input_tokens, output_tokens = api.chat(messages, enable_thinking=enable_thinking)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(qwen) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(qwen) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0

def _call_skywork_api(messages, max_retries, base_delay, enable_thinking):
    """Call Skywork SkyClaw API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = SkyworkAPI()
            result, input_tokens, output_tokens = api.chat(messages, enable_thinking=enable_thinking)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(skywork) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(skywork) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0

def _call_deepseek_api(messages, model_name, thinking_type="enabled", reasoning_effort="high"):
    """Call DeepSeek API"""
    api = DeepSeekAPI()
    return api.chat(
        messages,
        model=model_name,
        thinking_type=thinking_type,
        reasoning_effort=reasoning_effort
    )

def _call_gpt55_api(messages, model_name, enable_thinking=False):
    """Call GPT-5.5 API"""
    api = GPT55API()
    return api.chat(messages, model=model_name, enable_thinking=enable_thinking)

def _call_mimo_api(messages, max_retries, base_delay, enable_thinking):
    """Call Mimo API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = MimoAPI()
            result, input_tokens, output_tokens = api.chat(messages)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(mimo) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(mimo) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_kimi_api(messages, max_retries, base_delay, enable_thinking):
    """Call Kimi API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = KimiAPI()
            result, input_tokens, output_tokens = api.chat(messages)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(kimi) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(kimi) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_doubao_api(messages, max_retries, base_delay, enable_thinking):
    """Call Doubao API"""
    last_error = None
    for attempt in range(max_retries):
        try:
            api = DoubaoAPI()
            result, input_tokens, output_tokens = api.chat(messages, enable_thinking=enable_thinking)
            return result, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(doubao) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(doubao) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_gemini_api(messages, max_retries, base_delay):
    """Call Gemini API"""
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not GEMINI_API_KEY:
        raise ValueError("API_KEY environment variable not set for Gemini API.")
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url=os.environ.get("GEMINI_API_URL", "https://api.example.com/v1")
    )
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gemini-3.5-flash",
                messages=messages,
                stream=False
            )
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0

            # Safety check: handle content being None
            content = response.choices[0].message.content
            if content is None:
                print(f"[LLM Warning] Service(gemini) returned None content, attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                    print(f"Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Gemini API returned empty content", input_tokens, output_tokens

            return content, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(gemini) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(gemini) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


def _call_qwen3_vl_api(messages, max_retries, base_delay, enable_thinking):
    """Call Qwen3-VL-225B API"""
    API_KEY = os.environ.get("API_KEY", "")
    if not API_KEY:
        raise ValueError("API_KEY environment variable not set for Qwen3-VL API.")
    BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.example.com/v1")
    MODEL_NAME = "city_guide_vl_235b"

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=32768,
                temperature=0.7,
                extra_body={"enable_thinking": enable_thinking}
            )
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage') and response.usage:
                input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
            return response.choices[0].message.content, input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service(Std) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service(Std) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


# Model config mapping
MODEL_CONFIGS = {
    "qwen3_30b_a3b": {"name": "Qwen3-30B-A3B-Instruct-2507", "token": os.environ.get("API_KEY", "")},
    "qwen3_235b_a22b": {"name": "Qwen3-235B-A22B-Instruct-2507-Pro", "token": os.environ.get("API_KEY", "")},
    "Qwen3.5-397B-A17B": {"name": "Qwen3.5-397B-A17B", "token": os.environ.get("API_KEY", "")},
    "DeepSeek-R1": {"name": "DeepSeek-R1", "token": os.environ.get("API_KEY", "")},
    "DeepSeek-V3": {"name": "DeepSeek-V3", "token": os.environ.get("API_KEY", "")},
    "DeepSeek-V3.2": {"name": "DeepSeek-V3.2", "token": os.environ.get("API_KEY", "")},
    "qwen3-vl-225b": {"name": "Qwen3-VL-235B-A22B-Instruct", "token": os.environ.get("API_KEY", "")},
    "glm-4.5v": {"name": "GLM-4.5V", "token": os.environ.get("API_KEY", "")},
}


def _call_generic_api(messages, model_name, max_retries, base_delay, enable_thinking):
    """Call Generic API (other models)"""
    url = os.environ.get("LLM_API_URL", "https://api.gpt.ge/v1")
    config = MODEL_CONFIGS.get(model_name, {})

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.get('token', '')}"
    }

    if "glm-4" in model_name:
        payload = {
            "model": config.get("name", model_name),
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}
        }
    else:
        payload = {
            "model": config.get("name", model_name),
            "messages": messages,
            "stream": False
        }

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            usage = result.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            return result['choices'][0]['message']['content'], input_tokens, output_tokens
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (base_delay * (2 ** attempt)) + random.uniform(0, 1)
                print(f"[LLM Retry] Service({model_name}) attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[LLM Error] Service({model_name}) failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0
