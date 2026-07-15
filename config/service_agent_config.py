"""
Service Agent Configuration Interface
======================================

This file configures the service agent model for the competition.
Participants should modify this file to use their preferred model for the service agent.

Default: Qwen3.5-397B-A17B (supports both API and local deployment)

Environment Variables Required:
- SERVICE_API_KEY: API key for the service model (or use API_KEY)
- SERVICE_API_BASE_URL: Base URL for the API endpoint (optional)
- VIDEO_MODE: "local" for local videos folder, "url" for public URLs (default: "local")

Environment Variables for Local Deployment:
- SERVICE_MODEL_NAME: Model name for local deployment
- SERVICE_API_BASE_URL: Your local model API endpoint

Environment Variables for Video URLs:
- VIDEO_URL_MAPPING: JSON string mapping video filenames to public URLs
                     Example: '{"retail1.mp4": "https://example.com/retail1.mp4"}'
"""
from openai import OpenAI
import os
import json

# ==============================================================================
# SERVICE AGENT MODEL CONFIGURATION
# ==============================================================================

# Model name for service agent
# Default: Qwen3.5-397B-A17B
# You can change this to any model you prefer
SERVICE_MODEL_NAME = os.environ.get("SERVICE_MODEL_NAME", "Qwen/Qwen3.5-397B-A17B") 

# API Key for service model
# This should be set as an environment variable: export SERVICE_API_KEY="your-api-key"
# If not set, will fall back to API_KEY
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", os.environ.get("API_KEY", "")) or os.environ.get("QWEN_API_KEY")

# Base URL for the API endpoint
# Default: https://api.example.com/v1/chat/completions
# For local deployment, set this to your local API endpoint
SERVICE_API_BASE_URL = os.environ.get("SERVICE_API_BASE_URL", os.environ.get("LLM_API_BASE_URL", "https://api.siliconflow.cn/v1")) 

# Maximum tokens for service model responses
SERVICE_MAX_TOKENS = 32768

# Temperature for service model (0.0 - 2.0)
SERVICE_TEMPERATURE = 0

# Whether to enable thinking mode (if supported by the model)
SERVICE_ENABLE_THINKING = False

# Supervisor agent system role prompt
SUPERVISOR_AGENT_SYSTEM_PROMPT = (
    "You are a service supervisor that routes user requests to the Visual Agent, Tool Agent, or ask_user: use the Visual Agent when the request needs identifying an unknown visible item, use the Tool Agent when the item is already known and the request needs tool/database actions, use ask_user when a required user_id is missing or when a task explicitly needs date-based reasoning and required current_date is missing, and if any required item is still visually described with an unknown exact name, use the Visual Agent first. Tax-related requests are not date-based and do not require current_date. Return only JSON with exactly two fields: agent_name and task."
)
# ==============================================================================
# VIDEO CONFIGURATION
# ==============================================================================

# Video storage mode: "local" or "url"
# "local" - Use videos from the local videos/ folder
# "url" - Use public URLs provided in VIDEO_URL_MAPPING
VIDEO_MODE = os.environ.get("VIDEO_MODE", "local")

# Base path for local videos
# Default: ./videos (relative to project root)
VIDEO_LOCAL_PATH = os.environ.get("VIDEO_LOCAL_PATH", "./videos")

# Video URL mapping (for url mode)
# This is a dictionary mapping video filenames to their public URLs
# Example: {"retail1.mp4": "https://example.com/videos/retail1.mp4"}
def _load_video_url_mapping():
    """Load video URL mapping from environment variable."""
    video_mapping_env = os.environ.get("VIDEO_URL_MAPPING", "")
    if video_mapping_env:
        try:
            return json.loads(video_mapping_env)
        except json.JSONDecodeError:
            print("[Warning] Failed to parse VIDEO_URL_MAPPING, using empty mapping")
    return {}

VIDEO_URL_MAPPING = _load_video_url_mapping()

# Default video URL mapping (fallback for common scenarios)
# Participants should update these with their actual video URLs
DEFAULT_VIDEO_URL_MAPPING = {
    # Retail scenarios
    "retail1.mp4": "https://example.com/videos/retail1.mp4",
    "retail2.mp4": "https://example.com/videos/retail2.mp4",
    "retail3.mp4": "https://example.com/videos/retail3.mp4",
    "retail4.mp4": "https://example.com/videos/retail4.mp4",
    "retail5.mp4": "https://example.com/videos/retail5.mp4",
    "retail6.mp4": "https://example.com/videos/retail6.mp4",
    "retail7.mp4": "https://example.com/videos/retail7.mp4",
    "retail8.mp4": "https://example.com/videos/retail8.mp4",
    "retail9.mp4": "https://example.com/videos/retail9.mp4",
    "retail10.mp4": "https://example.com/videos/retail10.mp4",

    # Kitchen scenarios
    "kitchen1.mp4": "https://example.com/videos/kitchen1.mp4",
    "deep_fried.mp4": "https://example.com/videos/deep_fried.mp4",
    "Green%20Pepper%20Chicken.mp4": "https://example.com/videos/Green_Pepper_Chicken.mp4",
    "dumplings.mp4": "https://example.com/videos/dumplings.mp4",

    # Restaurant scenarios
    "restaurant1.mp4": "https://example.com/videos/restaurant1.mp4",
    "restaurant2.mp4": "https://example.com/videos/restaurant2.mp4",
    "restaurant3.mp4": "https://example.com/videos/restaurant3.mp4",
    "restaurant4.mp4": "https://example.com/videos/restaurant4.mp4",
    "restaurant5.mp4": "https://example.com/videos/restaurant5.mp4",

    # Order scenarios
    "afrikana_annie_1.mp4": "https://example.com/videos/afrikana_annie_1.mp4",
    "annie_butcher_1.mp4": "https://example.com/videos/annie_butcher_1.mp4",
    "annie_meraki_1.mp4": "https://example.com/videos/annie_meraki_1.mp4",
    "annie_pauhana_1.mp4": "https://example.com/videos/annie_pauhana_1.mp4",
    "sunny_annie_1.mp4": "https://example.com/videos/sunny_annie_1.mp4",
    "afrikana_greek.mp4": "https://example.com/videos/afrikana_greek.mp4",
    "butcher_greek.mp4": "https://example.com/videos/butcher_greek.mp4",
    "greek_annie_1.mp4": "https://example.com/videos/greek_annie_1.mp4",
    "meraki_greek.mp4": "https://example.com/videos/meraki_greek.mp4",
    "pauhana_greek.mp4": "https://example.com/videos/pauhana_greek.mp4",
    "sunny_greek.mp4": "https://example.com/videos/sunny_greek.mp4",
}


def get_video_path(video_filename):
    """
    Get the video path or URL based on the configured VIDEO_MODE.

    Args:
        video_filename: The video filename (e.g., "retail1.mp4")

    Returns:
        str: The full path or URL to the video
    """
    if VIDEO_MODE == "url":
        # Use public URLs
        return VIDEO_URL_MAPPING.get(video_filename, DEFAULT_VIDEO_URL_MAPPING.get(video_filename, video_filename))
    else:
        # Use local videos folder
        return os.path.join(VIDEO_LOCAL_PATH, video_filename)


# ==============================================================================
# API CALL FUNCTION
# ==============================================================================

def call_service_model(messages, max_retries=3, enable_thinking=None):
    """
    Call the service model with the given messages.

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
        enable_thinking = SERVICE_ENABLE_THINKING

    BASE_DELAY = 10
    last_error = None

    for attempt in range(max_retries):
        try:
            client = OpenAI(
                api_key=SERVICE_API_KEY,
                base_url=SERVICE_API_BASE_URL
            )
            kwargs = {
                "model": SERVICE_MODEL_NAME,
                "messages": messages,
                "temperature":SERVICE_TEMPERATURE,
                "extra_body": {"enable_thinking": enable_thinking}
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
                print(f"[Service Model Retry] Attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)
            else:
                print(f"[Service Model Error] Failed after {max_retries} attempts: {str(e)}")
                return f"Error: {str(e)}", 0, 0


# ==============================================================================
# CONFIGURATION VALIDATION
# ==============================================================================

def validate_config():
    """
    Validate the service agent configuration.

    Returns:
        tuple: (is_valid, error_message)
    """
    if not SERVICE_API_KEY:
        return False, "SERVICE_API_KEY (or API_KEY) environment variable is not set. Please set it before running."

    if not SERVICE_API_BASE_URL:
        return False, "SERVICE_API_BASE_URL is not configured."

    if VIDEO_MODE not in ["local", "url"]:
        return False, f"Invalid VIDEO_MODE: {VIDEO_MODE}. Must be 'local' or 'url'."

    if VIDEO_MODE == "url" and not VIDEO_URL_MAPPING:
        return False, "VIDEO_MODE is 'url' but VIDEO_URL_MAPPING is not set."

    return True, None


def print_config():
    """Print current service agent configuration."""
    print(f"Service Agent Configuration:")
    print(f"  Model: {SERVICE_MODEL_NAME}")
    print(f"  API Base URL: {SERVICE_API_BASE_URL}")
    print(f"  Max Tokens: {SERVICE_MAX_TOKENS}")
    print(f"  Temperature: {SERVICE_TEMPERATURE}")
    print(f"  Thinking Mode: {SERVICE_ENABLE_THINKING}")
    print(f"  Video Mode: {VIDEO_MODE}")
    if VIDEO_MODE == "local":
        print(f"  Video Local Path: {VIDEO_LOCAL_PATH}")
    else:
        print(f"  Video URL Mapping: {len(VIDEO_URL_MAPPING)} entries")


if __name__ == "__main__":
    # Test configuration
    is_valid, error_msg = validate_config()
    if is_valid:
        print_config()
    else:
        print(f"Configuration Error: {error_msg}")
