"""
Configuration module for EgoBench Competition
==============================================

This module contains the configuration interfaces that participants
need to modify for the competition.

Files:
- user_agent_config.py: Configuration for the simulated user model
- service_agent_config.py: Configuration for the service agent model
"""

from .user_agent_config import (
    USER_MODEL_NAME,
    USER_API_KEY,
    USER_API_BASE_URL,
    USER_MAX_TOKENS,
    USER_TEMPERATURE,
    USER_ENABLE_THINKING,
    USER_CORRECTOR_MODEL_NAME,
    USER_CORRECTOR_API_KEY,
    USER_CORRECTOR_API_BASE_URL,
    USER_CORRECTOR_MAX_TOKENS,
    USER_CORRECTOR_TEMPERATURE,
    USER_CORRECTOR_ENABLE_THINKING,
    call_user_model,
    call_user_corrector_model,
    validate_config as validate_user_config
)

from .service_agent_config import (
    SERVICE_MODEL_NAME,
    SERVICE_API_KEY,
    SERVICE_API_BASE_URL,
    SERVICE_MAX_TOKENS,
    SERVICE_TEMPERATURE,
    SERVICE_ENABLE_THINKING,
    VIDEO_MODE,
    VIDEO_LOCAL_PATH,
    VIDEO_URL_MAPPING,
    get_video_path,
    call_service_model,
    validate_config as validate_service_config,
    print_config
)

__all__ = [
    # User agent config
    "USER_MODEL_NAME",
    "USER_API_KEY",
    "USER_API_BASE_URL",
    "USER_MAX_TOKENS",
    "USER_TEMPERATURE",
    "USER_ENABLE_THINKING",
    "USER_CORRECTOR_MODEL_NAME",
    "USER_CORRECTOR_API_KEY",
    "USER_CORRECTOR_API_BASE_URL",
    "USER_CORRECTOR_MAX_TOKENS",
    "USER_CORRECTOR_TEMPERATURE",
    "USER_CORRECTOR_ENABLE_THINKING",
    "call_user_model",
    "call_user_corrector_model",
    "validate_user_config",
    # Service agent config
    "SERVICE_MODEL_NAME",
    "SERVICE_API_KEY",
    "SERVICE_API_BASE_URL",
    "SERVICE_MAX_TOKENS",
    "SERVICE_TEMPERATURE",
    "SERVICE_ENABLE_THINKING",
    "VIDEO_MODE",
    "VIDEO_LOCAL_PATH",
    "VIDEO_URL_MAPPING",
    "get_video_path",
    "call_service_model",
    "validate_service_config",
    "print_config",
]
