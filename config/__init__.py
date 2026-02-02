"""
Configuration module for LearnPath chatbot

Key features:
- settings: singleton Settings instance (from .env)
- Settings: Pydantic settings class for GEMINI_* and LOG_*
- messages: user-facing message keys and provider
- Constants: MAX_INPUT_LENGTH, DEFAULT_CONTEXT_MESSAGES
"""

from .settings import settings, Settings
from .messages import (
    MessageKey,
    MessageProvider,
    DefaultMessageProvider,
    default_messages,
)
from .constants import (
    MAX_INPUT_LENGTH,
    DEFAULT_CONTEXT_MESSAGES,
)

__all__ = [
    "settings",
    "Settings",
    "MessageKey",
    "MessageProvider",
    "DefaultMessageProvider",
    "default_messages",
    "MAX_INPUT_LENGTH",
    "DEFAULT_CONTEXT_MESSAGES",
]
