"""
Configuration module for LearnPath chatbot

Key features:
- settings: singleton Settings instance (from .env)
- Settings: Pydantic settings class for GEMINI_* and LOG_*
- messages: user-facing message keys and provider
"""

from .settings import settings, Settings
from .messages import (
    MessageKey,
    MessageProvider,
    DefaultMessageProvider,
    default_messages,
)

__all__ = [
    "settings",
    "Settings",
    "MessageKey",
    "MessageProvider",
    "DefaultMessageProvider",
    "default_messages",
]
