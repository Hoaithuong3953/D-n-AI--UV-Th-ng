"""
AI layer for LearnPath chatbot
"""

from .llm_client import LLMClient
from .gemini_client import GeminiClient
from .prompts import SYSTEM_PROMPT, ROADMAP_PROMPT_TEMPLATE

__all__ = [
    "LLMClient",
    "GeminiClient",
    "SYSTEM_PROMPT",
    "ROADMAP_PROMPT_TEMPLATE"
]