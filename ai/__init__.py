"""
AI layer for LearnPath chatbot

Key features:
- LLMClient: protocol for LLM implementations
- GeminiClient: Gemini API client (generate_text, stream_chat)
- SYSTEM_PROMPT, ROADMAP_PROMPT_TEMPLATE: prompts for chat and roadmap generation
- PROFILE_EXTRACT_PROMPT: prompt for extracting user profile from recent chat
"""

from .llm_client import LLMClient
from .gemini_client import GeminiClient
from .prompts import SYSTEM_PROMPT, ROADMAP_PROMPT_TEMPLATE, PROFILE_EXTRACT_PROMPT

__all__ = [
    "LLMClient",
    "GeminiClient",
    "SYSTEM_PROMPT",
    "ROADMAP_PROMPT_TEMPLATE",
    "PROFILE_EXTRACT_PROMPT",
]