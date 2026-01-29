"""
Protocol definition for LLM client implementations (Gemini, OpenAI, etc.)
"""

from typing import Protocol, List, Generator
from domain import ChatMessage

class LLMClient(Protocol):
    """
    Interface for LLM Clients (Gemini, OpenAI,...)
    """

    def generate_text(self, prompt: str) -> str:
        """
        Generate simple text from a prompt
        """
        ...

    def stream_chat(self, history: List[ChatMessage], new_message: str) -> Generator[str, None, None]:
        """
        Generate a streaming response for chat
        """
        ...