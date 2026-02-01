"""
intent_detector.py

Intent detection for chat routing; uses LLM to classify user message intent

Key features:
- IntentDetector: classify roadmap vs chat intent via LLM
- is_roadmap_intent: returns True when user asks for a learning roadmap
"""
from typing import Optional

from ai import LLMClient
from domain import Intent
from utils import logger

INTENT_PROMPT = """
Phân loại intent của người dùng. Chỉ trả về MỘT trong các giá trị:

- ROADMAP: người dùng muốn tạo lộ trình học, kế hoạch học, learning path
- CHAT: trò chuyện thông thường

User: {text}

Trả về duy nhất 1 từ:
"""

class IntentDetector:
    """
    Detect user intent (roadmap vs chat) using LLM classification

    Responsibilities:
    - Classify user message as ROADMAP or CHAT via LLM
    - Return False on empty input or LLM failure (fail-safe to chat)
    """
    def __init__(self, llm_client: LLMClient):
        """Initialize with LLM client used for classification"""
        self.llm = llm_client

    def is_roadmap_intent(self, text: str) -> bool:
        """
        Classify whether the user message expresses intent to create or get a learning roadmap

        Args:
            text: Raw user message (empty/whitespace is treated as non-roadmap)

        Returns:
            True if intent is ROADMAP; False otherwise (including on LLM failure)
        """
        text = (text or "").strip()
        if not text:
            return False
        
        try:
            prompt = INTENT_PROMPT.format(text=text)
            response = self.llm.generate_text(prompt)
            return self._parse_roadmap_intent(response)
        except Exception as e:
            logger.warning(f"Intent detection failed, treating as non-roadmap: {e}")
            return False
    
    def detect(self, text: str) -> Intent:
        """
        Classify user message intent (CHAT or ROADMAP).

        Args:
            text: Raw user message.

        Returns:
            Intent.ROADMAP if user asks for a learning roadmap; Intent.CHAT otherwise
        """
        return Intent.ROADMAP if self.is_roadmap_intent(text) else Intent.CHAT

    @staticmethod
    def _parse_roadmap_intent(response: Optional[str]) -> bool:
        """Return True if LLM response indicates ROADMAP intent"""
        if not response:
            return False
        normalized = response.strip().upper()
        return "ROADMAP" in normalized