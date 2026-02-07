"""
intent_detector.py

Two tier intent detection: rule (keywords) first, LLM fallback

Key features:
- IntentDetector: classify roadmap vs chat intent
- ROADMAP_KEYWORDS for rule-based, INTENT PROMPT for LLM fallback
"""
from ai import LLMClient
from domain import Intent
from utils import logger

ROADMAP_KEYWORDS = [
    "lộ trình",
    "roadmap",
    "kế hoạch học",
    "learning path",
    "tạo lộ trình",
]

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
        Classify intent: keyword match -> ROADMAP; else LLM fallback -> CHAT or ROADMAP

        Args:
            text: Raw user message

        Returns:
            Intent.CHAT or Intent.ROADMAP; empty text returns Intent.CHAT
        """
        text = (text or "").strip()
        if not text:
            return Intent.CHAT
        lower = text.lower()
        if any(k in lower for k in ROADMAP_KEYWORDS):
            logger.info("intent detect: keyword match -> ROADMAP")
            return Intent.ROADMAP
        
        out = self._detect_by_llm(text)
        logger.info(f"intent detect: llm fallback -> {out.value}")
        return out
    
    def _detect_by_llm(self, text: str) -> Intent:
        """Fallback: call LLM with short timeout; parse response for ROADMAP, default CHAT"""
        try:
            prompt = INTENT_PROMPT.format(text=text)
            response = self.llm.generate_text(prompt)
            if response and "ROADMAP" in response.strip().upper():
                return Intent.ROADMAP
            
        except Exception as e:
            logger.warning(f"Intent detection LLM fallback failed: {e}")
        return Intent.CHAT