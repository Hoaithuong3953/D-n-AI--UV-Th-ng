"""
intent_detector.py

Two tier intent detection: rule-based keywords first, LLM fallback

Key features:
- detect(text) -> IntentResult with intent, decision score and detection method
- Rule-based: Fast keyword matching for ROADMAP intent
- LLM fallback: When no keyword match
"""
from ai import LLMClient
from domain import Intent, IntentResult, IntentDetectionMethod, ConfidenceLevel
from utils import logger

ROADMAP_KEYWORDS = [
    "lộ trình",
    "roadmap",
    "kế hoạch học",
    "learning path",
    "tạo lộ trình",
    "muốn học",
    "học trong",
    "bắt đầu học",
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
    Detect user intent (CHAT, ROADMAP) with decision score

    Two-tier detection strategy:
    1. Keyword matching: Fast, high confidence
    2. LLM fallback: Slower, lower confidence

    Confidence levels are heuristic utilities for monitoring
    Levels indicate detection method quality, not true probability

    Attributes:
        Keyword match: HIGH confidence (strong signal)
        LLM on short text: LOW confidence (weak signal)
        LLM on medium text: MEDIUM confidence (medium signal)
        Empty input: LOW confidence (default to CHAT)
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize IntentDetector with LLM client

        Args:
            llm_client: LLM client for fallback intent detection
        """
        self.llm = llm_client

    def detect(self, text: str) -> IntentResult:
        """
        Detect user intent with two-tier strategy (keyword -> LLM)

        Args:
            text: Raw user message

        Returns:
            intent, method and confidence level for IntentResult
        """
        text = text.strip()
        if not text or "".strip():
            logger.debug("intent detect: empty -> CHAT (confidence=LOW)")
            return IntentResult(
                intent=Intent.CHAT,
                method=IntentDetectionMethod.LLM,
                confidence=ConfidenceLevel.LOW,
            )
        
        text_lower = text.lower()

        # Rule 1: Keyword match
        for keyword in ROADMAP_KEYWORDS:
            if keyword in text_lower:
                logger.info(f"intent detect: keyword '{keyword}' -> ROADMAP (confidence=HIGH)")
                return IntentResult(
                    intent=Intent.ROADMAP,
                    method=IntentDetectionMethod.KEYWORD,
                    confidence=ConfidenceLevel.HIGH,
                )
        
        # Rule 2: LLM fallback
        llm_intent = self._detect_by_llm(text)

        # Estimate score based on text length (heuristic)
        word_count = len(text.split())
        if word_count < 5:
            confidence = ConfidenceLevel.LOW
        else:
            confidence = ConfidenceLevel.MEDIUM

        logger.info(
            f"intent detect: llm -> {llm_intent.value} "
            f"confidence={confidence.name}, words={word_count}"
        )

        return IntentResult(
            intent=llm_intent,
            method=IntentDetectionMethod.LLM,
            confidence=confidence
        )
    
    def _detect_by_llm(self, text: str) -> Intent:
        """
        LLM fallback for intent detection

        Args:
            text: User message text

        Returns:
            Intent.ROADMAP if response contains "ROADMAP"
            Intent.CHAT otherwise (default, safe fallback)
        """
        try:
            prompt = INTENT_PROMPT.format(text=text)
            response = self.llm.generate_text(prompt)
            
            if response and "ROADMAP" in response.strip().upper():
                return Intent.ROADMAP
            
        except Exception as e:
            logger.warning(f"Intent detection LLM fallback failed: {e}")
        return Intent.CHAT