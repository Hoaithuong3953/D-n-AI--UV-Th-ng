"""
test_intent.py

Unit tests for domain.intent (Intent, IntentDetectionMethod, IntentResult, ConfidenceLevel)
"""
import pytest
from pydantic import ValidationError

from domain import (
    ConfidenceLevel,
    Intent,
    IntentDetectionMethod,
    IntentResult,
)

class TestIntent:
    """Tests for Intent enum values"""

    def test_intent_values(self):
        """Intent enum should expose CHAT and ROADMAP with expected values"""
        assert Intent.CHAT.value == "chat"
        assert Intent.ROADMAP.value == "roadmap"

class TestIntentDetectionMethod:
    """Tests for IntentDetectionMethod enum values"""

    def test_intent_detection_method_values(self):
        """IntentDetectionMethod should expose KEYWORD and LLM with expected values"""
        assert IntentDetectionMethod.KEYWORD.value == "keyword"
        assert IntentDetectionMethod.LLM.value == "llm"

class TestConfidenceLevel:
    """Tests for ConfidenceLevel enum"""

    def test_confidence_level_values(self):
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"

class TestIntentResult:
    """Tests for IntentResult model"""

    def test_intent_result_valid_creation(self):
        """Valid IntentResult fields create instance successfully"""
        result = IntentResult(
            intent=Intent.CHAT,
            method=IntentDetectionMethod.KEYWORD,
            confidence=ConfidenceLevel.HIGH,
        )

        assert result.intent is Intent.CHAT
        assert result.method is IntentDetectionMethod.KEYWORD
        assert result.confidence is ConfidenceLevel.HIGH

    def test_intent_result_accepts_all_confidence_levels(self):
        """Each ConfidenceLevel is valid for IntentResult"""
        for level in ConfidenceLevel:
            r = IntentResult(
                intent=Intent.ROADMAP,
                method=IntentDetectionMethod.LLM,
                confidence=level,
            )
            assert r.confidence is level

    def test_intent_result_missing_confidence_invalid(self):
        """confidence is required"""
        with pytest.raises(ValidationError) as exc_info:
            IntentResult(
                intent=Intent.CHAT,
                method=IntentDetectionMethod.KEYWORD,
            )
        assert any(err["loc"] == ("confidence",) for err in exc_info.value.errors())