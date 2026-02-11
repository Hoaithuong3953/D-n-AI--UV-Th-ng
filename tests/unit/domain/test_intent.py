"""
test_intent.py

Unit tests for domain.intent (Intent, IntentDetectionMethod, IntentResult)
"""
import pytest
from pydantic import ValidationError

from domain import Intent, IntentDetectionMethod, IntentResult

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

class TestIntentResult:
    """Tests for IntentResult model"""

    def test_intent_result_valid_creation(self):
        """Valid IntentResult fields create instance successfully"""
        result = IntentResult(
            intent=Intent.CHAT,
            score=0.9,
            method=IntentDetectionMethod.KEYWORD,
        )

        assert result.intent is Intent.CHAT
        assert result.score == 0.9
        assert result.method is IntentDetectionMethod.KEYWORD

    def test_intent_result_score_boundaries(self):
        """Score accepts boundary values 0.0 and 1.0"""
        low = IntentResult(
            intent=Intent.CHAT,
            score=0.0,
            method=IntentDetectionMethod.LLM,
        )
        high = IntentResult(
            intent=Intent.ROADMAP,
            score=1.0,
            method=IntentDetectionMethod.KEYWORD,
        )

        assert low.score == 0.0
        assert high.score == 1.0

    @pytest.mark.parametrize("score", [-0.01, -1.0])
    def test_intent_result_score_below_zero_invalid(self, score: float):
        """Score below 0.0 should raise ValidationError"""
        with pytest.raises(ValidationError):
            IntentResult(
                intent=Intent.CHAT,
                score=score,
                method=IntentDetectionMethod.LLM,
            )

    @pytest.mark.parametrize("score", [1.01, 2.0])
    def test_intent_result_score_above_one_invalid(self, score: float):
        """Score above 1.0 should raise ValidationError"""
        with pytest.raises(ValidationError):
            IntentResult(
                intent=Intent.ROADMAP,
                score=score,
                method=IntentDetectionMethod.KEYWORD,
            )