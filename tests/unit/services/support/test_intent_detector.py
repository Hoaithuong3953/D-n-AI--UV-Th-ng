"""
Unit tests for services.support.intent_detector (keyword + LLM fallback, confidence levels)
"""
import pytest
from unittest.mock import Mock

from domain import Intent, IntentDetectionMethod, ConfidenceLevel
from services.support.intent_detector import IntentDetector, ROADMAP_KEYWORDS

@pytest.fixture
def mock_llm():
    return Mock()

@pytest.fixture
def detector(mock_llm):
    return IntentDetector(mock_llm)

class TestEmptyInput:
    def test_empty_string_returns_chat_low_confidence(self, detector):
        result = detector.detect("")
        assert result.intent == Intent.CHAT
        assert result.confidence == ConfidenceLevel.LOW
        assert result.method == IntentDetectionMethod.LLM

    def test_whitespace_only_returns_chat_low_confidence(self, detector):
        result = detector.detect("    ")
        assert result.intent == Intent.CHAT
        assert result.confidence == ConfidenceLevel.LOW

class TestKeywordDetection:
    def test_lo_trinh_keyword(self, detector):
        result = detector.detect("tôi muốn tạo lộ trình Python")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_roadmap_keyword(self, detector):
        result = detector.detect("tôi muốn tạo roadmap học Python trong 3 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_ke_hoach_hoc_keyword(self, detector):
        result = detector.detect("cho tôi kế hoạch học Python")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_learning_path_keyword(self, detector):
        result = detector.detect("tôi cần một learning path cho web dev")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_tao_lo_trinh_keyword(self, detector):
        result = detector.detect("tạo lộ trình học 4 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_muon_hoc_keyword(self, detector):
        result = detector.detect("tôi muốn học Python")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_bat_dau_hoc_keyword(self, detector):
        result = detector.detect("bắt đầu học React")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_hoc_trong_keyword(self, detector):
        result = detector.detect("tôi muốn học trong 3 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_roadmap_keywords_list_non_empty(self):
        assert isinstance(ROADMAP_KEYWORDS, list)
        assert len(ROADMAP_KEYWORDS) > 0

class TestLLMFallback:
    def test_llm_short_text_low_confidence(self, detector, mock_llm):
        mock_llm.generate_text.return_value = "CHAT"

        result = detector.detect("Python")
        assert result.intent == Intent.CHAT
        assert result.confidence == ConfidenceLevel.LOW
        assert result.method == IntentDetectionMethod.LLM

    def test_llm_medium_text_medium_confidence(self, detector, mock_llm):
        mock_llm.generate_text.return_value = "CHAT"

        result = detector.detect("Giải thích về OOP trong Python")
        assert result.intent == Intent.CHAT
        assert result.confidence == ConfidenceLevel.MEDIUM
        assert result.method == IntentDetectionMethod.LLM

class TestDecisionConfidence:
    def test_confidence_low_for_short_text(self, detector, mock_llm):
        mock_llm.generate_text.return_value = "CHAT"
        result = detector.detect("Python")
        assert result.confidence == ConfidenceLevel.LOW
        assert result.method == IntentDetectionMethod.LLM

    def test_confidence_medium_for_medium_text(self, detector, mock_llm):
        mock_llm.generate_text.return_value = "CHAT"
        result = detector.detect("Giải thích cho tôi về Python")
        assert result.confidence == ConfidenceLevel.MEDIUM
        assert result.method == IntentDetectionMethod.LLM

    def test_confidence_high_for_keyword_match(self, detector):
        result = detector.detect("Tạo lộ trình Python")
        assert result.confidence == ConfidenceLevel.HIGH
        assert result.method == IntentDetectionMethod.KEYWORD
