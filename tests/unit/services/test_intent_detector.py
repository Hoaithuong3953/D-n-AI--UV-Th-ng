"""
Unit tests for intent_detectot module with decision scores
"""
import pytest
from unittest.mock import Mock

from services.intent_detector import IntentDetector, ROADMAP_KEYWORDS
from domain import Intent, IntentDetectionMethod

@pytest.fixture
def mock_llm():
    """Mock LLM client for testing"""
    return Mock()

@pytest.fixture
def detector(mock_llm):
    """IntentDetector instance with mock LLM"""
    return IntentDetector(mock_llm)

class TestEmptyInput:
    """Tests for empty or whitespace-only input handling"""

    def test_empty_string_returns_chat_with_zero_score(self, detector):
        """Empty string should return CHAT with zero score"""
        result = detector.detect("")
        assert result.intent == Intent.CHAT
        assert result.score == 0.0
        assert result.method == IntentDetectionMethod.LLM

    def test_whitespace_only_returns_chat_with_zero_score(self, detector):
        """Whitespace-only input should return CHAT with zero score"""
        result = detector.detect("    ")
        assert result.intent == Intent.CHAT
        assert result.score == 0.0

class TestKeywordDetection:
    """Tests for rule-based keyword detection"""

    def test_lo_trinh_keyword(self, detector):
        """Text containing 'lộ trình' should trigger ROADMAP"""
        result = detector.detect("tôi muốn tạo lộ trình Python")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_roadmap_keyword(self, detector):
        """Text containing 'roadmap' should trigger ROADMAP"""
        result = detector.detect("tôi muốn tạo roadmap học Python trong 3 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_ke_hoach_hoc_keyword(self, detector):
        """Text containing 'kế hoạch học' should trigger ROADMAP"""
        result = detector.detect("cho tôi kế hoạch học Python")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_learning_path_keyword(self, detector):
        """Text containing 'learning path' should trigger ROADMAP"""
        result = detector.detect("tôi cần một learning path cho web dev")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_tao_lo_trinh_keyword(self, detector):
        """Text containing 'tạo lộ trình' should trigger ROADMAP"""
        result = detector.detect("tạo lộ trình học 4 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_muon_hoc_keyword(self, detector):
        """Text containing 'muốn học' should trigger ROADMAP"""
        result = detector.detect("tôi muốn học Python")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_bat_dau_hoc_keyword(self, detector):
        """Text containing 'bắt đầu học' should trigger ROADMAP"""
        result = detector.detect("bắt đầu học React")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

    def test_hoc_trong_keyword(self, detector):
        """Text containing 'học trong' should trigger ROADMAP"""
        result = detector.detect("tôi muốn học trong 3 tháng")
        assert result.intent == Intent.ROADMAP
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD

class TestLLMFallback:
    """Tests for LLM fallback detection when no keyword is matched"""
    def test_llm_short_text_low_score(self, detector, mock_llm):
        """Short text via LLM should return low confidence score (0.45)"""
        mock_llm.generate_text.return_value = "CHAT"

        result = detector.detect("Python")
        assert result.intent == Intent.CHAT
        assert result.score == 0.45
        assert result.method == IntentDetectionMethod.LLM

    def test_llm_medium_text_high_score(self, detector, mock_llm):
        """Medium-length text via LLM should return higher score (0.55)"""
        mock_llm.generate_text.return_value = "CHAT"

        result = detector.detect("Giải thích về OOP trong Python")
        assert result.intent == Intent.CHAT
        assert result.score == 0.55
        assert result.method == IntentDetectionMethod.LLM

class TestDecisionScore:
    """"""
    def test_score_low_for_short_text(self, detector, mock_llm):
        """Short text (<5 words) via LLM gets low score (0.45)"""
        mock_llm.generate_text.return_value = "CHAT"
        result = detector.detect("Python")
        assert result.score == 0.45
        assert result.method == IntentDetectionMethod.LLM

    def test_score_higher_for_medium_text(self, detector, mock_llm):
        """Medium text (5+ words) via LLM gets higher score (0.55)"""
        mock_llm.generate_text.return_value = "CHAT"
        result = detector.detect("Giải thích cho tôi về Python")
        assert result.score == 0.55
        assert result.method == IntentDetectionMethod.LLM

    def test_score_high_for_keyword_match(self, detector):
        """Keyword match gets high score (0.95)"""
        result = detector.detect("Tạo lộ trình Python")
        assert result.score == 0.95
        assert result.method == IntentDetectionMethod.KEYWORD