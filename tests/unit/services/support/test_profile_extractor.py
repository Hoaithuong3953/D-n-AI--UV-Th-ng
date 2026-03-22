"""
Unit tests for ProfileExtractor (rule fast-path + LLM path)
"""
import json
from unittest.mock import MagicMock

import pytest

from domain import ChatMessage, UserProfile
from services.support.profile_extractor import ProfileExtractor, _parse_profile_json

class TestParseProfileJson:
    def test_plain_json_object(self):
        raw = '{"goal": "X", "current_level": "beginner"}'
        assert _parse_profile_json(raw) == {"goal": "X", "current_level": "beginner"}

    def test_json_in_markdown_fence(self):
        raw = """Here:
```json
{"goal": "Y", "current_level": "intermediate"}
```
"""
        d = _parse_profile_json(raw)
        assert d == {"goal": "Y", "current_level": "intermediate"}

    def test_invalid_returns_none(self):
        assert _parse_profile_json("not json") is None

class TestProfileExtractorExtract:
    def test_empty_history_returns_none(self):
        ext = ProfileExtractor(MagicMock())
        assert ext.extract([]) is None

    def test_no_user_messages_returns_none(self):
        ext = ProfileExtractor(MagicMock())
        assert ext.extract([ChatMessage(role="assistant", content="hi")]) is None

    def test_fast_path_rule_based_full_profile(self):
        llm = MagicMock()
        ext = ProfileExtractor(llm)
        text = "tôi muốn học Python, mới bắt đầu, 1 giờ/ngày"
        history = [ChatMessage(role="user", content=text)]

        profile = ext.extract(history)

        assert isinstance(profile, UserProfile)
        assert profile.goal == "Python"
        assert profile.current_level == "beginner"
        assert "giờ" in profile.time_commitment
        llm.generate_text.assert_not_called()

    def test_llm_path_builds_profile(self):
        llm = MagicMock()
        llm.generate_text.return_value = json.dumps(
            {
                "goal": "Học Rust",
                "current_level": "intermediate",
                "time_commitment": "2 giờ",
            }
        )
        ext = ProfileExtractor(llm)
        history = [ChatMessage(role="user", content="something vague without markers")]

        profile = ext.extract(history)

        assert profile is not None
        assert profile.goal == "Học Rust"
        assert profile.current_level == "intermediate"
        llm.generate_text.assert_called_once()

    def test_llm_service_error_propagates(self):
        llm = MagicMock()
        from utils import LLMServiceError

        llm.generate_text.side_effect = LLMServiceError(message="fail")
        ext = ProfileExtractor(llm)
        history = [ChatMessage(role="user", content="no fast path keywords here at all")]

        with pytest.raises(LLMServiceError):
            ext.extract(history)

    def test_missing_required_sets_last_missing_fields(self):
        llm = MagicMock()
        llm.generate_text.return_value = json.dumps({"goal": "only goal"})
        ext = ProfileExtractor(llm)
        history = [ChatMessage(role="user", content="xyz no level time")]

        assert ext.extract(history) is None
        assert ext.last_missing_fields
