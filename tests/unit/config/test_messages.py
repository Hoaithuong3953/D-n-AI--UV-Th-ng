"""
test_messages.py

Unit tests for config.messages: MessageKey, DefaultMessageProvider, default_messages

Key features:
- Ensure MessageKey values match the public contract
- Ensure DefaultMessageProvider/default_messages templates and formatting behave consistently
"""
import pytest

from config import MessageKey, DefaultMessageProvider, default_messages

class TestMessageKey:
    """MessageKey enum values"""

    EXPECTED_MESSAGE_KEYS = {
        "LLM_ERROR": "llm_error",
        "LLM_STREAM_INTERRUPTED": "llm_stream_interrupted",
        "UNEXPECTED_ERROR": "unexpected_error",

        "EMPTY_INPUT": "empty_input",
        "INPUT_TOO_LONG": "input_too_long",
        "SESSION_EXPIRED": "session_expired",
        "FILL_PROFILE": "fill_profile",

        "THINKING": "thinking",

        "PROFILE_ANALYZING": "profile_analyzing",
        "PROFILE_EXTRACTED": "profile_extracted",

        "ROADMAP_LOADING": "roadmap_loading",
        "ROADMAP_CREATED": "roadmap_created",
        "ROADMAP_ERROR": "roadmap_error",
        "ROADMAP_INVALID_JSON": "roadmap_invalid_json",
        "ROADMAP_INVALID_SCHEMA": "roadmap_invalid_schema",
        "ROADMAP_GENERATION_FAILED": "roadmap_generation_failed",
    }
    
    def test_every_message_key_has_expected_value(self):
        """Every MessageKey member has expected .value"""
        for name, expected_value in self.EXPECTED_MESSAGE_KEYS.items():
            key = getattr(MessageKey, name)
            assert key.value == expected_value, (
                f"MessageKey.{name}.value should be {expected_value!r}, got {key.value!r}"
            )

    def test_message_key_has_no_extra_members(self):
        """MessageKey has no extra members beyond the contract"""
        actual = {m.name for m in MessageKey}
        expected = set(self.EXPECTED_MESSAGE_KEYS)
        assert actual == expected, (
            f"MessageKey mismatch: extra {actual - expected!r}, missing {expected - actual!r}"
        )

class TestDefaultMessageProvider:
    """DefaultMessageProvider get/format"""
    def test_get_returns_extract_template_for_empty_input(self):
        """get(EMPTY_INPUT) returns correct template"""
        provider = DefaultMessageProvider()
        assert provider.get(MessageKey.EMPTY_INPUT) == "Vui lòng nhập nội dung tin nhắn."

    def test_format_input_too_long_substitutes_max(self):
        """format(INPUT_TOO_LONG, max=...) substitutes {max} correctly"""
        provider = DefaultMessageProvider()
        assert provider.format(MessageKey.INPUT_TOO_LONG, max="500") == (
            "Tin nhắn quá dài. Vui lòng giới hạn trong 500 kí tự."
        )

    def test_format_profile_extracted_substitutes_goal_level_time(self):
        """format(PROFILE_EXTRACTED, goal=..., level=..., time=...) substitutes correctly"""
        provider = DefaultMessageProvider()
        formatted_message = provider.format(
            MessageKey.PROFILE_EXTRACTED,
            goal="Học lập trình Python",
            level="Người mới bắt đầu",
            time="6 tháng"
        )
        assert formatted_message == (
            "Đã trích xuất thông tin hồ sơ: mục tiêu Học lập trình Python, trình độ Người mới bắt đầu, thời gian học 6 tháng."
        )

    def test_format_without_kwargs_returns_same_as_get(self):
        """format(key) without kwargs returns same string as get(key)"""
        provider = DefaultMessageProvider()
        assert provider.format(MessageKey.THINKING) == provider.get(MessageKey.THINKING)

    def test_all_message_keys_have_non_empty_template(self):
        """Every MessageKey has a template in DefaultMessageProvider"""
        provider = DefaultMessageProvider()
        for key in MessageKey:
            template = provider.get(key)
            assert template, f"Missing template for MessageKey.{key!r}"

    def test_default_messages_is_default_message_provider_instance(self):
        """default_messages is a singleton instance of DefaultMessageProvider"""
        assert isinstance(default_messages, DefaultMessageProvider)

    def test_default_messages_get_same_as_fresh_provider_for_all_keys(self):
        """default_messages.get(key) returns same as DefaultMessageProvider().get(key) for all keys"""
        fresh = DefaultMessageProvider()
        for key in MessageKey:
            assert default_messages.get(key) == fresh.get(key), (
                f"default_messages.get({key!r}) differs from fresh provider"
            )

    def test_default_messages_get_returns_thinking_text(self):
        """default_messages.get(THINKING) returns the correct thinking text"""
        assert default_messages.get(MessageKey.THINKING) == "Đang suy nghĩ..."