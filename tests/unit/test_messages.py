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
        "UNEXPECTED_ERROR": "unexpected_error",
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