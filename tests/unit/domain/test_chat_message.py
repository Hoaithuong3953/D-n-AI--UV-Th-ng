"""
test_chat_message.py

Unit tests for ChatMessage model (roles, content, timestamp, role literal validation)
"""
import pytest
from pydantic import ValidationError
from datetime import datetime

from domain import ChatMessage

class TestChatMessage:
    """Tests for ChatMessage model"""

    def test_create_chat_message_user(self):
        """ChatMessage with role='user' has correct role, content, timestamp"""
        message = ChatMessage(
            role="user",
            content="Hello"
        )
        assert message.role == "user"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)

    def test_create_chat_message_assistant(self):
        """ChatMessage with role='assistant' has correct role, content, timestamp"""
        message = ChatMessage(
            role="assistant",
            content="Hello"
        )
        assert message.role == "assistant"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)

    def test_create_chat_message_system(self):
        """ChatMessage can store system-like content with role='user' (system stored as user)"""
        message = ChatMessage(
            role="user",
            content="System error"
        )
        assert message.role == "user"
        assert message.content == "System error"
        assert isinstance(message.timestamp, datetime)

    def test_chat_message_role_literal(self):
        """role must be one of system, user, assistant; invalid raises ValidationError"""
        # Valid roles
        valid_roles = ["system", "user", "assistant"]
        for role in valid_roles:
            message = ChatMessage(role=role, content="Test")
            assert message.role == role

        # Invalid role should raise ValidationError
        with pytest.raises(ValidationError):
            ChatMessage(
                role="invalid_role",
                content="Test"
            )
