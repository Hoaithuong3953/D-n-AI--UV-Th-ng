"""
Unit test for ChatMessage model
"""
import pytest
from pydantic import ValidationError
from datetime import datetime

from domain.models import ChatMessage

class TestChatMessage:
    """Tests for ChatMessage model"""
    def test_create_chat_message_user(self):
        """Test creating a user chat message"""
        message = ChatMessage(
            role="user",
            content="Hello"
        )
        assert message.role == "user"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)

    def test_create_chat_message_assistant(self):
        """Test creating an assistant chat message"""
        message = ChatMessage(
            role="assistant",
            content="Hello"
        )
        assert message.role == "assistant"
        assert message.content == "Hello"
        assert isinstance(message.timestamp, datetime)

    def test_create_chat_message_system(self):
        """Test creating a system chat message"""
        message = ChatMessage(
            role="user",
            content="System error"
        )
        assert message.role == "user"
        assert message.content == "System error"
        assert isinstance(message.timestamp, datetime)

    def test_chat_message_role_literal(self):
        """Test that role must be valid literal"""
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
