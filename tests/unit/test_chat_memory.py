"""
test_chat_memory.py

Unit tests for ChatMemory (in-memory history storage)

Key features:
- Initial empty history, add_message, clean_history
"""
from domain import ChatMessage
from memory import ChatMemory

class TestChatMemory:
    """Tests for ChatMemory (in-memory history storage)"""

    def test_initial_empty_history(self):
        """New ChatMemory with empty storage has no messages"""
        memory = ChatMemory()

        history = memory.load_history()

        assert history == []

    def test_add_message_appends_to_history(self):
        """add_message appends ChatMessage to internal storage"""
        memory = ChatMemory()
        msg = ChatMessage(role="user", content="Hello")

        memory.add_message(msg)

        history = memory.load_history()
        assert len(history) == 1

    def test_clean_history_clears_all_messages(self):
        """clean_history removes all stored messages"""
        memory = ChatMemory()
        memory.add_message(ChatMessage(role="user", content="msg1"))
        memory.add_message(ChatMessage(role="assistant", content="msg2"))

        history = memory.load_history()
        assert len(history) == 2

        memory.clean_history()
        history = memory.load_history()
        assert len(history) == 0