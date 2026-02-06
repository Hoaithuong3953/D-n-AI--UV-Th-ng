"""
test_chat_flow_memory.py

Integration tests for memory/history management in chat flow

Tests:
- Context window limiting (recent N messages)
- History building across multiple messages
- Memory persistence
"""
import pytest
from unittest.mock import MagicMock

from domain import ChatMessage

class TestChatFlowMemoryIntegration:
    """Test memory integration in chat flow"""
    
    def test_context_window_limits_history(self, app_service):
        """Only recent N messages sent to ChatService"""
        app = app_service(
            llm_client=MagicMock(),
            chat_context_messages=2,
        )
        
        app._memory.add_message(ChatMessage(role="user", content="Old message 1"))
        app._memory.add_message(ChatMessage(role="assistant", content="Old response 1"))
        app._memory.add_message(ChatMessage(role="user", content="Old message 2"))
        app._memory.add_message(ChatMessage(role="assistant", content="Old response 2"))
        
        list(app.handle_message("New message"))
        
        recent = app._get_recent_history()
        assert len(recent) <= 2

    def test_full_history_preserved_while_context_limited(self, app_service):
        """Full history is preserved in memory, but only recent N sent to LLM"""
        app = app_service(
            llm_client=MagicMock(),
            chat_context_messages=2,
        )
        
        for i in range(5):
            app._memory.add_message(ChatMessage(role="user", content=f"Message {i}"))
            app._memory.add_message(ChatMessage(role="assistant", content=f"Response {i}"))
        
        full_history = app._memory.load_history()
        assert len(full_history) == 10
        
        recent = app._get_recent_history()
        assert len(recent) == 2
        assert recent[0].content == "Message 4"
        assert recent[1].content == "Response 4"
