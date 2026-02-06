"""
test_chat_flow_session.py

Integration tests for session management in chat flow

Tests:
- Session expiration handling
- Session state persistence (to_session)
- Session reset
"""
import pytest
import time
from unittest.mock import MagicMock

from domain import ChatMessage
from domain.events import SessionExpired

class TestChatFlowSessionExpiration:
    """Test session expiration scenarios"""
    
    def test_expired_session_returns_session_expired_event(self, app_service):
        """Expired session -> SessionExpired event + history cleared"""
        app = app_service(
            llm_client=MagicMock(),
            timeout_minutes=0,
        )
        
        app._memory.add_message(ChatMessage(role="user", content="Old message"))
        app._session.touch_activity()
        
        time.sleep(0.1)
        
        events = list(app.handle_message("New message"))

        session_expired_events = [e for e in events if isinstance(e, SessionExpired)]
        assert len(session_expired_events) >= 1
        
        history = app._memory.load_history()
        assert len(history) == 0

class TestChatFlowSessionState:
    """Test session state persistence"""
    
    def test_to_session_saves_history_and_activity(self, app_service, fake_llm_client):
        """to_session() saves history and last activity to session_state"""
        llm = fake_llm_client(["Test "])
        app = app_service(llm_client=llm)
        list(app.handle_message("Test message"))
        
        session_state = {}
        app.to_session(session_state)
        
        assert "app_history" in session_state
        assert len(session_state["app_history"]) == 2
        
        assert "app_session_last_activity" in session_state
        assert session_state["app_session_last_activity"] is not None
    
    def test_reset_session_clears_history_and_session(self, app_service, fake_llm_client):
        """reset_session() clears all history and resets session"""
        llm = fake_llm_client(["Message 1", "Message 2"])
        app = app_service(llm_client=llm)

        list(app.handle_message("Message 1"))
        list(app.handle_message("Message 2"))
        
        app.reset_session()
        
        history = app._memory.load_history()
        assert len(history) == 0
