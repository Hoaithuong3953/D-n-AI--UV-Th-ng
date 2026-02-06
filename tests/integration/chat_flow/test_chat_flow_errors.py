"""
test_chat_flow_errors.py

Integration tests for error handling in chat flow

Tests:
- LLM errors (API failures, timeouts)
- Unexpected errors
- StreamError from ChatService
- Error message persistence in history
"""
import pytest
from unittest.mock import MagicMock

from services import AppService, ChatService, SessionManager
from services.chat_service import StreamError
from ai import GeminiClient
from memory import ChatMemory
from config import default_messages, MessageKey, DEFAULT_CONTEXT_MESSAGES
from domain.events import ErrorOccurred, StatusUpdate, TextChunk
from utils import LLMServiceError

class TestChatFlowErrorHandling:
    """Test error handling in chat flow"""
    def test_llm_error_returns_error_event_and_saves_error_message(self, app_service):
        """LLM error -> ErrorOccurred event + error message saved to history"""
        # Mock LLM client that raises error
        mock_llm = MagicMock(spec=GeminiClient)
        mock_llm.stream_chat.side_effect = LLMServiceError(
            code="API_ERROR", 
            message="API call failed"
        )
        
        app = app_service(llm_client=mock_llm)
        
        events = list(app.handle_message("What is Python?"))
        
        status_events = [e for e in events if isinstance(e, StatusUpdate)]
        error_events = [e for e in events if isinstance(e, ErrorOccurred)]
        
        assert len(status_events) >= 1
        assert len(error_events) == 1
        assert error_events[0].error_type == "llm"
        
        history = app._memory.load_history()
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert "kết nối" in history[1].content.lower() or "lỗi" in history[1].content.lower()

    def test_unexpected_error_handled_gracefully(self, app_service):
        """Unexpected errors -> ErrorOccurred event with generic message"""
        mock_llm = MagicMock(spec=GeminiClient)
        mock_llm.stream_chat.side_effect = RuntimeError("Something went wrong")
        
        app = app_service(llm_client=mock_llm)
        
        events = list(app.handle_message("Test"))
        
        error_events = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(error_events) == 1
        assert error_events[0].error_type == "unexpected"

    def test_stream_error_from_chat_service_handled(self):
        """StreamError from ChatService -> ErrorOccurred event"""
        mock_chat_service = MagicMock(spec=ChatService)
        
        def fake_stream_response(user_input, history):
            yield "Partial "
            yield StreamError(key=MessageKey.LLM_ERROR)
        
        mock_chat_service.stream_response = fake_stream_response
        
        session_manager = SessionManager(timeout_minutes=30)
        memory = ChatMemory()
        
        app = AppService(
            chat_service=mock_chat_service,
            session_manager=session_manager,
            messages=default_messages,
            memory=memory,
            chat_context_messages=DEFAULT_CONTEXT_MESSAGES,
        )
        
        events = list(app.handle_message("Test"))
        
        text_chunks = [e for e in events if isinstance(e, TextChunk)]
        error_events = [e for e in events if isinstance(e, ErrorOccurred)]
        
        assert len(text_chunks) == 1
        assert len(error_events) == 1
        assert error_events[0].error_type == "llm"