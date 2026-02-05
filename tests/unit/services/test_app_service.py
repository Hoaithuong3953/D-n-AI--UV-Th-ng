import pytest
from unittest.mock import MagicMock
from datetime import datetime

from config.messages import DefaultMessageProvider, MessageKey
from config.constants import DEFAULT_CONTEXT_MESSAGES, MAX_INPUT_LENGTH
from memory import ChatMemory
from services import AppService
from services.chat_service import StreamError
from domain.events import (
    Event,
    ErrorOccurred,
    SessionExpired,
    StatusUpdate,
    TextChunk,
)
from domain.models import ChatMessage
from utils import LLMServiceError

def _make_app_service(
        *,
        chat_stream = None,
        session_expired = False,
):
    """Build AppService with mocks (chat, session, memory)"""
    mock_chat = MagicMock()
    if chat_stream is not None:
        mock_chat.stream_response.return_value = iter(chat_stream)

    mock_session = MagicMock()
    mock_session.is_expired.return_value = session_expired
    mock_session.get_last_activity.return_value = None

    messages = DefaultMessageProvider()
    memory = ChatMemory()

    app = AppService(
        chat_service=mock_chat,
        session_manager=mock_session,
        messages=messages,
        memory=memory,
        chat_context_messages=DEFAULT_CONTEXT_MESSAGES,
    )
    return app, mock_chat, mock_session

class TestAppServiceValidation:
    """Tests for handle_message validation (empty, whitespace, too long)"""

    def test_empty_input_yields_error_occurred(self):
        """Empty input -> ErrorOccurred(validation)"""
        app, mock_chat, _ = _make_app_service(chat_stream=[])
        events = list(app.handle_message(""))

        assert len(events) == 1
        assert isinstance(events[0], ErrorOccurred)
        assert events[0].error_type == "validation"
        assert events[0].user_message
        mock_chat.stream_response.assert_not_called()

    def test_whitespace_only_yields_error_occurred(self):
        """Whitespace only input -> ErrorOccurred(validation)"""
        app, mock_chat, _ = _make_app_service(chat_stream=[])
        events = list(app.handle_message(""))

        assert len(events) == 1
        assert isinstance(events[0], ErrorOccurred)
        assert events[0].error_type == "validation"
        mock_chat.stream_response.assert_not_called()

    def test_input_too_long_yields_error_occurred(self):
        """Input exceeds MAX_INPUT_LENGTH -> ErrorOccurred(validation)"""
        app, mock_chat, _ = _make_app_service(chat_stream=[])
        long_input = "x" * (MAX_INPUT_LENGTH + 1)
        events = list(app.handle_message(long_input))

        assert len(events) == 1
        assert isinstance(events[0], ErrorOccurred)
        assert events[0].error_type == "validation"
        assert str(MAX_INPUT_LENGTH) in events[0].user_message
        mock_chat.stream_response.assert_not_called()

    def test_valid_input_with_whitespace_trimmed(self):
        """Input with leading/trailing whitespace is trimmed"""
        app, mock_chat, _ = _make_app_service(chat_stream=["response"])
        events = list(app.handle_message("  hello  "))

        assert not any(
            isinstance(e, ErrorOccurred) and e.error_type == "validation" for e in events
        )
        call_args = mock_chat.stream_response.call_args
        assert call_args is not None

class TestAppServiceSessionExpired:
    """Tests for session expired (yields SessionExpired, clears state)"""

    def test_session_expired_yields_session_expired_event(self):
        """Session expired -> SessionExpired event"""
        app, mock_chat, mock_session = _make_app_service(
            session_expired=True,
            chat_stream=[]
        )
        events = list(app.handle_message("hello"))
        
        assert len(events) == 1
        assert isinstance(events[0], SessionExpired)
        assert events[0].message
        mock_chat.stream_response.assert_not_called()

    def test_session_expired_clears_memory(self):
        """Session expired -> memory cleared"""
        app, _, _ = _make_app_service(
            session_expired=True,
            chat_stream=[]
        )
        app._memory.add_message(ChatMessage(role="user", content="old message"))
        assert len(app._memory.load_history()) == 1

        list(app.handle_message("hello"))
        assert len(app._memory.load_history()) == 0

    def test_session_expired_calls_session_reset(self):
        """Session expired -> session.reset() called"""
        app, _, mock_session = _make_app_service(
            session_expired=True,
            chat_stream=[]
        )
        list(app.handle_message("hello"))
        mock_session.reset.assert_called_once()

    def test_session_expired_does_not_touch_activity(self):
        """Session expired -> touch_activity not called (session reset instead)"""
        app, _, mock_session = _make_app_service(
            session_expired=True,
            chat_stream=[]
        )
        list(app.handle_message("hello"))
        mock_session.touch_activity.assert_not_called()

class TestAppServiceChatFlow:
    """Tests for chat flow (StatusUpdate, TextChunk, StreamError)"""

    def test_chat_flow_yields_status_update_thinking(self):
        """Chat flow starts with StatusUpdate(loading, THINKING)"""
        app, _, _ = _make_app_service(chat_stream=["response"])
        events = list(app.handle_message("hi"))
        status_events = [e for e in events if isinstance(e, StatusUpdate)]

        assert len(status_events) == 1
        assert status_events[0].status == "loading"
        assert status_events[0].message

    def test_chat_flow_yields_test_chunks(self):
        """Chat flow yields TextChunk events for each chunk"""
        app, _, _ = _make_app_service(chat_stream=["Hello", "world"])
        events = list(app.handle_message("hi"))
        text_events = [e for e in events if isinstance(e, TextChunk)]

        assert len(text_events) == 2
        assert text_events[0].text == "Hello"
        assert text_events[1].text == "world"

    def test_chat_flow_saves_user_message_to_memory(self):
        """User message saved to memory before streaming"""
        app, _, _ = _make_app_service(chat_stream=["response"])
        list(app.handle_message("hi"))
        history = app._memory.load_history()

        assert len(history) >= 1
        assert history[0].role == "user"
        assert history[0].content == "hi"

    def test_chat_flow_saves_assistant_response_to_memory(self):
        """Full assistant response saved to memory after streaming"""
        app, _, _ = _make_app_service(chat_stream=["Hello ", "world"])
        list(app.handle_message("hi"))
        history = app._memory.load_history()

        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "hi"
        assert history[1].role == "assistant"
        assert history[1].content == "Hello world"

    def test_chat_flow_touches_session_activity(self):
        """Session activity touched before processing message"""
        app, _, mock_session = _make_app_service(chat_stream=["response"])
        list(app.handle_message("hi"))
        mock_session.touch_activity.assert_called_once()

    def test_chat_flow_passes_recent_history_to_chat_service(self):
        """Recent chat history passed to chat history"""
        app, mock_chat, _ = _make_app_service(chat_stream="response")

        app._memory.add_message(ChatMessage(role="user", content="msg1"))
        app._memory.add_message(ChatMessage(role="assistant", content="resp1"))

        list(app.handle_message("msg2"))

        call_args = mock_chat.stream_response.call_args
        assert call_args is not None
        history_arg = call_args[0][1]
        assert len(history_arg) >= 2
        assert history_arg[0].content == "msg1"
        assert history_arg[1].content == "resp1"

    def test_chat_flow_empty_response_not_saved(self):
        """Empty response (no chunks) not saved to memory"""
        app, _, _ = _make_app_service(chat_stream=[])
        list(app.handle_message("hi"))
        history = app._memory.load_history()

        assert len(history) == 1
        assert history[0].role == "user"

class TestAppServiceStreamError:
    """Tests for StreamError handling in chat flow"""

    def test_stream_error_llm_error_yields_error_occurred_llm(self):
        """StreamError(LLM_ERROR) -> ErrorOccurred(llm)"""
        app, _, _ = _make_app_service(
            chat_stream=[StreamError(key=MessageKey.LLM_ERROR)]
        )
        events = list(app.handle_message("hi"))
        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) == 1
        assert errs[0].error_type == "llm"
        assert errs[0].user_message

    def test_stream_error_unexpected_error_yields_error_occurred_unexpected(self):
        """StreamError(UNEXPECTED_ERROR) -> ErrorOccurred(unexpected)"""
        app, _, _ = _make_app_service(
            chat_stream=[StreamError(key=MessageKey.UNEXPECTED_ERROR)]
        )
        events = list(app.handle_message("hi"))
        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) == 1
        assert errs[0].error_type == "unexpected"

    def test_stream_error_saves_error_message_to_memory(self):
        """StreamError -> error message saved to memory as assistant"""
        app, _, _ = _make_app_service(
            chat_stream=[StreamError(key=MessageKey.LLM_ERROR)]
        )
        list(app.handle_message("hi"))
        history = app._memory.load_history()

        assert len(history) == 2
        assert history[1].role == "assistant"
        assert history[1].content

    def test_stream_error_after_chunks_saves_chunks_and_error(self):
        """StreamError after chunks -> chunks yielded, then error"""
        app, _, _ = _make_app_service(
            chat_stream=[
                "Hello ",
                "world",
                StreamError(key=MessageKey.LLM_ERROR)
            ]
        )
        events = list(app.handle_message("hi"))
        text_events = [e for e in events if isinstance(e, TextChunk)]
        err_events = [e for e in events if isinstance(e, ErrorOccurred)]

        assert len(text_events) == 2
        assert len(err_events) == 1

        history = app._memory.load_history()
        assert len(history) == 2
        assert history[1].role == "assistant"

class TestAppServiceErrorHandling:
    """Tests for exception handling (LLMServiceError, general exceptions)"""

    def test_chat_service_raise_llm_service_error_yields_error_occurred_llm(self):
        """chat_service.stream_response raises LLMServiceError -> ErrorOccurred(llm)"""
        app, mock_chat, _ = _make_app_service()
        mock_chat.stream_response.side_effect = LLMServiceError("Connection failed")

        events = list(app.handle_message("hi"))
        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) == 1
        assert errs[0].error_type == "llm"

        history = app._memory.load_history()
        assert len(history) == 2
        assert history[1].role == "assistant"

    def test_chat_service_raises_general_exception_yields_error_occurred_unexpected(self):
        """chat_service.stream_response raises Exception -> ErrorOccurred(unexpected)"""
        app, mock_chat, _ = _make_app_service()
        mock_chat.stream_response.side_effect = ValueError("Something broke")

        events = list(app.handle_message("hi"))
        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) == 1
        assert errs[0].error_type == "unexpected"

        history = app._memory.load_history()
        assert len(history) == 2
        assert history[1].role == "assistant"

    def test_error_handling_still_touches_session(self):
        """Even on error, session activity is touched (before error)"""
        app, mock_chat, mock_session = _make_app_service()
        mock_chat.stream_response.side_effect = LLMServiceError("error")

        list(app.handle_message("hi"))
        mock_session.touch_activity.assert_called_once()

class TestAppServiceMemoryManager:
    """Tests for memory/history managerment"""

    def test_get_recent_history_limits_to_chat_context_messages(self):
        """_get_recent_history limits to chat_context_messages"""
        app, _, _ = _make_app_service(chat_stream=["response"])
        app._chat_context_messages = 2

        for i in range(10):
            app._memory.add_message(ChatMessage(role="user", content=f"msg{i}"))
        
        history = app._get_recent_history()
        assert len(history) == 2
        assert history[0].content == "msg8"
        assert history[1].content == "msg9"

    def test_get_recent_history_returns_all_if_less_than_limit(self):
        """_get_recent_history returns all if count < limit"""
        app, _, _ = _make_app_service(chat_stream=["response"])
        app._chat_context_messages = 10

        for i in range(3):
            app._memory.add_message(ChatMessage(role="user", content=f"msg{i}"))
        
        history = app._get_recent_history()
        assert len(history) == 3

    def test_get_recent_histoty_empty_returns_empty_list(self):
        """_get_recent_history with no history returns []"""
        app, _, _ = _make_app_service(chat_stream=["response"])
        history = app._get_recent_history()
        assert history == []

class TestAppServiceResetSession:
    """Tests for reset_session method"""

    def test_reset_session_clears_memory(self):
        """reset_session clears chat history"""
        app, _, _ = _make_app_service(chat_stream=[])
        app._memory.add_message(ChatMessage(role="user", content="test"))
        assert len(app._memory.load_history()) == 1

        app.reset_session()
        assert len(app._memory.load_history()) == 0

    def test_reset_session_calls_session_reset(self):
        """reset_session calls session_manager.reset()"""
        app, _, mock_session = _make_app_service(chat_stream=[])
        app.reset_session()
        mock_session.reset.assert_called_once()

class TestAppServicePersistence:
    """Tests for to_session (session state persistence)"""

    def test_to_session_saves_history(self):
        """to_session saves chat history to session_state"""
        app, _, _ = _make_app_service(chat_stream=[])
        app._memory.add_message(ChatMessage(role="user", content="hello"))
        app._memory.add_message(ChatMessage(role="assistant", content="hi"))

        session_state = {}
        app.to_session(session_state)

        assert "app_history" in session_state
        assert len(session_state["app_history"]) == 2
        assert session_state["app_history"][0]["role"] == "user"
        assert session_state["app_history"][0]["content"] == "hello"

    def test_to_session_saves_last_activity_timestamp(self):
        """to_session saves last_activity timestamp"""
        app, _, mock_session = _make_app_service(chat_stream=[])
        mock_datetime = datetime(2026, 2, 1, 12, 0, 0)
        mock_session.get_last_activity.return_value = mock_datetime

        session_state = {}
        app.to_session(session_state)

        assert "app_session_last_activity" in session_state
        assert session_state["app_session_last_activity"] == mock_datetime.timestamp()

    def test_to_session_last_activity_none_saves_none(self):
        """to_session with no last_activity saves None"""
        app, _, mock_session = _make_app_service(chat_stream=[])
        mock_session.get_last_activity.return_value = None

        session_state = {}
        app.to_session(session_state)

        assert session_state["app_session_last_activity"] is None

    def test_to_session_empty_history_saves_empty_list(self):
        """to_session with empty history saves []"""
        app, _, _ = _make_app_service(chat_stream=[])

        session_state = {}
        app.to_session(session_state)

        assert session_state["app_history"] == []