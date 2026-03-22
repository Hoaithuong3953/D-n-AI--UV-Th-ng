"""
Unit tests for services.flows.chat_flow.handle_chat_request
"""
from unittest.mock import MagicMock

from config.messages import DefaultMessageProvider, MessageKey
from domain import ChatMessage
from domain.events import ErrorOccurred, TextChunk
from memory import ChatMemory
from services.core.chat_service import StreamError
from services.flows import chat_flow

class TestHandleChatRequest:
    def test_yields_text_chunks_and_appends_assistant_message(self):
        app = MagicMock()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = []
        app._chat.stream_response.return_value = iter(["Hello ", "world"])
        app.messages = DefaultMessageProvider()

        events = list(chat_flow.handle_chat_request(app, "hi"))

        texts = [e.text for e in events if isinstance(e, TextChunk)]
        assert texts == ["Hello ", "world"]
        hist = app._memory.load_history()
        assert len(hist) == 1
        assert hist[0].role == "assistant"
        assert hist[0].content == "Hello world"

    def test_stream_error_llm_yields_error_and_stops(self):
        app = MagicMock()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = []
        app._chat.stream_response.return_value = iter(
            [StreamError(key=MessageKey.LLM_ERROR)]
        )
        app.messages = DefaultMessageProvider()

        events = list(chat_flow.handle_chat_request(app, "hi"))

        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) == 1
        assert errs[0].error_type == "llm"
        hist = app._memory.load_history()
        assert len(hist) == 1
        assert hist[0].role == "assistant"

    def test_empty_stream_no_assistant_message(self):
        app = MagicMock()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = []
        app._chat.stream_response.return_value = iter([])
        app.messages = DefaultMessageProvider()

        list(chat_flow.handle_chat_request(app, "hi"))

        assert app._memory.load_history() == []