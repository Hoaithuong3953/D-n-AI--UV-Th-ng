"""
test_chat_service.py

Unit tests for services.chat_service (stream yields chunks or StreamError)
"""
import pytest
from unittest.mock import MagicMock

from domain import ChatMessage
from services.chat_service import ChatService, StreamError
from config.messages import MessageKey
from utils import LLMServiceError

class TestStreamError:
    """Tests for StreamError dataclass"""

    def test_stream_error_holds_key(self):
        err = StreamError(key=MessageKey.LLM_ERROR)
        assert err.key == MessageKey.LLM_ERROR

class TestChatServiceStream:
    """Tests for ChatService.stream (chunks, StreamError on LLM error)"""

    def test_stream_yield_chunks_then_returns(self):
        """Happy path: stream yields chunks successfully"""
        mock_llm = MagicMock()
        def gen():
            yield "Hello"
            yield "world"

        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        history = [ChatMessage(role="user", content="m0")]
        out = list(svc.stream_response("hi", history=history))

        assert out == ["Hello", "world"]
        mock_llm.stream_chat.assert_called_once()

        call_kw = mock_llm.stream_chat.call_args[1]
        assert call_kw["new_message"] == "hi"
        assert call_kw["history"] == history

    def test_stream_retry_then_success(self):
        """First attempt fails before any chunks, second attempt succeeds"""
        mock_llm = MagicMock()
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise LLMServiceError(message="Timeout")
            
            yield "Retry"
            yield "success"

        mock_llm.stream_chat.side_effect = side_effect
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 2
        assert out == ["Retry", "success"]

    def test_stream_empty_response_yields_error(self):
        """Empty stream yields error after max retries"""
        mock_llm = MagicMock()
        def gen():
            return
            yield

        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 2
        assert len(out) == 1
        assert isinstance(out[0], StreamError)
        assert out[0].key == MessageKey.LLM_ERROR

    def test_stream_empty_first_attempt_success_second(self):
        """First attempt returns empty, second attempt succeeds"""
        mock_llm = MagicMock()
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return
                yield
            yield "Success"
        mock_llm.stream_chat.side_effect = side_effect
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 2
        assert out == ["Success"]

    def test_stream_yield_stream_error_on_llm_service_error(self):
        """LLMServiceError before any chunks yields StreamError with LLM_ERROR key"""
        mock_llm = MagicMock()
        mock_llm.stream_chat.side_effect = LLMServiceError(message="Connection failed")
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))
        assert len(out) == 1
        assert isinstance(out[0], StreamError)
        assert out[0].key == MessageKey.LLM_ERROR

    def test_stream_general_exception_yields_unexpected_error(self):
        """Non-LLMServiceError before any chunks yields StreamError with LLM_ERROR key"""
        mock_llm = MagicMock()
        mock_llm.stream_chat.side_effect = ValueError("Something broke")
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 2
        assert len(out) == 1
        assert isinstance(out[0], StreamError)
        assert out[0].key == MessageKey.UNEXPECTED_ERROR

    def test_stream_mid_stream_failure_no_retry(self):
        """Mid-stream failure does not trigger retry"""
        mock_llm = MagicMock()
        def gen():
            yield "Hello"
            yield "world"
            raise LLMServiceError(message="Connection lost")
        
        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 1
        assert len(out) == 3
        assert out[0] == "Hello"
        assert out[1] == "world"
        assert isinstance(out[2], StreamError)
        assert out[2].key == MessageKey.LLM_STREAM_INTERRUPTED

    def test_stream_mid_stream_general_exception_no_retry(self):
        """General exception after chunks does not trigger retry"""
        mock_llm = MagicMock()
        def gen():
            yield "Chunk 1"
            raise ValueError("Unexpected error mid-stream")
        
        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 1
        assert len(out) == 2
        assert out[0] == "Chunk 1"
        assert isinstance(out[1], StreamError)
        assert out[1].key == MessageKey.LLM_STREAM_INTERRUPTED

    def test_stream_first_chunk_then_failure_is_mid_stream(self):
        """Single chunk before error counts as mid-stream failure"""
        mock_llm = MagicMock()
        def gen():
            yield "A"
            raise LLMServiceError(message="Error")
        
        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 1
        assert out[0] == "A"
        assert isinstance(out[1], StreamError)
        assert out[1].key == MessageKey.LLM_STREAM_INTERRUPTED

    def test_stream_quota_error_no_retry(self):
        """429 quota error before chunks: no retry, immediate StreamError"""
        mock_llm = MagicMock()
        mock_llm.stream_chat.side_effect = LLMServiceError(message="429 Quota exceeded")
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 1
        assert len(out) == 1
        assert isinstance(out[0], StreamError)
        assert out[0].key == MessageKey.LLM_ERROR

    def test_stream_quota_error_mid_stream_no_retry(self):
        """Quota error after chunks yields interrupted error"""
        mock_llm = MagicMock()
        def gen():
            yield "Started"
            raise LLMServiceError(message="429 Quota exceeded")
        mock_llm.stream_chat.return_value = gen()
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 1
        assert out[0] == "Started"
        assert isinstance(out[1], StreamError)
        assert out[1].key == MessageKey.LLM_ERROR

    def test_stream_transient_error_retries_before_chunks(self):
        """Transient LLMServiceError before chunks: retry once before failing"""
        mock_llm = MagicMock()
        mock_llm.stream_chat.side_effect = LLMServiceError(message="Timeout")
        svc = ChatService(llm_client=mock_llm)
        out = list(svc.stream_response("hi", history=[]))

        assert mock_llm.stream_chat.call_count == 2
        assert len(out) == 1
        assert isinstance(out[0], StreamError)
        assert out[0].key == MessageKey.LLM_ERROR

class TestChatServiceMessageKeyMapping:
    """Test _stream_error_key method directly"""

    def test_stream_error_key_llm_service_error(self):
        """LLMServiceError maps to LLM_ERROR"""
        mock_llm = MagicMock()
        svc = ChatService(llm_client=mock_llm)
        key = svc._stream_error_key(LLMServiceError("test"))
        assert key == MessageKey.LLM_ERROR

    def test_stream_error_key_general_exception(self):
        """General exception maps to UNEXPECTED_ERROR"""
        mock_llm = MagicMock()
        svc = ChatService(llm_client=mock_llm)
        key = svc._stream_error_key(ValueError("test"))
        assert key == MessageKey.UNEXPECTED_ERROR

    def test_stream_error_key_runtime_error(self):
        """RuntimeError maps to UNEXPECTED_ERROR"""
        mock_llm = MagicMock()
        svc = ChatService(llm_client=mock_llm)
        key = svc._stream_error_key(RuntimeError("test"))
        assert key == MessageKey.UNEXPECTED_ERROR