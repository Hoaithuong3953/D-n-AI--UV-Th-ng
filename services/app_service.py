"""
app_service.py

Application Service: handle user message, coordinate chat and session services

Key features:
- handle_message(user_input) yields Event stream (TextChunk, StatusUpdate, ErrorOccurred, SessionExpired)
- Manages chat history, session expiration, error handling
- Orchestrates domain services (ChatService, SessionManager)
"""
from __future__ import annotations

from typing import Generator, List, TYPE_CHECKING

from domain import (
    ChatMessage,
)
from domain.events import (
    Event,
    TextChunk,
    StatusUpdate,
    ErrorOccurred,
    SessionExpired,
)
from config import (
    MAX_INPUT_LENGTH,
    MessageKey,
    MessageProvider,
)
from utils import LLMServiceError, logger
from services.chat_service import StreamError

if TYPE_CHECKING:
    from services.chat_service import ChatService
    from services.session_manager import SessionManager
    from memory import ChatHistory

class AppService:
    """
    Application Service: orchestrate use cases and domain services
    
    Responsibilities:
    - Handle user message use case
    - Coordinate domain services (chat, session, memory)
    - Manage cross-cutting concerns (validation, error handling)
    - Translate domain events to UI events
    """
    def __init__(
        self,
        chat_service: ChatService,
        session_manager: SessionManager,
        messages: MessageProvider,
        memory: ChatHistory,
        *,
        chat_context_messages: int
    ):
        self._chat = chat_service
        self._session = session_manager
        self._memory = memory
        self.messages = messages
        self._chat_context_messages = chat_context_messages

    def handle_message(self, user_input: str) -> Generator[Event, None, None]:
        """
        Handle user message: validate, check session, stream chat response

        Args:
            user_input: The user's message input

        Yields:
            Event: Stream of events (TextChunk, StatusUpdate, ErrorOccurred, SessionExpired)
        """
        logger.info(f"handle_message start (input_len={len(user_input)})")
        user_input = user_input.strip()

        if not user_input:
            yield ErrorOccurred(
                "validation", 
                self.messages.get(MessageKey.EMPTY_INPUT)
            )
            return
        if len(user_input) > MAX_INPUT_LENGTH:
            yield ErrorOccurred(
                "validation",
                self.messages.format(MessageKey.INPUT_TOO_LONG, max=str(MAX_INPUT_LENGTH))
            )
            return
        
        if self._session.is_expired():
            self._memory.clean_history()
            self._session.reset()
            yield SessionExpired(
                self.messages.get(MessageKey.SESSION_EXPIRED)
            )
            return
        
        self._session.touch_activity()
        self._memory.add_message(ChatMessage(role="user", content=user_input))
        yield StatusUpdate(
            "loading", 
            self.messages.get(MessageKey.THINKING)
        )

        try:
            yield from self._handle_chat_request(user_input)
        except LLMServiceError as e:
            msg = self.messages.get(MessageKey.LLM_ERROR)
            yield ErrorOccurred("llm", msg)
            self._memory.add_message(ChatMessage(role="assistant", content=msg))
        except Exception as e:
            logger.exception(f"handle_message error: {e}")
            msg = self.messages.get(MessageKey.UNEXPECTED_ERROR)
            yield ErrorOccurred("unexpected", msg)
            self._memory.add_message(ChatMessage(role="assistant", content=msg))
        logger.info("handle_message end")

    def _handle_chat_request(self, user_input: str) -> Generator[Event, None, None]:
        """Chat request handler: stream chat response, yield TextChunk and ErrorOccurred events"""
        logger.info("_handle_chat_request start")
        full_response = ""
        history = self._get_recent_history()
        for item in self._chat.stream_response(user_input, history):
            if isinstance(item, str):
                full_response += item
                yield TextChunk(item)
            else:
                assert isinstance(item, StreamError)
                msg = self.messages.get(item.key)
                error_type = "llm" if item.key == MessageKey.LLM_ERROR else "unexpected"
                yield ErrorOccurred(error_type, msg)
                self._memory.add_message(ChatMessage(role="assistant", content=msg))
                return
        if full_response:
            self._memory.add_message(ChatMessage(role="assistant", content=full_response))
        logger.info(f"handle_chat_request end (response len={len(full_response)})")

    def _get_recent_history(self) -> List[ChatMessage]:
        """Return recent chat history for ChatService and RoadmapService"""
        history = self._memory.load_history()
        if not history:
            return []
        
        n = self._chat_context_messages
        return history[-n:] if len(history) > n else history
    
    def reset_session(self):
        """Clear chat history and reset session state"""
        self._memory.clean_history()
        self._session.reset()

    def to_session(self, session_state) -> None:
        """Save application state to session_state dict"""
        history = self._memory.load_history()
        session_state["app_history"] = [
            m.model_dump(mode="json") for m in history
        ]
        la = self._session.get_last_activity()
        session_state["app_session_last_activity"] = (
            la.timestamp() if la else None
        )
