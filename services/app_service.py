"""
app_service.py

Application Service: handle user message, coordinate chat and session services

Key features:
- handle_message(user_input) yields Event stream (TextChunk, StatusUpdate, ErrorOccurred, SessionExpired)
- Manages chat history, session expiration and error handling
- Routes by intent and orchestrates profile extraction + roadmap generation flows
- Coordinates domain services (ChatService, SessionManager, IntentDetector, ProfileExtractor, RoadmapService)
"""
from __future__ import annotations

from typing import Generator, List, TYPE_CHECKING

from domain import (
    ChatMessage,
    Intent,
    UserProfile,
    Roadmap,
    ConversationState,
)
from domain.events import (
    Event,
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
from services.flows import chat_flow, roadmap_flow

if TYPE_CHECKING:
    from services.core.chat_service import ChatService
    from services.support.session_manager import SessionManager
    from services.support.intent_detector import IntentDetector
    from services.support.profile_extractor import ProfileExtractor
    from services.core.roadmap_service import RoadmapService
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
        intent_detector: IntentDetector,
        profile_extractor: ProfileExtractor,
        roadmap_service: RoadmapService,
        *,
        chat_context_messages: int
    ):
        self._chat = chat_service
        self._session = session_manager
        self._memory = memory
        self._intent = intent_detector
        self._profile_extractor = profile_extractor
        self._roadmap = roadmap_service
        self.messages = messages
        self.current_roadmap: Roadmap | None = None
        self.current_profile: UserProfile | None = None
        self.conversation_state = ConversationState.NORMAL
        self._chat_context_messages = chat_context_messages

    def handle_message(self, user_input: str) -> Generator[Event, None, None]:
        """
        Handle user message: validate, check session, stream chat response

        Flow:
        1. Validate input (empty, too long)
        2. Check session expiration
        3. Smart routing:
            - If AWAITING_PROFILE_INFO: re-extract profile
            - Otherwise: detect intent -> chat or roadmap
        4. Yield events for UI consumption

        Args:
            user_input: The user's message input

        Yields:
            Event: Stream of events (TextChunk, StatusUpdate, ErrorOccurred, SessionExpired)
        """
        logger.info(f"handle_message start (input_len={len(user_input)})")
        user_input = user_input.strip()

        # Validation
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
        
        # Session expiration check
        if self._session.is_expired():
            self._memory.clean_history()
            self.current_profile = None
            self.current_roadmap = None
            self._session.reset()
            yield SessionExpired(
                self.messages.get(MessageKey.SESSION_EXPIRED)
            )
            return
        
        self._session.touch_activity()
        self._memory.add_message(ChatMessage(role="user", content=user_input))
        
        # Smart re-extraction: if user input indicates profile update, re-extract profile
        if self.conversation_state == ConversationState.AWAITING_PROFILE_INFO:
            logger.info("State: AWAITING_PROFILE_INFO, attempting re-extraction")
            yield from roadmap_flow.handle_profile_reextraction(self)
            return
        
        # Route (intent -> chat | roadmap)
        yield StatusUpdate(
            "loading", 
            self.messages.get(MessageKey.THINKING)
        )
        try:
            result = self._intent.detect(user_input)
            logger.info(
                f"intent={result.intent.value}, "
                f"confidence={getattr(result.confidence, 'name', str(result.confidence))}, "
                f"method={result.method.name}"
            )

            if result.intent == Intent.ROADMAP:
                yield from roadmap_flow.handle_roadmap_request(self)
            else:
                yield from chat_flow.handle_chat_request(self, user_input)
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

    def _get_recent_history(self) -> List[ChatMessage]:
        """Return recent chat history for ChatService and RoadmapService"""
        history = self._memory.load_history()
        if not history:
            return []
        
        n = self._chat_context_messages
        return history[-n:] if len(history) > n else history
    
    def reset_session(self):
        """Clear all conversation state for a truly fresh session"""
        self._memory.clean_history()
        self.current_profile = None
        self.current_roadmap = None
        self.conversation_state = ConversationState.NORMAL
        if hasattr(self._profile_extractor, "last_missing_fields"):
            self._profile_extractor.last_missing_fields = []
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
