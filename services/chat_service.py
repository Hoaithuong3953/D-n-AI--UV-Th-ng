"""
chat_service.py

Chat service layer for LearnPath chatbot. Handles user messages: validation, session,
streaming LLM response, history persistence and error handling

Key features:
- Input validation and session expiry; context window limit
- Streaming response and history persistence
- User-facing error messages on LLM/session failure
"""
from typing import Generator, List, Union
from dataclasses import dataclass

from utils import logger, LLMServiceError
from ai import LLMClient
from memory import ChatHistory
from domain import ChatMessage
from config import MessageKey
from .session_manager import SessionManager

@dataclass(frozen=True)
class StreamError:
    """Stream error result; Application resolves key to user message"""
    key: MessageKey

class ChatService:
    """
    Process user chat messages and stream AI responses with session and history

    Responsibilities:
    - Isolate UI from AI state and API calls
    - Apply business rules (input validation, context window, session expiry)
    - Stream response and persist history; surface user-friendly errors
    """

    MAX_INPUT_LENGTH = 2000
    MAX_CONTEXT_MESSAGES = 20

    def __init__(self, ai_client: LLMClient, history: ChatHistory, session: SessionManager):
        """
        Initialize ChatService with required dependencies

        Args:
            ai_client: LLM client implementation (generate_text, stream_chat)
            history: Chat history storage (add_message, load_history, clean_history)
            session: Manages lifetime of the current chat session (touch_activity, is_expired, reset)
        """
        self.ai = ai_client
        self.history = history
        self.session = session

    def process_message(self, user_input: str) -> Generator[str, None, None]:
        """
        Process a user chat message and stream the AI response

        Validates input length and session; appends message to history then streams
        chunks from the LLM. Errors are yielded as user-facing messages

        Args:
            user_input: Raw message text from the user

        Yields:
            Chunks of the AI response, or an error message string on failure
        """
        user_input = user_input.strip()

        if not user_input:
            logger.warning("Empty user input received in ChatService")
            yield "Vui lòng nhập nội dung tin nhắn."
            return
        
        if len(user_input) > self.MAX_INPUT_LENGTH:
            logger.warning(f"Input too long: {len(user_input)} chars")
            yield f"Tin nhắn quá dài. Vui lòng giới hạn trong {self.MAX_INPUT_LENGTH} kí tự"
            return
        
        if self.session.is_expired():
            logger.info("Session expired due to inactivity. Resetting history")
            self.history.clean_history()
            self.session.reset()
            yield "Phiên làm việc đã hết hạn do không hoạt động. Đã bắt đầu phiên mới"

        self.session.touch_activity()

        user_message = ChatMessage(role="user", content=user_input)
        self.history.add_message(user_message)

        yield from self._stream_response(user_input)

    def _stream_error_key(self, error: Exception) -> MessageKey:
        """Map streaming exception to MessageKey error code"""
        if isinstance(error, LLMServiceError):
            logger.error(f"AI Service error: {error}")
            return MessageKey.LLM_ERROR
        logger.error(f"Unexpected Chat Error: {error}")
        return MessageKey.UNEXPECTED_ERROR

    def _stream_response(self, user_input: str) -> Generator[Union[str, StreamError], None, None]:
        """
        Handle the LLM streaming lifecycle: load history, stream response, save result

        Args:
            user_input: The user's message that triggered that response

        Yields:
            str: Chunks of the AI-generated response as they arrive
        """
        raw_history: List[ChatMessage] = self.history.load_history()
        if len(raw_history) > 1:
            history_context = raw_history[:-1]
        else:
            history_context = []

        if len(history_context) > self.MAX_CONTEXT_MESSAGES:
            history_context = history_context[-self.MAX_CONTEXT_MESSAGES :]

        if not history_context:
            logger.info("No usable history context for this request")

        logger.info(f"Processing chat. Context length: {len(history_context)}")

        full_response = ""
        error_occurred = False

        try:
            stream_generation = self.ai.stream_chat(
                history=history_context,
                new_message=user_input   
            )

            for chunk in stream_generation:
                full_response += chunk
                yield chunk

        except LLMServiceError as e:
            error_key = self._stream_error_key(e)
            yield StreamError(key=error_key)
            error_occurred = True

        except Exception as e:
            error_key = self._stream_error_key(e)
            yield StreamError(key=error_key)
            error_occurred = True

        finally:
            if full_response and not error_occurred:
                bot_message = ChatMessage(role="assistant", content=full_response)
                self.history.add_message(bot_message)
                logger.info(f"Saved bot response (Length: {len(full_response)}), Error: {error_occurred}")
            elif error_occurred:
                logger.warning("Error occurred, not saving to history")