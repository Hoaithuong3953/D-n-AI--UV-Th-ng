from typing import Generator, List

from utils import logger, LLMServiceError
from ai import LLMClient
from memory import ChatHistory
from domain import ChatMessage
from .session_manager import SessionManager

class ChatService:
    """
    Service Layer for Chat Operations
    Responsibility: 
    - Isolate the UI from the complexity of managing AI state and API calls
    - Apply business rules around input validation, context window and error handling
    """

    MAX_INPUT_LENGTH = 2000

    def __init__(self, ai_client: LLMClient, history: ChatHistory, session: SessionManager):
        """
        Initialize ChatService with required dependencies

        Args:
            ai_client: LLM Client implementation
            history: Chat history storage
            session: Manages the lifetime of the current chat session
        """
        self.ai = ai_client
        self.history = history
        self.session = session

    def process_message(self, user_input: str) -> Generator[str, None, None]:
        """
        Processing a user's chat message and stream the AI response

        Args:
            user_input: The user's message text

        Yields:
            str: Chunks of the AI response as they are generated
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

    def _handle_stream_error(self, error: Exception, error_type: str) -> str:
        """
        Handle streaming errors and return user-friendly error messages

        Args:
            error: The exception that occurred during streaming
            error_type: Type of error for routing

        Returns:
            str: User-friendly error message formatted for display
        """
        if error_type == "llm_service":
            logger.error(f"AI Service error: {error}")
            return "\n\n[Lỗi kết nối: Không thể tải toàn bộ tin nhắn, vui lòng thử lại]"
        else:
            logger.error(f"Unexpected Chat Error: {error}")
            return "\n\n[Lỗi hệ thống: Đã xảy ra sự cố không mong muốn]"

    def _stream_response(self, user_input: str) -> Generator[str, None, None]:
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
            logger.info("No previous history, starting fresh conversation")

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
            error_message = self._handle_stream_error(e, "llm_service")
            full_response += error_message
            yield error_message
            error_occurred = True

        except Exception as e:
            error_message = self._handle_stream_error(e, "unexpected")
            full_response += error_message
            yield error_message
            error_occurred = True

        finally:
            if full_response and not error_occurred:
                bot_message = ChatMessage(role="assistant", content=full_response)
                self.history.add_message(bot_message)
                logger.info(f"Saved bot response (Length: {len(full_response)}), Error: {error_occurred}")
            elif error_occurred:
                logger.warning("Error occurred, not saving to history")