from typing import Generator

from utils.logger import logger
from utils.exceptions import LLMServiceError
from ai.llm_client import LLMClient
from memory.chat_history import ChatHistory
from domain.models import ChatMessage

class ChatService:
    """
    Service Layer for Chat Operations.
    Responsibility: Isolate the UI from the complexity of managing AI state and API calls.
    """
    def __init__(self, ai_client: LLMClient, history: ChatHistory):
        """
        Initialize ChatService with AI client and memory storage

        Args:
            ai_client: LLM Client implementation
            history: Chat history storage
        """
        self.ai = ai_client
        self.history = history

    def process_message(self, user_input: str) -> Generator[str, None, None]:
        """
        Processing a user's chat message and stream the AI response

        Args:
            user_input: The user's message text

        Yields:
            str: Chunks of the AI response as they are generated
        """
        if not user_input or not user_input.strip():
            logger.warning("Empty user input received in ChatService")
            yield "Vui lòng nhập nội dung tin nhắn."
            return

        user_message = ChatMessage(role="user", content=user_input)
        self.history.add_message(user_message)

        yield from self._stream_response(user_input)

    def _handle_stream_error(self, error: Exception, error_type: str) -> str:
        """
        Handle streaming errors and return user-friendly error messages

        Args:
            error: The exception that occured during streaming
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
            user_input: The user's message that trigged that response

        Yields:
            str: Chunks of the AI-generated response as they arrive
        """
        raw_history = self.history.load_history()
        history_context = raw_history[:-1]

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
            if full_response:
                bot_message = ChatMessage(role="assistant", content=full_response)
                self.history.add_message(bot_message)
                logger.info(f"Saved bot response (Length: {len(full_response)}), Error: {error_occurred}")