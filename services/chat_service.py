"""
chat_service.py

Chat service: load history, stream LLM response

Key features:
- stream_response(user_input) yields str chunks or StreamError(key); Application resolves key to message
- No MessageProvider; facade owns message resolution
"""
from typing import Generator, List, Union, Union
from dataclasses import dataclass

from utils import logger, LLMServiceError
from ai import LLMClient
from domain import ChatMessage
from config import MessageKey

@dataclass(frozen=True)
class StreamError:
    """Stream error result; Application resolves key to user message"""
    key: MessageKey

class ChatService:
    """
    Execute chat: load history, stream LLM response

    Responsibilities:
    - stream_response(user_input): yield str chunks or StreamError(key)
    - Application (facade) resolves key to message via MessageProvider
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize ChatService with required dependencies

        Args:
            llm_client: LLM client implementation for streaming chat responses
        """
        self.llm = llm_client

    def _stream_error_key(self, error: Exception) -> MessageKey:
        """Map streaming exception to MessageKey error code"""
        if isinstance(error, LLMServiceError):
            logger.error(f"AI Service error: {error}")
            return MessageKey.LLM_ERROR
        logger.error(f"Unexpected Chat Error: {error}")
        return MessageKey.UNEXPECTED_ERROR

    def stream_response(
        self, 
        user_input: str,
        history: List[ChatMessage],
    ) -> Generator[Union[str, StreamError], None, None]:
        """
        Stream chat response for the given user input

        Args:
            user_input: The user's message that triggered that response
            history: Recent chat history

        Yields:
            str: Response chunks from LLM
            StreamError: On failure; Application resolves key to user message
        """
        logger.info(f"Chat stream start (context_len={len(history)})")
        max_attempts = 2
        
        for attempt in range(1, max_attempts + 1):
            try: 
                first = True
                stream_generation = self.llm.stream_chat(
                    history=history,
                    new_message=user_input
                )

                for chunk in stream_generation:
                    if first:
                        logger.info("Chat first chunk received")
                        first = False
                    yield chunk
                if not first:
                    logger.info("Chat stream end")
                    return
            except LLMServiceError as e:
                is_quota = "429" in str(e) or "quota" in str(e).lower()
                if is_quota or attempt >= max_attempts:
                    yield StreamError(key=self._stream_error_key(e))
                    return
                logger.warning(f"Chat stream attempt {attempt} failed, retrying: {e}")
            except Exception as e:
                if attempt >= max_attempts:
                    yield StreamError(key=self._stream_error_key(e))
                    return
                logger.warning(f"Chat stream attempt {attempt} failed, retrying: {e}")