"""
Chat flow handlers for AppService
"""
from __future__ import annotations

from typing import Generator, TYPE_CHECKING

from domain import ChatMessage
from domain.events import Event, TextChunk, ErrorOccurred
from config import MessageKey
from utils import logger
from services.chat_service import StreamError

if TYPE_CHECKING:
    from services.app_service import AppService

def handle_chat_request(app: AppService, user_input: str) -> Generator[Event, None, None]:
    """
    Chat request handler: stream chat response, yield TextChunk and ErrorOccurred events
        
    Args:
        user_input: User message input

    Yields:
        TextChunk: Response text chunks
        ErrorOccurred: On LLM or unexpected error
    """
    logger.info("chat request start")
    full_response = ""
    history = app._get_recent_history()
    for item in app._chat.stream_response(user_input, history):
        if isinstance(item, str):
            full_response += item
            yield TextChunk(item)
        elif isinstance(item, StreamError):
            msg = app.messages.get(item.key)
            error_type = "llm" if item.key == MessageKey.LLM_ERROR else "unexpected"
            yield ErrorOccurred(error_type, msg)
            app._memory.add_message(ChatMessage(role="assistant", content=msg))
            return

    if full_response:
        app._memory.add_message(ChatMessage(role="assistant", content=full_response))
    logger.info(f"chat request done (response_len={len(full_response)})")