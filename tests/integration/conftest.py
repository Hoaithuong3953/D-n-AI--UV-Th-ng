"""
conftest.py

Fixtures for integration tests (chat flow)

Key features:
- fake_llm_client: factory to create a fake GeminiClient with arbitrary chunks
- app_service: factory to create AppService with arbitrary LLM + config
"""
import pytest
from unittest.mock import MagicMock

from services import AppService, ChatService, SessionManager
from ai import GeminiClient
from memory import ChatMemory
from config import default_messages, DEFAULT_CONTEXT_MESSAGES

@pytest.fixture
def fake_llm_client():
    """
    Fake LLM client factory
    - Accepts a list of chunks
    - Returns a GeminiClient fake stream with the correct chunks
    """
    def _factory(chunks):
        mock = MagicMock(spec=GeminiClient)

        def fake_stream_chat(history, new_message):
            for c in chunks:
                yield c

        mock.stream_chat = fake_stream_chat
        return mock
    return _factory

@pytest.fixture
def app_service():
    """
    AppService factory:

    Usage:
        app = app_service(
            llm_client=custom_mock,
            timeout_minutes=0,
            chat_context_messages=2,
        )
    """
    def _factory(
        llm_client,
        timeout_minutes: int = 30,
        chat_context_messages=DEFAULT_CONTEXT_MESSAGES,
    ):
        chat_service = ChatService(llm_client=llm_client)
        session_manager = SessionManager(timeout_minutes=timeout_minutes)
        memory = ChatMemory()

        return AppService(
            chat_service=chat_service,
            session_manager=session_manager,
            messages=default_messages,
            memory=memory,
            chat_context_messages=chat_context_messages,
        )
    return _factory