"""Shared pytest fixtures for ``tests/integration``."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai import GeminiClient
from config import DEFAULT_CONTEXT_MESSAGES, default_messages
from memory import ChatMemory
from services import AppService, ChatService, SessionManager
from services.core.roadmap_service import RoadmapService
from services.support.intent_detector import IntentDetector
from services.support.profile_extractor import ProfileExtractor

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def sample_roadmaps_data():
    return json.loads((_FIXTURES_DIR / "sample_roadmaps.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_roadmap_llm_response_first(sample_roadmaps_data):
    return json.dumps(sample_roadmaps_data[0], ensure_ascii=False)


@pytest.fixture
def fake_llm_client():
    """Returns a callable ``(chunks, *, generate_text_return=..., generate_text_side_effect=...) -> mock``."""

    def _factory(chunks, *, generate_text_return=None, generate_text_side_effect=None):
        mock = MagicMock(spec=GeminiClient)

        def fake_stream_chat(history, new_message):
            for c in chunks:
                yield c

        mock.stream_chat = fake_stream_chat
        if generate_text_side_effect is not None:
            mock.generate_text.side_effect = generate_text_side_effect
        elif generate_text_return is not None:
            mock.generate_text.return_value = generate_text_return
        return mock

    return _factory


@pytest.fixture
def app_service():
    """Returns a callable ``(llm_client, **kwargs) -> AppService``."""

    def _factory(llm_client, timeout_minutes=30, chat_context_messages=DEFAULT_CONTEXT_MESSAGES):
        return AppService(
            chat_service=ChatService(llm_client=llm_client),
            session_manager=SessionManager(timeout_minutes=timeout_minutes),
            messages=default_messages,
            memory=ChatMemory(),
            intent_detector=IntentDetector(llm_client=llm_client),
            profile_extractor=ProfileExtractor(llm_client=llm_client),
            roadmap_service=RoadmapService(llm_client=llm_client),
            chat_context_messages=chat_context_messages,
        )

    return _factory
