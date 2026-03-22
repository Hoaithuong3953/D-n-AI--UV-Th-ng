"""
Unit tests for services.flows.roadmap_flow helpers and generate_roadmap
"""
from unittest.mock import MagicMock

from config.messages import DefaultMessageProvider, MessageKey
from domain import ChatMessage, ConversationState, Milestone, Resource, Roadmap, UserProfile
from domain.events import ErrorOccurred, ProfileExtracted, RoadmapCreated, StatusUpdate, TextChunk
from memory import ChatMemory
from services.flows import roadmap_flow

class TestMissingFieldPrompt:
    def test_empty_missing_uses_fill_profile_template(self):
        app = MagicMock()
        app.messages = DefaultMessageProvider()
        msg = roadmap_flow._missing_field_prompt(app, [])
        assert msg == app.messages.get(MessageKey.FILL_PROFILE)

    def test_missing_goal_and_level_lists_numbered_lines(self):
        app = MagicMock()
        app.messages = DefaultMessageProvider()
        msg = roadmap_flow._missing_field_prompt(app, ["goal", "level"])
        assert "1." in msg and "2." in msg

class TestGenerateRoadmap:
    def test_success_yields_status_roadmap_created_and_chunk(self):
        app = MagicMock()
        app.current_profile = UserProfile(
            goal="g",
            current_level="beginner",
            time_commitment="1h",
        )
        app.messages = DefaultMessageProvider()
        app._memory = ChatMemory()

        rm = Roadmap(
            topic="T",
            duration_week=1,
            milestones=[
                Milestone(
                    week=1,
                    topic="w",
                    description="d",
                    resources=[
                        Resource(title="r", url="https://example.com", type="documentation")
                    ],
                )
            ],
        )
        app._roadmap.generate_roadmap.return_value = rm
        app._get_recent_history.return_value = [
            ChatMessage(role="user", content="học trong 2 tháng"),
        ]

        events = list(roadmap_flow.generate_roadmap(app))

        assert any(isinstance(e, StatusUpdate) for e in events)
        assert any(isinstance(e, RoadmapCreated) for e in events)
        assert any(isinstance(e, TextChunk) for e in events)
        app._roadmap.generate_roadmap.assert_called_once()
        call_kw = app._roadmap.generate_roadmap.call_args[1]
        assert call_kw["duration_week"] == 8

    def test_llm_error_yields_error_occurred(self):
        from utils import LLMServiceError

        app = MagicMock()
        app.current_profile = UserProfile(
            goal="g",
            current_level="beginner",
            time_commitment="1h",
        )
        app.messages = DefaultMessageProvider()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = []
        app._roadmap.generate_roadmap.side_effect = LLMServiceError(message="x")

        events = list(roadmap_flow.generate_roadmap(app))
        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert errs and errs[0].error_type == "llm"

class TestHandleRoadmapRequest:
    def test_no_profile_extractor_llm_error(self):
        from utils import LLMServiceError

        app = MagicMock()
        app.current_profile = None
        app.messages = DefaultMessageProvider()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = [ChatMessage(role="user", content="hi")]
        app._profile_extractor.extract.side_effect = LLMServiceError(message="fail")

        events = list(roadmap_flow.handle_roadmap_request(app))
        assert any(isinstance(e, ErrorOccurred) and e.error_type == "llm" for e in events)

    def test_no_profile_after_extract_sets_awaiting_state(self):
        app = MagicMock()
        app.current_profile = None
        app.messages = DefaultMessageProvider()
        app._memory = ChatMemory()
        app._get_recent_history.return_value = [ChatMessage(role="user", content="x")]
        app._profile_extractor.extract.return_value = None
        app._profile_extractor.last_missing_fields = ["goal"]

        list(roadmap_flow.handle_roadmap_request(app))

        assert app.conversation_state == ConversationState.AWAITING_PROFILE_INFO
