"""
test_roadmap_flow_integration.py

Integration tests for roadmap flow (ROADMAP intent, profile, roadmap generation)

Tests:
- ProfileExtracted and RoadmapCreated events after successful generate_text
- Validation ErrorOccurred when roadmap generation fails after retries
"""
from unittest.mock import MagicMock

from ai import GeminiClient
from domain import ConversationState
from domain.events import ErrorOccurred, ProfileExtracted, RoadmapCreated, StatusUpdate
from utils import LLMServiceError

class TestRoadmapFlowIntegration:
    """Integration tests for roadmap flow via AppService"""

    def test_roadmap_intent_yields_profile_and_roadmap_events(
        self, app_service, fake_llm_client, sample_roadmap_llm_response_first
    ):
        """ROADMAP intent with rule-based profile and sample roadmap JSON from fixtures"""
        user_input = "Tôi muốn học Python, mới bắt đầu, 1 giờ/ngày"
        llm = fake_llm_client(
            [],
            generate_text_return=sample_roadmap_llm_response_first,
        )
        app = app_service(llm_client=llm)
        events = list(app.handle_message(user_input))

        assert any(isinstance(e, StatusUpdate) for e in events)
        assert any(isinstance(e, ProfileExtracted) for e in events)
        assert any(isinstance(e, RoadmapCreated) for e in events)

        assert app.current_profile is not None
        assert app.current_profile.goal == "Python"
        assert app.current_roadmap is not None
        assert app.current_roadmap.topic == "Học Python cơ bản"
        assert app.current_roadmap.duration_week == 4

        llm.generate_text.assert_called()

    def test_roadmap_generation_failure_yields_validation_error(
        self, app_service, fake_llm_client
    ):
        """generate_text keeps failing -> RoadmapService exhausts retries -> ErrorOccurred validation"""
        user_input = "Tôi muốn học Python, mới bắt đầu, 1 giờ/ngày"
        llm = fake_llm_client(
            [],
            generate_text_side_effect=LLMServiceError(message="roadmap failed"),
        )
        app = app_service(llm_client=llm)
        events = list(app.handle_message(user_input))

        errs = [e for e in events if isinstance(e, ErrorOccurred)]
        assert len(errs) >= 1
        assert any(e.error_type == "validation" for e in errs)

    def test_roadmap_profile_extract_llm_error_yields_llm_error_occurred(self, app_service):
        """ROADMAP intent -> profile extract calls generate_text; LLM failure -> ErrorOccurred(llm), state stays NORMAL"""
        mock_llm = MagicMock(spec=GeminiClient)
        mock_llm.generate_text.side_effect = LLMServiceError(
            code="PROFILE_EXTRACT",
            message="extract failed",
        )
        app = app_service(llm_client=mock_llm)
        events = list(app.handle_message("Tôi muốn tạo lộ trình"))

        llm_errs = [
            e for e in events if isinstance(e, ErrorOccurred) and e.error_type == "llm"
        ]
        assert len(llm_errs) == 1
        assert app.conversation_state == ConversationState.NORMAL
        assert app.current_profile is None
        history = app._memory.load_history()
        assert len(history) == 2
        assert history[1].role == "assistant"