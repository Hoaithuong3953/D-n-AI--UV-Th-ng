"""
test_roadmap_flow_re_extraction.py

Integration tests: incomplete profile -> AWAITING_PROFILE_INFO -> re-extract -> roadmap

Tests:
- First ROADMAP message with LLM returning partial profile sets awaiting state
- Second message completes profile and generates roadmap (LLM profile + LLM roadmap)
"""
import json

from domain import ConversationState
from domain.events import ErrorOccurred, ProfileExtracted, RoadmapCreated
from utils import LLMServiceError

class TestRoadmapFlowReExtraction:
    """Profile gap flow then successful roadmap"""

    def test_awaiting_profile_then_roadmap(
        self, app_service, fake_llm_client, sample_roadmap_llm_response_first
    ):
        """Turn 1: partial JSON from extract -> validation + AWAITING; turn 2: full profile + roadmap JSON"""
        incomplete = '{"goal": "Học web"}'
        complete = json.dumps(
            {
                "goal": "Học web",
                "current_level": "beginner",
                "time_commitment": "2 giờ/ngày",
            },
            ensure_ascii=False,
        )
        llm = fake_llm_client(
            [],
            generate_text_side_effect=[
                incomplete,
                complete,
                sample_roadmap_llm_response_first,
            ],
        )
        app = app_service(llm_client=llm)

        list(app.handle_message("Tôi muốn tạo lộ trình"))

        assert app.conversation_state == ConversationState.AWAITING_PROFILE_INFO
        assert app.current_profile is None

        events2 = list(app.handle_message("beginner, 2 giờ mỗi ngày"))

        assert app.conversation_state == ConversationState.NORMAL
        assert app.current_profile is not None
        assert app.current_roadmap is not None
        assert any(isinstance(e, ProfileExtracted) for e in events2)
        assert any(isinstance(e, RoadmapCreated) for e in events2)
        assert llm.generate_text.call_count == 3

class TestRoadmapFlowLLMProfileSingleMessage:
    """ROADMAP intent without rule-based profile: LLM extract + LLM roadmap in one turn"""

    def test_roadmap_with_llm_profile_only(
        self, app_service, fake_llm_client, sample_roadmap_llm_response_first
    ):
        """Keyword ROADMAP, no fast-path; two generate_text: profile JSON then roadmap JSON"""
        profile_json = json.dumps(
            {
                "goal": "Học Rust",
                "current_level": "beginner",
                "time_commitment": "1 giờ/ngày",
            },
            ensure_ascii=False,
        )
        llm = fake_llm_client(
            [],
            generate_text_side_effect=[profile_json, sample_roadmap_llm_response_first],
        )
        app = app_service(llm_client=llm)

        events = list(
            app.handle_message(
                "Tôi muốn tạo lộ trình học ngôn ngữ mới chưa có trong danh sách goal"
            )
        )

        assert any(isinstance(e, ProfileExtracted) for e in events)
        assert any(isinstance(e, RoadmapCreated) for e in events)
        assert app.current_profile.goal == "Học Rust"
        assert llm.generate_text.call_count == 2

class TestRoadmapFlowReExtractionError:
    """LLM failure during profile re-extraction after AWAITING_PROFILE_INFO"""

    def test_profile_reextraction_llm_error(self, app_service, fake_llm_client):
        """Turn 1: incomplete profile -> AWAITING; turn 2: extract raises -> ErrorOccurred(llm), NORMAL state"""
        incomplete = '{"goal": "Học web"}'
        llm = fake_llm_client(
            [],
            generate_text_side_effect=[
                incomplete,
                LLMServiceError(code="REEXTRACT", message="re-extract failed"),
            ],
        )
        app = app_service(llm_client=llm)

        list(app.handle_message("Tôi muốn tạo lộ trình"))
        assert app.conversation_state == ConversationState.AWAITING_PROFILE_INFO

        events2 = list(app.handle_message("beginner, 2 giờ mỗi ngày"))
        llm_errs = [
            e for e in events2 if isinstance(e, ErrorOccurred) and e.error_type == "llm"
        ]
        assert len(llm_errs) == 1
        assert app.conversation_state == ConversationState.NORMAL
        assert app.current_profile is None
        assert llm.generate_text.call_count == 2