"""
Unit tests for services.core.roadmap_service.RoadmapService
"""
import json
from unittest.mock import MagicMock

import pytest

from domain import Milestone, Resource, Roadmap, UserProfile
from services.core.roadmap_service import RoadmapService
from utils import LLMServiceError, ValidationError
from config.messages import MessageKey

def _minimal_roadmap_dict():
    return {
        "topic": "Python",
        "duration_week": 1,
        "milestones": [
            {
                "week": 1,
                "topic": "Week 1",
                "description": "Basics",
                "resources": [
                    {
                        "title": "Docs",
                        "url": "https://example.com",
                        "type": "documentation",
                    }
                ],
            }
        ],
    }


class TestRoadmapServiceGenerate:
    def test_generate_roadmap_success_first_attempt(self):
        llm = MagicMock()
        llm.generate_text.return_value = json.dumps(_minimal_roadmap_dict())
        svc = RoadmapService(llm_client=llm, max_retries=2)

        profile = UserProfile(
            goal="Học Python",
            current_level="beginner",
            time_commitment="1 giờ/ngày",
        )
        rm = svc.generate_roadmap(profile, duration_week=1)

        assert isinstance(rm, Roadmap)
        assert rm.topic == "Python"
        assert rm.duration_week == 1
        llm.generate_text.assert_called_once()

    def test_generate_roadmap_retries_on_invalid_json_then_success(self):
        llm = MagicMock()
        good = json.dumps(_minimal_roadmap_dict())
        llm.generate_text.side_effect = ["not json {", good]
        svc = RoadmapService(llm_client=llm, max_retries=2)

        profile = UserProfile(
            goal="g",
            current_level="beginner",
            time_commitment="1h",
        )
        rm = svc.generate_roadmap(profile, duration_week=1)
        assert rm.topic == "Python"
        assert llm.generate_text.call_count == 2

    def test_generate_roadmap_raises_after_max_retries(self):
        llm = MagicMock()
        llm.generate_text.return_value = "not valid json"
        svc = RoadmapService(llm_client=llm, max_retries=2)

        profile = UserProfile(
            goal="g",
            current_level="beginner",
            time_commitment="1h",
        )
        with pytest.raises(ValidationError) as exc_info:
            svc.generate_roadmap(profile, duration_week=1)
        assert exc_info.value.code == MessageKey.ROADMAP_GENERATION_FAILED.value
        assert llm.generate_text.call_count == 2

    def test_generate_roadmap_wraps_llm_failure_after_retries(self):
        """LLM failures are retried; final error is ValidationError ROADMAP_GENERATION_FAILED"""
        llm = MagicMock()
        llm.generate_text.side_effect = LLMServiceError(message="down")
        svc = RoadmapService(llm_client=llm, max_retries=2)

        profile = UserProfile(
            goal="g",
            current_level="beginner",
            time_commitment="1h",
        )
        with pytest.raises(ValidationError) as exc_info:
            svc.generate_roadmap(profile, duration_week=1)
        assert exc_info.value.code == MessageKey.ROADMAP_GENERATION_FAILED.value
        assert llm.generate_text.call_count == 2

class TestRoadmapServicePrompt:
    def test_build_prompt_includes_profile_fields(self):
        llm = MagicMock()
        llm.generate_text.return_value = json.dumps(_minimal_roadmap_dict())
        svc = RoadmapService(llm_client=llm)

        profile = UserProfile(
            goal="Mục tiêu X",
            current_level="intermediate",
            time_commitment="2 giờ",
            learning_style="video",
            background="none",
            constraints=["free"],
        )
        svc.generate_roadmap(profile, duration_week=6)
        prompt = llm.generate_text.call_args[0][0]
        assert "Mục tiêu X" in prompt
        assert "intermediate" in prompt
        assert "2 giờ" in prompt
        assert "6" in prompt