"""
Unit tests for services.support.profile_inference heuristics
"""
import pytest

from services.support import profile_inference as pi

class TestInferGoal:
    def test_python_marker(self):
        assert pi.infer_goal_from_user_text("I want to learn python basics") == "Python"

    def test_no_match(self):
        assert pi.infer_goal_from_user_text("12345 99999") is None

class TestInferLevel:
    def test_beginner_vietnamese(self):
        assert pi.infer_level_from_user_text("tôi mới bắt đầu") == "beginner"

    def test_intermediate(self):
        assert pi.infer_level_from_user_text("intermediate level") == "intermediate"

    def test_advanced(self):
        assert pi.infer_level_from_user_text("nâng cao") == "advanced"

class TestInferTimeCommitment:
    def test_hours_per_day(self):
        assert pi.infer_time_commitment_from_user_text("2 giờ/ngày") == "2 giờ/ngày"

    def test_minutes_per_day(self):
        assert pi.infer_time_commitment_from_user_text("30 phút mỗi ngày") == "30 phút/ngày"

class TestInferDurationWeeks:
    def test_months_to_weeks(self):
        assert pi.infer_duration_weeks_from_user_text("trong 3 tháng") == 12

    def test_weeks_literal(self):
        assert pi.infer_duration_weeks_from_user_text("8 tuần") == 8

    def test_none_when_absent(self):
        assert pi.infer_duration_weeks_from_user_text("no duration here") is None

class TestResolveHelpers:
    def test_resolve_goal_prefers_raw(self):
        assert pi.resolve_goal("  My goal  ", "python") == "My goal"

    def test_resolve_goal_fallback_infer(self):
        assert pi.resolve_goal(None, "học javascript") == "JavaScript"

    def test_resolve_level_from_user_text(self):
        assert pi.resolve_level(None, "beginner here") == "beginner"

    def test_resolve_time_prefers_raw(self):
        assert pi.resolve_time_commitment("1 giờ", "") == "1 giờ"
