"""
Unit test for UserProfile model
"""
import pytest
from pydantic import ValidationError

from domain import UserProfile

class TestUserProfile:
    """Test for UserProfile model"""
    def test_create_user_profile_minimal(self):
        profile = UserProfile(
            goal="Learn Python",
            current_level="beginner",
            time_commitment="2 giờ/ngày"
        )
        assert profile.goal == "Learn Python"
        assert profile.current_level == "beginner"
        assert profile.time_commitment == "2 giờ/ngày"
        assert profile.learning_style is None
        assert profile.background is None
        assert profile.constraints is None

    def test_create_user_profile_full(self):
        """Test creating user profile with all fields"""
        profile = UserProfile(
            goal="Learn Python",
            current_level="intermedia",
            time_commitment="10 hours/week",
            learning_style="Visual",
            background="Software engineer",
            constraints=["Free only", "Weekends only"]
        )
        assert profile.learning_style == "Visual"
        assert profile.background == "Software engineer"
        assert len(profile.constraints) == 2

    def test_user_profile_goal_max_length(self):
        """Test that goal max_length constraint is enforced"""
        # Valid length
        profile = UserProfile(
            goal="A"*500,
            current_level="beginner",
            time_commitment="2 hours/day"
        )
        assert len(profile.goal) == 500

        # Exceeds max_length should raise ValidationError
        with pytest.raises(ValidationError):
            UserProfile(
                goal="A"*501,
                current_level="beginner",
                time_commitment="2 hours/day"
            )
