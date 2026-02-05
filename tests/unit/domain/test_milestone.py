"""
test_milestone.py

Unit tests for Milestone model (week, topic, resources, validation constraints)
"""
import pytest
from pydantic import ValidationError

from domain import Milestone, Resource

class TestMilestone:
    """Tests for Milestone model"""

    def test_create_milestone_minimal(self):
        """Milestone with minimal required fields is valid"""
        milestone = Milestone(
            week=1,
            topic="Python Basics",
            description="Learn Python basics",
            resources=[
                Resource(
                    title="Tutorial",
                    url="https://example.com",
                    type="documentation"
                )
            ]
        )

        assert milestone.week == 1
        assert milestone.topic == "Python Basics"
        assert len(milestone.resources) == 1
        assert milestone.estimated_time is None
        assert milestone.learning_objectives is None

    def test_create_milestone_full(self):
        """Milestone with all fields (estimated_time, learning_objectives) is valid"""
        milestone = Milestone(
            week=1,
            topic="Python Basics",
            description="Learn Python basics",
            resources=[
                Resource(
                    title="Tutorial",
                    url="https://example.com",
                    type="documentation"
                )
            ],
            estimated_time="5 giờ",
            learning_objectives=["Understand variables", "Learn functions"]
        )

        assert milestone.estimated_time == "5 giờ"
        assert len(milestone.learning_objectives) == 2

    def test_milestone_week_validation(self):
        """week must be >= 1; invalid raises ValidationError"""
        # Valid week
        milestone = Milestone(
            week=1,
            topic="Test",
            description="Test description",
            resources=[
                Resource(
                    title="Test resource",
                    url="https://example.com",
                    type="documentation"
                )
            ]
        )

        assert milestone.week == 1

        # Week < 1 should raise ValidationError
        with pytest.raises(ValidationError):
            Milestone(
                week=0,
                topic="Test",
                description="Test",
                resources=[
                    Resource(
                        title="Test resource",
                        url="https://example.com",
                        type="documentation"
                    )
                ]
            )

    def test_milestone_resources_min_length(self):
        """resources list must have at least 1 item; empty raises ValidationError"""
        # Valid: 1 resource
        milestone = Milestone(
            week=1,
            topic="Test",
            description="Test",
            resources=[
                Resource(
                    title="Test resource",
                    url="https://example.com",
                    type="documentation"
                )
            ]
        )
        assert len(milestone.resources) == 1

        # Empty resource list should raise ValidationError
        with pytest.raises(ValidationError):
            Milestone(
                week=1,
                topic="Test",
                description="Test",
                resources=[]
            )

    def test_milestone_topic_max_length(self):
        """topic max_length (200) is enforced; excess raises ValidationError"""
        # Valid length
        milestone = Milestone(
            week=1,
            topic="a" * 200,
            description="Test",
            resources=[
                Resource(
                    title="Test",
                    url="https://example.com",
                    type="documentation"
                )
            ]
        )
        assert len(milestone.topic) == 200

        # Exceeds max_length should raise ValidationError
        with pytest.raises(ValidationError):
            Milestone(
                week=1,
                topic="a"*201,
                description="Test",
                resources=[
                    Resource(
                        title="Test",
                        url="https://example.com",
                        type="documentation"
                    )
                ]
            )

    def test_milestone_description_max_length(self):
        """description max_length (1000) is enforced; excess raises ValidationError"""
        # Valid length
        milestone = Milestone(
            week=1,
            topic="Test",
            description="a"*1000,
            resources=[
                Resource(
                    title="Test",
                    url="https://example.com",
                    type="documentation"
                )
            ]
        )
        assert len(milestone.description) == 1000

        # Exceeds max_length should raise ValidationError
        with pytest.raises(ValidationError):
            Milestone(
                week=1,
                topic="Test",
                description="a"*1001,
                resources=[
                    Resource(
                        title="Test",
                        url="https://example.com",
                        type="documentation"
                    )
                ]
            )
