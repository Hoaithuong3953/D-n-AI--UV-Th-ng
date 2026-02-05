"""
test_roadmap.py

Unit tests for Roadmap model (topic, title, duration_week, milestones, validation)
"""
import pytest
from pydantic import ValidationError
from datetime import datetime

from domain import Roadmap, Milestone, Resource

class TestRoadmap:
    """Tests for Roadmap model"""

    def test_create_roadmap_minimal(self):
        """Roadmap with minimal required fields is valid"""
        roadmap = Roadmap(
            topic="Learn Python",
            duration_week=1,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
            ]
        )

        assert roadmap.topic == "Learn Python"
        assert roadmap.duration_week == 1
        assert roadmap.description is None
        assert roadmap.prerequisites is None
        assert isinstance(roadmap.created_at, datetime)

    def test_create_roadmap_full(self):
        """Roadmap with all fields (title, description, prerequisites) is valid"""
        roadmap = Roadmap(
            topic="Learn Python",
            title="Python Learning Path",
            description="Complete Python roadmap",
            duration_week=2,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
                Milestone(week=2, topic="W2", description="D2", resources=[
                    Resource(title="R2", url="https://example.com/2", type="documentation")
                ]),
            ],
            prerequisites=["Basic programming knowledge"]
        )

        assert roadmap.title == "Python Learning Path"
        assert roadmap.description == "Complete Python roadmap"
        assert len(roadmap.prerequisites) == 1

    def test_roadmap_auto_set_title(self):
        """title is auto-set to topic when not provided; provided title is kept"""
        # Title not provided
        roadmap = Roadmap(
            topic="Learn Python",
            duration_week=1,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
            ]
        )
        assert roadmap.title == "Learn Python"

        # Title provided
        roadmap2 = Roadmap(
            topic="Learn Python",
            title="Custom title",
            duration_week=1,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
            ]
        )
        assert roadmap2.title == "Custom title"

    def test_roadmap_sequential_weeks(self):
        """milestone weeks must be sequential from 1; mismatch raises ValidationError"""
        # Valid: Sequential weeks
        roadmap = Roadmap(
            topic="Test",
            duration_week=3,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
                Milestone(week=2, topic="W2", description="D2", resources=[
                    Resource(title="R2", url="https://example.com/2", type="documentation")
                ]),
                Milestone(week=3, topic="W3", description="D3", resources=[
                    Resource(title="R3", url="https://example.com/3", type="documentation")
                ])
            ]
        )
        assert len(roadmap.milestones) == 3

        # Invalid: Non-sequential weeks
        with pytest.raises(ValidationError) as exc_info:
            Roadmap(
                topic="Test",
                duration_week=3,
                milestones=[
                    Milestone(week=1, topic="W1", description="D1", resources=[
                        Resource(title="R1", url="https://example.com/1", type="documentation")
                    ]),
                    Milestone(week=3, topic="W3", description="D3", resources=[
                        Resource(title="R3", url="https://example.com/3", type="documentation")
                    ])
                ]
            )
        assert "sequential" in str(exc_info.value).lower()

        # Invalid: Starting from wrong number
        with pytest.raises(ValidationError):
            Roadmap(
                topic="Test",
                duration_week=2,
                milestones=[
                    Milestone(week=2, topic="W2", description="D2", resources=[
                        Resource(title="R2", url="https://example.com/2", type="documentation")
                    ]),
                    Milestone(week=3, topic="W3", description="D3", resources=[
                        Resource(title="R3", url="https://example.com/3", type="documentation")
                    ])
                ]
            )
    
    def test_roadmap_milestone_count_matches_duration(self):
        """Number of milestones must match duration_week; mismatch raises ValidationError"""
        # Valid: Count matches
        roadmap = Roadmap(
            topic="Test",
            duration_week=3,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
                Milestone(week=2, topic="W2", description="D2", resources=[
                    Resource(title="R2", url="https://example.com/2", type="documentation")
                ]),
                Milestone(week=3, topic="W3", description="D3", resources=[
                    Resource(title="R3", url="https://example.com/3", type="documentation")
                ])
            ]
        )
        assert len(roadmap.milestones) == roadmap.duration_week

        # Invalid: Count doesn't match
        with pytest.raises(ValidationError) as exc_info:
            Roadmap(
                topic="Test",
                duration_week=2,
                milestones=[
                    Milestone(week=1, topic="W1", description="D1", resources=[
                        Resource(title="R1", url="https://example.com/1", type="documentation")
                    ]),
                    Milestone(week=2, topic="W2", description="D2", resources=[
                        Resource(title="R2", url="https://example.com/2", type="documentation")
                    ]),
                    Milestone(week=3, topic="W3", description="D3", resources=[
                        Resource(title="R3", url="https://example.com/3", type="documentation")
                    ])
                ]
            )
        assert "must match duration_week" in str(exc_info.value)

    def test_roadmap_duration_week_validation(self):
        """duration_week must be >= 1; invalid raises ValidationError"""
        # Valid duration
        roadmap = Roadmap(
            topic="Test",
            duration_week=1,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
            ]
        )
        assert roadmap.duration_week == 1

        # Invalid: duration_week < 1
        with pytest.raises(ValidationError):
            Roadmap(
                topic="Test",
                duration_week=0,
                milestones=[
                    Milestone(week=1, topic="W1", description="D1", resources=[
                        Resource(title="R1", url="https://example.com/1", type="documentation")
                    ]),
                ]
            )

    def test_roadmap_milestones_min_length(self):
        """milestones list must have at least 1 item; empty raises ValidationError"""
        # Valid: 1 milestone
        roadmap = Roadmap(
            topic="Test",
            duration_week=1,
            milestones=[
                Milestone(week=1, topic="W1", description="D1", resources=[
                    Resource(title="R1", url="https://example.com/1", type="documentation")
                ]),
            ]
        )
        assert len(roadmap.milestones) == 1

        # Invalid: Empty milestones
        with pytest.raises(ValidationError):
            Roadmap(
                topic="Test",
                duration_week=1,
                milestones=[]
            )
