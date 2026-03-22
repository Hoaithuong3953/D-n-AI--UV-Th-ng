"""
test_snapshots.py

Unit tests for domain.snapshots: ProfileSnapshot, RoadmapSnapshot, from_user_profile, from_roadmap
"""
from dataclasses import FrozenInstanceError

import pytest

from domain import Milestone, Resource, Roadmap, UserProfile
from domain.snapshots import (
    ProfileSnapshot,
    from_roadmap,
    from_user_profile,
)

class TestFromUserProfile:
    """from_user_profile builds immutable ProfileSnapshot"""

    def test_maps_required_and_optional_fields(self):
        profile = UserProfile(
            goal="Học Python",
            current_level="beginner",
            time_commitment="1 giờ",
            learning_style="video",
            background="none",
            constraints=["miễn phí"],
        )
        snap = from_user_profile(profile)

        assert snap.goal == "Học Python"
        assert snap.current_level == "beginner"
        assert snap.time_commitment == "1 giờ"
        assert snap.learning_style == "video"
        assert snap.background == "none"
        assert snap.constraints == ("miễn phí",)

    def test_constraints_none_when_profile_has_none(self):
        profile = UserProfile(
            goal="g",
            current_level="intermediate",
            time_commitment="t",
            constraints=None,
        )
        assert from_user_profile(profile).constraints is None

    def test_profile_snapshot_is_frozen(self):
        profile = UserProfile(
            goal="g",
            current_level="advanced",
            time_commitment="t",
        )
        snap = from_user_profile(profile)
        with pytest.raises(FrozenInstanceError):
            snap.goal = "other"

class TestFromRoadmap:
    """from_roadmap builds RoadmapSnapshot aligned with Roadmap aggregate"""

    def _minimal_roadmap(self) -> Roadmap:
        return Roadmap(
            topic="Learn Python",
            duration_week=1,
            milestones=[
                Milestone(
                    week=1,
                    topic="W1",
                    description="D1",
                    resources=[
                        Resource(
                            title="R1",
                            url="https://example.com/r1",
                            type="documentation",
                            description="desc",
                            difficulty="beginner",
                        )
                    ],
                    estimated_time="5h",
                    learning_objectives=["obj1", "obj2"],
                )
            ],
            title="Title",
            description="Desc",
            prerequisites=["pre"],
        )

    def test_maps_topic_metadata_and_created_at(self):
        roadmap = self._minimal_roadmap()
        snap = from_roadmap(roadmap)

        assert snap.topic == "Learn Python"
        assert snap.title == "Title"
        assert snap.description == "Desc"
        assert snap.duration_week == 1
        assert snap.prerequisites == ("pre",)
        assert snap.created_at == roadmap.created_at

    def test_milestones_and_resources(self):
        roadmap = self._minimal_roadmap()
        snap = from_roadmap(roadmap)

        assert len(snap.milestones) == 1
        ms = snap.milestones[0]
        assert ms.week == 1
        assert ms.topic == "W1"
        assert ms.description == "D1"
        assert ms.estimated_time == "5h"
        assert ms.learning_objectives == ("obj1", "obj2")

        assert len(ms.resources) == 1
        rs = ms.resources[0]
        assert rs.title == "R1"
        assert rs.url == "https://example.com/r1"
        assert rs.type == "documentation"
        assert rs.description == "desc"
        assert rs.difficulty == "beginner"

    def test_prerequisites_none_when_roadmap_has_none(self):
        roadmap = Roadmap(
            topic="T",
            duration_week=1,
            milestones=[
                Milestone(
                    week=1,
                    topic="W",
                    description="D",
                    resources=[
                        Resource(
                            title="R",
                            url="https://example.com/x",
                            type="article",
                        )
                    ],
                )
            ],
            prerequisites=None,
        )
        assert from_roadmap(roadmap).prerequisites is None

    def test_milestone_without_learning_objectives(self):
        roadmap = Roadmap(
            topic="T",
            duration_week=1,
            milestones=[
                Milestone(
                    week=1,
                    topic="W",
                    description="D",
                    resources=[
                        Resource(title="R", url="https://example.com/x", type="book")
                    ],
                )
            ],
        )
        ms = from_roadmap(roadmap).milestones[0]
        assert ms.learning_objectives is None

def test_profile_snapshot_constructible_directly():
    """ProfileSnapshot dataclass fields match domain usage"""
    snap = ProfileSnapshot(
        goal="g",
        current_level="beginner",
        time_commitment="t",
    )
    assert snap.learning_style is None
