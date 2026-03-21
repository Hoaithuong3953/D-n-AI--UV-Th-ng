"""
snapshots.py

Immutable snapshots for domain events

"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple, Literal

if TYPE_CHECKING:
    from domain.roadmap import Roadmap
    from domain.user import UserProfile

@dataclass(frozen=True)
class ProfileSnapshot:
    """Immutable snapshot of user profile at extraction time"""
    goal: str
    current_level: Literal["beginner", "intermediate", "advanced"]
    time_commitment: str
    learning_style: Optional[str] = None
    background: Optional[str] = None
    constraints: Optional[Tuple[str, ...]] = None

@dataclass(frozen=True)
class ResourceSnapshot:
    """Immutable snapshot of learning resource"""
    title: str
    url: str
    type: Literal[
        "video",
        "article",
        "book",
        "course",
        "practice",
        "project",
        "documentation"
    ]
    description: Optional[str] = None
    difficulty: Optional[
        Literal["beginner", "intermediate", "advanced"]
    ] = None

@dataclass(frozen=True)
class MilestoneSnapshot:
    """Immutable snapshot of roadmap milestone"""
    week: int
    topic: str
    description: str
    resources: Tuple[ResourceSnapshot, ...]
    estimated_time: Optional[str] = None
    learning_objectives: Optional[Tuple[str, ...]] = None

@dataclass(frozen=True)
class RoadmapSnapshot:
    """Immutable snapshot of roadmap at creation time"""
    topic: str
    created_at: datetime
    title: Optional[str] = None
    description: Optional[str] = None
    duration_week: int = 1
    milestones: Tuple[MilestoneSnapshot, ...] = ()
    prerequisites: Optional[Tuple[str, ...]] = None

def from_user_profile(profile: UserProfile) -> ProfileSnapshot:
    """Build ProfileSnapshot from UserProfile aggregate (at event time)"""
    return ProfileSnapshot(
        goal=profile.goal,
        current_level=profile.current_level,
        time_commitment=profile.time_commitment,
        learning_style=profile.learning_style,
        background=profile.background,
        constraints=tuple(profile.constraints) if profile.constraints else None,
    )

def from_roadmap(roadmap: Roadmap) -> RoadmapSnapshot:
    """Build RoadmapSnapshot from Roadmap aggregate (at event time)"""
    def resource_snapshot(r) -> ResourceSnapshot:
        return ResourceSnapshot(
            title=r.title,
            url=str(r.url),
            type=r.type,
            description=r.description,
            difficulty=r.difficulty,
        )
    
    def milestone_snapshot(m) -> MilestoneSnapshot:
        return MilestoneSnapshot(
            week=m.week,
            topic=m.topic,
            description=m.description,
            resources=tuple(resource_snapshot(res) for res in m.resources),
            estimated_time=m.estimated_time,
            learning_objectives=(
                tuple(m.learning_objectives)
                if m.learning_objectives
                else None
            ),
        )
    
    return RoadmapSnapshot(
        topic=roadmap.topic,
        title=roadmap.title,
        description=roadmap.description,
        duration_week=roadmap.duration_week,
        milestones=tuple(milestone_snapshot(m) for m in roadmap.milestones),
        prerequisites=tuple(roadmap.prerequisites) if roadmap.prerequisites else None,
        created_at=roadmap.created_at,
    )