"""
roadmap.py

Roadmap domain models: Resource, Milestone, Roadmap

Key features:
- Pydantic models with validation (HttpUrl, sequential weeks, duration matching)
- Domain invariants: milestone count must match duration_week
- Auto-generated title from topic if not provided
"""
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl, ValidationInfo, field_validator, model_validator


class Resource(BaseModel):
    """
    Represents a single learning resource included in a milestone
    """
    title: str = Field(
        ...,
        max_length=200,
        description="Title of the resource"
    )
    url: HttpUrl = Field(
        ...,
        description="Direct link to the resource"
    )
    type: Literal["video", "article", "book", "course", "practice", "project", "documentation"] = Field(
        ...,
        description="Type of resource"
    )
    description: Optional[str] = Field(
        None,
        description="Description of the resource"
    )
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        None,
        description="Difficulty level of the resource"
    )

class Milestone(BaseModel):
    """
    Represents a weekly milestone in the learning roadmap (topic, description, resources)
    """
    week: int = Field(
        ...,
        ge=1,
        description="Week number in the roadmap (starting from 1)"
    )
    topic: str = Field(
        ...,
        max_length=200,
        description="Main topic covered in this week"
    )
    description: str = Field(
        ...,
        max_length=1000,
        description="Detailed description of what will be learned in this week"
    )
    resources: List[Resource] = Field(
        ...,
        min_length=1,
        description="List of recommended resources for this milestone"
    )
    estimated_time: Optional[str] = Field(
        None,
        description="Estimated time for this week (e.g., '5 giờ')"
    )
    learning_objectives: Optional[List[str]] = Field(
        None,
        description="Learning objectives for this week"
    )

class Roadmap(BaseModel):
    """
    Complete personalized learning roadmap generated for a user
    """
    topic: str = Field(
        ...,
        max_length=200,
        description="Overall main topic of the learning path"
    )
    title: Optional[str] = Field(
        None,
        max_length=200,
        description="Display title (auto-set to topic if not provided)"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Overall description of roadmap content and goals"
    )
    duration_week: int = Field(
        ...,
        ge=1,
        description="Total duration of the roadmap in weeks"
    )
    milestones: List[Milestone] = Field(
        ...,
        min_length=1,
        description="List of weekly milestones"
    )
    prerequisites: Optional[List[str]] = Field(
        None,
        description="Prerequisites required before starting"
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the roadmap was generated"
    )

    @model_validator(mode="after")
    def set_default_title(self):
        """Auto-set title = topic if title is not provided"""
        if not self.title:
            self.title = self.topic
        return self

    @field_validator('milestones')
    @classmethod
    def validate_milestones(cls, v: List[Milestone], info: ValidationInfo) -> List[Milestone]:
        """
        Validate:
        - Week numbers are sequential starting from 1
        - Number of milestones matches duration_week
        """
        # Sequential weeks
        weeks = [m.week for m in v]
        expected = list(range(1, len(weeks) + 1))
        if weeks != expected:
            raise ValueError(
                f"Week numbers must be sequential starting from 1. "
                f"Got {weeks}, expected {expected}"
            )
        
        # Milestone count matches duration_week
        if 'duration_week' in info.data:
            duration_week = info.data['duration_week']
            if len(v) != duration_week:
                raise ValueError(f"Number of milestones ({len(v)}) must match duration_week ({duration_week})")
            
        return v