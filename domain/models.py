from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class Resource(BaseModel):
    """
    Represents a single learning resource included in a milestone
    """
    title: str = Field(..., description="Title of the resource")
    url: str = Field(..., description="Direct link to the resource")
    type: str = Field(..., description="Type of resource (e.g., 'video', 'article', 'book', 'course', 'documentation')")

class Milestone(BaseModel):
    """
    Represents a weekly milestone in the learning roadmap
    Each milestone covers a specific topic with description and recommended resources
    """
    week: int = Field(..., ge=1, description="Week number in the roadmap (starting from 1)")
    topic: str = Field(..., description="Main topic covered in this week")
    description: str = Field(..., description="Detailed description of what will be learned in this week")
    resources: List[Resource] = Field(
        default_factory=list,
        description="List of recommended resources for this milestone"
    )

class Roadmap(BaseModel):
    """
    Complete personalize learning roadmap generated for a user
    """
    topic: str = Field(..., description="Overall main topic of the learning path")
    duration_week: int = Field(..., ge=1, description="Total duration of the roadmap in weeks")
    milestones: List[Milestone] = Field(..., min_length=1, description="List of weekly milestones")
    created_at: datetime = Field(default_factory=datetime.now, description="Timestamp when the roadmap was generated")

class UserProfile(BaseModel):
    """
    User input profile used to generate a personalized roadmap
    """
    goal: str = Field(..., description="User's learning goal")
    current_level: str = Field(..., description="User's current skill level (e.g., 'beginner', 'intermediate', 'advanced')")
    time_commitment:str = Field(..., description="Daily time user can commit to learning (e.g., 30 minutes, 2 hours)")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created"
    )