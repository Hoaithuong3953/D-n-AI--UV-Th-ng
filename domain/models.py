from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from datetime import datetime

class Resource(BaseModel):
    """
    Represents a single learning resource included in a milestone
    """
    title: str = Field(..., description="Title of the resource")
    url: str = Field(..., description="Direct link to the resource")
    type: Literal["video", "article", "book", "cource", "practice", "project", "documentation"] = Field(
        ..., 
        description="Type of resource"
    )
    description: Optional[str] = Field(None, description="Description of the resource")
    difficulty: Optional[Literal["beginner", "intermediate", "advanced"]] = Field(
        None,
        description="Difficulty level of the resource"
    )

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
    estimated_time: Optional[str] = Field(None, description="Estimated time for this week (e.g., '5 giờ')")
    learning_objectives: Optional[List[str]] = Field(None, description="Learning objectives for this week")

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
    role: Literal["system", "user", "assistant"] = Field(
        ..., 
        description="Message role: 'system', 'user' or 'assistant'"
    )
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created"
    )