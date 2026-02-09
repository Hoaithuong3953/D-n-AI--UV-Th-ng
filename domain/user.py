"""
user.py

User domain models: UserProfile

Key features:
- Required fields: goal, current_level, time_commitment
- Optional fields: learning_style, background, constraints
- Pydantic validation with field constraints
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    """
    User input profile used to generate a personalized roadmap
    """
    goal: str = Field(
        ...,
        max_length=500,
        description="User's learning goal"
    )
    current_level: str = Field(
        ...,
        description="User's current skill level (e.g., 'beginner', 'intermediate', 'advanced')"
    )
    time_commitment: str = Field(
        ...,
        description="Daily time user can commit to learning (e.g., 30 minutes, 2 hours)"
    )
    learning_style: Optional[str] = Field(
        None,
        description="Learning style preference"
    )
    background: Optional[str] = Field(
        None,
        description="Personal background/context"
    )
    constraints: Optional[List[str]] = Field(
        None,
        description="Learner conditions/constraints (e.g., ['Free materials only', 'Weekends only'])"
    )
