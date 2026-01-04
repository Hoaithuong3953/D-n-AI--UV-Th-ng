from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class Resources(BaseModel):
    title: str
    url: str
    type: str

class Milestone(BaseModel):
    week: int
    topic: str
    describtion: str
    resources: List[Resources] = []

class Roadmap(BaseModel):
    topic: str
    duration_week: int
    milestones: List[Milestone]
    create_at: datetime = Field(default_factory=datetime.now)

class UserProfile(BaseModel):
    goal: str
    current_level: str
    time_commitment:str