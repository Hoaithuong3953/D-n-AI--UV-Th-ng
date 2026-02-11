"""
intent.py

Intent detection models: Intent enum

Key features:
- Intent: User intent enum (CHAT, ROADMAP)
"""

from enum import Enum

from pydantic import BaseModel, Field

class Intent(Enum):
    """
    User intent for routing conversation flow

    Values:
        CHAT: General conversation, questions,...
        ROADMAP: Request to create/update learning roadmap
    """
    CHAT = "chat"
    ROADMAP = "roadmap"

class IntentDetectionMethod(Enum):
    """
    Method used for intent detection

    Values:
        KEYWORD: Detected via keyword matching (high confidence, fast)
        LLM: Detected via LLM fallback
    """
    KEYWORD = "keyword"
    LLM = "llm"

class IntentResult(BaseModel):
    """
    Result from intent detection with decision score

    Attributes:
        intent: Detected user intent (CHAT, ROADMAP)
        score: Decision score 0.0-1.0 for monitoring/debugging
        method: Detection method used (KEYWORD or LLM)
    """
    intent: Intent = Field(
        ...,
        description="Detected intent (CHAT, ROADMAP)"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Decision score (0.0-1.0)"
    )
    method: IntentDetectionMethod = Field(
        ...,
        description="Detection method (keyword or llm)"
    )