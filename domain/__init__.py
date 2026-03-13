"""
Domain layer for LearnPath chatbot

Key features:
- Re-export Resource, Milestone, Roadmap from roadmap
- Re-export ChatMessage, ConversationState from chat
- Re-export UserProfile from user
- Re-export Intent from intent
- Re-export Event, TextChunk, StatusUpdate, ErrorOccurred, SessionExpired, ProfileExtractor, RoadmapCreated from events
- Independent of application and infrastructure layers
"""

from .chat import ChatMessage, ConversationState
from .intent import Intent, IntentDetectionMethod, IntentResult, ConfidenceLevel
from .roadmap import Resource, Milestone, Roadmap
from .user import UserProfile
from .events import (
    Event,
    TextChunk,
    StatusUpdate,
    ErrorOccurred,
    SessionExpired,
    ProfileExtracted,
    RoadmapCreated,
)

__all__ = [
    "Resource",
    "Milestone",
    "Roadmap",
    "UserProfile",
    "ChatMessage",
    "ConversationState",
    "Intent",
    "IntentDetectionMethod",
    "IntentResult",
    "ConfidenceLevel",
    "Event",
    "TextChunk",
    "StatusUpdate",
    "ErrorOccurred",
    "SessionExpired",
    "ProfileExtracted",
    "RoadmapCreated",
]