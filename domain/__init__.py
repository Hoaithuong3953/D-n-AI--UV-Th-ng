"""
Domain layer for LearnPath chatbot

Key features:
- Re-export Resource, Milestone, Roadmap from roadmap
- Re-export ChatMessage from chat
- Re-export UserProfile from user
- Re-export Intent from intent
- Re-export Event, TextChunk, StatusUpdate, ErrorOccurred, SessionExpired from events
- Independent of application and infrastructure layers
"""

from .chat import ChatMessage
from .intent import Intent
from .roadmap import Resource, Milestone, Roadmap
from .user import UserProfile
from .events import (
    Event,
    TextChunk,
    StatusUpdate,
    ErrorOccurred,
    SessionExpired
)

__all__ = [
    "Resource",
    "Milestone",
    "Roadmap",
    "UserProfile",
    "ChatMessage",
    "Intent",
    "Event",
    "TextChunk",
    "StatusUpdate",
    "ErrorOccurred",
    "SessionExpired",
]