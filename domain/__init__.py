"""
Domain layer for LearnPath chatbot

Key features:
- Re-export Resource, Milestone, Roadmap, UserProfile, ChatMessage, Intent from models
- Re-export Event, TextChunk, StatusUpdate, ErrorOccurred, SessionExpired from events
- Independent of application and infrastructure layers
"""

from .models import (
    Resource,
    Milestone,
    Roadmap,
    UserProfile,
    ChatMessage,
    Intent
)
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