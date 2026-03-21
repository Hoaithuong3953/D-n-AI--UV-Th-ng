"""
events.py

Unified event types for application stream

Key features:
- UI consumes handle_message() as Generator[Event]; single source of event semantics
- Event, TextChunk, StatusUpdate, ErrorOccurred, SessionExpired, ProfileExtracted, RoadmapCreated
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from domain.snapshots import ProfileSnapshot, RoadmapSnapshot

@dataclass(frozen=True)
class Event:
    """Base event; all stream events subclass this and are immutable"""
    pass

@dataclass(frozen=True)
class TextChunk(Event):
    """Chunk of text from chat or status message"""
    text: str

@dataclass(frozen=True)
class StatusUpdate(Event):
    """Status update for loading, analyzing or generating phases"""
    status: Literal["loading", "analyzing_profile", "generating_roadmap"]
    message: str

@dataclass(frozen=True)
class ErrorOccurred(Event):
    """An error occurred; carries type and user-facing message"""
    error_type: Literal["validation", "llm", "unexpected"]
    user_message: str

@dataclass(frozen=True)
class SessionExpired(Event):
    """Session expired due to inactivity"""
    message: str

@dataclass(frozen=True)
class ProfileExtracted(Event):
    """User profile successfully extracted from conversation"""
    profile: ProfileSnapshot

@dataclass(frozen=True)
class RoadmapCreated(Event):
    """Learning roadmap successfully created"""
    roadmap: RoadmapSnapshot