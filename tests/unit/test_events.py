"""
test_events.py

Unit tests for domain.events (Event, TextChunk, StatusUpdate, ErrorOccurred, SessionExpired)

Key features:
- Event subclasses have expected attributes; all events are frozen (immutable)
"""
import pytest

from domain import (
    Event,
    TextChunk,
    StatusUpdate,
    ErrorOccurred,
    SessionExpired
)

def test_text_chunk():
    """TextChunk has text attribute and is subclass of Event"""
    e = TextChunk(text="hello")
    assert e.text == "hello"
    assert isinstance(e, Event)

def test_status_update():
    """StatusUpdate has status and message; is subclass of Event"""
    e = StatusUpdate(status="loading", message="Thinking...")
    assert e.status == "loading"
    assert e.message == "Thinking..."

def test_error_occurred():
    """ErrorOccurred has error_type and user_message"""
    e = ErrorOccurred(error_type="validation", user_message="Please enter")
    assert e.error_type == "validation"
    assert e.user_message == "Please enter"

def test_session_expired():
    """SessionExpired has message attribute"""
    e = SessionExpired(message="Session expired")
    assert e.message == "Session expired"

def test_events_are_frozen():
    """Event instances are frozen (assigning attribute raises AttributeError)"""
    e = TextChunk(text="x")
    with pytest.raises(AttributeError):
        e.text = "y"