"""
chat.py

Chat domain models: ChatMessage

Key features:
- ChatMessage: role (user/assistant), content, timestamp
- ConversationState: track multi-turn flows (NORMAL, AWAITING_PROFILE_INFO)
- Auto-generated timestamp on message creation
"""

from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """
    Single message in a chat conversation

    Attributes:
        role: Message role - "system", "user" or "assistant" (required)
        content: Message text content (required)
        timestamp: Auto-generated timestamp when message was created
    """
    role: Literal["system", "user", "assistant"] = Field(
        ...,
        description="Message role: 'system', 'user' or 'assistant'"
    )
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the message was created"
    )

class ConversationState(Enum):
    """
    Conversation state for multi-turn interaction flows
    """
    NORMAL = "normal"
    AWAITING_PROFILE_INFO = "awaiting_profile_info"