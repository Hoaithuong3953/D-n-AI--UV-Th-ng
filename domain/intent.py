"""
intent.py

Intent detection models: Intent enum

Key features:
- Intent: User intent enum (CHAT, ROADMAP)
"""

from enum import Enum

class Intent(Enum):
    """
    User intent for routing conversation flow

    Values:
        CHAT: General conversation, questions,...
        ROADMAP: Request to create/update learning roadmap
    """
    CHAT = "chat"
    ROADMAP = "roadmap"