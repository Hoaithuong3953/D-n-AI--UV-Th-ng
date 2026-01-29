"""
In-memory implementation of chat history storage

This module provides a simple list-based storage
suitable for testing, prototypes and short-lived sessions

Data is lost on program restart
"""

from typing import List
from domain import ChatMessage

class ChatMemory:
    """
    Manages chat history
    Responsibilities:
    - Store chat history in memory
    - Provide methods to add and retrieve messages
    """
    def __init__(self, storage: List[ChatMessage]):
        """Initialize chat memory with storage"""
        self.storage = storage

    def load_history(self) -> List[ChatMessage]:
        """Load chat history from storage"""
        return self.storage

    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the chat history"""
        self.storage.append(message)
    
    def add_user_message(self, content: str):
        """Add a user message to the chat history"""
        msg = ChatMessage(role="user", content=content)
        self.add_message(msg)
    
    def add_bot_message(self, content: str):
        """Add a bot message to the chat history"""
        msg = ChatMessage(role="assistant", content=content)
        self.add_message(msg)

    def clean_history(self):
        """Clear the chat history"""
        self.storage.clear()