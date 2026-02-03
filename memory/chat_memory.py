"""
chat_memory.py

In-memory implementation of chat history storage (list-based)

Key features:
- ChatMemory: add/load/clear messages; data lost on restart
- Suitable for testing, prototypes and short-lived sessions
"""

from typing import List
from domain import ChatMessage

class ChatMemory:
    """
    Handle in-memory chat history (add, load, clear)

    Responsibilities:
    - Store messages in an in-memory list
    - Provide add_message, load_history, clean_history
    """
    def __init__(self):
        """Initialize chat memory with an empty storage list"""
        self._storage: List[ChatMessage] = []

    def load_history(self) -> List[ChatMessage]:
        """
        Load chat history from storage

        Returns:
            List of ChatMessage in chronological order
        """
        return list(self._storage)

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a message to the chat history

        Args:
            message: ChatMessage to append
        """
        self._storage.append(message)

    def clean_history(self) -> None:
        """Clear all messages from the chat history"""
        self._storage.clear()