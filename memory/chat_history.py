"""
Protocol defining the required interface for any chat history storage

Any class that implements these methods can be used as a valid chat history backend
This enables loose coupling and easy swapping between implementations (in-memory, Redis, database,...)
"""

from typing import Protocol, List

from domain import ChatMessage

class ChatHistory(Protocol):
    """
    Interface that every chat history storage must follow
    """
    def add_message(self, message: ChatMessage) -> None:
        """Append a new message to the end of the conversation history

        The message is added in chronological order (most recent last)
        """
        ...

    def load_history(self) -> List[ChatMessage]:
        """Return the complete list of messages in this conversation
        
        Returns messages in chronological order (oldest first)
        Implementations should return a copy or immutable view if possible
        to prevent accidental modification of internal storage
        """
        ...
    
    def clean_history(self):
        """Remove all messages from the current conversation history
        
        Used when:
        - User starts a new conversation
        - Session is being reset
        """
        ...