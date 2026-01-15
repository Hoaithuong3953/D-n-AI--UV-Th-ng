from typing import Protocol, List, Dict

class ChatHistory(Protocol):
    """
    Interface for chat history storage
    """
    def add_user_message(self, content: str) -> None:
        """
        Add user message
        """
        ...

    def add_bot_message(self, content: str) -> None:
        """
        Add bot message
        """
        ...

    def load_history(self) -> List[Dict[str, str]]:
        """
        Load full history
        """
        ...