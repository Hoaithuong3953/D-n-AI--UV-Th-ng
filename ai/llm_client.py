from typing import Protocol, List, Dict, Generator

class LLMClient(Protocol):
    """
    Interfacr for LLM Clients (Gemini, OpenAI,...)
    """

    def generate_text(self, prompt: str) -> str:
        """
        Generate simple text from a prompt
        """
        ...

    def stream_chat(self, history: List[Dict[str, str]], new_message: str) -> Generator[str, None, None]:
        """
        Generate a streaming response for chat
        """
        ...