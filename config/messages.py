"""
messages.py

User-facing message keys and provider

Key features:
- MessageKey: enum of message keys
- MessageProvider: protocol for message provider
- DefaultMessageProvider: default implementation with templates
"""
from enum import Enum
from typing import Protocol

class MessageKey(str, Enum):
    """Keys for user-facing messages; use with MessageProvider.get or .format"""
    # Errors
    LLM_ERROR = "llm_error"
    UNEXPECTED_ERROR = "unexpected_error"

class MessageProvider(Protocol):
    """Protocol for message provider"""
    def get(self, key: MessageKey) -> str:
        """Return message for key"""
        ...

    def format(self, key: MessageKey, **kwargs: str) -> str:
        """Return message formatted with kwargs (e.g. goal=..., max=...)"""
        ...

class DefaultMessageProvider:
    """
    Default message provider

    Responsibilities:
    - get(key): return template for MessageKey
    - format(key, **kwargs): substitute kwargs into template
    """

    _TEMPLATES: dict[MessageKey, str] = {
        MessageKey.LLM_ERROR: "Không thể kết nối hoặc tải tin nhắn. Vui lòng thử lại sau.",
        MessageKey.UNEXPECTED_ERROR: "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau.",
    }

    def get(self, key: MessageKey) -> str:
        return self._TEMPLATES.get(key, "")
    
    def format(self, key: MessageKey, **kwargs: str) -> str:
        template = self._TEMPLATES.get(key, "")
        return template.format(**kwargs) if kwargs else template
    
default_messages = DefaultMessageProvider()