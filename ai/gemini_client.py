"""
gemini_client.py

Gemini LLM client implementation and streaming helpers

Key features:
- Configure and validate Gemini API (api_key, model, system_prompt)
- generate_text with retry on transient errors
- stream_chat with history conversion to Gemini format
"""

from typing import Generator, List, Dict, Any
from google.api_core import exceptions as google_exceptions
import google.generativeai as genai

from utils import logger, LLMServiceError, ValidationError, gemini_retry
from ai.llm_client import LLMClient
from domain import ChatMessage

# Gemini safety settings (BLOCK_ONLY_HIGH threshold)
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]

class GeminiClient:
    """
    Gemini implementation of LLMClient

    Responsibilities:
    - Validate config (api_key, model_name, system_prompt) and init SDK model
    - Generate text and stream chat with timeout and retry
    - Convert domain ChatMessage history to Gemini message format
    """
    def __init__(
        self,
        api_key: str,
        model_name: str,
        request_timeout: int,
        stream_timeout: int,
        system_prompt: str
    ):
        """
        Initialize GeminiClient with API config and timeouts.

        Args:
            api_key: Gemini API key
            model_name: Gemini model name (e.g. gemini-2.5-flash)
            request_timeout: Timeout in seconds for non-streaming requests
            stream_timeout: Timeout in seconds for streaming
            system_prompt: System instruction for the model

        Raises:
            ValidationError: If api_key, model_name or system_prompt is invalid
            LLMServiceError: If SDK initialization fails unexpectedly
        """
        self._validate_config(api_key, model_name, system_prompt)

        self.api_key = api_key
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.stream_timeout = stream_timeout
        self.system_prompt = system_prompt

        self.model = self._init_model()
    
    def _validate_config(self, api_key: str, model_name: str, system_prompt: str) -> None:
        """Validate config before initializing the SDK; raise ValidationError if invalid"""
        if not api_key or not api_key.strip():
            raise ValidationError(message="Gemini API key must not be empty")
        
        if not model_name or not model_name.strip():
            raise ValidationError(message="Gemini model name must not be empty")
        
        if not system_prompt or not system_prompt.strip():
            raise ValidationError(message="System prompt must not be empty")
        
    def _init_model(self):
        """
        Initialize Gemini SDK model

        Returns:
            GenerativeModel: initialized Gemini model

        Raises:
            ValidationError: On invalid API key or model name
            LLMServiceError: On unexpected SDK initialization errors
        """
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_prompt
            )
            logger.info(f"GeminiClient initialized with model: {self.model_name}")
            return model
        except google_exceptions.InvalidArgument as e:
            raise ValidationError(
                message="Invalid Gemini API key or model name"
            ) from e
        except Exception as e:
            raise LLMServiceError(
                code="LLM_INIT_FAILED", 
                message=f"Failed to init Gemini client"
            ) from e
    
    @staticmethod
    def _to_gemini_history(history: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Convert domain ChatMessage history into Gemini-compatible message objects

        Args:
            history: List of ChatMessage objects

        Returns:
            List of dicts in Gemini format
            [
                {
                    "role": "user" | "model",
                    "parts": [<message content>],
                }
            ]
        """
        converted: List[Dict[str, Any]] = []

        for msg in history:
            if msg.role == "assistant":
                role = "model"
            else:
                role = "user"

            converted.append(
                {
                    "role": role,
                    "parts": [msg.content]
                }
            )
        return converted
        
    @gemini_retry(max_retries=3)
    def generate_text(self, prompt: str) -> str:
        """
        Generate text from prompt

        Args:
            prompt: Input text. If empty, returns empty string

        Returns:
            Generated content; empty string if blocked/filtered

        Raises:
            LLMServiceError: On transient errors (timeout, 5xx) after retries
        """
        if not prompt or not prompt.strip():
            raise ValidationError(message="Prompt must not be empty")
        
        try:
            response = self.model.generate_content(
                prompt, 
                safety_settings=_SAFETY_SETTINGS,
                request_options={"timeout": self.request_timeout}
            )
        except google_exceptions.GoogleAPICallError as e:
            raise LLMServiceError(
                code="GENERATION_FAILED",
                message="Failed to generate content from Gemini"
            ) from e
        
        if not getattr(response, "text", None):
            raise LLMServiceError(
                code="EMPTY_RESPONSE", 
                message="Gemini returned empty response"
            )
        
        return response.text.strip()
        
    def stream_chat(self, history: List[ChatMessage], new_message: str) -> Generator[str, None, None]:
        """
        Stream chat response from Gemini

        Args:
            history: List of previous chat messages (role/content)
            new_message: User's new message

        Yields:
            Chunks of generated text as they arrive

        Raises:
            ValidationError: If new_message empty
            LLMServiceError: On Gemini streaming failure
        """
        new_message = new_message.strip()
        if not new_message:
            raise ValidationError(message="New message must be not empty")
        
        gemini_history = self._to_gemini_history(history)
        
        try:
            chat = self.model.start_chat(history=gemini_history)
            stream = chat.send_message(
                new_message, 
                stream=True, 
                safety_settings=_SAFETY_SETTINGS,
                request_options={"timeout": self.stream_timeout}
            )

            for chunk in stream:
                if getattr(chunk, "text", None):
                    yield chunk.text
        
        except google_exceptions.GoogleAPICallError as e:
            raise LLMServiceError(
                code="STREAM_FAILED", 
                message="Failed to stream response from Gemini"
            ) from e