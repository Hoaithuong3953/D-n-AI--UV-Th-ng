import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from typing import Generator, Dict

from config.settings import settings
from utils.logger import logger
from utils.exceptions import LLMServiceError
from utils.retry import gemini_retry

class GeminiClient:

    _instance = None
    _initialized = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._perform_initialization()
            self._initialized = True

    def _perform_initialization(self):        
        if not settings.GEMINI_API_KEY or not settings.GEMINI_API_KEY.strip():
            logger.error("GEMINI_API_KEY is missing or empty on settings")
            raise LLMServiceError(code="INVALID_API_KEY", message="Gemini api key is not configured")

        try: 
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini Client initialized with model: {settings.GEMINI_MODEL}")
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Invalid Gemini API key or configuration: {e}")
            raise LLMServiceError(code="INVALID_API_KEY", message="Invalid or unauthorized Gemini api key") from e
        except Exception as e:
            logger.critical(f"Unexpected error initializing Gemini client: {e}")
            raise LLMServiceError(code="LLM_INIT_FAILED", message="Failed to initialize Gemini client") from e

    @gemini_retry(max_retries=3)
    def generate_text(self, prompt: str) -> str:
        """
        Generate text from Gemini with retry and response validation
        """
        logger.debug(f"Sending prompt to Gemini (Length: {len(prompt)}")
        response = self.model.generate_content(
            prompt,
            request_options={"timeout": 60}
        )
        
        # Validate response
        if not hasattr(response, "text") or not response.text:
            if hasattr(response, "prompt_feedback") and response.prompt_feedback:
                block_reason = response.prompt_feedback.block_reason
                logger.warning(f"Gemini blocked content: {block_reason}")
                raise LLMServiceError(code="CONTENT_BLOCK", message=f"Content blocked: {block_reason}")
            raise LLMServiceError(code="EMPTY_RESPONSE", message="Gemini return empty or invalid response")
        logger.info("Successfully generated text from Gemini")
        return response.text.strip()
        
    @gemini_retry(max_retries=3)
    def stream_chat(self, history: list[Dict], new_message: str) -> Generator[str, None, None]:
        chat_session = None
        try:
            chat_session = self.model.start_chat(history=history)
            logger.debug("Starting new chat stream session")

            response_stream = chat_session.send_message(new_message, stream=True, request_options={"timeout": 120})

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

            logger.info("Gemini chat stream completed successfully")
        
        except Exception as e:
            logger.warning(f"Chat stream interrupted - will retry if traisient error: {e}")
            raise
        finally:
            logger.debug("Chat session attempt finished")
        
def get_gemini_client() -> GeminiClient:
    return GeminiClient()