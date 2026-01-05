import google.generativeai as genai
from config.settings import settings
from utils.logger import logger
from utils.exceptions import LLMServiceError

class GeminiClient:
    def __init__(self):
        try: 
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info(f"Gemini Client initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.critical(f"Failed to initialize Gemini: {e}")
            raise LLMServiceError(f"Init failed: {e}")
        
    def generate_text(self, prompt: str) -> str:
        try:
            logger.debug(f"Sending prompt to Gemini (Length: {len(prompt)})")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise LLMServiceError(f"Generation failed: {e}")
        
    def stream_chat(self, history: list, new_message: str):
        try:
            chat_session = self.model.start_chat(history=history)
            response_stream = chat_session.send_message(new_message, stream=True)

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise LLMServiceError(f"Stream failed: {e}")
        
gemini_client = GeminiClient()