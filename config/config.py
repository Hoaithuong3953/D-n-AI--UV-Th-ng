import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

APP_CONFIG = {
    'name': 'LearnPath AI',
    'version': '1.0.0',
}

AI_CONFIG = {
    'model': 'gemini-2.5-flash',
    'max_tokens': 2048,
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40
}