"""
Unit test for Settings configuration
"""
import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from config.settings import Settings

# Test constants
VALID_GEMINI_API_KEY = "AIzaSy" + "a" * 30

class TestSettingsFields:
    """Test settings default values and required fields"""
    
    def test_default_values(self):
        """Test that Settings has correct default value when env vars not set"""
        with patch.dict(os.environ, {"GEMINI_API_KEY": VALID_GEMINI_API_KEY}):
            settings = Settings(_env_file=None)

            # API configuration
            assert settings.GEMINI_API_KEY == VALID_GEMINI_API_KEY
            assert settings.GEMINI_MODEL == "gemini-2.5-flash"

            # Logging configuration
            assert settings.LOG_LEVEL == "INFO"
            assert settings.LOG_TO_FILE is False
            assert settings.LOG_FILE_PATH == "logs/app.log"
            assert settings.LOG_FILE_RETENTION == 7

    def test_required_api_key(self):
        """Test that GEMINI_API_KEY is required"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)

            errors = exc_info.value.errors()
            assert any(
                error["loc"] == ("GEMINI_API_KEY",)
                for error in errors
            )

class TestSettingsFromEvironment:
    """Test settings loading from environment variables"""

    def test_load_model(self):
        """Test loading model settings from environment"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
            "GEMINI_MODEL": "gemini-2.5-pro"
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file = None)

            assert settings.GEMINI_MODEL == "gemini-2.5-pro"

    def test_load_logging(self):
        """Test loading logging settings from environment"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
            "LOG_LEVEL": "DEBUG",
            "LOG_TO_FILE": "true",
            "LOG_FILE_PATH": "custom/logs/app.log"
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)
            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.LOG_TO_FILE is True
            assert settings.LOG_FILE_PATH == "custom/logs/app.log"