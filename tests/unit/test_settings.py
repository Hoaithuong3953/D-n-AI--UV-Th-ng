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

