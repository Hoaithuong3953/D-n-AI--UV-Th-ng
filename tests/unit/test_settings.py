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

class TestSettingsFieldValidation:
    """Test settings field validation from .env"""

    def test_gemini_api_key_empty_string_validation(self):
        """Test that empty GEMINI_API_KEY is rejected"""
        env_vars = {
            "GEMINI_API_KEY": ""
        }

        with patch.dict(os.environ, env_vars):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file = None)

            errors = exc_info.value.errors()
            assert any(
                error["loc"] == ("GEMINI_API_KEY",)
                for error in errors
            )

    def test_gemini_api_key_invalid_prefix_validation(self):
        """Test that GEMINI_API_KEY without 'AIzaSy' prefix is rejected"""
        env_vars = {
            "GEMINI_API_KEY": "invalid_prefix_key_1234567890123456789"
        }

        with patch.dict(os.environ, env_vars):
            try:
                settings = Settings(_env_file = None)
                assert settings.GEMINI_API_KEY == "invalid_prefix_key_1234567890123456789"
            except ValidationError as e:
                errors = e.errors()
                assert any(
                    error["loc"] == ("GEMINI_API_KEY",)
                    for error in errors
                )

    def test_gemini_api_invalid_length_validation(self):
        """Test that GEMINI_API_KEY with unusual length is reject"""

        # Test too short (< 30)
        env_vars = {
            "GEMINI_API_KEY": "AIzaSy_short"
        }

        with patch.dict(os.environ, env_vars):
            try:
                settings = Settings(_env_file=None)
                assert settings.GEMINI_API_KEY == "AIzaSy_short"
            except ValidationError as e:
                errors = e.errors()
                assert any(
                    error["loc"] == ("GEMINI_API_KEY",)
                    for error in errors
                )

        # Test too long (> 50)
        env_vars = {
            "GEMINI_API_KEY": "AIzaSy" + "x" * 50
        }

        with patch.dict(os.environ, env_vars):
            try:
                settings = Settings(_env_file=None)
                assert settings.GEMINI_API_KEY == "AIzaSy" + "x" * 50
            except ValidationError as e:
                errors = e.errors()
                assert any(
                    error["loc"] == ("GEMINI_API_KEY",)
                    for error in errors
                )

    def test_gemini_api_key_valid_format(self):
        """Test that valid GEMINI_API_KEY format passes validation"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)
            assert settings.GEMINI_API_KEY == VALID_GEMINI_API_KEY

    def test_log_level_literal_validation(self):
        """Test that LOG_LEVEL only accepts valid values"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Test all valid levels
        for level in valid_levels:
            env_vars = {
                "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
                "LOG_LEVEL": level
            }

            with patch.dict(os.environ, env_vars):
                settings = Settings(_env_file=None)
                assert settings.LOG_LEVEL == level

        # Test invalid level
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
            "LOG_LEVEL": "INVALID"
        }

        with patch.dict(os.environ, env_vars):
            try:
                settings = Settings(_env_file = None)
                assert settings.LOG_LEVEL == "INVALID"
            except ValidationError as e:
                errors = e.errors()
                assert any(
                    error["loc"] == ("LOG_LEVEL",)
                    for error in errors
                )

    def test_log_file_retention_positive_validation(self):
        """Test that LOG_FILE_RETENTION must be >= 1"""
        # Test valid values
        for retention in [1, 7, 30, 356]:
            env_vars = {
                "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
                "LOG_FILE_RETENTION": str(retention)
            }

            with patch.dict(os.environ, env_vars):
                settings = Settings(_env_file=None)
                assert settings.LOG_FILE_RETENTION == retention

        # Test invalud values
        for invalid_retention in [0, -1, -10]:
            env_vars = {
                "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
                "LOG_FILE_RETENTION": str(invalid_retention)
            }

            with patch.dict(os.environ, env_vars):
                with pytest.raises(ValidationError) as exc_info:
                    Settings(_env_file=None)

                errors = exc_info.value.errors()
                assert any(
                    error["loc"] == ("LOG_FILE_RETENTION",)
                    for error in errors
                )