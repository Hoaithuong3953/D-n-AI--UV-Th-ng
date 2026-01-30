"""
Unit test for Settings configuration
"""
import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from config import Settings

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

class TestSettingsInstanceCreation:
    """Test settings instance creation and access"""
    def test_settings_instance_creation(self):
        """Test that settings instance can be created"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)

            assert isinstance(settings, Settings)

            assert hasattr(settings, "GEMINI_API_KEY")
            assert hasattr(settings, "GEMINI_MODEL")
            assert hasattr(settings, "LOG_LEVEL")
            assert hasattr(settings, "LOG_TO_FILE")
            assert hasattr(settings, "LOG_FILE_PATH")
            assert hasattr(settings, "LOG_FILE_RETENTION")

    def test_settings_field_types(self):
        """Test that settings fiedls have correct types"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)

            assert isinstance(settings.GEMINI_API_KEY, str)
            assert isinstance(settings.GEMINI_MODEL, str)
            assert isinstance(settings.LOG_LEVEL, str)
            assert isinstance(settings.LOG_TO_FILE, bool)
            assert isinstance(settings.LOG_FILE_PATH, str)
            assert isinstance(settings.LOG_FILE_RETENTION, int)

class TestSettingsIntegration:
    """Test settings integration with other components"""

    def test_settings_can_be_imported(self):
        """Test that settings can be import from config module"""
        from config import settings

        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_used_in_gemini_client(self):
        """Test that settings can be used to initialize GeminiClient"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)

            assert hasattr(settings, "GEMINI_API_KEY")
            assert hasattr(settings, "GEMINI_MODEL")

    def test_settings_load_from_env_file(self):
        """Test that settings can be loaded from .env file"""
        env_vars = {
            "GEMINI_API_KEY": VALID_GEMINI_API_KEY,
            "GEMINI_MODEL": "gemini-2.5-pro",
            "LOG_LEVEL": "DEBUG"
        }

        with patch.dict(os.environ, env_vars):
            settings = Settings(_env_file=None)

            assert settings.GEMINI_API_KEY == VALID_GEMINI_API_KEY
            assert settings.GEMINI_MODEL == "gemini-2.5-pro"
            assert settings.LOG_LEVEL == "DEBUG"