from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Literal

class Settings(BaseSettings):
    """
    Application configuration settings

    All settings can be configured via environment variables in .env file

    Required settings:
    - GEMINI_API_KEY: Gemini API key (required, no default)

    Optional settings:
    - GEMINI_MODEL: Model to use (default: "gemini-2.5-flash")
    - LOG_LEVEL: Logging level (default: "INFO")
    - LOG_TO_FILE: Enable file logging (default: False)
    - LOG_FILE_PATH: Path to log file (default: "logs/app.log")
    - LOG_FILE_RETENTION: Days to retain logs (default: 7)
    """
    
    APP_NAME: str = Field(
        default="LearnPath AI",
        description="Application name (optional, has default)"
    )
    APP_VERSION: str = Field(
        default="1.0.0",
        description="Application version (optional, has default)"
    )
    GEMINI_API_KEY: str = Field(
        ...,
        description="Gemini API key (required, no default, must be set in .env file)"
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model to use (optional, has default)"
    )

    @field_validator('GEMINI_API_KEY')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key not empty and has reasonable format"""
        v = v.strip()

        if not v:
            raise ValueError("GEMINI_API_KEY cannot be empty")
        
        if not v.startswith("AIzaSy"):
            raise ValueError("GEMINI_API_KEY should start with 'AIzaSy'. Please check if you're using a valid Google AI Studio API key")
        
        if len(v) < 30 or len(v) > 50:
            raise ValueError(f"GEMINI_API_KEY length ({len(v)}) seems unusual. Please check your key")
        
        return v


    # Logging settings
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level (optional, has default)"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(module)s:%(funcName)s:%(lineno)d - %(message)s",
        description="Default log format (can be customized if needed)"
    )
    LOG_DATE_FORMAT: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date/time format for logs"
    )
    LOG_TO_FILE: bool = Field(
        default=False,
        description="If true, enable logging to a file"
    )
    LOG_FILE_PATH: str = Field(
        default="logs/app.log",
        description="Path to log file"
    )
    LOG_FILE_ROTATION: str = Field(
        default="midnight",
        description="Log file rotation interval"
    )
    LOG_FILE_RETENTION: int = Field(
        default=7,
        ge=1,
        description="Number of days to retain log files"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()