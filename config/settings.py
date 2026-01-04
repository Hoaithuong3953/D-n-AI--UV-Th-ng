from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    APP_NAME: str = "LearnPath AI"
    APP_VERSION: str = "1.0.0"
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-flash"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()