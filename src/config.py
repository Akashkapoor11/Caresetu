# src/config.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    ENV: str = "dev"
    DB_PATH: str = ".data/care_setu.db"
    OPENAI_API_KEY: str | None = None
    MONGO_URI: str | None = None
    OCR_PROVIDER: str = "tesseract"
    TESSERACT_CMD: str | None = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
