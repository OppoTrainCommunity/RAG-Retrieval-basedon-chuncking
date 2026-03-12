from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenRouter
    openrouter_api_key: str = ""

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "cv-analysis-rag"

    # Storage
    chroma_persist_dir: str = "./data/chroma_db"
    upload_dir: str = "./data/uploads"

    # App
    app_version: str = "1.0.0"
    max_file_size: int = 20 * 1024 * 1024  # 20 MB

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
