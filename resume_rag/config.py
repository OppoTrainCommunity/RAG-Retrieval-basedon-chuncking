# resume_rag/config.py
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class AppConfig:
    persist_dir: str = "chroma_store"
    collection_name: str = "resumes"
    uploads_dir: str = "temp_uploads"
    index_file: str = "resume_index.json"  # inside persist_dir
    default_k: int = 8

def ensure_dirs(cfg: AppConfig) -> None:
    Path(cfg.persist_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.uploads_dir).mkdir(parents=True, exist_ok=True)

def get_openrouter_key() -> str | None:
    return os.getenv("OPENROUTER_API_KEY")

def get_openrouter_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
