"""
Configuration Module
====================

Centralized configuration using environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # LLM Provider Configuration
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama")
    )
    
    # OpenRouter LLM Configuration
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ollama Configuration
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral")
    )
    
    # Embedding Models
    embedding_model_a: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_A", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    embedding_model_b: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_B", "intfloat/e5-base-v2"
        )
    )
    
    # Paths
    chroma_persist_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "data/chroma"))
    )
    upload_dir: Path = field(
        default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    )
    
    # Retrieval Settings
    default_top_k: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5"))
    )
    
    # Chunking Settings
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150"))
    )
    
    # LLM Settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_request_timeout: int = 60
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def embedding_models(self) -> dict:
        """Return available embedding models."""
        return {
            "Model A (MiniLM)": self.embedding_model_a,
            "Model B (E5)": self.embedding_model_b
        }
    
    def get_collection_name(self, model_key: str) -> str:
        """Generate collection name based on model."""
        if model_key == "Model A (MiniLM)":
            return "cvs_model_a"
        elif model_key == "Model B (E5)":
            return "cvs_model_b"
        else:
            # Sanitize model name for collection
            safe_name = model_key.replace("/", "_").replace("-", "_").lower()
            return f"cvs_{safe_name}"
    
    def get_model_persist_dir(self, model_key: str) -> Path:
        """Get persist directory for a specific model."""
        collection_name = self.get_collection_name(model_key)
        return self.chroma_persist_dir / collection_name
    
    def is_openrouter_configured(self) -> bool:
        """Check if OpenRouter API key is set."""
        return bool(self.openrouter_api_key)


# Global settings instance
settings = Settings()


# Prompt template for RAG
RAG_PROMPT_TEMPLATE = """You are a professional HR assistant specialized in analyzing CVs/resumes.
Answer the question based ONLY on the provided context from uploaded CVs.

IMPORTANT RULES:
1. Only use information explicitly present in the context below.
2. If the information is not in the context, say "I could not find this information in the uploaded CVs" and suggest what additional information might help.
3. Provide concise, professional answers using bullet points when appropriate.
4. Always cite your sources using the format: (Source: filename, Page: X, Chunk: Y)
5. Do not make assumptions or hallucinate information not present in the context.
6. Provide a brief summary at the end of your answer.

CONTEXT FROM CVs:
{context}

QUESTION: {question}

ANSWER (with citations):"""


OPENROUTER_HEADERS = {
    "HTTP-Referer": "http://localhost:8501",
    "X-Title": "RAG CV Assistant"
}
