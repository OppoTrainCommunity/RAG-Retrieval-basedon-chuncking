"""
Configuration management for CV RAG System.
Loads configuration from YAML file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import yaml
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class LLMConfig:
    """LLM configuration."""
    model: str
    temperature: float = 0.2
    max_tokens: int = 800


@dataclass
class JudgeConfig:
    """Judge model configuration."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 400


@dataclass
class EmbeddingsConfig:
    """Embeddings configuration."""
    provider: str = "openrouter"
    model: str = "openai/text-embedding-3-small"
    local_model: str = "all-MiniLM-L6-v2"


@dataclass
class ChromaConfig:
    """ChromaDB configuration."""
    persist_dir: str = "./chroma_db"
    collection: str = "cv_rag"


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 6
    section_filter: Optional[str] = None
    similarity_metric: str = "cosine"


@dataclass
class DataConfig:
    """Data paths configuration."""
    cvs_path: str = "./data/cvs.csv"
    processed_dir: str = "./data/processed"
    chunks_path: str = "./data/processed/chunks.parquet"


@dataclass
class OutputsConfig:
    """Output configuration."""
    dir: str = "./outputs"
    eval_csv: str = "eval_results.csv"
    eval_json: str = "eval_results.json"
    logs_file: str = "run_logs.jsonl"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    console: bool = True
    file: bool = True


@dataclass
class Config:
    """Main configuration class."""
    llm: LLMConfig
    judge: JudgeConfig
    embeddings: EmbeddingsConfig
    chroma: ChromaConfig
    retrieval: RetrievalConfig
    data: DataConfig
    outputs: OutputsConfig
    logging: LoggingConfig
    
    # API Configuration
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    def __post_init__(self):
        """Ensure output directories exist."""
        Path(self.outputs.dir).mkdir(parents=True, exist_ok=True)
        Path(self.data.processed_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma.persist_dir).mkdir(parents=True, exist_ok=True)


def _get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)


def _load_yaml_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        # Try relative to this file
        config_file = Path(__file__).parent.parent / config_path
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML and environment variables.
    Environment variables take precedence over YAML values.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Config object with all settings.
    """
    # Load base config from YAML
    yaml_config = _load_yaml_config(config_path)
    
    # LLM Config with env override
    llm_yaml = yaml_config.get("llm", {})
    llm_config = LLMConfig(
        model=_get_env("LLM_MODEL", llm_yaml.get("model", "openai/gpt-4o-mini")),
        temperature=float(llm_yaml.get("temperature", 0.2)),
        max_tokens=int(llm_yaml.get("max_tokens", 800)),
    )
    
    # Judge Config with env override
    judge_yaml = yaml_config.get("judge", {})
    judge_config = JudgeConfig(
        model=_get_env("JUDGE_MODEL", judge_yaml.get("model", "openai/gpt-4o-mini")),
        temperature=float(judge_yaml.get("temperature", 0.0)),
        max_tokens=int(judge_yaml.get("max_tokens", 400)),
    )
    
    # Embeddings Config with env override
    emb_yaml = yaml_config.get("embeddings", {})
    use_local = _get_env("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    embeddings_config = EmbeddingsConfig(
        provider="local" if use_local else emb_yaml.get("provider", "openrouter"),
        model=_get_env("EMBEDDING_MODEL", emb_yaml.get("model", "openai/text-embedding-3-small")),
        local_model=emb_yaml.get("local_model", "all-MiniLM-L6-v2"),
    )
    
    # ChromaDB Config
    chroma_yaml = yaml_config.get("chroma", {})
    chroma_config = ChromaConfig(
        persist_dir=chroma_yaml.get("persist_dir", "./chroma_db"),
        collection=chroma_yaml.get("collection", "cv_rag"),
    )
    
    # Retrieval Config
    retrieval_yaml = yaml_config.get("retrieval", {})
    retrieval_config = RetrievalConfig(
        top_k=int(retrieval_yaml.get("top_k", 6)),
        section_filter=retrieval_yaml.get("section_filter"),
        similarity_metric=retrieval_yaml.get("similarity_metric", "cosine"),
    )
    
    # Data Config
    data_yaml = yaml_config.get("data", {})
    data_config = DataConfig(
        cvs_path=data_yaml.get("cvs_path", "./data/cvs.csv"),
        processed_dir=data_yaml.get("processed_dir", "./data/processed"),
        chunks_path=data_yaml.get("chunks_path", "./data/processed/chunks.parquet"),
    )
    
    # Outputs Config
    outputs_yaml = yaml_config.get("outputs", {})
    outputs_config = OutputsConfig(
        dir=outputs_yaml.get("dir", "./outputs"),
        eval_csv=outputs_yaml.get("eval_csv", "eval_results.csv"),
        eval_json=outputs_yaml.get("eval_json", "eval_results.json"),
        logs_file=outputs_yaml.get("logs_file", "run_logs.jsonl"),
    )
    
    # Logging Config
    logging_yaml = yaml_config.get("logging", {})
    logging_config = LoggingConfig(
        level=logging_yaml.get("level", "INFO"),
        console=logging_yaml.get("console", True),
        file=logging_yaml.get("file", True),
    )
    
    # Build main config
    config = Config(
        llm=llm_config,
        judge=judge_config,
        embeddings=embeddings_config,
        chroma=chroma_config,
        retrieval=retrieval_config,
        data=data_config,
        outputs=outputs_config,
        logging=logging_config,
        openrouter_api_key=_get_env("OPENROUTER_API_KEY", ""),
        openrouter_base_url=_get_env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    
    return config


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config(config_path)
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """Force reload the configuration."""
    global _config
    _config = load_config(config_path)
    return _config
