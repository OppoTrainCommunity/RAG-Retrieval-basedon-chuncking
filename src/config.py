"""
Configuration settings for the RAG Resume Analysis system.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories
for dir_path in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, CHROMA_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Models
EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large", 
    "openai/text-embedding-ada-002",
]

LLM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/devstral-2-2512:free",
    "meta-llama/llama-3.2-3b-instruct:free"
]

# Analysis settings
CHUNKING_STRATEGIES = ["fixed", "semantic", "recursive"]

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_MIN_CHUNK_SIZE = 100

# API Settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
RETRY_DELAY = 1.0
