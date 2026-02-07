"""
Embeddings Module
=================

Factory for creating local HuggingFace embedding models.
Supports switching between multiple embedding models.
"""

from typing import Optional, Dict, Any
from functools import lru_cache

from .logging_utils import get_logger
from .config import settings

logger = get_logger(__name__)

# Try different import paths for HuggingFace embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    EMBEDDINGS_SOURCE = "langchain-huggingface"
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        EMBEDDINGS_SOURCE = "langchain-community"
    except ImportError:
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            EMBEDDINGS_SOURCE = "langchain"
        except ImportError:
            HuggingFaceEmbeddings = None
            EMBEDDINGS_SOURCE = None
            logger.error(
                "HuggingFaceEmbeddings not available. "
                "Install: pip install langchain-huggingface sentence-transformers"
            )


class EmbeddingFactory:
    """
    Factory class for creating and caching embedding models.
    
    Supports two configurable embedding models that can be switched
    based on user selection.
    """
    
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def get_model_config(cls, model_key: str) -> Dict[str, Any]:
        """
        Get model configuration for a given model key.
        
        Args:
            model_key: Model identifier key
            
        Returns:
            Model configuration dict
        """
        models = settings.embedding_models
        
        if model_key not in models:
            raise ValueError(f"Unknown model key: {model_key}. Available: {list(models.keys())}")
        
        model_name = models[model_key]
        
        # Model-specific configurations
        if "e5" in model_name.lower():
            # E5 models benefit from query/passage prefixes
            return {
                "model_name": model_name,
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {
                    "normalize_embeddings": True,
                    "batch_size": 32
                },
                "query_instruction": "query: ",
                "embed_instruction": "passage: "
            }
        else:
            # Standard configuration for other models
            return {
                "model_name": model_name,
                "model_kwargs": {"device": "cpu"},
                "encode_kwargs": {
                    "normalize_embeddings": True,
                    "batch_size": 32
                }
            }
    
    @classmethod
    def create_embeddings(cls, model_key: str, use_cache: bool = True):
        """
        Create or retrieve a cached embedding model.
        
        Args:
            model_key: Model identifier key (e.g., "Model A (MiniLM)")
            use_cache: Whether to cache and reuse model instances
            
        Returns:
            HuggingFaceEmbeddings instance
        """
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "HuggingFaceEmbeddings not available. "
                "Install: pip install langchain-huggingface sentence-transformers"
            )
        
        # Check cache
        if use_cache and model_key in cls._instances:
            logger.info(f"Using cached embedding model: {model_key}")
            return cls._instances[model_key]
        
        # Get configuration
        config = cls.get_model_config(model_key)
        
        logger.info(f"Loading embedding model: {config['model_name']}")
        
        try:
            # Create embeddings instance
            embeddings = HuggingFaceEmbeddings(
                model_name=config["model_name"],
                model_kwargs=config["model_kwargs"],
                encode_kwargs=config["encode_kwargs"]
            )
            
            # Cache the instance
            if use_cache:
                cls._instances[model_key] = embeddings
            
            logger.info(f"Successfully loaded embedding model: {model_key}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_key}: {e}")
            raise
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        Get available embedding models.
        
        Returns:
            Dict mapping model keys to model names
        """
        return settings.embedding_models
    
    @classmethod
    def clear_cache(cls):
        """Clear the model cache."""
        cls._instances.clear()
        logger.info("Embedding model cache cleared")
    
    @classmethod
    def get_model_info(cls, model_key: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_key: Model identifier key
            
        Returns:
            Model information dict
        """
        config = cls.get_model_config(model_key)
        
        return {
            "key": model_key,
            "model_name": config["model_name"],
            "device": config["model_kwargs"].get("device", "cpu"),
            "normalize": config["encode_kwargs"].get("normalize_embeddings", False),
            "cached": model_key in cls._instances,
            "source_library": EMBEDDINGS_SOURCE
        }


def get_embeddings(model_key: str):
    """
    Convenience function to get embeddings.
    
    Args:
        model_key: Model identifier key
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    return EmbeddingFactory.create_embeddings(model_key)


def list_available_models() -> Dict[str, str]:
    """
    List available embedding models.
    
    Returns:
        Dict mapping model keys to model names
    """
    return EmbeddingFactory.get_available_models()
