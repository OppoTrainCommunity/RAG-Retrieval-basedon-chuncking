"""
Embeddings factory for CV RAG System.
Supports multiple embedding providers (OpenRouter, Local).
"""

import logging
from typing import Optional, List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class OpenRouterEmbeddings(Embeddings):
    """
    OpenRouter-compatible embeddings using OpenAI API format.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/text-embedding-3-small",
    ):
        """
        Initialize OpenRouter embeddings.
        
        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter base URL.
            model: Embedding model name.
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        # Use OpenAI client for API calls
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed.
        
        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
        )
        
        return response.data[0].embedding


class LocalEmbeddings(Embeddings):
    """
    Local embeddings using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded local embedding model: {model_name}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text to embed.
        
        Returns:
            Embedding vector.
        """
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


class EmbeddingsFactory:
    """
    Factory for creating embedding instances based on configuration.
    """
    
    @staticmethod
    def create(
        provider: str = "openrouter",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        local_model: str = "all-MiniLM-L6-v2",
    ) -> Embeddings:
        """
        Create an embeddings instance based on provider.
        
        Args:
            provider: Embedding provider ("openrouter" or "local").
            model: Model name for the provider.
            api_key: API key (required for openrouter).
            base_url: Base URL for API.
            local_model: Model name for local embeddings.
        
        Returns:
            Embeddings instance.
        
        Raises:
            ValueError: If provider is not supported.
        """
        if provider.lower() == "openrouter":
            if not api_key:
                raise ValueError("API key required for OpenRouter embeddings")
            
            return OpenRouterEmbeddings(
                api_key=api_key,
                base_url=base_url or "https://openrouter.ai/api/v1",
                model=model or "openai/text-embedding-3-small",
            )
        
        elif provider.lower() == "local":
            return LocalEmbeddings(model_name=local_model)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def from_config(config) -> Embeddings:
        """
        Create embeddings from a Config object.
        
        Args:
            config: Config object with embeddings settings.
        
        Returns:
            Embeddings instance.
        """
        return EmbeddingsFactory.create(
            provider=config.embeddings.provider,
            model=config.embeddings.model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            local_model=config.embeddings.local_model,
        )


def get_embeddings(
    provider: str = "openrouter",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    local_model: str = "all-MiniLM-L6-v2",
) -> Embeddings:
    """
    Convenience function to get embeddings instance.
    
    Args:
        provider: Embedding provider.
        model: Model name.
        api_key: API key.
        base_url: Base URL.
        local_model: Local model name.
    
    Returns:
        Embeddings instance.
    """
    return EmbeddingsFactory.create(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        local_model=local_model,
    )
