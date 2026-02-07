"""
Vector Store Module
===================

ChromaDB vector store wrapper with model-specific collections.
Supports persistent storage and deduplication.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .logging_utils import get_logger
from .config import settings
from .chunking import get_chunk_ids

logger = get_logger(__name__)

# LangChain Document import
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

# ChromaDB imports with fallbacks
try:
    from langchain_chroma import Chroma
    CHROMA_SOURCE = "langchain-chroma"
except ImportError:
    try:
        from langchain_community.vectorstores import Chroma
        CHROMA_SOURCE = "langchain-community"
    except ImportError:
        from langchain.vectorstores import Chroma
        CHROMA_SOURCE = "langchain"

import chromadb


class VectorStoreManager:
    """
    Manager for ChromaDB vector stores with model-specific collections.
    
    Handles persistence, collection management, and document ingestion.
    """
    
    def __init__(
        self,
        persist_dir: Optional[Path] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            persist_dir: Base directory for persistence
        """
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._stores: Dict[str, Chroma] = {}
        
        logger.info(f"VectorStoreManager initialized with persist_dir: {self.persist_dir}")
    
    def get_collection_persist_dir(self, model_key: str) -> Path:
        """
        Get the persistence directory for a specific model/collection.
        
        Args:
            model_key: Embedding model key
            
        Returns:
            Path to collection's persist directory
        """
        collection_name = settings.get_collection_name(model_key)
        return self.persist_dir / collection_name
    
    def create_or_load_collection(
        self,
        model_key: str,
        embeddings
    ) -> Chroma:
        """
        Create or load a ChromaDB collection for a specific model.
        
        Args:
            model_key: Embedding model key
            embeddings: Embedding model instance
            
        Returns:
            Chroma vector store instance
        """
        collection_name = settings.get_collection_name(model_key)
        persist_dir = self.get_collection_persist_dir(model_key)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if model_key in self._stores:
            logger.info(f"Using cached vector store: {collection_name}")
            return self._stores[model_key]
        
        try:
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=str(persist_dir)
            )
            
            self._stores[model_key] = vectorstore
            
            # Get collection stats
            stats = self.get_collection_stats(model_key)
            logger.info(
                f"Loaded collection '{collection_name}' with {stats['document_count']} documents"
            )
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create/load collection: {e}")
            raise
    
    def add_documents(
        self,
        model_key: str,
        documents: List[Document],
        embeddings
    ) -> Dict[str, int]:
        """
        Add documents to the vector store with deduplication.
        
        Args:
            model_key: Embedding model key
            documents: List of LangChain Documents
            embeddings: Embedding model instance
            
        Returns:
            Stats dict with new/skipped counts
        """
        stats = {"new": 0, "skipped": 0, "total": len(documents)}
        
        if not documents:
            return stats
        
        # Get or create vector store
        vectorstore = self.create_or_load_collection(model_key, embeddings)
        
        try:
            # Get chunk IDs
            chunk_ids = get_chunk_ids(documents)
            
            # Get existing IDs
            existing_ids = self._get_existing_ids(vectorstore)
            
            # Filter out duplicates
            new_docs = []
            new_ids = []
            
            for doc, chunk_id in zip(documents, chunk_ids):
                if chunk_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(chunk_id)
                    stats["new"] += 1
                else:
                    stats["skipped"] += 1
            
            # Add new documents
            if new_docs:
                vectorstore.add_documents(documents=new_docs, ids=new_ids)
                logger.info(f"Added {len(new_docs)} new documents to collection")
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _get_existing_ids(self, vectorstore: Chroma) -> set:
        """
        Get existing document IDs from a collection.
        
        Args:
            vectorstore: Chroma instance
            
        Returns:
            Set of existing IDs
        """
        try:
            collection = vectorstore._collection
            existing_data = collection.get()
            
            if existing_data and existing_data.get("ids"):
                return set(existing_data["ids"])
            
            return set()
            
        except Exception:
            return set()
    
    def get_collection_stats(self, model_key: str) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            model_key: Embedding model key
            
        Returns:
            Stats dict
        """
        stats = {
            "collection_name": settings.get_collection_name(model_key),
            "document_count": 0,
            "has_data": False,
            "persist_dir": str(self.get_collection_persist_dir(model_key))
        }
        
        if model_key not in self._stores:
            return stats
        
        try:
            vectorstore = self._stores[model_key]
            collection = vectorstore._collection
            count = collection.count()
            
            stats["document_count"] = count
            stats["has_data"] = count > 0
            
        except Exception as e:
            logger.warning(f"Could not get collection stats: {e}")
        
        return stats
    
    def reset_collection(self, model_key: str) -> bool:
        """
        Reset/delete a specific collection.
        
        Args:
            model_key: Embedding model key
            
        Returns:
            True if successful
        """
        persist_dir = self.get_collection_persist_dir(model_key)
        
        try:
            # Remove from cache
            if model_key in self._stores:
                del self._stores[model_key]
            
            # Delete persist directory
            if persist_dir.exists():
                shutil.rmtree(persist_dir)
            
            # Recreate empty directory
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Reset collection for {model_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def reset_all_collections(self) -> bool:
        """
        Reset all collections.
        
        Returns:
            True if successful
        """
        try:
            # Clear cache
            self._stores.clear()
            
            # Delete entire persist directory
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
            
            # Recreate
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Reset all collections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset all collections: {e}")
            return False
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections.
        
        Returns:
            List of collection info dicts
        """
        collections = []
        
        for model_key in settings.embedding_models.keys():
            stats = self.get_collection_stats(model_key)
            collections.append({
                "model_key": model_key,
                **stats
            })
        
        return collections
    
    def get_vectorstore(self, model_key: str) -> Optional[Chroma]:
        """
        Get a vector store instance if it exists.
        
        Args:
            model_key: Embedding model key
            
        Returns:
            Chroma instance or None
        """
        return self._stores.get(model_key)


# Global instance
_vector_store_manager: Optional[VectorStoreManager] = None


def get_vector_store_manager() -> VectorStoreManager:
    """
    Get the global vector store manager instance.
    
    Returns:
        VectorStoreManager instance
    """
    global _vector_store_manager
    
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    
    return _vector_store_manager


def reset_vector_store_manager():
    """Reset the global vector store manager."""
    global _vector_store_manager
    _vector_store_manager = None
