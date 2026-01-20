"""
ChromaDB vector store for CV RAG System.
Handles indexing, loading, and querying of CV chunks.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import pandas as pd

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB vector store wrapper for CV chunks.
    """
    
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "cv_rag",
        embeddings: Optional[Embeddings] = None,
    ):
        """
        Initialize ChromaDB store.
        
        Args:
            persist_dir: Directory for persistent storage.
            collection_name: Name of the Chroma collection.
            embeddings: Embeddings instance for vectorization.
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embeddings = embeddings
        
        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Chroma client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Get or create the vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client=self._client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._vectorstore
    
    def build_index(
        self,
        chunks_df: pd.DataFrame,
        text_col: str = "chunk_text",
        id_col: str = "chunk_id",
        metadata_cols: Optional[List[str]] = None,
    ) -> None:
        """
        Build the vector index from chunks DataFrame.
        
        Args:
            chunks_df: DataFrame with chunk data.
            text_col: Name of the text column.
            id_col: Name of the ID column.
            metadata_cols: List of metadata columns to include.
        """
        if metadata_cols is None:
            metadata_cols = [
                "candidate_id", "section_name", "chunk_index",
                "name", "email", "role", "location", "years_experience",
            ]
        
        # Filter to existing columns
        metadata_cols = [c for c in metadata_cols if c in chunks_df.columns]
        
        logger.info(f"Building index with {len(chunks_df)} chunks...")
        
        # Create documents
        documents = []
        for _, row in chunks_df.iterrows():
            # Build metadata, handling NaN values
            metadata = {}
            for col in metadata_cols:
                value = row.get(col)
                if pd.notna(value):
                    # Convert to string for Chroma compatibility
                    if isinstance(value, (int, float)):
                        metadata[col] = value
                    else:
                        metadata[col] = str(value)
            
            doc = Document(
                page_content=str(row[text_col]),
                metadata=metadata,
            )
            documents.append(doc)
        
        # Get chunk IDs
        ids = chunks_df[id_col].astype(str).tolist()
        
        # Add to vectorstore
        self._vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            client=self._client,
            ids=ids,
        )
        
        logger.info(f"Built index with {len(documents)} documents")
    
    def load_index(self) -> bool:
        """
        Load existing index from disk.
        
        Returns:
            True if index exists and was loaded, False otherwise.
        """
        try:
            # Check if collection exists
            collections = self._client.list_collections()
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                self._vectorstore = Chroma(
                    client=self._client,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                )
                count = self._vectorstore._collection.count()
                logger.info(f"Loaded existing index with {count} documents")
                return True
            else:
                logger.warning(f"Collection '{self.collection_name}' not found")
                return False
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def reset_index(self) -> None:
        """
        Reset (delete) the index.
        """
        try:
            self._client.delete_collection(self.collection_name)
            self._vectorstore = None
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")
    
    def get_retriever(
        self,
        top_k: int = 6,
        search_type: str = "similarity",
        filter_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Get a LangChain retriever for the vectorstore.
        
        Args:
            top_k: Number of documents to retrieve.
            search_type: Type of search ("similarity" or "mmr").
            filter_dict: Optional metadata filter.
        
        Returns:
            LangChain retriever.
        """
        search_kwargs = {"k": top_k}
        
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
    
    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query text.
            k: Number of results.
            filter_dict: Optional metadata filter.
        
        Returns:
            List of matching documents.
        """
        if filter_dict:
            return self.vectorstore.similarity_search(
                query, k=k, filter=filter_dict
            )
        return self.vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Query text.
            k: Number of results.
            filter_dict: Optional metadata filter.
        
        Returns:
            List of (document, score) tuples.
        """
        if filter_dict:
            return self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the index."""
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
    
    def get_all_candidate_ids(self) -> List[str]:
        """
        Get all unique candidate IDs in the store.
        """
        if not self._vectorstore:
            return []
            
        try:
            # Fetch all metadata
            # Note: For large collections, this might be slow. 
            # In a production system, we would maintain a separate index of candidates.
            result = self._vectorstore._collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])
            
            candidate_ids = set()
            for m in metadatas:
                if m and "candidate_id" in m:
                    candidate_ids.add(str(m["candidate_id"]))
            
            return sorted(list(candidate_ids))
        except Exception as e:
            logger.error(f"Error getting candidate IDs: {e}")
            return []

    @classmethod
    def from_config(cls, config, embeddings: Embeddings) -> "ChromaStore":
        """
        Create ChromaStore from configuration.
        
        Args:
            config: Config object.
            embeddings: Embeddings instance.
        
        Returns:
            ChromaStore instance.
        """
        return cls(
            persist_dir=config.chroma.persist_dir,
            collection_name=config.chroma.collection,
            embeddings=embeddings,
        )
