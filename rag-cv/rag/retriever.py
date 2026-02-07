"""
Retriever Module
================

Configurable retriever with top-k and optional filtering.
"""

from typing import List, Dict, Any, Optional

from .logging_utils import get_logger
from .config import settings

logger = get_logger(__name__)

# LangChain imports
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


class RetrieverConfig:
    """Configuration for the retriever."""
    
    def __init__(
        self,
        top_k: int = None,
        search_type: str = "similarity",
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retriever configuration.
        
        Args:
            top_k: Number of documents to retrieve
            search_type: "similarity" or "mmr"
            score_threshold: Minimum similarity score (0-1)
            filter_metadata: Metadata filters (e.g., {"source": "resume.pdf"})
        """
        self.top_k = top_k or settings.default_top_k
        self.search_type = search_type
        self.score_threshold = score_threshold
        self.filter_metadata = filter_metadata or {}


class CVRetriever:
    """
    Retriever for CV documents with configurable options.
    """
    
    def __init__(
        self,
        vectorstore,
        config: Optional[RetrieverConfig] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            vectorstore: Chroma vector store instance
            config: Retriever configuration
        """
        self.vectorstore = vectorstore
        self.config = config or RetrieverConfig()
        self._retriever = None
        
        self._build_retriever()
    
    def _build_retriever(self):
        """Build the LangChain retriever."""
        search_kwargs = {"k": self.config.top_k}
        
        # Add filter if specified
        if self.config.filter_metadata:
            search_kwargs["filter"] = self.config.filter_metadata
        
        # MMR configuration
        if self.config.search_type == "mmr":
            self._retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    **search_kwargs,
                    "fetch_k": self.config.top_k * 4,
                    "lambda_mult": 0.5  # Balance relevance and diversity
                }
            )
        else:
            # Standard similarity search
            self._retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs
            )
        
        logger.info(
            f"Built retriever: type={self.config.search_type}, "
            f"top_k={self.config.top_k}"
        )
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant Documents
        """
        try:
            docs = self._retriever.invoke(query)
            
            # Apply score threshold if configured
            if self.config.score_threshold is not None:
                docs = self._filter_by_score(query, docs)
            
            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def _filter_by_score(
        self,
        query: str,
        docs: List[Document]
    ) -> List[Document]:
        """
        Filter documents by similarity score.
        
        Args:
            query: Original query
            docs: Retrieved documents
            
        Returns:
            Filtered documents
        """
        # Get scores
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=len(docs) * 2
            )
            
            # Filter by threshold
            filtered = [
                doc for doc, score in results
                if score >= self.config.score_threshold
            ]
            
            return filtered[:self.config.top_k]
            
        except Exception as e:
            logger.warning(f"Score filtering failed: {e}")
            return docs
    
    def retrieve_with_scores(
        self,
        query: str
    ) -> List[tuple]:
        """
        Retrieve documents with similarity scores.
        
        Args:
            query: Search query
            
        Returns:
            List of (Document, score) tuples
        """
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query,
                k=self.config.top_k
            )
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Scored retrieval failed: {e}")
            return []
    
    def get_retriever(self):
        """
        Get the underlying LangChain retriever.
        
        Returns:
            LangChain retriever instance
        """
        return self._retriever
    
    def update_config(self, config: RetrieverConfig):
        """
        Update retriever configuration.
        
        Args:
            config: New configuration
        """
        self.config = config
        self._build_retriever()


def create_retriever(
    vectorstore,
    top_k: int = None,
    search_type: str = "similarity",
    score_threshold: Optional[float] = None,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> CVRetriever:
    """
    Create a retriever with specified configuration.
    
    Args:
        vectorstore: Chroma vector store instance
        top_k: Number of documents to retrieve
        search_type: "similarity" or "mmr"
        score_threshold: Minimum similarity score
        filter_metadata: Metadata filters
        
    Returns:
        CVRetriever instance
    """
    config = RetrieverConfig(
        top_k=top_k,
        search_type=search_type,
        score_threshold=score_threshold,
        filter_metadata=filter_metadata
    )
    
    return CVRetriever(vectorstore, config)


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents for display or context.
    
    Args:
        docs: List of Documents
        
    Returns:
        Formatted string
    """
    if not docs:
        return "No relevant documents found."
    
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = meta.get("source", "Unknown")
        page = meta.get("page_start", meta.get("page", "?"))
        chunk_id = meta.get("chunk_id", "?")
        candidate = meta.get("candidate_name", "Unknown")
        
        header = f"[Source {i}: {source}, Page {page}, Chunk: {chunk_id}]"
        if candidate != "Unknown":
            header = f"[Source {i}: {source} ({candidate}), Page {page}]"
        
        content = doc.page_content.strip()
        formatted_parts.append(f"{header}\n{content}")
    
    return "\n\n---\n\n".join(formatted_parts)


def get_sources_summary(docs: List[Document]) -> List[Dict[str, Any]]:
    """
    Get a summary of sources from retrieved documents.
    
    Args:
        docs: List of Documents
        
    Returns:
        List of source summary dicts
    """
    sources = []
    
    for doc in docs:
        meta = doc.metadata
        sources.append({
            "source": meta.get("source", "Unknown"),
            "page": meta.get("page_start", meta.get("page", "?")),
            "chunk_id": meta.get("chunk_id", "?"),
            "candidate_name": meta.get("candidate_name", "Unknown"),
            "preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        })
    
    return sources
