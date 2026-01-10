"""
Semantic chunking implementation.

This module implements semantic chunking using the Kamradt modified algorithm,
which groups text based on semantic similarity of embeddings.
"""
from typing import List, Dict, Tuple
from chromadb.utils import embedding_functions
from chunking_evaluation.chunking import KamradtModifiedChunker
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    Semantic chunking using Kamradt's modified algorithm.
    
    This chunker uses sentence embeddings to identify semantic boundaries
    in text and creates chunks that are semantically coherent.
    
    Attributes:
        name (str): Name of the chunking strategy
        embedding_fn: Embedding function for computing text embeddings
        chunker: KamradtModifiedChunker instance
        
    Methods:
        chunk_texts: Perform semantic chunking on input texts
    """
    
    def __init__(self, avg_chunk_size: int = 400, min_chunk_size: int = 50,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic chunker.
        
        Args:
            avg_chunk_size (int, optional): Target average chunk size in tokens.
                Defaults to 400.
            min_chunk_size (int, optional): Minimum initial split size in tokens.
                Defaults to 50.
            embedding_model (str, optional): Name of the sentence transformer model
                to use for embeddings. Defaults to "all-MiniLM-L6-v2".
                
        Note:
            The chunker will attempt to create chunks close to avg_chunk_size
            while respecting semantic boundaries detected through embeddings
        """
        super().__init__("Semantic Chunking")
        
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.chunker = KamradtModifiedChunker(
            avg_chunk_size=avg_chunk_size,
            min_chunk_size=min_chunk_size,
            embedding_function=self.embedding_fn,
        )
    
    def chunk_texts(self, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Perform semantic chunking on texts.
        
        Splits each input text into semantically coherent chunks using
        embedding-based boundary detection.
        
        Args:
            texts (List[str]): List of text documents to chunk
            metadatas (List[Dict]): List of metadata dictionaries corresponding
                to each text document
                
        Returns:
            Tuple[List[str], List[Dict], List[str]]: A tuple containing:
                - List of text chunks
                - List of metadata dictionaries for each chunk with added fields:
                    - 'chunk_id': Sequential ID within the document
                    - 'chunk_type': Set to "semantic"
                - List of unique chunk IDs in format "{doc_id}_sem_{chunk_num}"
                
        Examples:
            >>> chunker = SemanticChunker()
            >>> texts = ["Long document text..."]
            >>> metas = [{"doc_id": "doc1", "file_name": "test.pdf"}]
            >>> chunks, chunk_metas, chunk_ids = chunker.chunk_texts(texts, metas)
        """
        chunks = []
        chunk_metas = []
        chunk_ids = []
        
        for text, meta in zip(texts, metadatas):
            text_chunks = self.chunker.split_text(text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                chunk_metas.append({
                    **meta, 
                    "chunk_id": i, 
                    "chunk_type": "semantic"
                })
                chunk_ids.append(f"{meta['doc_id']}_sem_{i}")
        
        return chunks, chunk_metas, chunk_ids

