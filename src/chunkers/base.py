"""
Base chunker abstract class.

This module defines the abstract interface that all chunking strategies
must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from src.preprocessors.base import BasePreprocessor


class BaseChunker(ABC):
    """
    Abstract base class for chunking strategies.
    
    This class defines the interface that all concrete chunking implementations
    must follow and provides common evaluation functionality.
    
    Attributes:
        name (str): Human-readable name of the chunking strategy
        
    Methods:
        chunk_texts: Abstract method to chunk texts into smaller pieces
        evaluate: Evaluate the chunking results
    """
    
    def __init__(self, name: str):
        """
        Initialize the base chunker.
        
        Args:
            name (str): Name of the chunking strategy for identification
        """
        self.name = name
    
    @abstractmethod
    def chunk_texts(self, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Chunk texts and return chunks with metadata.
        
        This abstract method must be implemented by concrete chunker classes
        to define their specific chunking strategy.
        
        Args:
            texts (List[str]): List of text documents to chunk
            metadatas (List[Dict]): List of metadata dictionaries corresponding
                to each text document
                
        Returns:
            Tuple[List[str], List[Dict], List[str]]: A tuple containing:
                - List of text chunks
                - List of metadata dictionaries for each chunk
                - List of unique chunk IDs
                
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    def evaluate(self, chunks: List[str], chunk_idx1: int = 3, 
                chunk_idx2: int = 6, use_tokens: bool = True) -> None:
        """
        Evaluate chunking strategy by analyzing sample chunks.
        
        Prints detailed information about the chunking results including
        chunk content, overlap analysis, and token counts.
        
        Args:
            chunks (List[str]): List of text chunks to evaluate
            chunk_idx1 (int, optional): Index of first chunk to analyze. 
                Defaults to 3.
            chunk_idx2 (int, optional): Index of second chunk to analyze. 
                Defaults to 6.
            use_tokens (bool, optional): If True, analyze at token level,
                otherwise at character level. Defaults to True.
                
        Returns:
            None: Prints evaluation results to stdout
            
        Note:
            Ensure chunk_idx1 and chunk_idx2 are within the valid range
            of the chunks list
        """
        print(f"\n{'='*60}")
        print(f"{self.name} Evaluation")
        print(f"{'='*60}\n")
        
        BasePreprocessor.analyze_chunks(chunks, chunk_idx1, chunk_idx2, use_tokens)
        
        print(f"\nTokens in chunk[{chunk_idx1}]:", 
              BasePreprocessor.count_tokens(chunks[chunk_idx1]))
        print(f"\nTokens in chunk[{chunk_idx2}]:", 
              BasePreprocessor.count_tokens(chunks[chunk_idx2]))

