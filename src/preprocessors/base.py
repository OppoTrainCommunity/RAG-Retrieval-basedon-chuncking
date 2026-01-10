import re
import tiktoken
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

"""
Base preprocessor abstract class for text preprocessing operations.

This module defines the abstract interface and common utilities for all
preprocessor implementations.
"""

class BasePreprocessor(ABC):
    """
    Abstract base class for text preprocessing.
    
    This class provides common preprocessing utilities and defines the interface
    that all concrete preprocessors must implement.
    
    Attributes:
        None
    
    Methods:
        clean_pdf_text: Static method to clean PDF text formatting
        prepare_data: Abstract method to load and prepare data from source
        count_tokens: Static method to count tokens in text
        analyze_chunks: Static method to analyze and compare chunks
    """    
    @staticmethod
    def clean_pdf_text(text: str) -> str:
        """
        Clean PDF text by removing extra spaces and newlines.
        
        This method normalizes whitespace and removes excessive line breaks
        that are common in PDF extractions.
        
        Args:
            text (str): Raw text extracted from PDF
            
        Returns:
            str: Cleaned text with normalized whitespace
            
        Examples:
            >>> BasePreprocessor.clean_pdf_text("Hello    World\n\n\n\nTest")
            'Hello World\n\nTest'
        """
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    @abstractmethod
    def prepare_data(self, data_path: str) -> Tuple[List[str], List[Dict]]:
        """
        Load and prepare data from source.
        
        This abstract method must be implemented by concrete preprocessor classes
        to handle their specific data format.
        
        Args:
            data_path (str): Path to the data file
            
        Returns:
            Tuple[List[str], List[Dict]]: A tuple containing:
                - List of text documents
                - List of metadata dictionaries corresponding to each document
                
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @staticmethod
    def count_tokens(text: str, model: str = "cl100k_base") -> int:
        """
        Count tokens in a text string using tiktoken.
        
        Args:
            text (str): Text to tokenize and count
            model (str, optional): Tiktoken encoding model name. 
                Defaults to "cl100k_base".
                
        Returns:
            int: Number of tokens in the text
            
        Examples:
            >>> BasePreprocessor.count_tokens("Hello world")
            2
        """
        encoder = tiktoken.get_encoding(model)
        num_tokens = len(encoder.encode(text))
        return num_tokens
    
    @staticmethod
    def analyze_chunks(chunks: List[str], chunk_idx1: int, chunk_idx2: int, 
                      use_tokens: bool = False) -> None:
        """
        Analyze and print chunk details including overlap detection.
        
        This method prints the content of two specified chunks and identifies
        any overlapping content between them, either at the token or character level.
        
        Args:
            chunks (List[str]): List of text chunks to analyze
            chunk_idx1 (int): Index of the first chunk to analyze
            chunk_idx2 (int): Index of the second chunk to analyze
            use_tokens (bool, optional): If True, analyze overlap at token level,
                otherwise at character level. Defaults to False.
                
        Returns:
            None: Prints analysis results to stdout
            
        Note:
            Assumes chunk_idx2 comes after chunk_idx1 when checking for overlap
        """
        print("\nNumber of Chunks:", len(chunks))
        print("\n", "="*50, f"Chunk {chunk_idx1}", "="*50)
        print(chunks[chunk_idx1])
        print("\n", "="*50, f"Chunk {chunk_idx2}", "="*50)
        print(chunks[chunk_idx2])
        
        chunk1, chunk2 = chunks[chunk_idx1], chunks[chunk_idx2]
        
        if use_tokens:
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens1 = encoding.encode(chunk1)
            tokens2 = encoding.encode(chunk2)
            
            for i in range(len(tokens1), 0, -1):
                if tokens1[-i:] == tokens2[:i]:
                    overlap = encoding.decode(tokens1[-i:])
                    print("\n", "="*50)
                    print(f"Overlapping text ({i} tokens):", overlap)
                    return
            print("\nNo token overlap found")
        else:
            for i in range(min(len(chunk1), len(chunk2)), 0, -1):
                if chunk1[-i:] == chunk2[:i]:
                    print("\n", "="*50)
                    print(f"Overlapping text ({i} chars):", chunk1[-i:])
                    return
            print("\nNo character overlap found")

