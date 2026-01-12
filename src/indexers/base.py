"""
Base indexer abstract class.

This module defines the abstract interface that all indexer implementations
must follow.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseIndexer(ABC):
    """
    Abstract base class for vector database indexers.
    
    This class defines the interface that all concrete indexer implementations
    must follow for storing and retrieving text chunks.
    
    Attributes:
        collection_name (str): Name of the collection/index
        
    Methods:
        add_chunks: Abstract method to add chunks to the index
        search: Abstract method to search for similar chunks
        get_collection_stats: Abstract method to get collection statistics
        delete_collection: Abstract method to delete the collection
    """
    
    def __init__(self, collection_name: str):
        """
        Initialize the base indexer.
        
        Args:
            collection_name (str): Name of the collection to create or use
        """
        self.collection_name = collection_name
    
    @abstractmethod
    def add_chunks(self, chunks: List[str], metadatas: List[Dict], 
                   chunk_ids: List[str]) -> None:
        """
        Add text chunks to the index.
        
        Args:
            chunks (List[str]): List of text chunks to index
            metadatas (List[Dict]): List of metadata dictionaries for each chunk
            chunk_ids (List[str]): List of unique IDs for each chunk
            
        Returns:
            None
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def search(self, query: str, n_results: int = 5) -> Dict:
        """
        Search for similar chunks.
        
        Args:
            query (str): Search query text
            n_results (int, optional): Number of results to return. Defaults to 5.
            
        Returns:
            Dict: Search results containing chunks, distances, and metadata
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dict: Dictionary containing collection statistics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """
        Delete the collection from the database.
        
        Returns:
            None
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass