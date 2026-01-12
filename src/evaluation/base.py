"""
Base evaluator abstract class.

This module defines the abstract interface for evaluation implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseEvaluator(ABC):
    """
    Abstract base class for RAG evaluation.
    
    This class defines the interface that all concrete evaluator implementations
    must follow for assessing retrieval quality.
    
    Attributes:
        name (str): Name of the evaluator
        
    Methods:
        load_queries: Abstract method to load evaluation queries
        evaluate: Abstract method to run evaluation
        generate_report: Abstract method to generate evaluation report
    """
    
    def __init__(self, name: str):
        """
        Initialize the base evaluator.
        
        Args:
            name (str): Name of the evaluator for identification
        """
        self.name = name
    
    @abstractmethod
    def load_queries(self, queries_path: str) -> List[Tuple[str, str, str]]:
        """
        Load evaluation queries from file.
        
        Args:
            queries_path (str): Path to the queries file
            
        Returns:
            List[Tuple[str, str, str]]: List of (query, source_file, expected_answer)
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def evaluate(self, indexer, queries: List[Tuple[str, str, str]], 
                k: int = 5) -> Dict:
        """
        Run evaluation on queries.
        
        Args:
            indexer: Indexer instance to query
            queries: List of evaluation queries
            k (int): Number of results to retrieve
            
        Returns:
            Dict: Evaluation results and metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
