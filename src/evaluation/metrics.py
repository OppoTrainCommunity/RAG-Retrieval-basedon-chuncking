"""
Retrieval evaluation metrics.

This module provides implementations of various retrieval quality metrics
such as Precision, Recall, MRR, Hit Rate, etc.
"""
from typing import List, Dict, Set
import time


class RetrievalMetrics:
    """
    Collection of retrieval evaluation metrics.
    
    This class provides static methods for calculating various metrics used
    to evaluate information retrieval quality.
    
    Methods:
        calculate_precision: Calculate precision at k
        calculate_recall: Calculate recall at k
        calculate_mrr: Calculate Mean Reciprocal Rank
        calculate_hit_rate: Calculate hit rate
        calculate_average_precision: Calculate Average Precision
        calculate_ndcg: Calculate Normalized Discounted Cumulative Gain
    """
    
    @staticmethod
    def calculate_precision(retrieved_docs: List[str], 
                           relevant_docs: Set[str]) -> float:
        """
        Calculate precision: ratio of relevant documents in retrieved set.
        
        Precision = |Retrieved ∩ Relevant| / |Retrieved|
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            relevant_docs (Set[str]): Set of relevant document IDs
            
        Returns:
            float: Precision score between 0 and 1
            
        Examples:
            >>> retrieved = ["doc1", "doc2", "doc3"]
            >>> relevant = {"doc1", "doc3", "doc5"}
            >>> RetrievalMetrics.calculate_precision(retrieved, relevant)
            0.6666666666666666
        """
        if not retrieved_docs:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        return relevant_retrieved / len(retrieved_docs)
    
    @staticmethod
    def calculate_recall(retrieved_docs: List[str], 
                        relevant_docs: Set[str]) -> float:
        """
        Calculate recall: ratio of relevant documents that were retrieved.
        
        Recall = |Retrieved ∩ Relevant| / |Relevant|
        
        Args:
            retrieved_docs (List[str]): List of retrieved document IDs
            relevant_docs (Set[str]): Set of relevant document IDs
            
        Returns:
            float: Recall score between 0 and 1
            
        Examples:
            >>> retrieved = ["doc1", "doc2", "doc3"]
            >>> relevant = {"doc1", "doc3", "doc5"}
            >>> RetrievalMetrics.calculate_recall(retrieved, relevant)
            0.6666666666666666
        """
        if not relevant_docs:
            return 0.0
        
        relevant_retrieved = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        return relevant_retrieved / len(relevant_docs)
    
    @staticmethod
    def calculate_mrr(retrieved_docs_list: List[List[str]], 
                     relevant_docs_list: List[Set[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank across multiple queries.
        
        MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is position of first relevant doc
        
        Args:
            retrieved_docs_list (List[List[str]]): List of retrieved doc lists for each query
            relevant_docs_list (List[Set[str]]): List of relevant doc sets for each query
            
        Returns:
            float: MRR score between 0 and 1
            
        Examples:
            >>> retrieved = [["doc1", "doc2"], ["doc3", "doc1"]]
            >>> relevant = [{"doc2"}, {"doc1"}]
            >>> RetrievalMetrics.calculate_mrr(retrieved, relevant)
            0.75
        """
        reciprocal_ranks = []
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            rank = None
            for i, doc in enumerate(retrieved, 1):
                if doc in relevant:
                    rank = i
                    break
            
            if rank:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def calculate_hit_rate(retrieved_docs_list: List[List[str]], 
                          relevant_docs_list: List[Set[str]]) -> float:
        """
        Calculate hit rate: percentage of queries with at least one relevant result.
        
        Args:
            retrieved_docs_list (List[List[str]]): List of retrieved doc lists for each query
            relevant_docs_list (List[Set[str]]): List of relevant doc sets for each query
            
        Returns:
            float: Hit rate between 0 and 1
            
        Examples:
            >>> retrieved = [["doc1", "doc2"], ["doc3", "doc4"]]
            >>> relevant = [{"doc2"}, {"doc5"}]
            >>> RetrievalMetrics.calculate_hit_rate(retrieved, relevant)
            0.5
        """
        hits = 0
        
        for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
            if any(doc in relevant for doc in retrieved):
                hits += 1
        
        return hits / len(retrieved_docs_list) if retrieved_docs_list else 0.0
    
    @staticmethod
    def calculate_f1(precision: float, recall: float) -> float:
        """
        Calculate F1 score (harmonic mean of precision and recall).
        
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            precision (float): Precision score
            recall (float): Recall score
            
        Returns:
            float: F1 score between 0 and 1
            
        Examples:
            >>> RetrievalMetrics.calculate_f1(0.8, 0.6)
            0.6857142857142857
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
