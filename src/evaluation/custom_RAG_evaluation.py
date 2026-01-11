import json
import numpy as np
from typing import List, Dict
import chromadb
from src.chunkers.base import BaseChunker
from src.evaluation.Evaluation_metrics import EvaluationMetrics

class CustomRAGEvaluation:
    """
    Custom evaluation framework for RAG retrieval systems.
    
    This class evaluates chunking strategies in the context of retrieval
    using your own question-answer pairs and domain-specific data.
    
    Attributes:
        questions (List[str]): List of evaluation questions
        ground_truth_docs (List[List[str]]): Ground truth document IDs for each question
        k_values (List[int]): K values for top-k evaluation
        
    Methods:
        load_qa_data: Load question-answer pairs from file
        evaluate_retrieval: Evaluate a chunker's retrieval performance
        calculate_metrics: Calculate all evaluation metrics
    """
    
    def __init__(self, k_values: List[int] = [1, 3, 5, 10]):
        """
        Initialize the custom RAG evaluation.
        
        Args:
            k_values (List[int], optional): K values for evaluating top-k retrieval.
                Defaults to [1, 3, 5, 10].
        """
        self.questions = []
        self.ground_truth_docs = []
        self.k_values = k_values
    
    def load_qa_data(self, qa_file_path: str) -> None:
        """
        Load question-answer pairs from JSON file.
        
        Expected JSON format:
        [
            {
                "question": "What is X?",
                "answer": "X is...",
                "relevant_doc_ids": ["doc1_chunk_3", "doc1_chunk_4"]
            },
            ...
        ]
        
        Args:
            qa_file_path (str): Path to the JSON file containing QA pairs
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
            KeyError: If required fields are missing
        """
        with open(qa_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        for item in qa_data:
            self.questions.append(item['question'])
            self.ground_truth_docs.append(item.get('relevant_doc_ids', []))
    
    def load_qa_from_dict(self, qa_data: List[Dict]) -> None:
        """
        Load question-answer pairs from dictionary.
        
        Args:
            qa_data (List[Dict]): List of QA dictionaries with keys:
                - 'question': Question text
                - 'relevant_doc_ids': List of relevant chunk IDs
        """
        for item in qa_data:
            self.questions.append(item['question'])
            self.ground_truth_docs.append(item.get('relevant_doc_ids', []))
    
    def _create_vector_store(self, chunks: List[str], chunk_ids: List[str],
                            chunk_metas: List[Dict], embedding_fn) -> chromadb.Collection:
        """
        Create a ChromaDB collection with chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            chunk_ids (List[str]): List of chunk IDs
            chunk_metas (List[Dict]): List of chunk metadata
            embedding_fn: Embedding function to use
            
        Returns:
            chromadb.Collection: Collection containing the chunks
        """
        client = chromadb.Client()
        collection = client.create_collection(
            name="eval_collection",
            embedding_function=embedding_fn
        )
        
        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=chunk_metas
        )
        
        return collection
    
    def _calculate_precision_at_k(self, retrieved_ids: List[str], 
                                  relevant_ids: List[str], k: int) -> float:
        """
        Calculate precision at k.
        
        Precision@k = (relevant items in top-k) / k
        
        Args:
            retrieved_ids (List[str]): Retrieved chunk IDs (ordered by relevance)
            relevant_ids (List[str]): Ground truth relevant chunk IDs
            k (int): Number of top results to consider
            
        Returns:
            float: Precision at k
        """
        if k == 0 or len(retrieved_ids) == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_ids))
        return relevant_in_top_k / k
    
    def _calculate_recall_at_k(self, retrieved_ids: List[str], 
                               relevant_ids: List[str], k: int) -> float:
        """
        Calculate recall at k.
        
        Recall@k = (relevant items in top-k) / (total relevant items)
        
        Args:
            retrieved_ids (List[str]): Retrieved chunk IDs (ordered by relevance)
            relevant_ids (List[str]): Ground truth relevant chunk IDs
            k (int): Number of top results to consider
            
        Returns:
            float: Recall at k
        """
        if len(relevant_ids) == 0:
            return 0.0
        
        top_k = retrieved_ids[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_ids))
        return relevant_in_top_k / len(relevant_ids)
    
    def _calculate_mrr(self, retrieved_ids: List[str], 
                       relevant_ids: List[str]) -> float:
        """
        Calculate Mean Reciprocal Rank.
        
        MRR = 1 / (rank of first relevant item)
        
        Args:
            retrieved_ids (List[str]): Retrieved chunk IDs (ordered by relevance)
            relevant_ids (List[str]): Ground truth relevant chunk IDs
            
        Returns:
            float: Reciprocal rank (0 if no relevant items found)
        """
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0
    
    def _calculate_ndcg_at_k(self, retrieved_ids: List[str], 
                            relevant_ids: List[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        NDCG measures ranking quality, giving more weight to relevant items
        appearing earlier in the results.
        
        Args:
            retrieved_ids (List[str]): Retrieved chunk IDs (ordered by relevance)
            relevant_ids (List[str]): Ground truth relevant chunk IDs
            k (int): Number of top results to consider
            
        Returns:
            float: NDCG at k
        """
        def dcg_at_k(relevances, k):
            relevances = np.array(relevances[:k])
            if relevances.size == 0:
                return 0.0
            return np.sum(relevances / np.log2(np.arange(2, relevances.size + 2)))
        
        # Create binary relevance list (1 if relevant, 0 otherwise)
        top_k = retrieved_ids[:k]
        relevances = [1 if doc_id in relevant_ids else 0 for doc_id in top_k]
        
        # Calculate DCG
        dcg = dcg_at_k(relevances, k)
        
        # Calculate IDCG (best possible DCG)
        ideal_relevances = [1] * min(len(relevant_ids), k)
        idcg = dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _calculate_hit_rate_at_k(self, retrieved_ids: List[str], 
                                 relevant_ids: List[str], k: int) -> float:
        """
        Calculate hit rate at k.
        
        Hit rate = 1 if at least one relevant item in top-k, else 0
        
        Args:
            retrieved_ids (List[str]): Retrieved chunk IDs
            relevant_ids (List[str]): Ground truth relevant chunk IDs
            k (int): Number of top results to consider
            
        Returns:
            float: 1.0 if hit, 0.0 otherwise
        """
        top_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0
    
    def evaluate_retrieval(self, chunker: BaseChunker, texts: List[str], 
                          metadatas: List[Dict], embedding_fn) -> EvaluationMetrics:
        """
        Evaluate retrieval performance of a chunking strategy.
        
        Args:
            chunker (BaseChunker): Chunking strategy to evaluate
            texts (List[str]): Source texts to chunk
            metadatas (List[Dict]): Metadata for source texts
            embedding_fn: Embedding function for vector store
            
        Returns:
            EvaluationMetrics: Comprehensive evaluation metrics
            
        Examples:
            >>> evaluation = CustomRAGEvaluation()
            >>> evaluation.load_qa_data("qa_pairs.json")
            >>> metrics = evaluation.evaluate_retrieval(chunker, texts, metas, embed_fn)
            >>> print(f"MRR: {metrics.mrr}")
        """
        # Chunk the texts
        chunks, chunk_metas, chunk_ids = chunker.chunk_texts(texts, metadatas)
        
        # Create vector store
        collection = self._create_vector_store(chunks, chunk_ids, chunk_metas, embedding_fn)
        
        # Initialize metric accumulators
        precision_scores = {k: [] for k in self.k_values}
        recall_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        hit_rate_scores = {k: [] for k in self.k_values}
        mrr_scores = []
        
        # Evaluate each question
        for question, relevant_ids in zip(self.questions, self.ground_truth_docs):
            # Query the collection
            results = collection.query(
                query_texts=[question],
                n_results=max(self.k_values)
            )
            
            retrieved_ids = results['ids'][0]
            
            # Calculate metrics for each k
            for k in self.k_values:
                precision_scores[k].append(
                    self._calculate_precision_at_k(retrieved_ids, relevant_ids, k)
                )
                recall_scores[k].append(
                    self._calculate_recall_at_k(retrieved_ids, relevant_ids, k)
                )
                ndcg_scores[k].append(
                    self._calculate_ndcg_at_k(retrieved_ids, relevant_ids, k)
                )
                hit_rate_scores[k].append(
                    self._calculate_hit_rate_at_k(retrieved_ids, relevant_ids, k)
                )
            
            # Calculate MRR
            mrr_scores.append(self._calculate_mrr(retrieved_ids, relevant_ids))
        
        # Aggregate metrics
        metrics = EvaluationMetrics(
            precision_at_k={k: np.mean(scores) for k, scores in precision_scores.items()},
            recall_at_k={k: np.mean(scores) for k, scores in recall_scores.items()},
            mrr=np.mean(mrr_scores),
            ndcg_at_k={k: np.mean(scores) for k, scores in ndcg_scores.items()},
            hit_rate_at_k={k: np.mean(scores) for k, scores in hit_rate_scores.items()}
        )
        
        return metrics
    
    def print_results(self, metrics: EvaluationMetrics, chunker_name: str = "Chunker") -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            metrics (EvaluationMetrics): Metrics to print
            chunker_name (str, optional): Name of the chunker. Defaults to "Chunker".
        """
        print(f"\n{'='*60}")
        print(f"{chunker_name} - Retrieval Evaluation Results")
        print(f"{'='*60}\n")
        
        print(f"Mean Reciprocal Rank (MRR): {metrics.mrr:.4f}")
        print()
        
        print("Precision @ K:")
        for k, score in sorted(metrics.precision_at_k.items()):
            print(f"  P@{k}: {score:.4f}")
        print()
        
        print("Recall @ K:")
        for k, score in sorted(metrics.recall_at_k.items()):
            print(f"  R@{k}: {score:.4f}")
        print()
        
        print("NDCG @ K:")
        for k, score in sorted(metrics.ndcg_at_k.items()):
            print(f"  NDCG@{k}: {score:.4f}")
        print()
        
        print("Hit Rate @ K:")
        for k, score in sorted(metrics.hit_rate_at_k.items()):
            print(f"  HR@{k}: {score:.4f}")

