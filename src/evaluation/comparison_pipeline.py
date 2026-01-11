from typing import List, Dict
from src.chunkers.base import BaseChunker
from src.evaluation.Evaluation_metrics import EvaluationMetrics
from src.evaluation.custom_RAG_evaluation import CustomRAGEvaluation


class ComparisonPipeline:
    """
    Pipeline for comparing multiple chunking strategies.
    
    Attributes:
        evaluation (CustomRAGEvaluation): Evaluation instance
        
    Methods:
        compare_chunkers: Compare multiple chunking strategies
    """
    
    def __init__(self, evaluation: CustomRAGEvaluation):
        """
        Initialize the comparison pipeline.
        
        Args:
            evaluation (CustomRAGEvaluation): Configured evaluation instance
        """
        self.evaluation = evaluation
    
    def compare_chunkers(self, chunkers: List[BaseChunker], texts: List[str],
                        metadatas: List[Dict], embedding_fn) -> Dict[str, EvaluationMetrics]:
        """
        Compare multiple chunking strategies.
        
        Args:
            chunkers (List[BaseChunker]): List of chunkers to compare
            texts (List[str]): Source texts
            metadatas (List[Dict]): Source metadata
            embedding_fn: Embedding function
            
        Returns:
            Dict[str, EvaluationMetrics]: Results for each chunker
        """
        results = {}
        
        for chunker in chunkers:
            print(f"\nEvaluating {chunker.name}...")
            metrics = self.evaluation.evaluate_retrieval(
                chunker, texts, metadatas, embedding_fn
            )
            results[chunker.name] = metrics
            self.evaluation.print_results(metrics, chunker.name)
        
        # Print comparison summary
        self._print_comparison(results)
        
        return results
    
    def _print_comparison(self, results: Dict[str, EvaluationMetrics]) -> None:
        """
        Print a comparison table of all chunkers.
        
        Args:
            results (Dict[str, EvaluationMetrics]): Results for each chunker
        """
        print(f"\n{'='*60}")
        print("Comparison Summary")
        print(f"{'='*60}\n")
        
        print(f"{'Chunker':<30} {'MRR':<10} {'P@5':<10} {'R@5':<10} {'NDCG@5':<10}")
        print("-" * 60)
        
        for name, metrics in results.items():
            print(f"{name:<30} {metrics.mrr:<10.4f} "
                  f"{metrics.precision_at_k[5]:<10.4f} "
                  f"{metrics.recall_at_k[5]:<10.4f} "
                  f"{metrics.ndcg_at_k[5]:<10.4f}")

