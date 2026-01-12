"""
Retrieval evaluation implementation.

This module provides concrete implementation for evaluating RAG retrieval
performance using the queries format: query ||| source_file ||| expected_answer
"""
import time
from typing import List, Dict, Tuple, Set
from .base import BaseEvaluator
from .metrics import RetrievalMetrics


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for RAG retrieval performance.
    
    This class evaluates retrieval quality by comparing retrieved chunks
    against expected source documents and computing various metrics.
    
    Attributes:
        name (str): Name of the evaluator
        metrics: RetrievalMetrics instance for computing scores
        
    Methods:
        load_queries: Load queries from file
        evaluate: Run evaluation on indexer
        generate_report: Generate detailed evaluation report
    """
    
    def __init__(self):
        """
        Initialize the retrieval evaluator.
        """
        super().__init__("Retrieval Evaluator")
        self.metrics = RetrievalMetrics()
    
    def load_queries(self, queries_path: str) -> List[Tuple[str, str, str]]:
        """
        Load evaluation queries from file.
        
        Expected format: query ||| source_file ||| expected_answer
        
        Args:
            queries_path (str): Path to the queries file
            
        Returns:
            List[Tuple[str, str, str]]: List of (query, source_file, expected_answer)
            
        Raises:
            FileNotFoundError: If queries file doesn't exist
            ValueError: If file format is invalid
            
        Examples:
            >>> evaluator = RetrievalEvaluator()
            >>> queries = evaluator.load_queries("data/queries.txt")
            >>> len(queries)
            30
        """
        queries = []
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('|||')
                if len(parts) != 3:
                    print(f"Warning: Skipping invalid line {line_num}: {line[:50]}...")
                    continue
                
                query = parts[0].strip()
                source_file = parts[1].strip()
                expected_answer = parts[2].strip()
                
                queries.append((query, source_file, expected_answer))
        
        print(f"Loaded {len(queries)} queries from {queries_path}")
        return queries
    
    def _extract_source_files(self, metadatas: List[Dict]) -> Set[str]:
        """
        Extract source file names from metadata.
        
        Args:
            metadatas (List[Dict]): List of metadata dictionaries
            
        Returns:
            Set[str]: Set of unique source file names
        """
        return {meta.get('file_name', '') for meta in metadatas if meta.get('file_name')}
    
    def evaluate(self, indexer, queries: List[Tuple[str, str, str]], 
                k: int = 5, verbose: bool = True) -> Dict:
        """
        Run evaluation on queries using the provided indexer.
        
        For each query, retrieves top-k chunks and checks if they come from
        the expected source document.
        
        Args:
            indexer: ChromaDB indexer instance to query
            queries (List[Tuple[str, str, str]]): List of (query, source_file, answer)
            k (int, optional): Number of results to retrieve per query. Defaults to 5.
            verbose (bool, optional): Print progress during evaluation. Defaults to True.
            
        Returns:
            Dict: Dictionary containing:
                - 'precision@k': Average precision at k
                - 'recall@k': Average recall at k
                - 'f1@k': Average F1 score at k
                - 'mrr': Mean Reciprocal Rank
                - 'hit_rate@k': Hit rate at k
                - 'avg_latency_ms': Average retrieval latency in milliseconds
                - 'total_queries': Total number of queries evaluated
                - 'detailed_results': Per-query results
                
        Examples:
            >>> evaluator = RetrievalEvaluator()
            >>> queries = evaluator.load_queries("data/queries.txt")
            >>> results = evaluator.evaluate(indexer, queries, k=5)
            >>> print(f"MRR: {results['mrr']:.3f}")
        """
        detailed_results = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_latencies = []
        retrieved_docs_list = []
        relevant_docs_list = []
        
        for i, (query, expected_source, expected_answer) in enumerate(queries, 1):
            if verbose and i % 10 == 0:
                print(f"Evaluating query {i}/{len(queries)}...")
            
            # Measure retrieval latency
            start_time = time.time()
            search_results = indexer.search(query, n_results=k)
            latency_ms = (time.time() - start_time) * 1000
            all_latencies.append(latency_ms)
            
            # Extract retrieved documents
            retrieved_metadatas = search_results['metadatas'][0]
            retrieved_sources = self._extract_source_files(retrieved_metadatas)
            retrieved_docs_list.append(list(retrieved_sources))
            
            # Define relevant documents (expected source)
            relevant_docs = {expected_source}
            relevant_docs_list.append(relevant_docs)
            
            # Calculate metrics
            precision = self.metrics.calculate_precision(
                list(retrieved_sources), relevant_docs
            )
            recall = self.metrics.calculate_recall(
                list(retrieved_sources), relevant_docs
            )
            f1 = self.metrics.calculate_f1(precision, recall)
            
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            
            # Store detailed results
            detailed_results.append({
                'query': query,
                'expected_source': expected_source,
                'expected_answer': expected_answer,
                'retrieved_sources': list(retrieved_sources),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'latency_ms': latency_ms,
                'top_chunks': search_results['documents'][0][:3],  # Top 3 chunks
                'distances': search_results['distances'][0][:3]
            })
        
        # Calculate aggregate metrics
        mrr = self.metrics.calculate_mrr(retrieved_docs_list, relevant_docs_list)
        hit_rate = self.metrics.calculate_hit_rate(retrieved_docs_list, relevant_docs_list)
        
        results = {
            'precision@k': sum(all_precisions) / len(all_precisions),
            'recall@k': sum(all_recalls) / len(all_recalls),
            'f1@k': sum(all_f1s) / len(all_f1s),
            'mrr': mrr,
            'hit_rate@k': hit_rate,
            'avg_latency_ms': sum(all_latencies) / len(all_latencies),
            'total_queries': len(queries),
            'k': k,
            'detailed_results': detailed_results
        }
        
        if verbose:
            print("\n" + "="*60)
            print("Evaluation Complete!")
            print("="*60)
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = None) -> str:
        """
        Generate detailed evaluation report.
        
        Creates a formatted report with overall metrics and per-query breakdown.
        
        Args:
            results (Dict): Results from evaluate() method
            output_path (str, optional): Path to save report. If None, returns string.
            
        Returns:
            str: Formatted evaluation report
            
        Examples:
            >>> report = evaluator.generate_report(results)
            >>> print(report)
            >>> # Or save to file
            >>> evaluator.generate_report(results, "evaluation_report.txt")
        """
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("RAG RETRIEVAL EVALUATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("OVERALL METRICS:")
        report_lines.append("-"*80)
        report_lines.append(f"Total Queries Evaluated: {results['total_queries']}")
        report_lines.append(f"Top-K Retrieved: {results['k']}")
        report_lines.append("")
        report_lines.append(f"Precision@{results['k']}: {results['precision@k']:.4f}")
        report_lines.append(f"Recall@{results['k']}: {results['recall@k']:.4f}")
        report_lines.append(f"F1@{results['k']}: {results['f1@k']:.4f}")
        report_lines.append(f"Mean Reciprocal Rank (MRR): {results['mrr']:.4f}")
        report_lines.append(f"Hit Rate@{results['k']}: {results['hit_rate@k']:.4f}")
        report_lines.append(f"Average Latency: {results['avg_latency_ms']:.2f} ms")
        report_lines.append("")
        
        # Failed queries (where recall = 0)
        failed_queries = [r for r in results['detailed_results'] if r['recall'] == 0]
        report_lines.append(f"Failed Queries (Recall=0): {len(failed_queries)}")
        report_lines.append("")
        
        # Perfect queries (where recall = 1)
        perfect_queries = [r for r in results['detailed_results'] if r['recall'] == 1]
        report_lines.append(f"Perfect Queries (Recall=1): {len(perfect_queries)}")
        report_lines.append("")
        
        # Sample failed queries
        if failed_queries:
            report_lines.append("SAMPLE FAILED QUERIES:")
            report_lines.append("-"*80)
            for i, failed in enumerate(failed_queries[:5], 1):
                report_lines.append(f"\n{i}. Query: {failed['query'][:100]}...")
                report_lines.append(f"   Expected Source: {failed['expected_source']}")
                report_lines.append(f"   Retrieved Sources: {failed['retrieved_sources']}")
                report_lines.append(f"   Top Distance: {failed['distances'][0]:.4f}")
            report_lines.append("")
        
        # Latency distribution
        latencies = [r['latency_ms'] for r in results['detailed_results']]
        report_lines.append("LATENCY STATISTICS:")
        report_lines.append("-"*80)
        report_lines.append(f"Min Latency: {min(latencies):.2f} ms")
        report_lines.append(f"Max Latency: {max(latencies):.2f} ms")
        report_lines.append(f"Avg Latency: {sum(latencies)/len(latencies):.2f} ms")
        report_lines.append("")
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text
