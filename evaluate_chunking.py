"""
Main script for evaluating chunking strategies.

This script loads queries, runs evaluation on different chunking strategies,
and compares their retrieval performance.
"""
from src.indexers import ChromaIndexer
from src.evaluation import RetrievalEvaluator
import json


def evaluate_strategy(collection_name: str, queries_path: str, 
                     k: int = 5) -> dict:
    """
    Evaluate a single chunking strategy.
    
    Args:
        collection_name (str): Name of the ChromaDB collection
        queries_path (str): Path to queries file
        k (int, optional): Number of results to retrieve. Defaults to 5.
        
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {collection_name}")
    print(f"{'='*80}")
    
    # Initialize indexer
    indexer = ChromaIndexer(
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator()
    
    # Load queries
    queries = evaluator.load_queries(queries_path)
    
    # Run evaluation
    results = evaluator.evaluate(indexer, queries, k=k, verbose=True)
    
    # Generate report
    report = evaluator.generate_report(results)
    print(report)
    
    # Save report to file
    report_filename = f"evaluation_reports/evaluation_report_{collection_name}.txt"
    evaluator.generate_report(results, report_filename)
    
    return results


def compare_strategies(strategies: list, queries_path: str, k: int = 5):
    """
    Compare multiple chunking strategies.
    
    Args:
        strategies (list): List of collection names to evaluate
        queries_path (str): Path to queries file
        k (int, optional): Number of results to retrieve. Defaults to 5.
    """
    all_results = {}
    
    for strategy in strategies:
        results = evaluate_strategy(strategy, queries_path, k)
        all_results[strategy] = results
    
    # Print comparison
    print("\n" + "="*80)
    print("STRATEGY COMPARISON")
    print("="*80)
    print(f"{'Strategy':<30} {'Precision@k':<15} {'Recall@k':<15} {'MRR':<15} {'Hit Rate':<15}")
    print("-"*80)
    
    for strategy, results in all_results.items():
        print(f"{strategy:<30} "
              f"{results['precision@k']:<15.4f} "
              f"{results['recall@k']:<15.4f} "
              f"{results['mrr']:<15.4f} "
              f"{results['hit_rate@k']:<15.4f}")
    
    # Save comparison to JSON
    comparison_data = {
        strategy: {
            'precision@k': results['precision@k'],
            'recall@k': results['recall@k'],
            'f1@k': results['f1@k'],
            'mrr': results['mrr'],
            'hit_rate@k': results['hit_rate@k'],
            'avg_latency_ms': results['avg_latency_ms']
        }
        for strategy, results in all_results.items()
    }
    
    with open('evaluation_reports/strategy_comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

    print("\nComparison saved to evaluation_reports/strategy_comparison.json")

    # Determine best strategy
    best_strategy = max(all_results.items(), 
                       key=lambda x: x[1]['f1@k'])
    
    print(f"\n🏆 Best Strategy (by F1@{k}): {best_strategy[0]}")
    print(f"   F1 Score: {best_strategy[1]['f1@k']:.4f}")


def main():
    """
    Main function to run chunking strategy evaluation.
    """
    QUERIES_PATH = "data/queries.txt"
    
    # List of strategies to evaluate
    strategies = [
        "semantic_chunking",
        "paragraph_chunking"
    ]
    
    # Compare all strategies
    compare_strategies(strategies, QUERIES_PATH, k=10)


if __name__ == "__main__":
    main()
