#!/usr/bin/env python3
"""
RAG Retrieval Evaluation Script

Main script for evaluating different chunking strategies on your RAG system
using your own question-answer pairs.

Usage:
    python evaluate.py              # Full evaluation
    python evaluate.py --analyze    # Analyze QA data only
    python evaluate.py --quick      # Quick re-evaluation
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.qa_data_processor import QADataProcessor
from src.evaluation.custom_RAG_evaluation import CustomRAGEvaluation
from src.evaluation.comparison_pipeline import ComparisonPipeline
from src.preprocessors import CVDataPreprocessor
from src.chunkers import SemanticChunker, ParagraphChunker
from chromadb.utils import embedding_functions


def analyze_qa_data(qa_file: str = "data/queries.txt"):
    """
    Analyze your QA data to understand its characteristics.
    
    This helps you understand what you're working with before evaluation.
    
    Args:
        qa_file (str): Path to QA text file
    """
    processor = QADataProcessor()
    processor.parse_qa_file(qa_file)
    
    print("\n" + "="*70)
    print("QA DATA ANALYSIS")
    print("="*70)
    
    print(f"\n📊 Total questions: {len(processor.qa_triplets)}")
    
    # Count questions per source file
    source_counts = {}
    for qa in processor.qa_triplets:
        source = qa['source_file']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n📁 Questions per source file:")
    for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {source}: {count} questions")
    
    # Calculate average question and answer lengths
    avg_q_len = sum(len(qa['question'].split()) for qa in processor.qa_triplets) / len(processor.qa_triplets)
    avg_a_len = sum(len(qa['answer'].split()) for qa in processor.qa_triplets) / len(processor.qa_triplets)
    
    print(f"\n📏 Average lengths:")
    print(f"  • Question: {avg_q_len:.1f} words")
    print(f"  • Answer: {avg_a_len:.1f} words")
    
    # Show sample questions
    print(f"\n📝 Sample questions:\n")
    for i, qa in enumerate(processor.qa_triplets[:5], 1):
        print(f"{i}. Q: {qa['question']}")
        print(f"   📄 Source: {qa['source_file']}")
        print(f"   ✓ A: {qa['answer'][:80]}{'...' if len(qa['answer']) > 80 else ''}\n")


def run_complete_evaluation(data_path: str = "data/CVs.json", 
                           qa_file: str = "data/queries.txt",
                           output_file: str = "data/qa_processed.json"):
    """
    Complete evaluation workflow for your RAG system.
    
    This function:
    1. Loads and preprocesses your CV data
    2. Parses your QA file
    3. Chunks the documents using different strategies
    4. Maps questions to relevant chunks
    5. Evaluates retrieval performance
    
    Args:
        data_path (str): Path to CVs JSON file
        qa_file (str): Path to QA text file
        output_file (str): Path to save processed QA data
        
    Returns:
        Dict: Evaluation results for each chunking strategy
    """
    print("="*70)
    print("RAG SYSTEM RETRIEVAL EVALUATION")
    print("="*70)
    
    # Step 1: Load and preprocess CV data
    print("\n[1/6] 📂 Loading CV data...")
    preprocessor = CVDataPreprocessor()
    texts, metadatas = preprocessor.prepare_data(data_path)
    print(f"      ✓ Loaded {len(texts)} CV documents")
    
    # Step 2: Initialize chunkers
    print("\n[2/6] 🔧 Initializing chunking strategies...")
    semantic_chunker = SemanticChunker(
        avg_chunk_size=400,
        min_chunk_size=50,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    paragraph_chunker = ParagraphChunker(
        tokens_per_chunk=200,
        chunk_overlap=30,
        short_paragraph_threshold=120
    )
    print("      ✓ Initialized Semantic Chunking")
    print("      ✓ Initialized Paragraph Chunking")
    
    # Step 3: Parse QA data
    print(f"\n[3/6] 📝 Parsing QA data from {qa_file}...")
    processor = QADataProcessor()
    processor.parse_qa_file(qa_file)
    print(f"      ✓ Parsed {len(processor.qa_triplets)} question-answer pairs")
    
    # Step 4: Chunk documents with one strategy to create ground truth
    print("\n[4/6] 🎯 Creating ground truth mappings...")
    print("      This may take a moment...")
    
    # Use semantic chunker to create initial chunks for mapping
    chunks, chunk_metas, chunk_ids = semantic_chunker.chunk_texts(texts, metadatas)
    print(f"      ✓ Created {len(chunks)} chunks")
    
    # Map questions to relevant chunks
    qa_data = processor.map_to_chunks(chunks, chunk_metas, chunk_ids, search_window=2)
    
    # Save the processed QA data
    processor.save_to_json(qa_data, output_file)
    print(f"      ✓ Mapped questions to relevant chunks")
    
    # Step 5: Initialize evaluation
    print("\n[5/6] ⚙️  Setting up evaluation framework...")
    evaluation = CustomRAGEvaluation(k_values=[1, 3, 5, 10])
    evaluation.load_qa_from_dict(qa_data)
    
    # Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print("      ✓ Evaluation framework ready")
    
    # Step 6: Run comparison
    print("\n[6/6] 🚀 Evaluating chunking strategies...")
    print("      This will take a few minutes...\n")
    
    pipeline = ComparisonPipeline(evaluation)
    results = pipeline.compare_chunkers(
        chunkers=[semantic_chunker, paragraph_chunker],
        texts=texts,
        metadatas=metadatas,
        embedding_fn=embedding_fn
    )
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    
    return results


def run_quick_evaluation(data_path: str = "data/CVs.json",
                        qa_processed: str = "data/qa_processed.json"):
    """
    Run evaluation using pre-processed QA JSON file.
    
    Use this if you've already created qa_processed.json and want to
    quickly re-run evaluation with different chunkers.
    
    Args:
        data_path (str): Path to CVs JSON file
        qa_processed (str): Path to processed QA JSON file
        
    Returns:
        Dict: Evaluation results
    """
    print("="*70)
    print("QUICK EVALUATION MODE")
    print("="*70)
    
    # Check if processed QA file exists
    if not Path(qa_processed).exists():
        print(f"\n❌ Error: {qa_processed} not found!")
        print("   Run full evaluation first (without --quick flag)")
        return None
    
    print(f"\n📂 Loading data...")
    
    # Load data
    preprocessor = CVDataPreprocessor()
    texts, metadatas = preprocessor.prepare_data(data_path)
    print(f"   ✓ Loaded {len(texts)} CV documents")
    
    # Initialize evaluation with existing QA data
    evaluation = CustomRAGEvaluation(k_values=[1, 3, 5, 10])
    evaluation.load_qa_data(qa_processed)
    print(f"   ✓ Loaded {len(evaluation.questions)} questions")
    
    # Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Initialize chunkers
    print(f"\n🔧 Initializing chunkers...")
    semantic_chunker = SemanticChunker(avg_chunk_size=400)
    paragraph_chunker = ParagraphChunker(tokens_per_chunk=200)
    print("   ✓ Ready")
    
    # Run evaluation
    print(f"\n🚀 Running evaluation...\n")
    pipeline = ComparisonPipeline(evaluation)
    results = pipeline.compare_chunkers(
        chunkers=[semantic_chunker, paragraph_chunker],
        texts=texts,
        metadatas=metadatas,
        embedding_fn=embedding_fn
    )
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    
    return results


def main():
    """
    Main entry point for the evaluation script.
    
    Parses command-line arguments and runs the appropriate evaluation mode.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate RAG retrieval system with different chunking strategies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py                    # Full evaluation
  python evaluate.py --analyze          # Analyze QA data only
  python evaluate.py --quick            # Quick re-evaluation
  python evaluate.py --data custom.json # Use custom data file
        """
    )
    
    parser.add_argument(
        '--analyze', 
        action='store_true',
        help='Analyze QA data only (no evaluation)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Quick evaluation using existing qa_processed.json'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/CVs.json',
        help='Path to CV data JSON file (default: data/CVs.json)'
    )
    
    parser.add_argument(
        '--qa',
        type=str,
        default='data/queries.txt',
        help='Path to QA text file (default: data/queries.txt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/qa_processed.json',
        help='Path to save processed QA data (default: data/qa_processed.json)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.analyze:
            analyze_qa_data(args.qa)
        elif args.quick:
            results = run_quick_evaluation(args.data, args.output)
        else:
            results = run_complete_evaluation(args.data, args.qa, args.output)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your data files are in the correct location:")
        print(f"  • CV data: {args.data}")
        print(f"  • QA data: {args.qa}")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()