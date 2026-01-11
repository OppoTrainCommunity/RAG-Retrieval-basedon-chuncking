"""
QA Data Processor and Evaluation Script

This script processes your QA data format and runs comprehensive retrieval evaluation.
"""

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path

from src.evaluation.comparison_pipeline import ComparisonPipeline


class QADataProcessor:
    """
    Process QA data from text file format to evaluation-ready format.
    
    This class handles parsing of QA triplets (question ||| source ||| answer)
    and maps them to relevant document chunks for evaluation.
    
    Attributes:
        qa_triplets (List[Dict]): Parsed QA triplets
        
    Methods:
        parse_qa_file: Parse QA data from text file
        map_to_chunks: Map questions to relevant chunk IDs
        save_to_json: Save processed data to JSON
    """
    
    def __init__(self):
        """Initialize the QA data processor."""
        self.qa_triplets = []
    
    def parse_qa_file(self, file_path: str) -> List[Dict]:
        """
        Parse QA data from text file.
        
        Expected format: question ||| source_file ||| answer
        
        Args:
            file_path (str): Path to the QA text file
            
        Returns:
            List[Dict]: List of parsed QA items with keys:
                - 'question': Question text
                - 'source_file': Source document filename
                - 'answer': Expected answer
                
        Examples:
            >>> processor = QADataProcessor()
            >>> qa_data = processor.parse_qa_file("queries.txt")
            >>> len(qa_data)
            30
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        
        for line in lines:
            if '|||' in line:
                parts = [p.strip() for p in line.split('|||')]
                if len(parts) == 3:
                    self.qa_triplets.append({
                        'question': parts[0],
                        'source_file': parts[1],
                        'answer': parts[2]
                    })
        
        return self.qa_triplets
    
    def map_to_chunks(self, chunks: List[str], chunk_metas: List[Dict], 
                     chunk_ids: List[str], search_window: int = 3) -> List[Dict]:
        """
        Map questions to relevant chunk IDs based on source file and answer content.
        
        This method finds which chunks from the source document are most likely
        to contain the answer by searching for answer text in chunks.
        
        Args:
            chunks (List[str]): List of text chunks
            chunk_metas (List[Dict]): List of chunk metadata
            chunk_ids (List[str]): List of chunk IDs
            search_window (int, optional): Number of surrounding chunks to include
                when answer is found. Defaults to 3.
                
        Returns:
            List[Dict]: QA items with added 'relevant_doc_ids' field
            
        Note:
            Uses fuzzy matching to find answer text in chunks and includes
            surrounding chunks for context
        """
        qa_with_chunks = []
        
        for qa in self.qa_triplets:
            source_file = qa['source_file']
            answer = qa['answer']
            
            # Find chunks from the same source file
            relevant_chunk_indices = []
            
            for i, (chunk, meta) in enumerate(zip(chunks, chunk_metas)):
                # Check if chunk is from the source file
                if meta.get('file_name') == source_file:
                    # Check if answer text appears in chunk
                    if self._text_similarity(answer, chunk) > 0.5:
                        relevant_chunk_indices.append(i)
            
            # If we found matching chunks, include surrounding context
            if relevant_chunk_indices:
                expanded_indices = set()
                for idx in relevant_chunk_indices:
                    # Add surrounding chunks for context
                    for offset in range(-search_window, search_window + 1):
                        context_idx = idx + offset
                        if 0 <= context_idx < len(chunks):
                            # Ensure it's from the same document
                            if chunk_metas[context_idx].get('file_name') == source_file:
                                expanded_indices.add(context_idx)
                
                relevant_ids = [chunk_ids[i] for i in sorted(expanded_indices)]
            else:
                # Fallback: use all chunks from the source file
                relevant_ids = [
                    chunk_ids[i] for i, meta in enumerate(chunk_metas)
                    if meta.get('file_name') == source_file
                ]
            
            qa_with_chunks.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'source_file': source_file,
                'relevant_doc_ids': relevant_ids
            })
        
        return qa_with_chunks
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate simple text similarity based on word overlap.
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def create_ground_truth_from_chunks(self, chunks: List[str], 
                                       chunk_metas: List[Dict],
                                       chunk_ids: List[str]) -> List[Dict]:
        """
        Create ground truth by mapping each question to chunks from its source file.
        
        This is a simpler approach that assumes all chunks from the source document
        are potentially relevant.
        
        Args:
            chunks (List[str]): List of text chunks
            chunk_metas (List[Dict]): List of chunk metadata
            chunk_ids (List[str]): List of chunk IDs
            
        Returns:
            List[Dict]: QA items with 'relevant_doc_ids' field
        """
        qa_with_chunks = []
        
        for qa in self.qa_triplets:
            source_file = qa['source_file']
            
            # Get all chunks from the source file
            relevant_ids = [
                chunk_ids[i] for i, meta in enumerate(chunk_metas)
                if meta.get('file_name') == source_file
            ]
            
            qa_with_chunks.append({
                'question': qa['question'],
                'answer': qa['answer'],
                'source_file': source_file,
                'relevant_doc_ids': relevant_ids
            })
        
        return qa_with_chunks
    
    def save_to_json(self, qa_data: List[Dict], output_path: str) -> None:
        """
        Save processed QA data to JSON file.
        
        Args:
            qa_data (List[Dict]): Processed QA data
            output_path (str): Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(qa_data)} QA pairs to {output_path}")


# =============================================================================
# Complete Evaluation Script
# =============================================================================

def run_complete_evaluation():
    """
    Complete evaluation workflow for your RAG system.
    
    This function:
    1. Loads and preprocesses your CV data
    2. Parses your QA file
    3. Chunks the documents using different strategies
    4. Maps questions to relevant chunks
    5. Evaluates retrieval performance
    
    Returns:
        Dict: Evaluation results for each chunking strategy
    """
    from src.preprocessors import CVDataPreprocessor
    from src.chunkers import SemanticChunker, ParagraphChunker
    from chromadb.utils import embedding_functions
    
    # Import the custom evaluation (assuming you saved it as custom_rag_evaluation.py)
    import sys
    sys.path.append('.')
    from src.evaluation.custom_RAG_evaluation import CustomRAGEvaluation
    from src.evaluation.comparison_pipeline import ComparisonPipeline
    
    print("="*60)
    print("RAG System Retrieval Evaluation")
    print("="*60)
    
    # Step 1: Load and preprocess CV data
    print("\n[1/6] Loading CV data...")
    DATA_PATH = "data/CVs.json"
    preprocessor = CVDataPreprocessor()
    texts, metadatas = preprocessor.prepare_data(DATA_PATH)
    print(f"✓ Loaded {len(texts)} CVs")
    
    # Step 2: Initialize chunkers
    print("\n[2/6] Initializing chunking strategies...")
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
    print("✓ Initialized Semantic and Paragraph chunkers")
    
    # Step 3: Parse QA data
    print("\n[3/6] Parsing QA data...")
    QA_FILE = "data/queries.txt"
    processor = QADataProcessor()
    processor.parse_qa_file(QA_FILE)
    print(f"✓ Parsed {len(processor.qa_triplets)} questions")
    
    # Step 4: Chunk documents with one strategy to create ground truth
    print("\n[4/6] Creating ground truth mappings...")
    # Use semantic chunker to create initial chunks for mapping
    chunks, chunk_metas, chunk_ids = semantic_chunker.chunk_texts(texts, metadatas)
    
    # Map questions to relevant chunks
    qa_data = processor.map_to_chunks(chunks, chunk_metas, chunk_ids, search_window=2)
    
    # Save the processed QA data
    processor.save_to_json(qa_data, "data/qa_processed.json")
    print(f"✓ Mapped questions to chunks")
    
    # Step 5: Initialize evaluation
    print("\n[5/6] Setting up evaluation framework...")
    evaluation = CustomRAGEvaluation(k_values=[1, 3, 5, 10])
    evaluation.load_qa_from_dict(qa_data)
    
    # Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    print("✓ Evaluation framework ready")
    
    # Step 6: Run comparison
    print("\n[6/6] Evaluating chunking strategies...")
    print("This may take a few minutes...\n")
    
    pipeline = ComparisonPipeline(evaluation)
    results = pipeline.compare_chunkers(
        chunkers=[semantic_chunker, paragraph_chunker],
        texts=texts,
        metadatas=metadatas,
        embedding_fn=embedding_fn
    )
    
    return results


def run_with_existing_qa_json():
    """
    Run evaluation using pre-processed QA JSON file.
    
    Use this if you've already created qa_processed.json and want to
    quickly re-run evaluation with different chunkers.
    
    Returns:
        Dict: Evaluation results
    """
    from src.preprocessors import CVDataPreprocessor
    from src.chunkers import SemanticChunker, ParagraphChunker
    from chromadb.utils import embedding_functions
    from src.evaluation.custom_RAG_evaluation import CustomRAGEvaluation
    from src.evaluation.comparison_pipeline import ComparisonPipeline
    
    # Load data
    preprocessor = CVDataPreprocessor()
    texts, metadatas = preprocessor.prepare_data("data/CVs.json")
    
    # Initialize evaluation with existing QA data
    evaluation = CustomRAGEvaluation(k_values=[1, 3, 5, 10])
    evaluation.load_qa_data("data/qa_processed.json")
    
    # Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Initialize chunkers
    semantic_chunker = SemanticChunker(avg_chunk_size=400)
    paragraph_chunker = ParagraphChunker(tokens_per_chunk=200)
    
    # Run evaluation
    pipeline = ComparisonPipeline(evaluation)
    results = pipeline.compare_chunkers(
        chunkers=[semantic_chunker, paragraph_chunker],
        texts=texts,
        metadatas=metadatas,
        embedding_fn=embedding_fn
    )
    
    return results


def analyze_qa_data():
    """
    Analyze your QA data to understand its characteristics.
    
    This helps you understand what you're working with before evaluation.
    """
    processor = QADataProcessor()
    processor.parse_qa_file("data/queries.txt")
    
    print("\n" + "="*60)
    print("QA Data Analysis")
    print("="*60)
    
    print(f"\nTotal questions: {len(processor.qa_triplets)}")
    
    # Count questions per source file
    source_counts = {}
    for qa in processor.qa_triplets:
        source = qa['source_file']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\nQuestions per source file:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} questions")
    
    # Show sample questions
    print(f"\nSample questions:")
    for i, qa in enumerate(processor.qa_triplets[:3], 1):
        print(f"\n{i}. Q: {qa['question']}")
        print(f"   Source: {qa['source_file']}")
        print(f"   A: {qa['answer'][:100]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG retrieval system')
    parser.add_argument('--analyze', action='store_true', 
                       help='Analyze QA data only')
    parser.add_argument('--quick', action='store_true',
                       help='Quick evaluation using existing qa_processed.json')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_qa_data()
    elif args.quick:
        results = run_with_existing_qa_json()
    else:
        results = run_complete_evaluation()