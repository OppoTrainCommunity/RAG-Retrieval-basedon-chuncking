"""
Main entry point for chunking evaluation.

This script demonstrates how to use the chunking evaluation pipeline with
different chunking strategies.
"""

from src.evaluation.comparison_pipeline import ComparisonPipeline
from src.evaluation.custom_RAG_evaluation import CustomRAGEvaluation
from src.preprocessors import CVDataPreprocessor
from src.chunkers import SemanticChunker, ParagraphChunker
from src.pipeline import ChunkingPipeline

def run_custom_evaluation():
    """
    Example function showing how to run custom RAG evaluation.
    
    This demonstrates the complete workflow for evaluating chunking strategies
    using your own question-answer pairs.
    """
    from src.preprocessors import CVDataPreprocessor
    from src.chunkers import SemanticChunker, ParagraphChunker
    from chromadb.utils import embedding_functions
    
    # Step 1: Prepare your data
    preprocessor = CVDataPreprocessor()
    texts, metadatas = preprocessor.prepare_data("data/CVs.json")
    
    # Step 2: Load your QA pairs
    evaluation = CustomRAGEvaluation(k_values=[1, 3, 5, 10])
    
    # Option A: Load from JSON file
    evaluation.load_qa_data("data/CVs.json")
    
    # Option B: Load from dictionary
    # qa_data = [
    #     {
    #         "question": "What programming languages does John know?",
    #         "relevant_doc_ids": ["doc1_sem_3", "doc1_sem_4"]
    #     },
    #     ...
    # ]
    # evaluation.load_qa_from_dict(qa_data)
    
    # Step 3: Initialize embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Step 4: Initialize chunkers
    semantic_chunker = SemanticChunker(avg_chunk_size=400)
    paragraph_chunker = ParagraphChunker(tokens_per_chunk=200)
    
    # Step 5: Run comparison
    pipeline = ComparisonPipeline(evaluation)
    results = pipeline.compare_chunkers(
        chunkers=[semantic_chunker, paragraph_chunker],
        texts=texts,
        metadatas=metadatas,
        embedding_fn=embedding_fn
    )
    
    return results
    
def main():
    """
    Main function to run the chunking evaluation pipeline.
    
    Initializes a CV preprocessor, creates semantic and paragraph chunking
    strategies, and runs them through the evaluation pipeline.
    
    Returns:
        None: Prints evaluation results to stdout
        
    Examples:
        To run the evaluation:
        $ python main.py
    """
    DATA_PATH = "data/CVs.json"
    
    # Initialize preprocessor
    preprocessor = CVDataPreprocessor()
    
    # Initialize chunkers with specific configurations
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
    
    # Create and run pipeline
    pipeline = ChunkingPipeline(
        preprocessor=preprocessor,
        chunkers=[semantic_chunker, paragraph_chunker]
    )
    
    results = pipeline.run(DATA_PATH)
    
    # Access results for further evaluation
    for strategy_name, (chunks, metas, ids) in results.items():
        print(f"\n{strategy_name}: {len(chunks)} total chunks")

    results = run_custom_evaluation()

if __name__ == "__main__":
    
    main()
