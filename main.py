"""
Main entry point for chunking evaluation.

This script demonstrates how to use the chunking evaluation pipeline with
different chunking strategies.
"""
from src.preprocessors import CVDataPreprocessor
from src.chunkers import SemanticChunker, ParagraphChunker
from src.pipeline import ChunkingPipeline


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


if __name__ == "__main__":
    main()
