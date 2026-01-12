"""
Main entry point for chunking evaluation with ChromaDB indexing.

This script demonstrates how to use the chunking evaluation pipeline with
different chunking strategies and index the results to ChromaDB.
"""
from src.preprocessors import CVDataPreprocessor
from src.chunkers import SemanticChunker, ParagraphChunker
from src.pipeline import ChunkingPipeline
from src.indexers import ChromaIndexer


def search_example():
    """
    Example function demonstrating how to search existing ChromaDB collections.
    
    Shows how to load existing collections and perform searches without
    re-indexing.
    
    Returns:
        None: Prints search results to stdout
    """
    # Connect to existing collection
    indexer = ChromaIndexer(
        collection_name="semantic_chunking",
        persist_directory="./chroma_db"
    )
    
    # Perform search
    query = "machine learning and data science skills"
    results = indexer.search(query, n_results=10)
    
    print("="*60)
    print(f"Search results for: '{query}'")
    print("="*60)
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        print(f"\nResult {i}:")
        print(f"Distance: {dist:.4f}")
        print(f"File: {meta.get('file_name', 'N/A')}")
        print(f"Text: {doc[:300]}...")
        print("-"*60)


def main():
    """
    Main function to run the chunking evaluation pipeline with ChromaDB indexing.
    
    Initializes a CV preprocessor, creates semantic and paragraph chunking
    strategies, runs them through the evaluation pipeline, and indexes the
    results to ChromaDB for semantic search.
    
    Returns:
        None: Prints evaluation results and indexing status to stdout
        
    Examples:
        To run the evaluation and indexing:
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
    
    # Index results to ChromaDB
    print("\n" + "="*60)
    print("Indexing chunks to ChromaDB")
    print("="*60)
    
    for strategy_name, (chunks, metas, ids) in results.items():
        print(f"\n{strategy_name}: {len(chunks)} total chunks")
        
        # Create collection name from strategy name
        collection_name = strategy_name.lower().replace(" ", "_")
        
        # Initialize ChromaDB indexer
        indexer = ChromaIndexer(
            collection_name=collection_name,
            embedding_model="all-MiniLM-L6-v2",
            persist_directory="./chroma_db"
        )
        
        # Add chunks to ChromaDB
        indexer.add_chunks(chunks, metas, ids)
        
        # Get and print collection stats
        stats = indexer.get_collection_stats()
        print(f"Collection stats: {stats}")
        
        # Example search
        print(f"\nExample search in {strategy_name}:")
        search_results = indexer.search("python programming experience", n_results=10)
        
        for i, (doc, meta, dist) in enumerate(zip(
            search_results['documents'][0],
            search_results['metadatas'][0],
            search_results['distances'][0]
        ), 1):
            print(f"\nResult {i} (distance: {dist:.4f}):")
            print(f"File: {meta.get('file_name', 'N/A')}")
            print(f"Chunk ID: {meta.get('chunk_id', 'N/A')}")
            print(f"Text preview: {doc[:200]}...")



if __name__ == "__main__":
    main()
    
    # Uncomment to run search example
    search_example()
