"""
Main chunking evaluation pipeline.

This module provides the ChunkingPipeline class that orchestrates the complete
workflow of preprocessing data and running multiple chunking strategies.
"""
from typing import List, Dict, Tuple
from src.preprocessors.base import BasePreprocessor
from src.chunkers.base import BaseChunker


class ChunkingPipeline:
    """
    Main pipeline for chunking evaluation.
    
    This class coordinates the preprocessing and chunking workflow, running
    multiple chunking strategies on the same preprocessed data and collecting
    results for evaluation.
    
    Attributes:
        preprocessor (BasePreprocessor): Preprocessor instance for data loading
        chunkers (List[BaseChunker]): List of chunking strategies to evaluate
        
    Methods:
        run: Execute the complete pipeline on input data
    """
    
    def __init__(self, preprocessor: BasePreprocessor, chunkers: List[BaseChunker]):
        """
        Initialize the chunking pipeline.
        
        Args:
            preprocessor (BasePreprocessor): Preprocessor instance to use for
                loading and preparing data
            chunkers (List[BaseChunker]): List of chunker instances to evaluate.
                Each chunker will be run on the same preprocessed data.
                
        Examples:
            >>> from src.preprocessors import CVDataPreprocessor
            >>> from src.chunkers import SemanticChunker, ParagraphChunker
            >>> preprocessor = CVDataPreprocessor()
            >>> chunkers = [SemanticChunker(), ParagraphChunker()]
            >>> pipeline = ChunkingPipeline(preprocessor, chunkers)
        """
        self.preprocessor = preprocessor
        self.chunkers = chunkers
    
    def run(self, data_path: str) -> Dict[str, Tuple[List[str], List[Dict], List[str]]]:
        """
        Run all chunking strategies and return results.
        
        Executes the complete pipeline: loads and preprocesses data, then runs
        each chunking strategy and evaluates the results.
        
        Args:
            data_path (str): Path to the input data file
            
        Returns:
            Dict[str, Tuple[List[str], List[Dict], List[str]]]: Dictionary mapping
                chunking strategy names to their results. Each result is a tuple of:
                - List of text chunks
                - List of chunk metadata dictionaries
                - List of unique chunk IDs
                
        Examples:
            >>> pipeline = ChunkingPipeline(preprocessor, [chunker1, chunker2])
            >>> results = pipeline.run("data/CVs.json")
            >>> for name, (chunks, metas, ids) in results.items():
            ...     print(f"{name}: {len(chunks)} chunks")
            
        Note:
            This method also prints evaluation results for each chunking strategy
            to stdout during execution
        """
        # Prepare data
        texts, metadatas = self.preprocessor.prepare_data(data_path)
        print(f"Loaded {len(texts)} documents")
        
        # Run each chunking strategy
        results = {}
        for chunker in self.chunkers:
            chunks, chunk_metas, chunk_ids = chunker.chunk_texts(texts, metadatas)
            results[chunker.name] = (chunks, chunk_metas, chunk_ids)
            
            # Evaluate
            chunker.evaluate(chunks)
        
        return results

