"""
Paragraph-based chunking implementation.

This module implements a chunking strategy that respects paragraph boundaries
and uses token-based splitting for paragraphs that exceed a threshold.
"""
import re
from typing import List, Dict, Tuple
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from .base import BaseChunker


class ParagraphChunker(BaseChunker):
    """
    Paragraph-based chunking with token splitting for long paragraphs.
    
    This chunker preserves natural paragraph boundaries when possible,
    and splits long paragraphs using token-based splitting.
    
    Attributes:
        name (str): Name of the chunking strategy
        token_splitter: Token-based text splitter for long paragraphs
        short_paragraph_threshold (int): Word count threshold for splitting
        
    Methods:
        chunk_texts: Perform paragraph-based chunking
        _split_into_paragraphs: Split text into paragraphs
        _chunk_paragraphs: Chunk paragraphs based on length
    """
    
    def __init__(self, tokens_per_chunk: int = 200, chunk_overlap: int = 30,
                 short_paragraph_threshold: int = 120):
        """
        Initialize the paragraph chunker.
        
        Args:
            tokens_per_chunk (int, optional): Target number of tokens per chunk
                for long paragraphs. Defaults to 200.
            chunk_overlap (int, optional): Number of overlapping tokens between
                chunks when splitting long paragraphs. Defaults to 30.
            short_paragraph_threshold (int, optional): Word count threshold.
                Paragraphs with fewer words are kept intact, longer ones are
                split using token splitter. Defaults to 120.
        """
        super().__init__("Paragraph Chunking")
        
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=tokens_per_chunk,
            chunk_overlap=chunk_overlap
        )
        self.short_paragraph_threshold = short_paragraph_threshold
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Uses regex to identify paragraph boundaries based on double newlines
        and filters out very short paragraphs.
        
        Args:
            text (str): Input text to split into paragraphs
            
        Returns:
            List[str]: List of paragraph strings, each stripped of whitespace
            
        Note:
            Paragraphs shorter than 50 characters are filtered out
        """
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if len(p.strip()) > 50]
    
    def _chunk_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """
        Chunk paragraphs, splitting long ones with token splitter.
        
        Short paragraphs (below threshold) are kept intact, while long
        paragraphs are split into smaller chunks using the token splitter.
        
        Args:
            paragraphs (List[str]): List of paragraph strings
            
        Returns:
            List[str]: List of text chunks, where short paragraphs remain
                intact and long paragraphs are split
                
        Note:
            Uses word count (split by whitespace) to determine if a paragraph
            is short or long
        """
        chunks = []
        for p in paragraphs:
            if len(p.split()) < self.short_paragraph_threshold:
                chunks.append(p)
            else:
                chunks.extend(self.token_splitter.split_text(p))
        return chunks
    
    def chunk_texts(self, texts: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict], List[str]]:
        """
        Perform paragraph-based chunking on texts.
        
        Splits each input text into paragraphs, then chunks those paragraphs
        based on their length.
        
        Args:
            texts (List[str]): List of text documents to chunk
            metadatas (List[Dict]): List of metadata dictionaries corresponding
                to each text document
                
        Returns:
            Tuple[List[str], List[Dict], List[str]]: A tuple containing:
                - List of text chunks
                - List of metadata dictionaries for each chunk with added fields:
                    - 'chunk_id': Sequential ID within the document
                    - 'chunk_type': Set to "paragraph"
                - List of unique chunk IDs in format "{doc_id}_par_{chunk_num}"
                
        Examples:
            >>> chunker = ParagraphChunker(tokens_per_chunk=200)
            >>> texts = ["Para 1\n\nPara 2\n\nLong para..."]
            >>> metas = [{"doc_id": "doc1", "file_name": "test.pdf"}]
            >>> chunks, chunk_metas, chunk_ids = chunker.chunk_texts(texts, metas)
        """
        chunks = []
        chunk_metas = []
        chunk_ids = []
        
        for text, meta in zip(texts, metadatas):
            paragraphs = self._split_into_paragraphs(text)
            text_chunks = self._chunk_paragraphs(paragraphs)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(chunk)
                chunk_metas.append({
                    **meta,
                    "chunk_id": i,
                    "chunk_type": "paragraph"
                })
                chunk_ids.append(f"{meta['doc_id']}_par_{i}")
        
        return chunks, chunk_metas, chunk_ids

