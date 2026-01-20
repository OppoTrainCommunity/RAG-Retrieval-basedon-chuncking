"""
Document processing module: PDF extraction and text chunking strategies.
"""
import re
from typing import List, Dict, Any
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from . import config

# Download NLTK data lazily
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts)
        return full_text.strip()
    except Exception as e:
        print(f"⚠️ Error extracting text from {pdf_path}: {e}")
        return ""


def fixed_length_chunking(text: str, chunk_size: int = config.DEFAULT_CHUNK_SIZE, overlap: int = config.DEFAULT_OVERLAP) -> List[Dict[str, Any]]:
    """Split text into fixed-size chunks with overlap."""
    if not text:
        return []
    
    chunks = []
    start = 0
    chunk_idx = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        
        chunks.append({
            "text": chunk_text,
            "chunk_index": chunk_idx,
            "start_char": start,
            "end_char": min(end, len(text)),
            "strategy": "fixed"
        })
        
        start += chunk_size - overlap
        chunk_idx += 1
    
    return chunks


def semantic_chunking(text: str, target_size: int = config.DEFAULT_CHUNK_SIZE) -> List[Dict[str, Any]]:
    """Split text at sentence boundaries."""
    if not text:
        return []
    
    # Split into sentences
    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_idx = 0
    current_start = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > target_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "chunk_index": chunk_idx,
                "start_char": current_start,
                "end_char": current_start + len(chunk_text),
                "strategy": "semantic",
                "sentence_count": len(current_chunk)
            })
            current_start += len(chunk_text) + 1
            chunk_idx += 1
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += sentence_length + 1
    
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "chunk_index": chunk_idx,
            "start_char": current_start,
            "end_char": current_start + len(chunk_text),
            "strategy": "semantic",
            "sentence_count": len(current_chunk)
        })
    
    return chunks


def recursive_chunking(text: str, max_chunk_size: int = config.DEFAULT_CHUNK_SIZE, level: int = 0) -> List[Dict[str, Any]]:
    """Hierarchically split text."""
    if not text or len(text) <= max_chunk_size:
        if text:
            return [{
                "text": text,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(text),
                "strategy": "recursive",
                "level": level
            }]
        return []
    
    chunks = []
    # Level 0: Paragraphs
    if level == 0:
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
    # Level 1: Sentences
    elif level == 1:
        try:
            paragraphs = sent_tokenize(text)
        except:
            paragraphs = re.split(r'(?<=[.!?])\s+', text)
    # Level 2: Fixed chunks fallback
    else:
        fixed = fixed_length_chunking(text, max_chunk_size, 50)
        for c in fixed:
            c["strategy"] = "recursive"
            c["level"] = level
        return fixed

    current_start = 0
    for para in paragraphs:
        if len(para) <= max_chunk_size:
            chunks.append({
                "text": para,
                "chunk_index": len(chunks),
                "start_char": current_start,
                "end_char": current_start + len(para),
                "strategy": "recursive",
                "level": level
            })
        else:
            sub = recursive_chunking(para, max_chunk_size, level + 1)
            for s in sub:
                s["chunk_index"] = len(chunks)
                s["start_char"] += current_start
                s["end_char"] += current_start
                chunks.append(s)
        current_start += len(para) + 2
        
    return chunks


def get_chunks(text: str, strategy: str = "fixed", **kwargs) -> List[Dict[str, Any]]:
    """Dispatcher function for chunking strategies."""
    strategy = strategy.lower()
    
    # Handle argument mapping
    if "chunk_size" in kwargs:
        if strategy == "semantic":
            kwargs["target_size"] = kwargs.pop("chunk_size")
        elif strategy == "recursive":
            kwargs["max_chunk_size"] = kwargs.pop("chunk_size")
            
    # Handle overlap removal for strategies that don't support it
    if strategy in ["semantic", "recursive"] and "overlap" in kwargs:
        kwargs.pop("overlap")
        
    if strategy == "fixed":
        return fixed_length_chunking(text, **kwargs)
    elif strategy == "semantic":
        return semantic_chunking(text, **kwargs)
    elif strategy == "recursive":
        return recursive_chunking(text, **kwargs)
    else:
        return fixed_length_chunking(text, **kwargs)
