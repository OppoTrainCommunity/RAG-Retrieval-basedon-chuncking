"""
Chunking Module
===============

Text chunking strategies for RAG pipeline.
Supports semantic chunking with fallback to recursive character splitting.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from .logging_utils import get_logger
from .pdf_loader import PDFDocument, ExtractedPage

logger = get_logger(__name__)

# LangChain imports
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Optional semantic chunker
try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    source: str
    doc_id: str
    page_start: int
    page_end: int
    chunk_id: str
    chunk_index: int
    char_start: int
    char_end: int
    candidate_name: Optional[str] = None


@dataclass
class TextChunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: ChunkMetadata


def create_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Create a stable, unique chunk ID.
    
    Args:
        doc_id: Document ID
        chunk_index: Index of chunk within document
        
    Returns:
        Unique chunk ID
    """
    return f"{doc_id}:chunk_{chunk_index:04d}"


def detect_sections(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect section headers in CV text.
    
    Args:
        text: Full document text
        
    Returns:
        List of (section_name, start_pos, end_pos) tuples
    """
    # Common CV section headers
    section_patterns = [
        r'^\s*(EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|PROFESSIONAL EXPERIENCE|HISTORY)',
        r'^\s*(EDUCATION|ACADEMIC|QUALIFICATIONS|ACADEMIC HISTORY)',
        r'^\s*(SKILLS|TECHNICAL SKILLS|COMPETENCIES|CORE COMPETENCIES|TECHNOLOGIES)',
        r'^\s*(PROJECTS|KEY PROJECTS|ACADEMIC PROJECTS|PERSONAL PROJECTS)',
        r'^\s*(CERTIFICATIONS|CERTIFICATES|AWARDS|HONORS|ACHIEVEMENTS)',
        r'^\s*(SUMMARY|PROFILE|OBJECTIVE|ABOUT|PROFESSIONAL SUMMARY)',
        r'^\s*(LANGUAGES|LANGUAGE SKILLS)',
        r'^\s*(REFERENCES|REFEREES)',
        r'^\s*(PUBLICATIONS|PAPERS|RESEARCH)',
        r'^\s*(INTERESTS|HOBBIES|ACTIVITIES|EXTRACURRICULAR)',
        r'^\s*(VOLUNTEER|VOLUNTEERING)',
        r'^\s*(CONTACT|CONTACT INFO|PERSONAL DETAILS)',
    ]
    
    sections = []
    combined_pattern = '|'.join(f'({p})' for p in section_patterns)
    
    for match in re.finditer(combined_pattern, text, re.MULTILINE | re.IGNORECASE):
        # Clean up the section name (remove leading/trailing chars matches might have caught if regex was complex)
        section_name = match.group().strip()
        sections.append((section_name, match.start(), match.end()))
    
    return sections


def chunk_by_sections(
    pdf_doc: PDFDocument,
    max_chunk_size: int = 1000,
    min_chunk_size: int = 100
) -> List[TextChunk]:
    """
    Chunk document by detected sections (semantic-like chunking).
    
    Args:
        pdf_doc: Extracted PDF document
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters per chunk
        
    Returns:
        List of TextChunk objects
    """
    chunks = []
    chunk_index = 0
    
    # Combine all pages into single text with page markers
    full_text = ""
    page_positions = []  # (start_pos, end_pos, page_num)
    
    for page in pdf_doc.pages:
        start_pos = len(full_text)
        full_text += page.text + "\n\n"
        end_pos = len(full_text)
        page_positions.append((start_pos, end_pos, page.page_number))
    
    # Detect sections
    sections = detect_sections(full_text)
    
    # If we found sections, use them
    if len(sections) >= 1:
        # Sort sections by start position just in case
        sections.sort(key=lambda x: x[1])
        
        # If the first section doesn't start at 0, create a "Intro/Contact" section for the beginning
        if sections[0][1] > 0:
            intro_text = full_text[0:sections[0][1]].strip()
            if intro_text:
                # Add synthetic section to start
                sections.insert(0, ("HEADER/CONTACT", 0, sections[0][1]))

        for i, (section_name, start, end) in enumerate(sections):
            # Get text until next section or end
            if i + 1 < len(sections):
                section_end = sections[i + 1][1]
            else:
                section_end = len(full_text)
            
            # The actual content usually starts after the header match
            # But we want to keep the header in the text for context, or cleaner: prepend it normalized
            raw_section_text = full_text[start:section_end].strip()
            
            # Clean section name for metadata/prefixing
            clean_section_name = re.sub(r'[^\w\s]', '', section_name).strip().upper()
            
            # Skip empty sections (really empty)
            if not raw_section_text:
                continue
            
            # If section is too large, split it further
            if len(raw_section_text) > max_chunk_size:
                # Split by paragraphs
                paragraphs = re.split(r'\n\n+', raw_section_text)
                current_chunk_text = ""
                
                # Helper to add chunk
                def add_chunk(text_content, s_start, s_end):
                    nonlocal chunk_index
                    # Prepend Section Name for Context Awareness
                    final_text = f"SECTION: {clean_section_name}\n{text_content}"
                    
                    p_start, p_end = get_page_range(s_start, s_end, page_positions)
                    chunks.append(TextChunk(
                        text=final_text,
                        metadata=ChunkMetadata(
                            source=pdf_doc.filename,
                            doc_id=pdf_doc.doc_id,
                            page_start=p_start,
                            page_end=p_end,
                            chunk_id=create_chunk_id(pdf_doc.doc_id, chunk_index),
                            chunk_index=chunk_index,
                            char_start=s_start,
                            char_end=s_end,
                            candidate_name=pdf_doc.candidate_name
                        )
                    ))
                    chunk_index += 1

                chunk_start_char = start
                
                for para in paragraphs:
                    if len(current_chunk_text) + len(para) < max_chunk_size:
                        current_chunk_text += para + "\n\n"
                    else:
                        if current_chunk_text.strip():
                            add_chunk(current_chunk_text.strip(), chunk_start_char, chunk_start_char + len(current_chunk_text))
                            chunk_start_char += len(current_chunk_text)
                        
                        current_chunk_text = para + "\n\n"
                
                # Last chunk
                if current_chunk_text.strip():
                    add_chunk(current_chunk_text.strip(), chunk_start_char, chunk_start_char + len(current_chunk_text))
            
            else:
                # Use entire section as chunk
                # Prepend Section Name for Context Awareness
                final_text = f"SECTION: {clean_section_name}\n{raw_section_text}"
                
                page_start, page_end = get_page_range(start, section_end, page_positions)
                
                chunks.append(TextChunk(
                    text=final_text,
                    metadata=ChunkMetadata(
                        source=pdf_doc.filename,
                        doc_id=pdf_doc.doc_id,
                        page_start=page_start,
                        page_end=page_end,
                        chunk_id=create_chunk_id(pdf_doc.doc_id, chunk_index),
                        chunk_index=chunk_index,
                        char_start=start,
                        char_end=section_end,
                        candidate_name=pdf_doc.candidate_name
                    )
                ))
                chunk_index += 1
    
    return chunks


def get_page_range(
    char_start: int,
    char_end: int,
    page_positions: List[Tuple[int, int, int]]
) -> Tuple[int, int]:
    """
    Determine page range for a character range.
    
    Args:
        char_start: Start character position
        char_end: End character position
        page_positions: List of (start_pos, end_pos, page_num)
        
    Returns:
        Tuple of (page_start, page_end)
    """
    page_start = 1
    page_end = 1
    
    for start_pos, end_pos, page_num in page_positions:
        if start_pos <= char_start < end_pos:
            page_start = page_num
        if start_pos < char_end <= end_pos:
            page_end = page_num
    
    return page_start, page_end


def chunk_with_recursive_splitter(
    pdf_doc: PDFDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[TextChunk]:
    """
    Chunk document using recursive character text splitter (fallback).
    
    Args:
        pdf_doc: Extracted PDF document
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of TextChunk objects
    """
    chunks = []
    chunk_index = 0
    
    # Create splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Process each page
    for page in pdf_doc.pages:
        if not page.has_content:
            continue
        
        # Split page text
        text_chunks = splitter.split_text(page.text)
        
        char_pos = 0
        for text in text_chunks:
            char_start = char_pos
            char_end = char_pos + len(text)
            
            chunks.append(TextChunk(
                text=text,
                metadata=ChunkMetadata(
                    source=pdf_doc.filename,
                    doc_id=pdf_doc.doc_id,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    chunk_id=create_chunk_id(pdf_doc.doc_id, chunk_index),
                    chunk_index=chunk_index,
                    char_start=char_start,
                    char_end=char_end,
                    candidate_name=pdf_doc.candidate_name
                )
            ))
            chunk_index += 1
            char_pos = char_end - chunk_overlap
    
    return chunks


def chunk_with_semantic_splitter(
    pdf_doc: PDFDocument,
    embeddings,
    breakpoint_threshold: int = 95
) -> List[TextChunk]:
    """
    Chunk document using semantic chunker (requires langchain-experimental).
    
    Args:
        pdf_doc: Extracted PDF document
        embeddings: Embedding model for semantic similarity
        breakpoint_threshold: Percentile threshold for breakpoints
        
    Returns:
        List of TextChunk objects
    """
    if not SEMANTIC_CHUNKER_AVAILABLE:
        logger.warning("SemanticChunker not available, using fallback")
        return chunk_with_recursive_splitter(pdf_doc)
    
    try:
        # Combine all text
        full_text = "\n\n".join(
            page.text for page in pdf_doc.pages if page.has_content
        )
        
        # Create semantic chunker
        chunker = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=breakpoint_threshold
        )
        
        # Split text
        docs = chunker.create_documents([full_text])
        
        chunks = []
        char_pos = 0
        
        for i, doc in enumerate(docs):
            text = doc.page_content
            char_start = char_pos
            char_end = char_pos + len(text)
            
            # Estimate page (simple heuristic)
            page_num = min(
                len(pdf_doc.pages),
                max(1, (char_start // 3000) + 1)
            )
            
            chunks.append(TextChunk(
                text=text,
                metadata=ChunkMetadata(
                    source=pdf_doc.filename,
                    doc_id=pdf_doc.doc_id,
                    page_start=page_num,
                    page_end=page_num,
                    chunk_id=create_chunk_id(pdf_doc.doc_id, i),
                    chunk_index=i,
                    char_start=char_start,
                    char_end=char_end,
                    candidate_name=pdf_doc.candidate_name
                )
            ))
            char_pos = char_end
        
        return chunks
        
    except Exception as e:
        logger.warning(f"Semantic chunking failed: {e}, using fallback")
        return chunk_with_recursive_splitter(pdf_doc)


def chunk_documents(
    pdf_docs: List[PDFDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    use_semantic: bool = False,
    embeddings=None
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Chunk multiple PDF documents into LangChain Documents.
    
    Args:
        pdf_docs: List of extracted PDF documents
        chunk_size: Target chunk size (for recursive splitter)
        chunk_overlap: Overlap between chunks
        use_semantic: Whether to try semantic chunking
        embeddings: Embedding model (required for semantic chunking)
        
    Returns:
        Tuple of (list of LangChain Documents, stats dict)
    """
    all_chunks = []
    stats = {
        "total_docs": len(pdf_docs),
        "total_chunks": 0,
        "chunking_method": "unknown",
        "chunks_per_doc": {}
    }
    
    for pdf_doc in pdf_docs:
        # Choose chunking strategy
        if use_semantic and embeddings and SEMANTIC_CHUNKER_AVAILABLE:
            chunks = chunk_with_semantic_splitter(pdf_doc, embeddings)
            stats["chunking_method"] = "semantic"
        else:
            # Try section-based first
            chunks = chunk_by_sections(pdf_doc, max_chunk_size=chunk_size)
            
            if len(chunks) < 2:
                # Fall back to recursive if sections didn't work
                chunks = chunk_with_recursive_splitter(
                    pdf_doc, chunk_size, chunk_overlap
                )
                stats["chunking_method"] = "recursive"
            else:
                stats["chunking_method"] = "section-based"
        
        stats["chunks_per_doc"][pdf_doc.filename] = len(chunks)
        all_chunks.extend(chunks)
    
    stats["total_chunks"] = len(all_chunks)
    
    # Convert to LangChain Documents
    documents = []
    for chunk in all_chunks:
        doc = Document(
            page_content=chunk.text,
            metadata={
                "source": chunk.metadata.source,
                "doc_id": chunk.metadata.doc_id,
                "page_start": chunk.metadata.page_start,
                "page_end": chunk.metadata.page_end,
                "chunk_id": chunk.metadata.chunk_id,
                "chunk_index": chunk.metadata.chunk_index,
                "char_start": chunk.metadata.char_start,
                "char_end": chunk.metadata.char_end,
                "candidate_name": chunk.metadata.candidate_name or "Unknown"
            }
        )
        documents.append(doc)
    
    logger.info(
        f"Chunked {len(pdf_docs)} docs into {len(documents)} chunks "
        f"using {stats['chunking_method']} method"
    )
    
    return documents, stats


def get_chunk_ids(documents: List[Document]) -> List[str]:
    """
    Extract chunk IDs from documents.
    
    Args:
        documents: List of LangChain Documents
        
    Returns:
        List of chunk IDs
    """
    return [doc.metadata.get("chunk_id", str(i)) for i, doc in enumerate(documents)]
