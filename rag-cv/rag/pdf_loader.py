"""
PDF Loader Module
=================

Robust PDF text extraction with metadata support.
Handles multi-page PDFs with fallback extraction methods.
"""

import io
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, BinaryIO
from dataclasses import dataclass

from .logging_utils import get_logger

logger = get_logger(__name__)

# Try to import PDF libraries with fallbacks
try:
    from pypdf import PdfReader
    PDF_LIBRARY = "pypdf"
except ImportError:
    try:
        from PyPDF2 import PdfReader
        PDF_LIBRARY = "PyPDF2"
    except ImportError:
        PdfReader = None
        PDF_LIBRARY = None
        logger.warning("No PDF library found. Install pypdf: pip install pypdf")

# PyMuPDF fallback
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class ExtractedPage:
    """Represents extracted content from a single PDF page."""
    text: str
    page_number: int
    char_count: int
    has_content: bool


@dataclass
class PDFDocument:
    """Represents a fully extracted PDF document."""
    filename: str
    doc_id: str
    pages: List[ExtractedPage]
    total_pages: int
    total_chars: int
    extraction_method: str
    candidate_name: Optional[str] = None


def compute_doc_id(file_bytes: bytes) -> str:
    """
    Compute a unique document ID from file content.
    
    Args:
        file_bytes: Raw PDF file bytes
        
    Returns:
        SHA256 hash of the file content
    """
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def normalize_text(text: str) -> str:
    """
    Normalize extracted text by cleaning whitespace.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple spaces/tabs with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


def extract_candidate_name(filename: str, first_page_text: str) -> Optional[str]:
    """
    Try to extract candidate name from filename or first page.
    
    Args:
        filename: PDF filename
        first_page_text: Text content of first page
        
    Returns:
        Candidate name if found, None otherwise
    """
    # Try to extract from filename (common patterns)
    # Remove extension and common prefixes
    name = Path(filename).stem
    
    # Remove common prefixes like "CV_", "Resume_", etc.
    prefixes_to_remove = [
        r'^cv[_\-\s]*',
        r'^resume[_\-\s]*',
        r'^curriculum[_\-\s]*vitae[_\-\s]*',
    ]
    for prefix in prefixes_to_remove:
        name = re.sub(prefix, '', name, flags=re.IGNORECASE)
    
    # Clean up remaining name
    name = re.sub(r'[_\-]+', ' ', name).strip()
    
    if name and len(name) > 2:
        return name.title()
    
    return None


def extract_with_pypdf(file_bytes: bytes, filename: str) -> Optional[PDFDocument]:
    """
    Extract text from PDF using pypdf/PyPDF2.
    
    Args:
        file_bytes: Raw PDF bytes
        filename: Original filename
        
    Returns:
        PDFDocument if successful, None otherwise
    """
    if PdfReader is None:
        return None
    
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = []
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = normalize_text(text)
            
            pages.append(ExtractedPage(
                text=text,
                page_number=page_num,
                char_count=len(text),
                has_content=len(text) > 10
            ))
        
        doc_id = compute_doc_id(file_bytes)
        total_chars = sum(p.char_count for p in pages)
        
        # Try to extract candidate name
        first_page_text = pages[0].text if pages else ""
        candidate_name = extract_candidate_name(filename, first_page_text)
        
        return PDFDocument(
            filename=filename,
            doc_id=doc_id,
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            extraction_method="pypdf",
            candidate_name=candidate_name
        )
        
    except Exception as e:
        logger.warning(f"pypdf extraction failed for {filename}: {e}")
        return None


def extract_with_pymupdf(file_bytes: bytes, filename: str) -> Optional[PDFDocument]:
    """
    Extract text from PDF using PyMuPDF (fallback).
    
    Args:
        file_bytes: Raw PDF bytes
        filename: Original filename
        
    Returns:
        PDFDocument if successful, None otherwise
    """
    if not PYMUPDF_AVAILABLE:
        return None
    
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text() or ""
            text = normalize_text(text)
            
            pages.append(ExtractedPage(
                text=text,
                page_number=page_num + 1,
                char_count=len(text),
                has_content=len(text) > 10
            ))
        
        doc.close()
        
        doc_id = compute_doc_id(file_bytes)
        total_chars = sum(p.char_count for p in pages)
        
        # Try to extract candidate name
        first_page_text = pages[0].text if pages else ""
        candidate_name = extract_candidate_name(filename, first_page_text)
        
        return PDFDocument(
            filename=filename,
            doc_id=doc_id,
            pages=pages,
            total_pages=len(pages),
            total_chars=total_chars,
            extraction_method="pymupdf",
            candidate_name=candidate_name
        )
        
    except Exception as e:
        logger.warning(f"PyMuPDF extraction failed for {filename}: {e}")
        return None


def extract_pdf(
    file_bytes: bytes,
    filename: str
) -> Optional[PDFDocument]:
    """
    Extract text from a PDF file using available libraries.
    
    Tries pypdf first, falls back to PyMuPDF if needed.
    
    Args:
        file_bytes: Raw PDF file bytes
        filename: Original filename
        
    Returns:
        PDFDocument if successful, None otherwise
    """
    # Try pypdf first
    doc = extract_with_pypdf(file_bytes, filename)
    
    # Check if extraction yielded content
    if doc and doc.total_chars > 50:
        logger.info(f"Extracted {doc.total_pages} pages ({doc.total_chars} chars) from {filename} using pypdf")
        return doc
    
    # Fallback to PyMuPDF
    if PYMUPDF_AVAILABLE:
        doc = extract_with_pymupdf(file_bytes, filename)
        if doc and doc.total_chars > 50:
            logger.info(f"Extracted {doc.total_pages} pages ({doc.total_chars} chars) from {filename} using PyMuPDF")
            return doc
    
    # If we got some content with pypdf but it was minimal, still return it
    if doc:
        logger.warning(f"Low content extraction from {filename}: {doc.total_chars} chars")
        return doc
    
    logger.error(f"Failed to extract any content from {filename}")
    return None


def load_pdfs_from_uploaded_files(
    uploaded_files: List[Any]
) -> Tuple[List[PDFDocument], Dict[str, Any]]:
    """
    Load multiple PDFs from Streamlit uploaded files.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        Tuple of (list of PDFDocuments, stats dict)
    """
    documents = []
    stats = {
        "total_files": len(uploaded_files),
        "successful": 0,
        "failed": 0,
        "failed_files": [],
        "total_pages": 0,
        "total_chars": 0
    }
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
        
        # Reset file pointer for potential re-read
        uploaded_file.seek(0)
        
        doc = extract_pdf(file_bytes, filename)
        
        if doc:
            documents.append(doc)
            stats["successful"] += 1
            stats["total_pages"] += doc.total_pages
            stats["total_chars"] += doc.total_chars
        else:
            stats["failed"] += 1
            stats["failed_files"].append(filename)
    
    logger.info(
        f"Loaded {stats['successful']}/{stats['total_files']} PDFs, "
        f"{stats['total_pages']} pages, {stats['total_chars']} chars"
    )
    
    return documents, stats


def load_pdfs_from_directory(
    directory: Path
) -> Tuple[List[PDFDocument], Dict[str, Any]]:
    """
    Load all PDFs from a directory.
    
    Args:
        directory: Path to directory containing PDFs
        
    Returns:
        Tuple of (list of PDFDocuments, stats dict)
    """
    pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
    
    documents = []
    stats = {
        "total_files": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "failed_files": [],
        "total_pages": 0,
        "total_chars": 0
    }
    
    for pdf_path in pdf_files:
        file_bytes = pdf_path.read_bytes()
        doc = extract_pdf(file_bytes, pdf_path.name)
        
        if doc:
            documents.append(doc)
            stats["successful"] += 1
            stats["total_pages"] += doc.total_pages
            stats["total_chars"] += doc.total_chars
        else:
            stats["failed"] += 1
            stats["failed_files"].append(pdf_path.name)
    
    return documents, stats


def save_uploaded_file(
    uploaded_file: Any,
    upload_dir: Path
) -> Path:
    """
    Save an uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        upload_dir: Directory to save to
        
    Returns:
        Path to saved file
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique filename
    file_bytes = uploaded_file.read()
    doc_id = compute_doc_id(file_bytes)
    filename = f"{doc_id}_{uploaded_file.name}"
    filepath = upload_dir / filename
    
    # Save file
    filepath.write_bytes(file_bytes)
    
    # Reset file pointer
    uploaded_file.seek(0)
    
    logger.info(f"Saved uploaded file to {filepath}")
    return filepath
