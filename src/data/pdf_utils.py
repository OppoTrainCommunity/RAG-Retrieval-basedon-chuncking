"""
PDF text extraction utilities for CV RAG System.
Extracts text from PDF files for ingestion into the RAG pipeline.
"""

import logging
from io import BytesIO
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    max_pages: Optional[int] = None,
) -> str:
    """
    Extract text from PDF bytes using pypdf.
    
    This is a best-effort extraction that works well with text-based PDFs.
    Scanned/image-based PDFs will return empty or minimal text.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file.
        max_pages: Optional maximum number of pages to extract.
                   If None, extracts all pages.
    
    Returns:
        Extracted text content as a single string.
        Empty string if extraction fails or PDF is scanned/image-based.
    
    Raises:
        ImportError: If pypdf is not installed.
        Exception: For other PDF parsing errors.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF extraction. "
            "Install it with: pip install pypdf"
        )
    
    text_parts = []
    
    try:
        # Create PDF reader from bytes
        pdf_file = BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file)
        
        # Determine number of pages to process
        total_pages = len(reader.pages)
        pages_to_process = total_pages if max_pages is None else min(max_pages, total_pages)
        
        logger.info(f"Extracting text from {pages_to_process}/{total_pages} pages")
        
        for page_num in range(pages_to_process):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            
            if page_text:
                text_parts.append(page_text)
        
        # Combine all pages
        full_text = "\n\n".join(text_parts)
        
        # Basic cleanup
        full_text = _clean_extracted_text(full_text)
        
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        
        return full_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise


def _clean_extracted_text(text: str) -> str:
    """
    Clean up extracted PDF text.
    
    Args:
        text: Raw extracted text.
    
    Returns:
        Cleaned text.
    """
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Fix common PDF extraction issues
    # Multiple newlines -> double newline (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove page numbers (common patterns)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*(?:of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def is_text_extractable(pdf_bytes: bytes, min_chars: int = 100) -> bool:
    """
    Check if a PDF has extractable text.
    
    This is a quick heuristic to determine if a PDF is text-based
    or scanned/image-based.
    
    Args:
        pdf_bytes: Raw bytes of the PDF file.
        min_chars: Minimum characters to consider the PDF text-extractable.
    
    Returns:
        True if PDF appears to have extractable text, False otherwise.
    """
    try:
        text = extract_text_from_pdf_bytes(pdf_bytes, max_pages=2)
        return len(text.strip()) >= min_chars
    except Exception:
        return False
