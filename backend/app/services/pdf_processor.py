import hashlib
import logging
from pathlib import Path
from typing import Optional

import pdfplumber
from PyPDF2 import PdfReader

from app.config import settings

logger = logging.getLogger("cv_analyzer.pdf_processor")

PDF_MAGIC_BYTES = b"%PDF-"
MIN_TEXT_LENGTH = 50


class PDFProcessor:
    def validate_file(self, content: bytes, filename: str) -> tuple[bool, str]:
        if not filename.lower().endswith(".pdf"):
            return False, "File must have a .pdf extension"

        if len(content) > settings.max_file_size:
            return False, f"File exceeds {settings.max_file_size // (1024*1024)} MB limit"

        if not content[:5] == PDF_MAGIC_BYTES:
            return False, "File does not appear to be a valid PDF"

        return True, ""

    def compute_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def extract_text(self, content: bytes) -> tuple[bool, str]:
        text = self._extract_with_pdfplumber(content)
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            text = self._extract_with_pypdf2(content)

        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return False, "Could not extract sufficient text from PDF"

        return True, text.strip()

    def _extract_with_pdfplumber(self, content: bytes) -> Optional[str]:
        try:
            import io
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                return "\n\n".join(pages_text)
        except Exception as e:
            logger.warning("pdfplumber extraction failed: %s", e)
            return None

    def _extract_with_pypdf2(self, content: bytes) -> Optional[str]:
        try:
            import io
            reader = PdfReader(io.BytesIO(content))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            return "\n\n".join(pages_text)
        except Exception as e:
            logger.warning("PyPDF2 extraction failed: %s", e)
            return None

    def save_file(self, content: bytes, filename: str) -> str:
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        filepath = upload_dir / filename
        # Avoid overwriting
        if filepath.exists():
            stem = filepath.stem
            suffix = filepath.suffix
            counter = 1
            while filepath.exists():
                filepath = upload_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        filepath.write_bytes(content)
        return str(filepath)
