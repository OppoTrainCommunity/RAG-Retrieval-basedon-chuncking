# resume_rag/parse_pdf.py
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    text = "\n".join(parts)
    text = text.replace("\x00", "").strip()
    return text
