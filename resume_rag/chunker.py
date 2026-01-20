# resume_rag/chunker.py
import re
from typing import List

def _clean(text: str) -> str:
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(
    text: str,
    mode: str = "fixed",
    chunk_size: int = 900,
    overlap: int = 150,
) -> List[str]:
    """
    mode:
      - fixed: chunks by characters, no overlap
      - window: chunks by characters with overlap
      - semantic: split by blank lines/headings then group to ~chunk_size
    """
    text = _clean(text)
    if not text:
        return []

    mode = (mode or "fixed").lower()

    if mode == "fixed":
        return _chunk_fixed(text, chunk_size)

    if mode == "window":
        return _chunk_window(text, chunk_size, overlap)

    if mode == "semantic":
        return _chunk_semantic(text, chunk_size)

    # fallback
    return _chunk_fixed(text, chunk_size)

def _chunk_fixed(text: str, chunk_size: int) -> List[str]:
    chunk_size = max(200, int(chunk_size))
    out = []
    for i in range(0, len(text), chunk_size):
        c = text[i : i + chunk_size].strip()
        if c:
            out.append(c)
    return out

def _chunk_window(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunk_size = max(200, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    step = chunk_size - overlap
    out = []
    i = 0
    while i < len(text):
        c = text[i : i + chunk_size].strip()
        if c:
            out.append(c)
        i += step
    return out

def _chunk_semantic(text: str, target: int) -> List[str]:
    """
    semantic (خفيف): نقسم إلى فقرات/مقاطع حسب الأسطر الفاضية أو headings،
    وبعدين نجمع المقاطع لحد target تقريباً.
    """
    target = max(300, int(target))

    # split by blank lines
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]

    # also split large blocks by sentence-ish
    refined = []
    for b in blocks:
        if len(b) <= target * 1.3:
            refined.append(b)
        else:
            # split by line or punctuation
            parts = re.split(r"(?<=[\.\!\?])\s+|\n", b)
            buff = ""
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(buff) + len(p) + 1 <= target:
                    buff = (buff + " " + p).strip()
                else:
                    if buff:
                        refined.append(buff)
                    buff = p
            if buff:
                refined.append(buff)

    # group to target
    out = []
    buf = ""
    for r in refined:
        if len(buf) + len(r) + 2 <= target:
            buf = (buf + "\n" + r).strip()
        else:
            if buf:
                out.append(buf)
            buf = r
    if buf:
        out.append(buf)

    return out
