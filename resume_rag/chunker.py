import re


def semantic_chunk(text: str):
    """
    Semantic chunking based on clean section headers:
    EDUCATION, EXPERIENCE, PROJECTS, SKILLS, CERTIFICATIONS, SUMMARY, PROFILE, LANGUAGES
    """
    headers = [
        "education",
        "experience",
        "work experience",
        "projects",
        "skills",
        "certifications",
        "summary",
        "profile",
        "languages",
    ]

    lines = text.split("\n")

    chunks = []
    current_header = None
    current_content = []

    def is_header(line: str) -> bool:
        clean = line.strip().lower()
        return clean in headers

    for line in lines:
        stripped = line.strip()

        if not stripped:
            # تجاهُل السطور الفارغة
            continue

        if is_header(stripped):
            # لو في تشنك سابقة → خزِّنها
            if current_header and current_content:
                chunks.append(f"{current_header.upper()}:\n" + "\n".join(current_content))

            current_header = stripped
            current_content = []
        else:
            if current_header:
                current_content.append(stripped)

    # آخر تشنك
    if current_header and current_content:
        chunks.append(f"{current_header.upper()}:\n" + "\n".join(current_content))

    # fallback لو ما اكتشفنا ولا هيدر
    return chunks if chunks else [text]


def sliding_window_chunk(text: str, window: int = 180, overlap: int = 40):
    """
    Fixed-size sliding window chunking:
    - window: عدد الكلمات في كل chunk
    - overlap: عدد الكلمات المشتركة بين chunk والتي تليها
    """
    words = text.split()
    chunks = []

    if not words:
        return []

    start = 0
    while start < len(words):
        end = start + window
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)

        # نتحرك للأمام مع overlap
        start += (window - overlap)

    return chunks
