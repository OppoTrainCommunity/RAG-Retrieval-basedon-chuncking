def chunk_fixed(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append(chunk_text)
        start += (chunk_size - overlap)
    return chunks


SECTION_HEADERS = ["education", "experience", "projects", "skills", "certifications", "training"]

def chunk_by_section(text):
    lines = text.splitlines()
    chunks = []
    current_section = []
    current_name = "general"

    def flush():
        if current_section:
            chunks.append({
                "section": current_name,
                "text": "\n".join(current_section).strip()
            })

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if any(h in lower for h in SECTION_HEADERS):
            flush()
            current_section = [stripped]
            current_name = stripped
        else:
            current_section.append(stripped)

    flush()
    return chunks
