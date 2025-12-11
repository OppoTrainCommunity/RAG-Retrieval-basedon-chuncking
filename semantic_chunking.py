import numpy as np
from nltk.tokenize import sent_tokenize

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)

def semantic_chunk_text(
    text: str,
    embedder,
    sim_threshold: float = 0.75,
    max_chars: int = 1200
):
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if not sentences:
        return []

    sent_embeddings = embedder(sentences)

    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_embeddings = [sent_embeddings[0]]
    current_len = len(sentences[0])

    for i in range(1, len(sentences)):
        sent = sentences[i]
        emb = sent_embeddings[i]

        last_emb = current_chunk_embeddings[-1]
        sim = cosine_similarity(last_emb, emb)

        if current_len + len(sent) > max_chars or sim < sim_threshold:
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sent]
            current_chunk_embeddings = [emb]
            current_len = len(sent)
        else:
            current_chunk_sentences.append(sent)
            current_chunk_embeddings.append(emb)
            current_len += len(sent)

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    return chunks
