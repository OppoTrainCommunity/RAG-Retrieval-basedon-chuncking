from pathlib import Path
import uuid
import hashlib

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
import fitz

from semantic_chunking import semantic_chunk_text

DB_PATH = "chroma_cvs_db"
COLLECTION_NAME = "cv_collection"

class LocalHashEmbeddingFunction(EmbeddingFunction):
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str):
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(h, 16) % self.dim
            vec[idx] += 1.0
        return vec

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed_one(t) for t in input]

    def name(self) -> str:
        return "local-hash-embedding"

embedding_func = LocalHashEmbeddingFunction(dim=256)

client = chromadb.PersistentClient(path=DB_PATH)

try:
    cv_collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )
except Exception:
    cv_collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            all_text.append(page_text)
    doc.close()
    return "\n".join(all_text)

def index_cv_pdf_in_chroma(pdf_path: str):
    pdf_path = Path(pdf_path)
    text = extract_text_from_pdf(str(pdf_path))
    if not text.strip():
        return None

    cv_id = str(uuid.uuid4())
    chunks = semantic_chunk_text(text, embedding_func)

    ids = []
    docs = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        ids.append(f"{cv_id}_chunk_{idx}")
        docs.append(chunk)
        metadatas.append({
            "cv_id": cv_id,
            "file_name": pdf_path.name,
            "source": "pdf",
            "chunk_index": idx
        })

    cv_collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas
    )

    return cv_id

def index_resume_text_in_chroma(resume_text: str, category: str, resume_id: str | None = None):
    if not resume_text.strip():
        return None

    if resume_id is None:
        resume_id = str(uuid.uuid4())

    chunks = semantic_chunk_text(resume_text, embedding_func)

    ids = []
    docs = []
    metadatas = []

    for idx, chunk in enumerate(chunks):
        ids.append(f"{resume_id}_chunk_{idx}")
        docs.append(chunk)
        metadatas.append({
            "resume_id": resume_id,
            "category": category,
            "source": "kaggle_resume_dataset",
            "chunk_index": idx
        })

    cv_collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas
    )

    return resume_id

def query_all_cvs(question: str, n_results: int = 5):
    results = cv_collection.query(
        query_texts=[question],
        n_results=n_results
    )
    return results

def query_single_cv(question: str, cv_id: str, n_results: int = 5):
    results = cv_collection.query(
        query_texts=[question],
        n_results=n_results,
        where={"cv_id": cv_id}
    )
    return results
