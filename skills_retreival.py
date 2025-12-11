#imports
import chromadb
from chromadb.utils import embedding_functions
import json
from pprint import pprint
from pypdf import PdfReader
import os
from typing import List, Dict
import pandas as pd

def read_pdf(path: str) -> str:
    """
    Read PDF file and return its full text
    """
    reader = PdfReader(path)
    text = ""
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text 

    return text

import re

SECTION_KEYWORDS = [
    "summary", "professional summary", "objective",
    "skills", "technical skills", "tech skills",
    "experience", "work experience", "professional experience",
    "projects", "personal projects",
    "education", "certifications", "achievements"
]

def semantic_chunk_text(
    text: str,
    max_chars: int = 800,
    min_chars: int = 200
) -> list[str]:
    """
    Heuristic semantic chunking for resumes / career text.

    - Splits into lines
    - Detects section-like headings (SKILLS, EDUCATION, EXPERIENCE, etc.)
    - Groups lines into semantically coherent chunks
    - Ensures chunk length is between min_chars and max_chars where possible
    """
    # Normalize newlines, collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    def is_heading(line: str) -> bool:
        # Example: "SKILLS", "WORK EXPERIENCE", "Projects", "Education:"
        normalized = line.strip().rstrip(":").lower()
        words = normalized.split()
        if len(words) <= 6 and (
            line.isupper() or
            normalized in SECTION_KEYWORDS
        ):
            return True
        return False

    for line in lines:
        # If this looks like a heading and we already have content,
        # start a new chunk
        if is_heading(line) and current_len >= min_chars:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line) + 1
            continue

        # If current chunk would become too large, cut it here
        if current_len + len(line) + 1 > max_chars and current_len >= min_chars:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line) + 1
        else:
            current_lines.append(line)
            current_len += len(line) + 1

    # Add last chunk if any
    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks

# NEW: paragraph-based chunking
def paragraph_chunk_text(
    text: str,
    max_chars: int = 800,
    min_chars: int = 200
) -> list[str]:
    """
    Simple paragraph-based chunking.

    - Split on blank lines to get paragraphs
    - Merge small paragraphs together
    - If a chunk gets too long, start a new one
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Split on blank lines => paragraphs
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for para in raw_paragraphs:
        # If this paragraph alone is huge, just cut it into pieces
        if len(para) > max_chars:
            # flush existing chunk
            if current_lines:
                chunks.append("\n\n".join(current_lines))
                current_lines = []
                current_len = 0

            # split big paragraph into smaller slices
            start = 0
            while start < len(para):
                end = min(start + max_chars, len(para))
                chunks.append(para[start:end])
                start = end
            continue

        # Normal case: try to add this paragraph to current chunk
        if current_len + len(para) + 2 <= max_chars:
            current_lines.append(para)
            current_len += len(para) + 2
        else:
            # end current chunk if it's big enough
            if current_len >= min_chars:
                chunks.append("\n\n".join(current_lines))
                current_lines = [para]
                current_len = len(para) + 2
            else:
                # if current chunk is too small, just force add
                current_lines.append(para)
                current_len += len(para) + 2

    if current_lines:
        chunks.append("\n\n".join(current_lines))

    return chunks

def build_pdf_corpus_json(input_dir: str, output_json_path: str) -> None:
    """
    Read all PDF files from input_dir, extract their text, and save them as a JSON list.
    Each entry will look like:
    {
        "id": "resume_0",
        "file_name": "samaShalabiCV(AI).pdf",
        "text": "full extracted text...",
        "source": "resume"
    }
    """
    data: List[Dict] = []
    idx = 0

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".pdf"):
            continue  # skip non-PDFs

        pdf_path = os.path.join(input_dir, filename)
        print(f"Reading PDF: {pdf_path}")

        text = read_pdf(pdf_path)

        doc = {
            "id": f"doc_{idx}",
            "file_name": filename,
            "text": text,
            "source": "pdf"
        }
        data.append(doc)
        idx += 1

    # Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} PDF documents into JSON: {output_json_path}")

def build_kaggle_resume_corpus(
    input_csv_path: str,
    output_json_path: str,
    text_column: str = "Resume",
) -> None:
    df = pd.read_csv(input_csv_path)
    df = df.dropna(subset=[text_column])

    data = []
    for idx, row in df.iterrows():
        text = str(row[text_column])
        doc = {
            "id": f"kaggle_{idx}",
            "file_name": f"kaggle_resume_{idx}.txt",   
            "text": text,
            "source": "kaggle_resume"
        }

        data.append(doc)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} Kaggle resumes into JSON corpus: {output_json_path}")

# NEW: generic indexer that accepts a chunking function
def index_json_with_chunker(
    json_path: str,
    collection_name: str,
    chunk_fn,
    persist_path: str = "chromadb_data/"
):
    chroma_client = chromadb.PersistentClient(path=persist_path)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    with open(json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    documents = []
    metadatas = []
    ids = []

    for d in docs:
        full_text = d["text"]
        base_id = d["id"]
        file_name = d.get("file_name", f"{base_id}.txt")
        source = d.get("source", "pdf")

        chunks = chunk_fn(full_text)

        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            chunk_id = f"{base_id}_chunk_{i}"
            documents.append(chunk)
            metadatas.append({
                "id": base_id,
                "chunk_index": i,
                "file_name": file_name,
                "source": source
            })
            ids.append(chunk_id)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"Indexed {len(documents)} chunks into Chroma collection '{collection_name}' using {chunk_fn.__name__}.")

# keep old API for semantic chunking
def build_pdfs_index_from_json(json_path: str, collection_name: str = "pdf_docs"):
    index_json_with_chunker(
        json_path=json_path,
        collection_name=collection_name,
        chunk_fn=semantic_chunk_text,
        persist_path="chromadb_data/"
    )

# NEW: index with paragraph chunking
def build_pdfs_index_paragraph_from_json(json_path: str, collection_name: str = "pdf_docs_paragraph"):
    index_json_with_chunker(
        json_path=json_path,
        collection_name=collection_name,
        chunk_fn=paragraph_chunk_text,
        persist_path="chromadb_data/"
    )

def get_pdf_collection(collection_name: str = "pdf_docs"):
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

    print("number of collection:" + str(collection.count()))
    return collection

def retrieve_from_pdfs(query_text: str, k: int = 5, collection_name: str = "pdf_docs"):
    """
    Retrieve top-k PDF documents (or chunks) relevant to the query_text.
    """
    collection = get_pdf_collection(collection_name)

    results = collection.query(
        query_texts=[query_text],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    out = []
    for doc, meta, _id in zip(docs, metas, ids):
        out.append({
            "id": _id,
            "file_name": meta.get("file_name"),
            "source": meta.get("source"),
            "text": doc
        })

    return out

import math
import random

def precision_at_k(ranked_ids, relevant_ids, k):
    if k == 0:
        return 0.0
    ranked_k = ranked_ids[:k]
    hits = sum(1 for d in ranked_k if d in relevant_ids)
    return hits / k

def recall_at_k(ranked_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0
    ranked_k = ranked_ids[:k]
    hits = sum(1 for d in ranked_k if d in relevant_ids)
    return hits / len(relevant_ids)

def average_precision(ranked_ids, relevant_ids):
    """Average Precision for a single query."""
    if not relevant_ids:
        return 0.0

    ap_sum = 0.0
    hits = 0
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            hits += 1
            ap_sum += hits / i  # precision at this rank

    if hits == 0:
        return 0.0

    return ap_sum / min(len(relevant_ids), len(ranked_ids))

def dcg_at_k(ranked_ids, relevant_ids, k):
    dcg = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranked_ids, relevant_ids, k):
    """nDCG@k for binary relevance."""
    if not relevant_ids:
        return 0.0

    dcg = dcg_at_k(ranked_ids, relevant_ids, k)

    # Ideal ranking: all relevant docs first
    ideal_order = list(relevant_ids)[:k]
    idcg = dcg_at_k(ideal_order, relevant_ids, k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg

def load_kaggle_id_to_category(
    csv_path: str,
    text_column: str = "Resume",
    label_column: str = "Category"
) -> Dict[str, str]:
    """
    Build a mapping from our synthetic ID (e.g. 'kaggle_12') to its category.
    Must mirror the logic used in build_kaggle_resume_corpus.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column])

    id_to_cat = {}
    for idx, row in df.iterrows():
        resume_id = f"kaggle_{idx}"
        category = str(row[label_column]) if label_column in df.columns else "Unknown"
        id_to_cat[resume_id] = category

    return id_to_cat

def evaluate_retrieval_system(
    corpus_json_path: str = "data/careers.json",
    kaggle_csv_path: str = "data/resumes/UpdatedResumeDataSet.csv",
    collection_name: str = "pdf_docs",
    k: int = 10,
    max_queries: int = 100,
) -> None:
    """
    Evaluate the current Chroma-based resume retrieval system using:
    - Precision@k
    - Recall@k
    - MAP (Mean Average Precision)
    - nDCG@k

    Query setup:
        For each resume R:
        - Use its text as the query
        - Relevant docs = other resumes with the same Category (from Kaggle CSV)
    """
    # 1) Load mapping from base resume id -> category (from Kaggle CSV)
    id_to_category = load_kaggle_id_to_category(kaggle_csv_path)

    # 2) Load the JSON corpus with ids + text (built by build_kaggle_resume_corpus)
    with open(corpus_json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # Keep only docs whose ids we have categories for
    docs = [d for d in docs if d["id"] in id_to_category]

    if not docs:
        print("No documents found that match Kaggle IDs. Check corpus_json_path and CSV path.")
        return

    # 3) Sample a subset of queries
    random.seed(42)
    if len(docs) <= max_queries:
        query_docs = docs
    else:
        query_docs = random.sample(docs, max_queries)

    # 4) Open Chroma collection once
    collection = get_pdf_collection(collection_name)

    precisions = []
    recalls = []
    aps = []
    ndcgs = []
    num_effective_queries = 0

    for q_doc in query_docs:
        q_id = q_doc["id"]
        q_cat = id_to_category.get(q_id)
        q_text = q_doc["text"]

        # Relevant docs = all other docs with the same category
        relevant_ids = {doc_id for doc_id, cat in id_to_category.items()
                        if cat == q_cat and doc_id != q_id}

        # If no other doc shares this category, skip query
        if not relevant_ids:
            continue

        # 5) Run retrieval (chunk-level)
        results = collection.query(
            query_texts=[q_text],
            n_results=5 * k  # get more chunks and then dedupe
        )

        metas = results["metadatas"][0]

        # Convert chunk results to a ranked list of base resume ids
        ranked_base_ids = []
        seen = set()

        for meta in metas:
            base_id = meta.get("id")  # base resume id you stored in metadatas
            if base_id is None:
                continue
            if base_id == q_id:
                # Optionally skip the query doc itself
                continue
            if base_id not in seen:
                seen.add(base_id)
                ranked_base_ids.append(base_id)
            if len(ranked_base_ids) >= k:
                break

        if not ranked_base_ids:
            continue

        p = precision_at_k(ranked_ids=ranked_base_ids, relevant_ids=relevant_ids, k=k)
        r = recall_at_k(ranked_ids=ranked_base_ids, relevant_ids=relevant_ids, k=k)
        ap = average_precision(ranked_ids=ranked_base_ids, relevant_ids=relevant_ids)
        nd = ndcg_at_k(ranked_ids=ranked_base_ids, relevant_ids=relevant_ids, k=k)

        precisions.append(p)
        recalls.append(r)
        aps.append(ap)
        ndcgs.append(nd)
        num_effective_queries += 1

    if num_effective_queries == 0:
        print("No queries had at least one relevant document. Evaluation skipped.")
        return

    print(f"\n[Collection: {collection_name}] Evaluation over {num_effective_queries} queries")
    print(f"Precision@{k}: {sum(precisions)/len(precisions):.4f}")
    print(f"Recall@{k}:    {sum(recalls)/len(recalls):.4f}")
    print(f"MAP:           {sum(aps)/len(aps):.4f}")
    print(f"nDCG@{k}:      {sum(ndcgs)/len(ndcgs):.4f}")

if __name__ == "__main__":
    
    # 1) Build Kaggle resume corpus (text -> JSON)
    build_kaggle_resume_corpus(
        input_csv_path="data/resumes/UpdatedResumeDataSet.csv",
        output_json_path="data/careers.json",
        text_column="Resume",     
    )

    # 2) Index the corpus into TWO collections:
    # semantic chunking
    build_pdfs_index_from_json(
        json_path="data/careers.json",
        collection_name="pdf_semantic"
    )

    # with paragraph chunking
    build_pdfs_index_paragraph_from_json(
        json_path="data/careers.json",
        collection_name="pdf_paragraph"
    )

    # 3) Evaluate both chunking strategies using the SAME eval code

    # Semantic chunking evaluation
    evaluate_retrieval_system(
        corpus_json_path="data/careers.json",
        kaggle_csv_path="data/resumes/UpdatedResumeDataSet.csv",
        collection_name="pdf_semantic",
        k=10,
        max_queries=100,   
    )

    # Paragraph chunking evaluation
    evaluate_retrieval_system(
        corpus_json_path="data/careers.json",
        kaggle_csv_path="data/resumes/UpdatedResumeDataSet.csv",
        collection_name="pdf_paragraph",
        k=10,
        max_queries=100,   
    )

    # Optional: example retrieval
    # query = "Python and backend skills with web development experience"
    # results = retrieve_from_pdfs(query, k=3, collection_name="pdf_semantic")
    # pprint(results)
