import json, time
from typing import Dict, List, Set

# imports
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
    "summary",
    "professional summary",
    "objective",
    "skills",
    "technical skills",
    "tech skills",
    "experience",
    "work experience",
    "professional experience",
    "projects",
    "personal projects",
    "education",
    "certifications",
    "achievements",
]


def semantic_chunk_text(
    text: str, max_chars: int = 800, min_chars: int = 200
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
        if len(words) <= 6 and (line.isupper() or normalized in SECTION_KEYWORDS):
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
    text: str, max_chars: int = 800, min_chars: int = 200
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

        doc = {"id": f"doc_{idx}", "file_name": filename, "text": text, "source": "pdf"}
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
        }

        data.append(doc)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} Kaggle resumes into JSON corpus: {output_json_path}")


# NEW: generic indexer that accepts a chunking function
def index_json_with_chunker(
    json_path: str, collection_name: str, chunk_fn, persist_path: str = "chromadb_data/"
):
    chroma_client = chromadb.PersistentClient(path=persist_path)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
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
            metadatas.append(
                {
                    "id": base_id,
                    "chunk_index": i,
                    "file_name": file_name,
                    "source": source,
                }
            )
            ids.append(chunk_id)

    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    print(
        f"Indexed {len(documents)} chunks into Chroma collection '{collection_name}' using {chunk_fn.__name__}."
    )


# semantic chunking
def build_pdfs_index_from_json(json_path: str, collection_name: str = "pdf_docs"):
    index_json_with_chunker(
        json_path=json_path,
        collection_name=collection_name,
        chunk_fn=semantic_chunk_text,
        persist_path="chromadb_data/",
    )


# NEW: index with paragraph chunking
def build_pdfs_index_paragraph_from_json(
    json_path: str, collection_name: str = "pdf_docs_paragraph"
):
    index_json_with_chunker(
        json_path=json_path,
        collection_name=collection_name,
        chunk_fn=paragraph_chunk_text,
        persist_path="chromadb_data/",
    )


def get_pdf_collection(collection_name: str = "pdf_docs"):
    chroma_client = chromadb.PersistentClient(path="chromadb_data/")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )

    print("number of collection:" + str(collection.count()))
    return collection


def retrieve_from_pdfs(
    query_text: str, k: int = 5, collection_name: str = "pdf_semantic"
):
    """
    Retrieve top-k PDF documents (or chunks) relevant to the query_text.
    """
    collection = get_pdf_collection(collection_name)

    results = collection.query(query_texts=[query_text], n_results=k)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    ids = results["ids"][0]

    out = []
    for doc, meta, _id in zip(docs, metas, ids):
        out.append(
            {
                "id": _id,
                "file_name": meta.get("file_name"),
                "source": meta.get("source"),
                "text": doc,
            }
        )

    return out


def hit_rate_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    return 1.0 if any(x in relevant for x in ranked[:k]) else 0.0


def reciprocal_rank(ranked: List[str], relevant: Set[str]) -> float:
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / k


def recall_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for x in ranked[:k] if x in relevant) / len(relevant)


def evaluate_rag_retrieval(
    collection, queries_path: str, qrels_path: str, k: int = 5, n_results: int = 20
):
    queries = json.load(open(queries_path, "r", encoding="utf-8"))
    qrels = json.load(open(qrels_path, "r", encoding="utf-8"))

    rows = []
    p_list, r_list, rr_list, hit_list, lat_list = [], [], [], [], []

    for q in queries:
        qid = q["qid"]
        query_text = q["query"]
        relevant = set(qrels.get(qid, {}).get("relevant_chunk_ids", []))

        t0 = time.perf_counter()
        res = collection.query(query_texts=[query_text], n_results=n_results)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ranked_chunk_ids = res["ids"][0]  # chunk IDs returned by Chroma

        p = precision_at_k(ranked_chunk_ids, relevant, k)
        r = recall_at_k(ranked_chunk_ids, relevant, k)
        rr = reciprocal_rank(ranked_chunk_ids, relevant)
        hit = hit_rate_at_k(ranked_chunk_ids, relevant, k)

        rows.append(
            {
                "qid": qid,
                "type": q.get("type"),
                "query": query_text,
                "P@k": p,
                "R@k": r,
                "RR": rr,
                "Hit@k": hit,
                "latency_ms": latency_ms,
            }
        )

        p_list.append(p)
        r_list.append(r)
        rr_list.append(rr)
        hit_list.append(hit)
        lat_list.append(latency_ms)

    summary = {
        f"avg_P@{k}": sum(p_list) / len(p_list) if p_list else 0.0,
        f"avg_R@{k}": sum(r_list) / len(r_list) if r_list else 0.0,
        "MRR": sum(rr_list) / len(rr_list) if rr_list else 0.0,
        f"HitRate@{k}": sum(hit_list) / len(hit_list) if hit_list else 0.0,
        "avg_latency_ms": sum(lat_list) / len(lat_list) if lat_list else 0.0,
    }

    return rows, summary


def label_helper(
    collection, query: str, qid: str, top_k: int = 20, show_chars: int = 350
):
    print("=" * 90)
    print(f"QID: {qid}")
    print(f"QUERY: {query}")
    print("=" * 90)

    res = collection.query(query_texts=[query], n_results=top_k)

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]

    for rank, (cid, text, meta) in enumerate(zip(ids, docs, metas), start=1):
        print(f"\n[RANK {rank}]  Chunk ID: {cid}")
        print(f"Candidate ID : {meta.get('id')}")
        print(f"File         : {meta.get('file_name')}")
        print("-" * 60)
        preview = text[:show_chars].replace("\n", " ")
        print(preview + ("..." if len(text) > show_chars else ""))
        print("-" * 60)

    print("\n👉 ACTION:")
    print("Copy the Chunk IDs that ACTUALLY contain the answer.")
    print("Add them under this QID in rag_eval_qrels.json.")


if __name__ == "__main__":

    # # 1) Build Kaggle resume corpus (text -> JSON)
    # build_kaggle_resume_corpus(
    #     input_csv_path="data/resumes/UpdatedResumeDataSet.csv",
    #     output_json_path="data/careers.json",
    #     text_column="Resume",
    # )

    # build_pdf_corpus_json(input_dir="data/resumes", output_json_path="data/CVs.json")

    # # 2) Index the corpus into TWO collections:
    # # semantic chunking
    # build_pdfs_index_from_json(
    #     json_path="data/CVs.json", collection_name="cvs_semantic"
    # )

    # # with paragraph chunking
    # build_pdfs_index_paragraph_from_json(
    #     json_path="data/CVs.json", collection_name="cvs_paragraph"
    # )

    # 3) Load collection
    collection = get_pdf_collection("cvs_paragraph")

    # 4) Label one query (prints top-20)
    # Example: label DIR_01 manually by reading printed chunks
    label_helper(
        collection=collection,
        query="Which university does Ibrahim attend?",
        qid="DIR_02",
        top_k=5,
    )

    # # 5) After you fill qrels.json, run evaluation
    # rows, summary = evaluate_rag_retrieval(
    #     collection=collection,
    #     queries_path="data/rag_eval_queries.json",
    #     qrels_path="data/rag_eval_qrels.json",
    #     k=5,
    #     n_results=20,
    # )

    # print("\n paragraph chunking evaluation SUMMARY:", summary)
