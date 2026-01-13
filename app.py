import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st

from src.services.retrieval_service import RetrievalService


# -----------------------------
# Helpers
# -----------------------------
def load_file_names_from_cvs_json(cvs_json_path: str) -> List[str]:
    """
    Loads unique file_name values from your data/CVs.json so the UI can filter results.
    """
    p = Path(cvs_json_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    names = sorted({d.get("file_name") for d in docs if d.get("file_name")})
    return names


def render_results(results: Dict[str, Any], max_show: int = 10):
    """
    Renders Chroma results returned by your ChromaIndexer.search() through the RetrievalService.
    Expected keys: documents, metadatas, distances (same as your main.py).
    """
    if not results or "documents" not in results or not results["documents"]:
        st.warning("No results returned.")
        return

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        st.warning("No documents returned. (Is the collection empty?)")
        return

    st.subheader("Top results")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        if i > max_show:
            break

        st.markdown(f"### Rank {i}  |  distance: `{dist:.4f}`")
        st.caption(
            f"file_name: {meta.get('file_name', 'N/A')}  |  "
            f"doc_id: {meta.get('doc_id', 'N/A')}  |  "
            f"chunk_id: {meta.get('chunk_id', 'N/A')}"
        )
        st.write(doc)
        st.divider()

    # Output context box (copy/paste into generation later)
    st.subheader("Output (concatenated context)")
    context = "\n\n---\n\n".join(docs)
    st.text_area("Retrieved context", value=context, height=260)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Resume RAG – Retrieval UI", layout="wide")
st.title("Resume RAG – Retrieval UI (Paragraph Chunking)")

# Defaults that match your project indexing
DEFAULT_COLLECTION = "paragraph_chunking"
DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"  # your main.py used embedding_models[0]

with st.sidebar:
    st.header("Chroma Settings (already indexed)")

    persist_dir = st.text_input("persist_directory", value=DEFAULT_PERSIST_DIR)
    collection_name = st.text_input("collection_name", value=DEFAULT_COLLECTION)
    embedding_model = st.text_input("embedding_model", value=DEFAULT_EMBEDDING_MODEL)

    st.divider()

    st.header("Data (optional)")
    cvs_json_path = st.text_input("CVs.json path (for file_name filter)", value="data/CVs.json")

    st.divider()

    st.header("Retrieval")
    top_k = st.slider("Top K", 1, 20, 10, 1)
    show_n = st.slider("Show results", 1, 20, 10, 1)

# init service (uses your project class)
service = RetrievalService(
    persist_dir=persist_dir,
    embedding_model=embedding_model,
)

# UI: search
st.subheader("Search & Retrieve Best Chunks")

query = st.text_input("Search query", value="machine learning and data science skills")

# Optional file_name filter
file_names = load_file_names_from_cvs_json(cvs_json_path)
use_filter = st.checkbox("Filter by file_name", value=False)

selected_file_name: Optional[str] = None
if use_filter:
    if file_names:
        selected_file_name = st.selectbox("Select CV file_name", file_names)
    else:
        st.info("No file names found (check CVs.json path). Filter will be disabled.")
        selected_file_name = None

# Search button
if st.button("🔎 Retrieve", type="primary"):
    where = {"file_name": selected_file_name} if selected_file_name else None

    try:
        results = service.search(
            collection_name=collection_name,
            query=query,
            k=top_k,
            where=where,
        )
        render_results(results, max_show=show_n)
    except Exception as e:
        st.error("Search failed. Common causes: wrong persist_directory, wrong collection name, or missing Chroma DB.")
        st.code(str(e))
        st.info(
            "Double-check:\n"
            f"- persist_directory: {persist_dir}\n"
            f"- collection_name: {collection_name}\n"
            "And confirm you already built the index into that directory."
        )

st.caption(
    "Note: This UI uses your project RetrievalService + ChromaIndexer (no custom chunking code here). "
    "It assumes the collection is already indexed."
)
