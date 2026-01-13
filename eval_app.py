import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
import chromadb

from src.preprocessors import CVDataPreprocessor
from src.chunkers import ParagraphChunker, SemanticChunker
from src.indexers import ChromaIndexer
from src.evaluation.retrieval_evaluator import RetrievalEvaluator


# ----------------------------
# Helpers
# ----------------------------
def list_collections(persist_dir: str) -> List[str]:
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        cols = client.list_collections()
        return sorted([c.name for c in cols])
    except Exception:
        return []


def load_cvs_json(data_path: str) -> List[dict]:
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def unique_file_names_from_cvs(data_path: str) -> List[str]:
    docs = load_cvs_json(data_path)
    names = sorted({d.get("file_name", "") for d in docs if d.get("file_name")})
    return names


def render_results(results: Dict, limit: int = 10):
    if not results or "documents" not in results or not results["documents"]:
        st.warning("No results.")
        return

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        if i > limit:
            break
        st.markdown(f"### Rank {i} — distance: `{dist:.4f}`")
        st.caption(f"file_name: {meta.get('file_name')} | doc_id: {meta.get('doc_id')} | chunk_id: {meta.get('chunk_id')}")
        st.write(doc)
        st.divider()


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Resume RAG – Retrieval UI", layout="wide")
st.title("Resume RAG – Retrieval UI (Chunking → Chroma → Search → Eval)")

with st.sidebar:
    st.header("Paths / Storage")

    data_path = st.text_input("CVs JSON path", value="data/CVs.json")
    queries_path = st.text_input("Queries path", value="data/queries.txt")
    persist_dir = st.text_input("Chroma persist dir", value="./chroma_db")

    st.header("Chunking Strategy")

    strategy = st.selectbox("Strategy", ["paragraph", "semantic"], index=0)

    embedding_models = [
        "all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "all-MiniLM-L6-v2",
    ]
    embedding_model = st.selectbox("Embedding model", embedding_models, index=0)

    if strategy == "paragraph":
        tokens_per_chunk = st.slider("tokens_per_chunk", 80, 600, 150, 10)
        chunk_overlap = st.slider("chunk_overlap", 0, 200, 50, 5)
        short_paragraph_threshold = st.slider("short_paragraph_threshold (words)", 30, 200, 90, 5)
    else:
        avg_chunk_size = st.slider("avg_chunk_size", 100, 800, 300, 25)
        min_chunk_size = st.slider("min_chunk_size", 20, 300, 75, 5)

    st.header("Retrieval / Eval")
    top_k = st.slider("Top K", 1, 20, 10, 1)

    st.caption("Tip: Use unique collection names per strategy to avoid mixing old/new chunks.")


tabs = st.tabs(["1) Index / Rebuild", "2) Search", "3) Evaluate"])


# ----------------------------
# Tab 1: Index / Rebuild
# ----------------------------
with tabs[0]:
    st.subheader("Build / Rebuild a Chroma collection")

    default_collection = f"{strategy}_chunking"
    collection_name = st.text_input("Collection name", value=default_collection)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**What this does:**")
        st.write(
            "- Loads CVs from your JSON\n"
            "- Chunks them using the selected strategy\n"
            "- Indexes chunks into ChromaDB\n"
        )

        delete_first = st.checkbox("Delete collection first (fresh rebuild)", value=False)

        if st.button("🚀 Build Index", type="primary"):
            if not Path(data_path).exists():
                st.error(f"CVs.json not found at: {data_path}")
            else:
                # Prepare data
                pre = CVDataPreprocessor()
                texts, metadatas = pre.prepare_data(data_path)

                # Select chunker
                if strategy == "paragraph":
                    chunker = ParagraphChunker(
                        tokens_per_chunk=tokens_per_chunk,
                        chunk_overlap=chunk_overlap,
                        short_paragraph_threshold=short_paragraph_threshold,
                    )
                else:
                    chunker = SemanticChunker(
                        avg_chunk_size=avg_chunk_size,
                        min_chunk_size=min_chunk_size,
                        embedding_model=embedding_model,
                    )

                # Chunk
                with st.spinner("Chunking CVs..."):
                    chunks, metas, ids = chunker.chunk_texts(texts, metadatas)

                st.success(f"Chunking complete ✅  | chunks: {len(chunks)}")

                # Index
                with st.spinner("Indexing into Chroma..."):
                    # Create indexer
                    indexer = ChromaIndexer(
                        collection_name=collection_name,
                        embedding_model=embedding_model,
                        persist_directory=persist_dir,
                        client_type="persistent",
                    )

                    # Optional: delete first
                    if delete_first:
                        try:
                            indexer.delete_collection()
                            indexer = ChromaIndexer(
                                collection_name=collection_name,
                                embedding_model=embedding_model,
                                persist_directory=persist_dir,
                                client_type="persistent",
                            )
                        except Exception as e:
                            st.warning(f"Could not delete collection (maybe doesn't exist): {e}")

                    indexer.add_chunks(chunks, metas, ids)

                stats = indexer.get_collection_stats()
                st.success(f"Indexed ✅  | collection count: {stats['count']}")

    with col2:
        st.markdown("**Available collections in persist dir:**")
        cols = list_collections(persist_dir)
        if cols:
            st.code("\n".join(cols))
        else:
            st.info("No collections found yet (or persist dir not created).")


# ----------------------------
# Tab 2: Search
# ----------------------------
with tabs[1]:
    st.subheader("Search an existing collection")

    cols = list_collections(persist_dir)
    if not cols:
        st.warning("No collections found. Build an index first.")
    else:
        search_collection = st.selectbox("Choose collection", cols, index=0)
        query = st.text_input("Query", value="python programming experience")

        # Optional file filter
        file_names = []
        if Path(data_path).exists():
            try:
                file_names = unique_file_names_from_cvs(data_path)
            except Exception:
                file_names = []

        filter_by_file = st.checkbox("Filter by file_name", value=False)
        selected_file: Optional[str] = None
        if filter_by_file and file_names:
            selected_file = st.selectbox("file_name", file_names)
        elif filter_by_file:
            st.info("Could not load file names from CVs.json; check the path.")

        if st.button("🔎 Search", type="primary"):
            indexer = ChromaIndexer(
                collection_name=search_collection,
                embedding_model=embedding_model,
                persist_directory=persist_dir,
                client_type="persistent",
            )

            where = {"file_name": selected_file} if selected_file else None
            results = indexer.search(query, n_results=top_k, where=where)
            render_results(results, limit=top_k)


# ----------------------------
# Tab 3: Evaluate
# ----------------------------
with tabs[2]:
    st.subheader("Evaluate retrieval using queries.txt")

    cols = list_collections(persist_dir)
    if not cols:
        st.warning("No collections found. Build an index first.")
    else:
        eval_collection = st.selectbox("Choose collection to evaluate", cols, index=0)

        if st.button("📊 Run Evaluation", type="primary"):
            if not Path(queries_path).exists():
                st.error(f"queries.txt not found at: {queries_path}")
            else:
                indexer = ChromaIndexer(
                    collection_name=eval_collection,
                    embedding_model=embedding_model,
                    persist_directory=persist_dir,
                    client_type="persistent",
                )

                evaluator = RetrievalEvaluator()
                queries = evaluator.load_queries(queries_path)

                with st.spinner("Evaluating..."):
                    results = evaluator.evaluate(indexer, queries, k=top_k, verbose=False)

                # Summary
                st.success("Evaluation complete ✅")
                st.metric(f"Recall@{top_k}", f"{results['recall@k']:.4f}")
                st.metric(f"MRR", f"{results['mrr']:.4f}")
                st.metric(f"Hit Rate@{top_k}", f"{results['hit_rate@k']:.4f}")
                st.metric(f"Precision@{top_k}", f"{results['precision@k']:.4f}")
                st.metric("Avg latency (ms)", f"{results['avg_latency_ms']:.2f}")

                # Report + download
                report_text = evaluator.generate_report(results)
                st.text_area("Full report", report_text, height=350)

                st.download_button(
                    label="⬇️ Download report.txt",
                    data=report_text.encode("utf-8"),
                    file_name=f"evaluation_report_{eval_collection}.txt",
                    mime="text/plain",
                )

                st.download_button(
                    label="⬇️ Download results.json",
                    data=json.dumps(results, indent=2).encode("utf-8"),
                    file_name=f"evaluation_results_{eval_collection}.json",
                    mime="application/json",
                )

                # Show a few failed queries
                failed = [r for r in results["detailed_results"] if r["recall"] == 0]
                st.markdown(f"### Failed queries (Recall=0): {len(failed)}")
                for item in failed[:5]:
                    st.write(f"- **Query:** {item['query']}")
                    st.write(f"  - expected: {item['expected_source']}")
                    st.write(f"  - retrieved: {item['retrieved_sources']}")
                    st.divider()
