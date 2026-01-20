"""
CV RAG System - Retrieval Evaluation Page
Evaluate retrieval quality without LLM generation.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Retrieval Eval - CV RAG System",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Retrieval Evaluation")
st.markdown("""
Evaluate retrieval quality using ground-truth labels.  
**No LLM required** — this tests only the vector search/retrieval stage.
""")

# Initialize components
from src.config import load_config

if "config" not in st.session_state:
    st.session_state.config = load_config("config.yaml")

config = st.session_state.config

# Load ChromaStore
@st.cache_resource
def get_chroma_store():
    from src.embeddings.factory import EmbeddingsFactory
    from src.vectordb.chroma_store import ChromaStore
    
    config = load_config("config.yaml")
    embeddings = EmbeddingsFactory.from_config(config)
    
    chroma_store = ChromaStore(
        persist_dir=config.chroma.persist_dir,
        collection_name=config.chroma.collection,
        embeddings=embeddings,
    )
    
    if not chroma_store.load_index():
        return None
    
    return chroma_store

chroma_store = get_chroma_store()

if chroma_store is None:
    st.warning("⚠️ Vector index not found. Please upload PDFs or run the indexing notebook first.")
    st.stop()

st.sidebar.success(f"✅ Index: {chroma_store.document_count} chunks")

# Evaluation dataset
st.header("1️⃣ Evaluation Dataset")

eval_data_path = st.text_input(
    "Evaluation dataset path",
    value="./data/eval/retrieval_eval.jsonl",
    help="Path to JSONL or CSV file with evaluation queries"
)

# Check if file exists
eval_path = Path(eval_data_path)
if not eval_path.exists():
    st.warning(f"⚠️ Evaluation file not found: {eval_data_path}")
    st.info("Create a JSONL file with format: `{\"query\": \"...\", \"expected_candidate_ids\": [...], \"expected_section_names\": [...]}`")
    st.stop()

# Load dataset
from src.eval.retrieval_eval import load_retrieval_eval_data, RetrievalEvaluator

try:
    dataset = load_retrieval_eval_data(eval_data_path)
    st.success(f"📂 Loaded **{len(dataset)}** evaluation queries")
    
    # Preview dataset
    with st.expander("Preview Evaluation Dataset"):
        preview_data = []
        for q in dataset:
            preview_data.append({
                "Query": q.query[:60] + "..." if len(q.query) > 60 else q.query,
                "Expected Candidates": ", ".join(q.expected_candidate_ids or []),
                "Expected Sections": ", ".join(q.expected_section_names or []),
                "Notes": q.notes or "",
            })
        st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        
except Exception as e:
    st.error(f"❌ Error loading dataset: {str(e)}")
    st.stop()

# Evaluation settings
st.header("2️⃣ Evaluation Settings")

col1, col2 = st.columns(2)

with col1:
    top_k = st.slider("Top-K Documents", 1, 20, config.retrieval.top_k)

with col2:
    st.metric("Queries to Evaluate", len(dataset))

# Run evaluation
st.header("3️⃣ Run Evaluation")

if st.button("🚀 Run Retrieval Evaluation", type="primary"):
    with st.spinner(f"Evaluating {len(dataset)} queries with top_k={top_k}..."):
        evaluator = RetrievalEvaluator(chroma_store, top_k=top_k)
        results_df = evaluator.evaluate(dataset)
        summary = evaluator.get_summary()
    
    # Store results in session state
    st.session_state.retrieval_results = results_df
    st.session_state.retrieval_summary = summary
    st.session_state.retrieval_evaluator = evaluator

# Display results if available
if "retrieval_results" in st.session_state:
    results_df = st.session_state.retrieval_results
    summary = st.session_state.retrieval_summary
    evaluator = st.session_state.retrieval_evaluator
    
    st.divider()
    st.header("📊 Results")
    
    # Aggregate metrics
    st.subheader("Aggregate Metrics")
    
    metric_cols = st.columns(5)
    metric_cols[0].metric("Hit@k", f"{summary['avg_hit_at_k']:.3f}")
    metric_cols[1].metric("Precision@k", f"{summary['avg_precision_at_k']:.3f}")
    metric_cols[2].metric("Recall@k", f"{summary['avg_recall_at_k']:.3f}")
    metric_cols[3].metric("MRR@k", f"{summary['avg_mrr_at_k']:.3f}")
    metric_cols[4].metric("nDCG@k", f"{summary['avg_ndcg_at_k']:.3f}")
    
    # Per-query results table
    st.subheader("Per-Query Results")
    
    display_cols = ["query", "hit_at_k", "precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k", "num_relevant_retrieved"]
    st.dataframe(
        results_df[display_cols].style.format({
            "hit_at_k": "{:.2f}",
            "precision_at_k": "{:.2f}",
            "recall_at_k": "{:.2f}",
            "mrr_at_k": "{:.2f}",
            "ndcg_at_k": "{:.2f}",
        }),
        use_container_width=True,
    )
    
    # Visualization
    st.subheader("Visualization")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ["avg_hit_at_k", "avg_precision_at_k", "avg_recall_at_k", "avg_mrr_at_k", "avg_ndcg_at_k"]
    labels = ["Hit@k", "Precision@k", "Recall@k", "MRR@k", "nDCG@k"]
    values = [summary[m] for m in metrics]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Average Score (0-1)")
    ax.set_title(f"Aggregate Retrieval Metrics (top_k={top_k})")
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", 
                ha="center", va="bottom", fontsize=10)
    
    st.pyplot(fig)
    
    # Download buttons
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="retrieval_eval_results.csv",
            mime="text/csv",
        )
    
    with col2:
        import json
        json_data = json.dumps({
            "aggregate_metrics": summary,
            "per_query_results": results_df.to_dict(orient="records"),
        }, indent=2)
        st.download_button(
            "📥 Download JSON",
            data=json_data,
            file_name="retrieval_eval_results.json",
            mime="application/json",
        )
    
    # Save to outputs
    if st.button("💾 Save to ./outputs/"):
        evaluator.save_results(
            output_dir="./outputs",
            csv_filename="retrieval_eval_results.csv",
            json_filename="retrieval_eval_results.json",
        )
        st.success("✅ Results saved to ./outputs/")

# Help section
st.divider()
st.header("ℹ️ Metrics Explained")

with st.expander("Understanding Retrieval Metrics"):
    st.markdown("""
    **Hit@k**: Binary metric — 1 if any relevant document appears in top-k results, else 0.
    
    **Precision@k**: (# relevant retrieved) / k — What fraction of retrieved docs are relevant?
    
    **Recall@k**: (# relevant retrieved) / (# expected relevant) — What fraction of all relevant docs did we find?
    
    **MRR@k (Mean Reciprocal Rank)**: 1/rank of first relevant result — How high does the first relevant doc rank?
    
    **nDCG@k (Normalized DCG)**: Accounts for position of relevant docs — Higher score if relevant docs rank higher.
    
    ---
    
    **Relevance Matching Priority**:
    1. `chunk_id` (most specific)
    2. `candidate_id` (matches any chunk from that candidate)
    3. `section_name` (matches any chunk from that section type)
    """)
