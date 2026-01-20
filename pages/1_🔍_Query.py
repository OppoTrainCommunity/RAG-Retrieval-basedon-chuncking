"""
CV RAG System - Query Page
Search and query candidate CVs.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

st.set_page_config(
    page_title="Query - CV RAG System",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Query CVs")
st.markdown("Search and query candidate CVs using AI-powered retrieval.")

# Initialize components
from src.config import load_config

if "config" not in st.session_state:
    st.session_state.config = load_config("config.yaml")

config = st.session_state.config

# Check API key
if not config.openrouter_api_key:
    st.error("⚠️ OpenRouter API Key not configured. Please set OPENROUTER_API_KEY in .env file.")
    st.stop()

# Load components
@st.cache_resource
def get_rag_components():
    from src.embeddings.factory import EmbeddingsFactory
    from src.vectordb.chroma_store import ChromaStore
    from src.rag.chain import RAGChain
    from src.eval.pipeline import EvaluationPipeline
    
    config = load_config("config.yaml")
    embeddings = EmbeddingsFactory.from_config(config)
    
    chroma_store = ChromaStore(
        persist_dir=config.chroma.persist_dir,
        collection_name=config.chroma.collection,
        embeddings=embeddings,
    )
    
    if not chroma_store.load_index():
        return None, None, None
    
    rag_chain = RAGChain.from_config(config, chroma_store)
    eval_pipeline = EvaluationPipeline.from_config(config)
    
    return chroma_store, rag_chain, eval_pipeline

chroma_store, rag_chain, eval_pipeline = get_rag_components()

if chroma_store is None:
    st.warning("⚠️ Vector index not found. Please upload a PDF first or run the indexing notebook.")
    st.stop()

# Sidebar settings
st.sidebar.header("Settings")

top_k = st.sidebar.slider("Top-K Documents", 1, 20, config.retrieval.top_k)

section_options = [None, "summary", "experience", "education", "skills", "projects", "certifications"]
section_filter = st.sidebar.selectbox(
    "Section Filter",
    options=section_options,
    format_func=lambda x: "All Sections" if x is None else x.title()
)

# Candidate filter
st.sidebar.subheader("Candidate Filter")
candidate_ids = chroma_store.get_all_candidate_ids()

if not candidate_ids:
    st.sidebar.info("No candidates found in index.")
    candidate_filter = None
else:
    options = ["All Candidates"] + candidate_ids
    selected_candidate = st.sidebar.selectbox("Filter by Candidate", options)
    
    candidate_filter = None
    if selected_candidate != "All Candidates":
        candidate_filter = selected_candidate

run_eval = st.sidebar.checkbox("Run Evaluation", value=True)

st.sidebar.success(f"✅ Index: {chroma_store.document_count} chunks")

# Query interface
query = st.text_area(
    "Enter your question about candidates:",
    placeholder="e.g., Who has experience with Python and machine learning?",
    height=100,
)

if st.button("🚀 Search", type="primary"):
    if query:
        with st.spinner("Searching and generating answer..."):
            response = rag_chain.invoke(
                query, 
                top_k=top_k, 
                section_filter=section_filter,
                candidate_filter=candidate_filter
            )
        
        st.subheader("💬 Answer")
        st.markdown(response.answer)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Retrieval Time", f"{response.retrieval_time:.2f}s")
        col2.metric("Generation Time", f"{response.generation_time:.2f}s")
        col3.metric("Sources", response.num_sources)
        
        # Sources
        st.subheader("📚 Sources")
        for i, source in enumerate(response.sources, 1):
            section_title = source.section_name.title() if source.section_name else "Unknown"
            with st.expander(f"[{i}] {source.candidate_id} | {section_title} | {source.chunk_id}"):
                if source.metadata.get("name"):
                    st.write(f"**Name:** {source.metadata['name']}")
                if source.metadata.get("filename"):
                    st.write(f"**File:** {source.metadata['filename']}")
                st.write(source.content)
        
        # Evaluation
        if run_eval:
            with st.spinner("Evaluating response..."):
                eval_result = eval_pipeline.evaluate_response(response)
            
            st.subheader("⚖️ Evaluation")
            eval_cols = st.columns(4)
            eval_cols[0].metric("Relevance", f"{eval_result.relevance_score}/5")
            eval_cols[1].metric("Faithfulness", f"{eval_result.faithfulness_score}/5")
            eval_cols[2].metric("Correctness", f"{eval_result.correctness_score}/5")
            eval_cols[3].metric("Average", f"{eval_result.average_score}/5")
    else:
        st.warning("Please enter a query.")
