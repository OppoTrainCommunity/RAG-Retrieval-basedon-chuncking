"""
CV RAG System - Streamlit Application
Main entry point for the web UI.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CV RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .score-box {
        background-color: #e8f4ea;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.8em;
        color: #666;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "config" not in st.session_state:
        from src.config import load_config
        st.session_state.config = load_config("config.yaml")
    
    if "chroma_store" not in st.session_state:
        st.session_state.chroma_store = None
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "eval_pipeline" not in st.session_state:
        st.session_state.eval_pipeline = None
    
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
    
    if "last_eval" not in st.session_state:
        st.session_state.last_eval = None


def load_components():
    """Load RAG components."""
    config = st.session_state.config
    
    if st.session_state.chroma_store is None:
        with st.spinner("Loading embeddings and vector store..."):
            from src.embeddings.factory import EmbeddingsFactory
            from src.vectordb.chroma_store import ChromaStore
            
            embeddings = EmbeddingsFactory.from_config(config)
            chroma_store = ChromaStore(
                persist_dir=config.chroma.persist_dir,
                collection_name=config.chroma.collection,
                embeddings=embeddings,
            )
            
            if not chroma_store.load_index():
                st.warning("⚠️ No index found. Please run the notebook first to build the index.")
                return False
            
            st.session_state.chroma_store = chroma_store
    
    if st.session_state.rag_chain is None:
        from src.rag.chain import RAGChain
        st.session_state.rag_chain = RAGChain.from_config(
            config, st.session_state.chroma_store
        )
    
    if st.session_state.eval_pipeline is None:
        from src.eval.pipeline import EvaluationPipeline
        st.session_state.eval_pipeline = EvaluationPipeline.from_config(config)
    
    return True


def render_sidebar():
    """Render sidebar with settings."""
    st.sidebar.title("⚙️ Settings")
    
    config = st.session_state.config
    
    # Model info
    st.sidebar.subheader("🤖 Model Configuration")
    st.sidebar.text_input(
        "LLM Model",
        value=config.llm.model,
        disabled=True,
        help="Configure in config.yaml"
    )
    st.sidebar.text_input(
        "Judge Model",
        value=config.judge.model,
        disabled=True,
        help="Configure in config.yaml"
    )
    
    st.sidebar.divider()
    
    # Retrieval settings
    st.sidebar.subheader("🔍 Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top-K Documents",
        min_value=1,
        max_value=20,
        value=config.retrieval.top_k,
        help="Number of documents to retrieve"
    )
    
    # Section filter
    section_options = [
        None, "summary", "experience", "education", 
        "skills", "projects", "certifications", "publications", "awards"
    ]
    section_filter = st.sidebar.selectbox(
        "Filter by Section",
        options=section_options,
        format_func=lambda x: "All Sections" if x is None else x.title(),
        help="Optional: filter retrieval to specific CV section"
    )
    
    st.sidebar.divider()
    
    # Run evaluation toggle
    run_eval = st.sidebar.checkbox(
        "Run Evaluation",
        value=True,
        help="Run LLM-as-judge evaluation after query"
    )
    
    st.sidebar.divider()
    
    # System status
    st.sidebar.subheader("📊 System Status")
    if st.session_state.chroma_store:
        doc_count = st.session_state.chroma_store.document_count
        st.sidebar.success(f"✅ Index loaded: {doc_count} chunks")
    else:
        st.sidebar.warning("⚠️ Index not loaded")
    
    # Navigation
    st.sidebar.divider()
    st.sidebar.subheader("📚 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        options=["🔍 Query", "📊 Evaluation History"],
        label_visibility="collapsed"
    )
    
    return {
        "top_k": top_k,
        "section_filter": section_filter,
        "run_eval": run_eval,
        "page": page,
    }


def render_query_page(settings):
    """Render main query page."""
    st.title("📄 CV RAG System")
    st.markdown("Ask questions about candidate CVs using AI-powered retrieval and generation.")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., Which candidates have experience with Kubernetes?",
        height=100,
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        run_button = st.button("🚀 Run Query", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.last_response = None
        st.session_state.last_eval = None
        st.rerun()
    
    if run_button and query:
        if not load_components():
            return
        
        # Run RAG query
        with st.spinner("🔍 Retrieving and generating answer..."):
            response = st.session_state.rag_chain.invoke(
                query,
                top_k=settings["top_k"],
                section_filter=settings["section_filter"],
            )
            st.session_state.last_response = response
        
        # Run evaluation if enabled
        if settings["run_eval"]:
            with st.spinner("⚖️ Evaluating response..."):
                eval_result = st.session_state.eval_pipeline.evaluate_response(response)
                st.session_state.last_eval = eval_result
    
    # Display results
    if st.session_state.last_response:
        render_response(st.session_state.last_response, st.session_state.last_eval)


def render_response(response, eval_result=None):
    """Render RAG response with sources and evaluation."""
    st.divider()
    
    # Answer section
    st.subheader("💬 Answer")
    st.markdown(response.answer)
    
    # Timing metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Retrieval Time", f"{response.retrieval_time:.3f}s")
    with col2:
        st.metric("Generation Time", f"{response.generation_time:.3f}s")
    with col3:
        st.metric("Total Time", f"{response.retrieval_time + response.generation_time:.3f}s")
    with col4:
        st.metric("Sources Retrieved", response.num_sources)
    
    st.divider()
    
    # Two columns: Sources and Evaluation
    col_sources, col_eval = st.columns([3, 2])
    
    with col_sources:
        st.subheader("📚 Retrieved Sources")
        
        for i, source in enumerate(response.sources, 1):
            with st.expander(
                f"Source {i}: {source.candidate_id} - {source.section_name.title()}",
                expanded=i <= 2  # Expand first 2 by default
            ):
                # Metadata
                meta_cols = st.columns(3)
                with meta_cols[0]:
                    st.markdown(f"**Candidate:** {source.candidate_id}")
                with meta_cols[1]:
                    st.markdown(f"**Section:** {source.section_name}")
                with meta_cols[2]:
                    if source.score:
                        st.markdown(f"**Score:** {source.score:.4f}")
                
                # Additional metadata
                if source.metadata.get("name"):
                    st.markdown(f"**Name:** {source.metadata['name']}")
                if source.metadata.get("role"):
                    st.markdown(f"**Role:** {source.metadata['role']}")
                
                # Content
                st.markdown("---")
                st.markdown(source.content)
    
    with col_eval:
        st.subheader("⚖️ Evaluation Scores")
        
        if eval_result:
            # Score metrics
            score_cols = st.columns(3)
            
            with score_cols[0]:
                st.metric(
                    "Relevance",
                    f"{eval_result.relevance_score}/5",
                    help="How relevant is the answer to the question?"
                )
            
            with score_cols[1]:
                st.metric(
                    "Faithfulness",
                    f"{eval_result.faithfulness_score}/5",
                    help="Is the answer faithful to the retrieved context?"
                )
            
            with score_cols[2]:
                st.metric(
                    "Correctness",
                    f"{eval_result.correctness_score}/5",
                    help="Overall correctness and quality"
                )
            
            # Average score with color
            avg_score = eval_result.average_score
            if avg_score >= 4:
                color = "green"
            elif avg_score >= 3:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(
                f"### Average Score: <span style='color: {color}'>{avg_score}/5</span>",
                unsafe_allow_html=True
            )
            
            st.divider()
            
            # Explanations
            st.markdown("**Explanations:**")
            
            with st.expander("Relevance", expanded=False):
                st.write(eval_result.relevance_explanation)
            
            with st.expander("Faithfulness", expanded=False):
                st.write(eval_result.faithfulness_explanation)
            
            with st.expander("Correctness", expanded=False):
                st.write(eval_result.correctness_explanation)
        else:
            st.info("Enable 'Run Evaluation' in the sidebar to see scores.")


def render_evaluation_history_page():
    """Render evaluation history page."""
    st.title("📊 Evaluation History")
    
    if not st.session_state.eval_pipeline or not st.session_state.eval_pipeline.results:
        st.info("No evaluation results yet. Run some queries first!")
        return
    
    # Get results DataFrame
    eval_df = st.session_state.eval_pipeline.to_dataframe()
    
    # Summary stats
    summary = st.session_state.eval_pipeline.get_summary_stats()
    
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Evaluations", summary["num_evaluations"])
    with col2:
        st.metric("Avg Relevance", f"{summary['avg_relevance']}/5")
    with col3:
        st.metric("Avg Faithfulness", f"{summary['avg_faithfulness']}/5")
    with col4:
        st.metric("Avg Correctness", f"{summary['avg_correctness']}/5")
    
    st.metric("Overall Average", f"{summary['avg_overall']}/5")
    
    st.divider()
    
    # Results table
    st.subheader("📋 Evaluation Results")
    
    display_cols = [
        "query", "relevance_score", "faithfulness_score",
        "correctness_score", "average_score", "timestamp"
    ]
    available_cols = [c for c in display_cols if c in eval_df.columns]
    
    st.dataframe(
        eval_df[available_cols],
        use_container_width=True,
        hide_index=True,
    )
    
    st.divider()
    
    # Chart
    st.subheader("📊 Score Distribution")
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ["relevance_score", "faithfulness_score", "correctness_score"]
    x = range(len(eval_df))
    width = 0.25
    
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    labels = ["Relevance", "Faithfulness", "Correctness"]
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], eval_df[metric], width, label=label, color=color)
    
    ax.set_xlabel("Query Index")
    ax.set_ylabel("Score (1-5)")
    ax.set_title("Evaluation Scores by Query")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}" for i in x])
    ax.legend()
    ax.set_ylim(0, 5.5)
    ax.axhline(y=4, color="green", linestyle="--", alpha=0.5)
    
    st.pyplot(fig)
    
    # Export buttons
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = eval_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            data=csv_data,
            file_name="eval_results.csv",
            mime="text/csv",
        )
    
    with col2:
        json_data = eval_df.to_json(orient="records", indent=2)
        st.download_button(
            "📥 Download JSON",
            data=json_data,
            file_name="eval_results.json",
            mime="application/json",
        )


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Check API key
    if not st.session_state.config.openrouter_api_key:
        st.error("""
        ⚠️ **OpenRouter API Key not found!**
        
        Please set your API key in one of these ways:
        1. Create a `.env` file with `OPENROUTER_API_KEY=your_key`
        2. Set the environment variable directly
        
        See `.env.example` for the template.
        """)
        st.stop()
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Route to appropriate page
    if settings["page"] == "🔍 Query":
        render_query_page(settings)
    elif settings["page"] == "📊 Evaluation History":
        render_evaluation_history_page()


if __name__ == "__main__":
    main()
