"""
RAG CV Assistant - Streamlit Application
========================================

A professional RAG system for analyzing CV/resume PDFs.
Improved UI/UX with modern design, animations, and better layout.

Usage:
    streamlit run app/streamlit_app.py
"""

import sys, os, time
import requests
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from typing import List, Dict, Any, Optional

from rag.config import settings
from rag.logging_utils import get_logger, configure_root_logger
from rag.pdf_loader import load_pdfs_from_uploaded_files, save_uploaded_file
from rag.chunking import chunk_documents
from rag.embeddings import EmbeddingFactory, get_embeddings
from rag.vectorstore import VectorStoreManager, get_vector_store_manager
from rag.retriever import create_retriever, get_sources_summary
from rag.chain import create_rag_chain, get_openrouter_llm

# Configure logging
configure_root_logger()
logger = get_logger(__name__)


# =============================================================================
# CUSTOM CSS — Modern, polished look
# =============================================================================

CUSTOM_CSS = """
<style>
/* ── Import Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    color: #e2e8f0;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #e2e8f0 !important;
}
section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: #e2e8f0 !important;
}

/* ── Stat cards ─────────────────────────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
}
.stat-card .stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1.2;
}
.stat-card .stat-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}
.stat-card.accent-blue   { border-top: 4px solid #3b82f6; }
.stat-card.accent-green  { border-top: 4px solid #22c55e; }
.stat-card.accent-purple { border-top: 4px solid #8b5cf6; }
.stat-card.accent-amber  { border-top: 4px solid #f59e0b; }
.stat-card.accent-rose   { border-top: 4px solid #f43f5e; }

/* ── Hero banner ────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #db2777 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    color: white;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner h1 {
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    position: relative;
}
.hero-banner p {
    font-size: 1rem;
    opacity: 0.9;
    margin: 0;
    position: relative;
}

/* ── Info badges ────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.badge-success { background: #dcfce7; color: #166534; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger  { background: #fce7f3; color: #9d174d; }
.badge-info    { background: #dbeafe; color: #1e40af; }

/* ── Source card ─────────────────────────────────────────────── */
.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: background 0.2s;
}
.source-card:hover { background: #f1f5f9; }
.source-card .source-title {
    font-weight: 600;
    color: #1e293b;
    font-size: 0.9rem;
}
.source-card .source-meta {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.2rem;
}
.source-card .source-preview {
    font-size: 0.82rem;
    color: #475569;
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: #fff;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    font-family: 'Menlo', 'Consolas', monospace;
}

/* ── Collection card ────────────────────────────────────────── */
.collection-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s;
}
.collection-card:hover {
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
}

/* ── Upload zone styling ────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 16px;
}
[data-testid="stFileUploader"] section {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    background: #f8fafc !important;
    transition: border-color 0.3s, background 0.3s;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
}

/* ── Buttons ────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    border: none;
    border-radius: 12px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: transform 0.15s, box-shadow 0.15s;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
}

/* ── Danger button override ─────────────────────────────────── */
.danger-btn button {
    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
}
.danger-btn button:hover {
    box-shadow: 0 6px 20px rgba(220, 38, 38, 0.35) !important;
}

/* ── Chat messages ──────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}

/* ── Expander ───────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    border-radius: 12px;
}

/* ── Progress steps ─────────────────────────────────────────── */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    font-size: 0.9rem;
}
.step-done { color: #22c55e; }
.step-active { color: #3b82f6; font-weight: 600; }
.step-pending { color: #94a3b8; }

/* ── Welcome empty state ────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #94a3b8;
}
.empty-state .icon {
    font-size: 4rem;
    margin-bottom: 1rem;
}
.empty-state h3 {
    color: #64748b;
    font-weight: 600;
}
.empty-state p {
    color: #94a3b8;
    max-width: 400px;
    margin: 0 auto;
}

/* ── Horizontal divider subtle ──────────────────────────────── */
hr {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* ── Scrollbar ──────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ── Toast success animation ────────────────────────────────── */
@keyframes slideIn {
    from { transform: translateY(-10px); opacity: 0; }
    to   { transform: translateY(0);     opacity: 1; }
}
.element-container .stSuccess, .element-container .stInfo {
    animation: slideIn 0.3s ease;
}
</style>
"""


# =============================================================================
# HELPER — HTML stat card
# =============================================================================

def stat_card(value, label, accent="blue"):
    """Render a styled stat card."""
    st.markdown(
        f"""<div class="stat-card accent-{accent}">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "selected_model": list(settings.embedding_models.keys())[0],
        "chat_history": [],
        "last_sources": [],
        "ingestion_stats": None,
        "chunking_stats": None,
        "vectorstore_manager": None,
        "current_retriever": None,
        "embeddings": None,
        "active_page": "chat",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_vectorstore_manager() -> VectorStoreManager:
    """Get or create vector store manager."""
    if st.session_state.vectorstore_manager is None:
        st.session_state.vectorstore_manager = VectorStoreManager()
    return st.session_state.vectorstore_manager


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the polished sidebar."""
    with st.sidebar:
        # Logo / branding
        st.markdown(
            """
            <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
                <span style="font-size:2.5rem;">📄</span>
                <h2 style="color:#e2e8f0; margin:0.3rem 0 0 0; font-size:1.3rem; font-weight:700;">
                    RAG CV Assistant
                </h2>
                <span style="color:#94a3b8; font-size:0.78rem;">Intelligent Resume Analysis</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigate",
            options=["💬  Chat", "📥  Ingest", "⚙️  Admin"],
            index=["💬  Chat", "📥  Ingest", "⚙️  Admin"].index(
                {"chat": "💬  Chat", "ingest": "📥  Ingest", "admin": "⚙️  Admin"}.get(
                    st.session_state.active_page, "💬  Chat"
                )
            ),
            label_visibility="collapsed",
        )

        # Map selection back
        page_map = {"💬  Chat": "chat", "📥  Ingest": "ingest", "⚙️  Admin": "admin"}
        st.session_state.active_page = page_map.get(page, "chat")
        
        st.markdown("---")
        
        # Always use Backend API
        st.session_state.use_api = True

        st.markdown("---")

        # Quick status panel
        st.markdown(
            '<p style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;">System Status</p>',
            unsafe_allow_html=True,
        )

        # API key
        if settings.is_openrouter_configured():
            st.markdown(
                '<span class="badge badge-success">API Key Active</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge badge-danger">API Key Missing</span>',
                unsafe_allow_html=True,
            )

        # Collection quick stats
        try:
            vs_manager = get_vectorstore_manager()
            total_docs = 0
            for mk in settings.embedding_models:
                try:
                    emb = get_embeddings(mk)
                    vs_manager.create_or_load_collection(mk, emb)
                    s = vs_manager.get_collection_stats(mk)
                    total_docs += s.get("document_count", 0)
                except Exception:
                    pass
            st.markdown(
                f'<span class="badge badge-info">{total_docs} chunks indexed</span>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        # Footer
        st.markdown("---")
        st.markdown(
            '<p style="text-align:center; color:#64748b; font-size:0.7rem;">v1.0.0 &middot; Built with Streamlit</p>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE: CHAT / SEARCH
# =============================================================================

def render_chat_page():
    """Render the Chat page with a modern conversational UI."""

    # Hero
    st.markdown(
        """<div class="hero-banner">
            <h1>💬 Chat with your CVs</h1>
            <p>Ask anything about the indexed resumes — skills, experience, education, and more.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Settings row (collapsed by default) ────────────────────
    with st.expander("🔧 Retrieval Settings", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            selected_model = st.selectbox(
                "Embedding Model",
                options=list(settings.embedding_models.keys()),
                index=list(settings.embedding_models.keys()).index(
                    st.session_state.selected_model
                ),
                key="chat_model_select",
            )

        with col2:
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=15,
                value=settings.default_top_k,
                help="Number of chunks to retrieve",
            )

        with col3:
            search_type = st.selectbox(
                "Search",
                options=["similarity", "mmr"],
                index=0,
                help="MMR adds diversity to results",
            )

        # LLM config
        col_a, col_b = st.columns([3, 1])
        with col_a:
            llm_model = st.text_input(
                "LLM Model (OpenRouter)",
                value=settings.openrouter_model,
                help="Model identifier on OpenRouter",
            )
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            if settings.is_openrouter_configured():
                st.markdown(
                    '<span class="badge badge-success">Key OK</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge badge-danger">No Key</span>',
                    unsafe_allow_html=True,
                )

    # ── Vector store readiness ─────────────────────────────────
    vs_manager = get_vectorstore_manager()

    try:
        embeddings = get_embeddings(selected_model)
        vectorstore = vs_manager.create_or_load_collection(selected_model, embeddings)
        stats = vs_manager.get_collection_stats(selected_model)

        if not stats.get("has_data", False):
            st.markdown(
                """<div class="empty-state">
                    <div class="icon">📂</div>
                    <h3>No documents indexed yet</h3>
                    <p>Head over to the <b>Ingest</b> page to upload and index your CV PDFs first.</p>
                </div>""",
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f'<span class="badge badge-info" style="margin-bottom:1rem;">{stats["document_count"]} chunks ready for search</span>',
            unsafe_allow_html=True,
        )

    except Exception as e:
        st.error(f"Failed to load vector store: {e}")
        return

    # ── Chat history ───────────────────────────────────────────
    for message in st.session_state.chat_history:
        avatar = "🧑‍💼" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    # ── Empty chat welcome ─────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown(
            """<div class="empty-state" style="padding:2rem 0;">
                <div class="icon">🗨️</div>
                <h3>Start a conversation</h3>
                <p>Type a question below to begin — e.g. <em>"Who has experience with Python?"</em></p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Chat input ─────────────────────────────────────────────
    if prompt := st.chat_input(
        "Ask a question about the CVs…",
        disabled=not settings.is_openrouter_configured(),
    ):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching & generating answer…"):
                try:
                    if st.session_state.get("use_api", False):
                        # API Mode
                        api_url = os.environ.get("API_URL", "http://localhost:8000") + "/synthesize/"
                        payload = {"question": prompt}
                        response = requests.post(api_url, json=payload)
                        
                        if response.status_code == 200:
                            data = response.json()
                            answer = data.get("response", "No answer received.")
                            sources = data.get("sources", [])
                        else:
                            answer = f"⚠️ API Error: {response.status_code} - {response.text}"
                            sources = []
                    else:
                        # Local Mode (Direct Import)
                        retriever = create_retriever(
                            vectorstore, top_k=top_k, search_type=search_type
                        )
                        rag_chain = create_rag_chain(retriever)
                        result = rag_chain.invoke(prompt)

                        answer = result["answer"]
                        sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        render_sources(sources)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                    st.session_state.last_sources = sources

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    if "Connection refused" in str(e):
                        error_msg += "\n\n(Make sure the FastAPI server is running: `uvicorn api.main:app`)"
                        
                    st.error(error_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg, "sources": []}
                    )

    # ── Clear chat (bottom-right) ──────────────────────────────
    if st.session_state.chat_history:
        cols = st.columns([5, 1])
        with cols[1]:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()


def render_sources(sources: List[Dict[str, Any]]):
    """Render source citations with styled cards."""
    if not sources:
        return

    with st.expander(f"📚 Sources ({len(sources)} chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            preview_html = ""
            if source.get("preview"):
                preview_html = f'<div class="source-preview">{source["preview"]}</div>'

            st.markdown(
                f"""<div class="source-card">
                    <div class="source-title">Source {i}: {source.get('source', 'Unknown')}</div>
                    <div class="source-meta">
                        Page {source.get('page', '?')} &bull;
                        Chunk {source.get('chunk_id', '?')} &bull;
                        Candidate: {source.get('candidate_name', 'Unknown')}
                    </div>
                    {preview_html}
                </div>""",
                unsafe_allow_html=True,
            )


# =============================================================================
# PAGE: INGEST / INDEX
# =============================================================================

def render_ingest_page():
    """Render the Ingest/Index page."""

    # Hero
    st.markdown(
        """<div class="hero-banner" style="background: linear-gradient(135deg, #0f766e 0%, #0ea5e9 100%);">
            <h1>📥 Ingest &amp; Index CVs</h1>
            <p>Upload PDF resumes, chunk them, and index into the vector database for searching.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Model + settings ───────────────────────────────────────
    col_model, col_info = st.columns([3, 1])

    with col_model:
        selected_model = st.selectbox(
            "Embedding Model",
            options=list(settings.embedding_models.keys()),
            index=list(settings.embedding_models.keys()).index(
                st.session_state.selected_model
            ),
            help="Choose the embedding model for vectorization",
        )
        st.session_state.selected_model = selected_model

    with col_info:
        model_name = settings.embedding_models[selected_model]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<span class="badge badge-info">{model_name.split("/")[-1]}</span>',
            unsafe_allow_html=True,
        )

    # ── Chunking settings ──────────────────────────────────────
    with st.expander("⚙️ Chunking Settings", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=2000,
                value=settings.chunk_size,
                step=100,
                help="Target size for text chunks",
            )

        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=settings.chunk_overlap,
                step=25,
                help="Overlap between consecutive chunks",
            )

        use_semantic = True

    st.markdown("---")

    # ── File upload ────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Drop your PDF CVs here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF CV files",
        label_visibility="visible",
    )

    # Show file list
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        file_cols = st.columns(min(len(uploaded_files), 4))
        for idx, f in enumerate(uploaded_files):
            with file_cols[idx % len(file_cols)]:
                size_kb = len(f.getvalue()) / 1024
                st.markdown(
                    f"""<div class="stat-card" style="padding:0.75rem; text-align:left;">
                        <div style="font-weight:600; font-size:0.85rem;">📄 {f.name}</div>
                        <div style="color:#64748b; font-size:0.75rem;">{size_kb:.0f} KB</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # Index button
    if st.button(
        "🚀 Start Indexing",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    ):
        if uploaded_files:
            process_uploaded_files(
                uploaded_files, selected_model, chunk_size, chunk_overlap, use_semantic
            )

    # ── Show stats if available ────────────────────────────────
    if st.session_state.ingestion_stats:
        st.markdown("---")
        render_ingestion_stats()


def process_uploaded_files(
    uploaded_files,
    model_key: str,
    chunk_size: int,
    chunk_overlap: int,
    use_semantic: bool,
):
    """Process and index uploaded PDF files with a progress bar."""
    progress = st.progress(0, text="Initializing…")

    try:
        # Step 1 — Extract
        progress.progress(10, text="📖 Extracting text from PDFs…")
        pdf_docs, extraction_stats = load_pdfs_from_uploaded_files(uploaded_files)

        if not pdf_docs:
            st.error("Failed to extract text from any PDF.")
            progress.empty()
            return

        # Step 2 — Embeddings
        progress.progress(30, text="🔤 Loading embedding model…")
        embeddings = get_embeddings(model_key)
        st.session_state.embeddings = embeddings

        # Step 3 — Chunk
        progress.progress(50, text="✂️ Chunking documents…")
        documents, chunking_stats = chunk_documents(
            pdf_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic=use_semantic,
            embeddings=embeddings if use_semantic else None,
        )
        st.session_state.chunking_stats = chunking_stats

        # Step 4 — Index
        progress.progress(75, text="📊 Indexing into vector store…")
        vs_manager = get_vectorstore_manager()
        ingestion_stats = vs_manager.add_documents(model_key, documents, embeddings)
        st.session_state.ingestion_stats = {
            **extraction_stats,
            **ingestion_stats,
            "chunking": chunking_stats,
        }

        # Save uploaded files
        for uploaded_file in uploaded_files:
            save_uploaded_file(uploaded_file, settings.upload_dir)

        progress.progress(100, text="✅ Indexing complete!")
        time.sleep(0.5)
        progress.empty()

        st.success(
            f"Successfully indexed **{len(pdf_docs)}** CV(s) with **{len(documents)}** chunks!"
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        progress.empty()
        st.error(f"Ingestion failed: {str(e)}")


def render_ingestion_stats():
    """Render ingestion statistics with styled cards."""
    stats = st.session_state.ingestion_stats

    st.markdown("#### 📊 Ingestion Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        stat_card(
            f"{stats.get('successful', 0)}/{stats.get('total_files', 0)}",
            "Files Processed",
            "blue",
        )
    with col2:
        stat_card(str(stats.get("total_pages", 0)), "Pages Extracted", "purple")
    with col3:
        stat_card(
            str(stats.get("chunking", {}).get("total_chunks", 0)),
            "Chunks Created",
            "green",
        )
    with col4:
        stat_card(
            f"{stats.get('new', 0)} / {stats.get('skipped', 0)}",
            "New / Skipped",
            "amber",
        )

    if stats.get("failed_files"):
        st.warning(f"Failed files: {', '.join(stats['failed_files'])}")


# =============================================================================
# PAGE: ADMIN
# =============================================================================

def render_admin_page():
    """Render the Admin page."""

    # Hero
    st.markdown(
        """<div class="hero-banner" style="background: linear-gradient(135deg, #475569 0%, #1e293b 100%);">
            <h1>⚙️ Admin Panel</h1>
            <p>Manage collections, view configuration, and perform maintenance tasks.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Collections overview ───────────────────────────────────
    st.markdown("#### 📦 Collections")

    vs_manager = get_vectorstore_manager()

    for model_key, model_name in settings.embedding_models.items():
        try:
            embeddings = get_embeddings(model_key)
            vs_manager.create_or_load_collection(model_key, embeddings)
            stats = vs_manager.get_collection_stats(model_key)
            doc_count = stats.get("document_count", 0)
            status_badge = (
                '<span class="badge badge-success">Active</span>'
                if doc_count > 0
                else '<span class="badge badge-warning">Empty</span>'
            )
        except Exception:
            doc_count = 0
            status_badge = '<span class="badge badge-danger">Error</span>'

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.markdown(
                f"""<div class="collection-card">
                    <div style="font-weight:600; font-size:0.95rem;">{model_key}</div>
                    <div style="color:#64748b; font-size:0.8rem; margin-top:0.15rem;">{model_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f'{status_badge} &nbsp; <span style="font-size:0.85rem; color:#475569;">{doc_count} chunks</span>',
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Reset", key=f"reset_{model_key}"):
                if vs_manager.reset_collection(model_key):
                    st.success(f"Reset {model_key}")
                    st.rerun()
                else:
                    st.error("Reset failed")

    # ── Danger Zone ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚠️ Danger Zone")

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(
                """<div class="collection-card" style="border-color:#fca5a5;">
                    <div style="font-weight:600; color:#dc2626;">Reset All Collections</div>
                    <div style="color:#64748b; font-size:0.82rem;">Permanently delete all indexed data for every embedding model.</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
            if st.button("🗑️ Reset All", key="reset_all_btn"):
                st.session_state.confirm_reset_all = True
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("confirm_reset_all"):
        st.warning("This will delete ALL indexed data. Are you sure?")
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            if st.button("Yes, Reset All", type="primary"):
                if vs_manager.reset_all_collections():
                    st.session_state.confirm_reset_all = False
                    st.success("All collections reset!")
                    st.rerun()
        with c2:
            if st.button("Cancel"):
                st.session_state.confirm_reset_all = False
                st.rerun()

    # ── System Configuration ───────────────────────────────────
    st.markdown("---")
    st.markdown("#### ℹ️ System Configuration")

    with st.expander("View Configuration", expanded=False):
        cfg = {
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "upload_dir": str(settings.upload_dir),
            "default_top_k": settings.default_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "openrouter_model": settings.openrouter_model,
            "openrouter_configured": settings.is_openrouter_configured(),
            "embedding_model_a": settings.embedding_model_a,
            "embedding_model_b": settings.embedding_model_b,
        }
        st.json(cfg)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application entry-point."""
    st.set_page_config(
        page_title="RAG CV Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # Sidebar
    render_sidebar()

    # Render active page
    page = st.session_state.active_page
    if page == "chat":
        render_chat_page()
    elif page == "ingest":
        render_ingest_page()
    elif page == "admin":
        render_admin_page()


if __name__ == "__main__":
    main()
