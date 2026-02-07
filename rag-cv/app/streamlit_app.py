"""
RAG CV Assistant - Streamlit Application
========================================

A professional RAG system for analyzing CV/resume PDFs.

Features:
- Multi-PDF upload and indexing
- Two switchable embedding models
- Question answering with citations
- Admin controls for database management

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

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
        "embeddings": None
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
# PAGE: INGEST / INDEX
# =============================================================================

def render_ingest_page():
    """Render the Ingest/Index page."""
    st.header("📥 Ingest / Index CVs")
    
    st.markdown("""
    Upload PDF CVs to index them into the vector database. 
    Choose an embedding model and upload your files.
    """)
    
    # Embedding model selector
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "🔤 Embedding Model",
            options=list(settings.embedding_models.keys()),
            index=list(settings.embedding_models.keys()).index(st.session_state.selected_model),
            help="Choose the embedding model for vectorization"
        )
        st.session_state.selected_model = selected_model
    
    with col2:
        st.markdown("**Model Info:**")
        model_name = settings.embedding_models[selected_model]
        st.caption(f"`{model_name}`")
    
    st.divider()
    
    # Chunking settings
    with st.expander("⚙️ Chunking Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.slider(
                "Chunk Size",
                min_value=200,
                max_value=2000,
                value=settings.chunk_size,
                step=100,
                help="Target size for text chunks"
            )
        
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=settings.chunk_overlap,
                step=25,
                help="Overlap between consecutive chunks"
            )
        
        use_semantic = st.checkbox(
            "Use Semantic Chunking (if available)",
            value=False,
            help="Try semantic chunking based on content similarity"
        )
    
    st.divider()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📄 Upload PDF CVs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF CV files"
    )
    
    # Index button
    if st.button("🚀 Index PDFs", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            process_uploaded_files(
                uploaded_files,
                selected_model,
                chunk_size,
                chunk_overlap,
                use_semantic
            )
    
    # Show stats if available
    if st.session_state.ingestion_stats:
        st.divider()
        render_ingestion_stats()


def process_uploaded_files(
    uploaded_files,
    model_key: str,
    chunk_size: int,
    chunk_overlap: int,
    use_semantic: bool
):
    """Process and index uploaded PDF files."""
    with st.status("Processing PDFs...", expanded=True) as status:
        try:
            # Step 1: Extract text from PDFs
            st.write("📖 Extracting text from PDFs...")
            pdf_docs, extraction_stats = load_pdfs_from_uploaded_files(uploaded_files)
            
            if not pdf_docs:
                st.error("❌ Failed to extract text from any PDF")
                status.update(label="Failed", state="error")
                return
            
            st.write(f"   ✓ Extracted {extraction_stats['total_pages']} pages from {extraction_stats['successful']} files")
            
            # Step 2: Load embeddings
            st.write("🔤 Loading embedding model...")
            embeddings = get_embeddings(model_key)
            st.session_state.embeddings = embeddings
            st.write(f"   ✓ Loaded {model_key}")
            
            # Step 3: Chunk documents
            st.write("✂️ Chunking documents...")
            documents, chunking_stats = chunk_documents(
                pdf_docs,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_semantic=use_semantic,
                embeddings=embeddings if use_semantic else None
            )
            st.session_state.chunking_stats = chunking_stats
            st.write(f"   ✓ Created {chunking_stats['total_chunks']} chunks using {chunking_stats['chunking_method']} method")
            
            # Step 4: Index into vector store
            st.write("📊 Indexing into vector store...")
            vs_manager = get_vectorstore_manager()
            ingestion_stats = vs_manager.add_documents(model_key, documents, embeddings)
            st.session_state.ingestion_stats = {
                **extraction_stats,
                **ingestion_stats,
                "chunking": chunking_stats
            }
            st.write(f"   ✓ Added {ingestion_stats['new']} new chunks, skipped {ingestion_stats['skipped']} duplicates")
            
            # Save uploaded files
            for uploaded_file in uploaded_files:
                save_uploaded_file(uploaded_file, settings.upload_dir)
            
            status.update(label="✅ Indexing complete!", state="complete")
            st.success(f"Successfully indexed {len(pdf_docs)} CVs with {len(documents)} chunks!")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            st.error(f"❌ Ingestion failed: {str(e)}")
            status.update(label="Failed", state="error")


def render_ingestion_stats():
    """Render ingestion statistics."""
    stats = st.session_state.ingestion_stats
    
    st.subheader("📊 Ingestion Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Files Processed", f"{stats.get('successful', 0)}/{stats.get('total_files', 0)}")
    
    with col2:
        st.metric("Total Pages", stats.get('total_pages', 0))
    
    with col3:
        st.metric("Total Chunks", stats.get('chunking', {}).get('total_chunks', 0))
    
    with col4:
        st.metric("New / Skipped", f"{stats.get('new', 0)} / {stats.get('skipped', 0)}")
    
    if stats.get('failed_files'):
        st.warning(f"⚠️ Failed files: {', '.join(stats['failed_files'])}")


# =============================================================================
# PAGE: CHAT / SEARCH
# =============================================================================

def render_chat_page():
    """Render the Chat/Search page."""
    st.header("💬 Chat with CVs")
    
    st.markdown("""
    Ask questions about the indexed CVs. The system will retrieve relevant 
    information and provide answers with citations.
    """)
    
    # Settings sidebar
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_model = st.selectbox(
            "🔤 Embedding Model (must match indexed data)",
            options=list(settings.embedding_models.keys()),
            index=list(settings.embedding_models.keys()).index(st.session_state.selected_model),
            key="chat_model_select"
        )
    
    with col2:
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=15,
            value=settings.default_top_k,
            help="Number of chunks to retrieve"
        )
    
    with col3:
        search_type = st.selectbox(
            "Search Type",
            options=["similarity", "mmr"],
            index=0,
            help="MMR adds diversity to results"
        )
    
    # LLM model override
    with st.expander("🤖 LLM Settings", expanded=False):
        llm_model = st.text_input(
            "OpenRouter Model",
            value=settings.openrouter_model,
            help="Model identifier for OpenRouter"
        )
        
        if settings.is_openrouter_configured():
            st.success("✅ OpenRouter API key is configured")
        else:
            st.error("❌ OpenRouter API key not set. Set OPENROUTER_API_KEY environment variable.")
    
    st.divider()
    
    # Check if vector store has data
    vs_manager = get_vectorstore_manager()
    
    try:
        embeddings = get_embeddings(selected_model)
        vectorstore = vs_manager.create_or_load_collection(selected_model, embeddings)
        stats = vs_manager.get_collection_stats(selected_model)
        
        if not stats.get('has_data', False):
            st.warning("⚠️ No documents indexed for this embedding model. Please ingest PDFs first.")
            return
        
        st.info(f"📊 {stats['document_count']} chunks available for search")
        
    except Exception as e:
        st.error(f"❌ Failed to load vector store: {e}")
        return
    
    # Chat history display
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the CVs...", disabled=not settings.is_openrouter_configured()):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Create retriever
                    retriever = create_retriever(
                        vectorstore,
                        top_k=top_k,
                        search_type=search_type
                    )
                    
                    # Create RAG chain
                    rag_chain = create_rag_chain(retriever)
                    
                    # Get response
                    result = rag_chain.invoke(prompt)
                    
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    render_sources(sources)
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    st.session_state.last_sources = sources
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


def render_sources(sources: List[Dict[str, Any]]):
    """Render source citations."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} chunks)", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"""
            **Source {i}: {source.get('source', 'Unknown')}**  
            Page: {source.get('page', '?')} | Chunk: {source.get('chunk_id', '?')}  
            Candidate: {source.get('candidate_name', 'Unknown')}
            """)
            
            if source.get('preview'):
                st.text(source['preview'])
            
            if i < len(sources):
                st.divider()


# =============================================================================
# PAGE: ADMIN
# =============================================================================

def render_admin_page():
    """Render the Admin page."""
    st.header("⚙️ Admin Panel")
    
    st.markdown("""
    Manage vector store collections and view system information.
    """)
    
    # Collection overview
    st.subheader("📊 Collections Overview")
    
    vs_manager = get_vectorstore_manager()
    
    for model_key, model_name in settings.embedding_models.items():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            try:
                # Try to get stats (may need to load first)
                embeddings = get_embeddings(model_key)
                vs_manager.create_or_load_collection(model_key, embeddings)
                stats = vs_manager.get_collection_stats(model_key)
                
                with col1:
                    st.markdown(f"**{model_key}**")
                    st.caption(f"`{model_name}`")
                
                with col2:
                    doc_count = stats.get('document_count', 0)
                    status = "🟢" if doc_count > 0 else "⚪"
                    st.markdown(f"{status} {doc_count} chunks")
                
                with col3:
                    if st.button(f"🗑️ Reset", key=f"reset_{model_key}"):
                        if vs_manager.reset_collection(model_key):
                            st.success(f"Reset {model_key}")
                            st.rerun()
                        else:
                            st.error("Reset failed")
                
            except Exception as e:
                with col1:
                    st.markdown(f"**{model_key}**")
                with col2:
                    st.markdown("⚪ Not loaded")
                with col3:
                    st.button(f"🗑️ Reset", key=f"reset_{model_key}", disabled=True)
            
            st.divider()
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Reset All Collections**")
        st.caption("Delete all indexed data for all embedding models")
        
        if st.button("🗑️ Reset All", type="secondary"):
            st.session_state.confirm_reset_all = True
    
    if st.session_state.get('confirm_reset_all'):
        st.warning("⚠️ This will delete ALL indexed data. Are you sure?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Yes, Reset All", type="primary"):
                if vs_manager.reset_all_collections():
                    st.session_state.confirm_reset_all = False
                    st.success("All collections reset!")
                    st.rerun()
        
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_reset_all = False
                st.rerun()
    
    # System info
    st.subheader("ℹ️ System Information")
    
    with st.expander("Configuration", expanded=False):
        st.json({
            "chroma_persist_dir": str(settings.chroma_persist_dir),
            "upload_dir": str(settings.upload_dir),
            "default_top_k": settings.default_top_k,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "openrouter_model": settings.openrouter_model,
            "openrouter_configured": settings.is_openrouter_configured(),
            "embedding_model_a": settings.embedding_model_a,
            "embedding_model_b": settings.embedding_model_b
        })


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="RAG CV Assistant",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("📄 RAG CV Assistant")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        options=["📥 Ingest / Index", "💬 Chat / Search", "⚙️ Admin"],
        index=0
    )
    
    # API key status
    st.sidebar.markdown("---")
    if settings.is_openrouter_configured():
        st.sidebar.success("🔑 API Key: Configured")
    else:
        st.sidebar.error("🔑 API Key: Missing")
        st.sidebar.caption("Set OPENROUTER_API_KEY")
    
    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("RAG CV Assistant v1.0.0")
    
    # Render selected page
    if page == "📥 Ingest / Index":
        render_ingest_page()
    elif page == "💬 Chat / Search":
        render_chat_page()
    elif page == "⚙️ Admin":
        render_admin_page()


if __name__ == "__main__":
    main()
