"""
CV RAG System - PDF Upload Page
Upload and ingest multiple CV PDFs into the vector database.
"""

import os
import sys
import time
from datetime import datetime
import hashlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Upload PDFs - CV RAG System",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Upload CV PDFs")
st.markdown("""
Upload one or more CV PDFs to ingest them into the vector database.
All CVs are appended to the existing collection — nothing is deleted.
""")

# Initialize components
from src.config import load_config

if "config" not in st.session_state:
    st.session_state.config = load_config("config.yaml")

config = st.session_state.config

# Check API key
if not config.openrouter_api_key:
    st.error("⚠️ OpenRouter API Key not configured. Please set OPENROUTER_API_KEY in .env file.")
    st.stop()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_candidate_id(filename: str) -> str:
    """
    Generate a deterministic candidate ID from filename with short hash.
    """
    safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ").replace(" ", "_")
    if "." in safe_filename:
        safe_filename = safe_filename.rsplit(".", 1)[0]
    
    # Create deterministic hash from filename + current timestamp
    hash_input = f"{filename}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    return f"{safe_filename}_{short_hash}"


def ingest_single_pdf(
    pdf_bytes: bytes,
    candidate_id: str,
    filename: str,
    chroma_store,
    chunker,
    metadata: dict = None,
) -> dict:
    """
    Ingest a single PDF into the vector database.
    
    Returns:
        dict with keys: success, candidate_id, num_chunks, error (if failed)
    """
    from src.data.loaders import df_from_pdf
    from src.data.chunking import chunk_dataframe
    from langchain_core.documents import Document
    
    result = {
        "success": False,
        "candidate_id": candidate_id,
        "filename": filename,
        "num_chunks": 0,
        "error": None,
    }
    
    try:
        # Step 1: Extract text and create DataFrame
        extra_meta = metadata or {}
        extra_meta["filename"] = filename
        
        cv_df = df_from_pdf(
            pdf_bytes=pdf_bytes,
            candidate_id=candidate_id,
            **extra_meta
        )
        
        # Step 2: Chunk the CV
        chunks_df = chunk_dataframe(cv_df, chunker=chunker)
        result["num_chunks"] = len(chunks_df)
        
        # Step 3: Build documents
        metadata_cols = [
            "candidate_id", "section_name", "chunk_index",
            "name", "email", "role", "location", "years_experience", "filename"
        ]
        metadata_cols = [c for c in metadata_cols if c in chunks_df.columns]
        
        documents = []
        ids = []
        
        for _, row in chunks_df.iterrows():
            doc_metadata = {}
            for col in metadata_cols:
                value = row.get(col)
                if pd.notna(value):
                    if isinstance(value, (int, float)):
                        doc_metadata[col] = value
                    else:
                        doc_metadata[col] = str(value)
            
            doc = Document(
                page_content=str(row["chunk_text"]),
                metadata=doc_metadata,
            )
            documents.append(doc)
            ids.append(str(row["chunk_id"]))
        
        # Step 4: Upsert to vectorstore
        chroma_store.vectorstore.add_documents(documents, ids=ids)
        
        result["success"] = True
        
    except ValueError as e:
        result["error"] = f"Text extraction failed: {str(e)}"
    except Exception as e:
        result["error"] = f"Ingestion error: {str(e)}"
    
    return result


# ============================================================================
# MAIN UI
# ============================================================================

# PDF Upload Section
st.header("1️⃣ Upload PDF Files")

uploaded_files = st.file_uploader(
    "Choose CV PDF files (you can select multiple)",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload text-based PDFs. Scanned/image-based PDFs are not supported without OCR."
)

if uploaded_files:
    st.success(f"📁 **{len(uploaded_files)}** PDF(s) selected for upload")
    
    # Show file list
    with st.expander("View Selected Files", expanded=True):
        file_info = []
        for f in uploaded_files:
            file_info.append({
                "Filename": f.name,
                "Size (KB)": f"{f.size / 1024:.1f}",
            })
        st.dataframe(pd.DataFrame(file_info), use_container_width=True)

# Candidate ID Options
st.header("2️⃣ Candidate ID Options")

id_option = st.radio(
    "How should Candidate IDs be generated?",
    options=[
        "Auto-generate from filename (recommended)",
        "Use a base ID with index suffix (e.g., base_1, base_2, ...)",
    ],
    index=0,
)

base_candidate_id = ""
if id_option.startswith("Use a base ID"):
    base_candidate_id = st.text_input(
        "Base Candidate ID",
        value="candidate",
        help="Each PDF will get this base ID with an index appended (e.g., candidate_1, candidate_2)"
    )

# Ingestion Section
st.header("3️⃣ Ingest into Vector Database")

col1, col2 = st.columns([2, 1])

with col1:
    ingest_button = st.button(
        "🚀 Ingest All PDFs",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

with col2:
    st.metric("PDFs to Process", len(uploaded_files) if uploaded_files else 0)

if ingest_button and uploaded_files:
    # Initialize components once
    from src.embeddings.factory import EmbeddingsFactory
    from src.vectordb.chroma_store import ChromaStore
    from src.data.chunking import CVChunker
    
    embeddings = EmbeddingsFactory.from_config(config)
    chroma_store = ChromaStore(
        persist_dir=config.chroma.persist_dir,
        collection_name=config.chroma.collection,
        embeddings=embeddings,
    )
    chroma_store.load_index()
    chunker = CVChunker()
    
    # Track results
    results = []
    total_chunks = 0
    succeeded = 0
    failed = 0
    
    # Progress UI
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    start_time = time.time()
    
    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx) / len(uploaded_files)
        progress_bar.progress(progress)
        status_container.info(f"⏳ Processing **{uploaded_file.name}** ({idx + 1}/{len(uploaded_files)})...")
        
        # Generate candidate ID
        if base_candidate_id:
            candidate_id = f"{base_candidate_id}_{idx + 1}"
        else:
            candidate_id = generate_candidate_id(uploaded_file.name)
        
        # Read PDF bytes
        pdf_bytes = uploaded_file.read()
        
        # Ingest
        result = ingest_single_pdf(
            pdf_bytes=pdf_bytes,
            candidate_id=candidate_id,
            filename=uploaded_file.name,
            chroma_store=chroma_store,
            chunker=chunker,
        )
        
        results.append(result)
        
        if result["success"]:
            succeeded += 1
            total_chunks += result["num_chunks"]
        else:
            failed += 1
    
    # Complete
    progress_bar.progress(1.0)
    status_container.empty()
    
    elapsed_time = time.time() - start_time
    
    # Clear cache to reload components on Query page
    st.cache_resource.clear()
    
    # Summary
    st.divider()
    st.header("📊 Ingestion Summary")
    
    summary_cols = st.columns(4)
    summary_cols[0].metric("Total PDFs", len(uploaded_files))
    summary_cols[1].metric("Succeeded", succeeded, delta=None if failed == 0 else f"-{failed} failed")
    summary_cols[2].metric("Total Chunks", total_chunks)
    summary_cols[3].metric("Time Taken", f"{elapsed_time:.1f}s")
    
    if succeeded > 0:
        st.success(f"✅ Successfully ingested **{succeeded}** CV(s) with **{total_chunks}** total chunks!")
    
    if failed > 0:
        st.warning(f"⚠️ **{failed}** PDF(s) failed to ingest (likely scanned/image-based).")
    
    # Detailed results table
    st.subheader("Detailed Results")
    
    results_df = pd.DataFrame(results)
    results_df["status"] = results_df["success"].apply(lambda x: "✅ Success" if x else "❌ Failed")
    
    # Reorder columns for display
    display_cols = ["filename", "candidate_id", "status", "num_chunks", "error"]
    display_df = results_df[[c for c in display_cols if c in results_df.columns]]
    
    st.dataframe(display_df, use_container_width=True)
    
    # Show failed files details
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        with st.expander("🔍 Failed Files Details"):
            for r in failed_results:
                st.error(f"**{r['filename']}**: {r['error']}")
    
    st.info(f"💡 **Next step:** Go to the **Query** page to search across all {chroma_store.document_count} chunks!")

# Help Section
st.divider()
st.header("ℹ️ Help")

with st.expander("Supported PDF Types"):
    st.markdown("""
    **✅ Supported:**
    - Text-based PDFs (created from Word, Google Docs, etc.)
    - PDFs with selectable text
    
    **❌ Not Supported (without OCR):**
    - Scanned documents
    - Image-based PDFs
    - PDFs where text is embedded as images
    
    **Tip:** Try selecting text in your PDF viewer. If you can select and copy text, it should work!
    """)

with st.expander("Candidate ID Guidelines"):
    st.markdown("""
    The **Candidate ID** is used to:
    - Uniquely identify each CV in the database
    - Filter search results by candidate
    - Include in citations when answering queries
    
    **Auto-generation (recommended):**
    - Creates IDs like `john_doe_resume_a1b2c3`
    - Deterministic hash ensures uniqueness
    
    **Base ID with index:**
    - Creates IDs like `batch_2024_1`, `batch_2024_2`, etc.
    - Useful for organized batch uploads
    """)

with st.expander("Batch Upload Tips"):
    st.markdown("""
    **For best results:**
    - Select all PDFs at once using Ctrl+Click or Shift+Click
    - Use text-based PDFs (not scanned)
    - Each PDF should contain one CV
    
    **Performance:**
    - Embedding is the slowest step
    - ~2-5 seconds per PDF depending on size
    - All chunks are persisted at the end
    """)
