"""
CV RAG System - Index Management Page
Build, view, and manage the vector index.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Index - CV RAG System",
    page_icon="🗂️",
    layout="wide",
)

st.title("🗂️ Index Management")
st.markdown("Manage the vector index for CV retrieval.")

from src.config import load_config

config = load_config("config.yaml")

# Display current configuration
st.subheader("⚙️ Current Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**ChromaDB Settings:**")
    st.code(f"""
Persist Directory: {config.chroma.persist_dir}
Collection Name: {config.chroma.collection}
""")

with col2:
    st.markdown("**Embeddings Settings:**")
    st.code(f"""
Provider: {config.embeddings.provider}
Model: {config.embeddings.model}
""")

st.divider()

# Check index status
chroma_path = Path(config.chroma.persist_dir)

if chroma_path.exists():
    st.success("✅ Vector index directory exists")
    
    # Try to load and get stats
    try:
        from src.embeddings.factory import EmbeddingsFactory
        from src.vectordb.chroma_store import ChromaStore
        
        @st.cache_resource
        def get_chroma_store():
            embeddings = EmbeddingsFactory.from_config(config)
            store = ChromaStore(
                persist_dir=config.chroma.persist_dir,
                collection_name=config.chroma.collection,
                embeddings=embeddings,
            )
            store.load_index()
            return store
        
        store = get_chroma_store()
        
        st.metric("Documents in Index", store.document_count)
        
    except Exception as e:
        st.warning(f"Could not load index details: {e}")

else:
    st.warning("⚠️ Vector index not found")

st.divider()

# Data preview
st.subheader("📄 CV Data Preview")

cvs_path = Path(config.data.cvs_path)
if cvs_path.exists():
    df = pd.read_csv(cvs_path)
    st.write(f"**{len(df)} CVs loaded from:** {cvs_path}")
    
    display_cols = ["candidate_id", "name", "role", "location", "years_experience"]
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols], use_container_width=True, hide_index=True)
else:
    st.warning(f"CV data not found at: {cvs_path}")

# Chunks preview
st.divider()
st.subheader("✂️ Chunks Preview")

chunks_path = Path(config.data.chunks_path)
if chunks_path.exists():
    chunks_df = pd.read_parquet(chunks_path)
    st.write(f"**{len(chunks_df)} chunks from:** {chunks_path}")
    
    # Section distribution
    section_counts = chunks_df["section_name"].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Chunks by Section:**")
        st.dataframe(section_counts.reset_index().rename(columns={"index": "Section", "section_name": "Count"}))
    
    with col2:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        section_counts.plot(kind="bar", ax=ax, color="#3498db")
        ax.set_xlabel("Section")
        ax.set_ylabel("Count")
        ax.set_title("Chunks per Section")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
else:
    st.info("No chunks file found. Run the notebook to generate chunks.")

st.divider()

# Rebuild index button
st.subheader("🔧 Index Operations")

st.warning("""
⚠️ **Note:** To rebuild the index, please use the main notebook (`main.ipynb`).

This ensures proper handling of:
- Data loading
- Chunking configuration
- Embedding generation
- Index building

The notebook provides step-by-step control over the process.
""")

if st.button("🗑️ Reset Index", type="secondary"):
    if st.checkbox("I understand this will delete the current index"):
        try:
            import shutil
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
                st.success("Index deleted. Run the notebook to rebuild.")
                st.cache_resource.clear()
        except Exception as e:
            st.error(f"Error deleting index: {e}")
