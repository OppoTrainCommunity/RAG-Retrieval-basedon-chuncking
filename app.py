"""
RAG Resume Analysis - Streamlit Application
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Import from source package
from src import config
from src import document_processor
from src import vector_store
from src import rag_engine

# ============================================================
# Streamlit App Config
# ============================================================
st.set_page_config(
    page_title="RAG Resume Analysis",
    page_icon="📄",
    layout="wide"
)

st.sidebar.title("📄 RAG Resume Analysis")
page = st.sidebar.radio("Navigation", ["⚙️ Configuration & Upload", "⛓️ LangChain Comparison"])

if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_valid" not in st.session_state:
    st.session_state.api_valid = False

# ============================================================
# Page 1: Configuration & Upload
# ============================================================
if page == "⚙️ Configuration & Upload":
    st.title("⚙️ Configuration & Upload")
    
    st.header("🔑 API Configuration")
    col1, col2 = st.columns([3, 1])
    with col1:
        api_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.api_key)
        st.session_state.api_key = api_key
    with col2:
        if st.button("Validate Key"):
            if api_key:
                with st.spinner("Validating..."):
                    # Quick validation using vector_store's embedding check logic roughly 
                    # or just try a dummy embedding
                    try:
                        vector_store.get_embedding("test", config.EMBEDDING_MODELS[0], api_key, use_cache=False)
                        st.session_state.api_valid = True
                        st.success("API Key is valid ✓")
                    except Exception as e:
                        st.session_state.api_valid = False
                        st.error(f"Invalid API Key: {e}")
            else:
                st.warning("Please enter an API key")
    
    if st.session_state.api_valid:
        st.success("✓ API Key validated")
    
    st.divider()
    
    st.header("🤖 Model & Strategy Selection")
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox("Embedding Model", config.EMBEDDING_MODELS)
    with col2:
        selected_strategy = st.selectbox("Chunking Strategy", config.CHUNKING_STRATEGIES)
    
    with st.expander("Advanced Chunking Parameters"):
        chunk_size = st.slider("Chunk Size", 100, 2000, config.DEFAULT_CHUNK_SIZE)
        overlap = st.slider("Overlap", 0, 200, config.DEFAULT_OVERLAP)
    
    st.divider()
    
    st.header("📁 Resume Upload")
    uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
    
    # Check existing
    existing_pdfs = list(config.DATA_DIR.glob("*.pdf"))
    if existing_pdfs:
        st.info(f"📂 Found {len(existing_pdfs)} PDF files in data/ folder")
        use_existing = st.checkbox("Use existing PDFs from data/ folder", value=True)
    else:
        use_existing = False
        st.info("💡 Add PDF files to the 'data/' folder or upload them above")
        
    if st.button("Process & Index Resumes", type="primary", disabled=not st.session_state.api_valid):
        pdf_paths = []
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = config.DATA_DIR / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                pdf_paths.append(file_path)
        
        if use_existing and existing_pdfs:
            pdf_paths.extend(existing_pdfs)
            
        pdf_paths = list(set(pdf_paths))
        
        if not pdf_paths:
            st.error("No PDF files to process!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            collection_name = vector_store.get_collection_name(selected_model, selected_strategy)
            collection = vector_store.initialize_collection(collection_name)
            
            processed_count = 0
            
            for i, pdf_path in enumerate(pdf_paths):
                status_text.text(f"Processing: {pdf_path.name}")
                text = document_processor.extract_text_from_pdf(str(pdf_path))
                
                if text:
                    chunks = document_processor.get_chunks(
                        text, selected_strategy, chunk_size=chunk_size, overlap=overlap
                    )
                    
                    chunk_texts = [c["text"] for c in chunks]
                    if chunk_texts:
                        embeddings = []
                        for ct in chunk_texts:
                            emb = vector_store.get_embedding(ct, selected_model, st.session_state.api_key)
                            embeddings.append(emb)
                        
                        vector_store.index_documents(
                            chunks, embeddings, pdf_path.name, collection, selected_model, selected_strategy
                        )
                        processed_count += 1
                
                progress_bar.progress((i + 1) / len(pdf_paths))
            
            st.success(f"Processed {processed_count} files into collection: {collection_name}")

# ============================================================
# Page 2: LangChain Comparison
# ============================================================
elif page == "⛓️ LangChain Comparison":
    st.title("⛓️ Multi-LLM RAG Comparison")
    
    if not st.session_state.api_valid:
        st.error("Please configure a valid API key in the 'Configuration' tab first.")
    else:
        st.header("1. Chain Configuration")
        col1, col2 = st.columns(2)
        with col1:
            emb_model = st.selectbox("Embedding Model", config.EMBEDDING_MODELS, key="lc_emb")
            strategy = st.selectbox("Chunking Strategy", config.CHUNKING_STRATEGIES, key="lc_strat")
            collection_name = vector_store.get_collection_name(emb_model, strategy)
            st.info(f"Using Collection: {collection_name}")
            
        with col2:
            model_a = st.selectbox("LLM Model A", config.LLM_MODELS, index=1)
            model_b = st.selectbox("LLM Model B", config.LLM_MODELS, index=2)
            
        st.divider()
        st.header("2. Run Comparison")
        
        query = st.text_input("Enter your question about the resumes:")
        ground_truth = st.text_input("Ground Truth Answer (Optional - for correctness eval):")
        
        if st.button("Generate & Compare", type="primary"):
            if not query:
                st.warning("Please enter a query.")
            else:
                with st.spinner("Initializing chains and generating responses..."):
                    # Initialize Manager
                    try:
                        rag = rag_engine.RAGChainManager(collection_name, emb_model, st.session_state.api_key)
                    except Exception as e:
                        st.error(f"Error initializing RAG: {e}")
                        st.stop()
                    
                    # Run Models parallel (simulated sequentially here)
                    col_a, col_b = st.columns(2)
                    
                    # Model A
                    res_a = rag.generate_response(query, model_a)
                    eval_a = rag.evaluate_answer(query, res_a["context"], res_a["answer"], ground_truth)
                    
                    # Model B
                    res_b = rag.generate_response(query, model_b)
                    eval_b = rag.evaluate_answer(query, res_b["context"], res_b["answer"], ground_truth)
                    
                    with col_a:
                        st.subheader(f"Model A: {model_a}")
                        if "error" in res_a:
                            st.error(res_a["error"])
                        else:
                            st.markdown(f"**Answer:**\n{res_a['answer']}")
                            st.caption(f"Latency: {res_a['latency_ms']:.2f} ms")
                            st.markdown("---")
                            st.metric("Relevance", eval_a.get("relevance", "N/A"))
                            st.metric("Faithfulness", eval_a.get("faithfulness", "N/A"))
                            if ground_truth:
                                st.metric("Correctness", f"{eval_a.get('correctness', 0)*100:.0f}%")
                            
                    with col_b:
                        st.subheader(f"Model B: {model_b}")
                        if "error" in res_b:
                            st.error(res_b["error"])
                        else:
                            st.markdown(f"**Answer:**\n{res_b['answer']}")
                            st.caption(f"Latency: {res_b['latency_ms']:.2f} ms")
                            st.markdown("---")
                            st.metric("Relevance", eval_b.get("relevance", "N/A"))
                            st.metric("Faithfulness", eval_b.get("faithfulness", "N/A"))
                            if ground_truth:
                                st.metric("Correctness", f"{eval_b.get('correctness', 0)*100:.0f}%")
                            
                    st.divider()
                    st.subheader("Retrieved Context")
                    with st.expander("Show Context"):
                        st.text(res_a["context"])
