# app.py
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
import streamlit as st
from pathlib import Path
import uuid
import os

from resume_rag.config import AppConfig, ensure_dirs, get_openrouter_key, get_openrouter_base_url
from resume_rag.pipeline_chain import index_resume_pdf, list_resumes, reset_index
from resume_rag.vector_store_chain import reset_collection
from resume_rag.llm_chain import compare_two_models

# MUST be first streamlit command
st.set_page_config(page_title="Resume → Career Path Advisor (RAG)", layout="wide")

cfg = AppConfig()
ensure_dirs(cfg)

st.title("📄 Resume → Career Path Advisor (RAG)")
st.caption("Upload PDFs, index into Chroma, then ask questions across ONE resume or ALL resumes via OpenRouter.")
import streamlit as st
import torch
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource
def load_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

# ---------------- Sidebar Settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    key_ok = bool(get_openrouter_key())
    st.success("OPENROUTER_API_KEY loaded ✅" if key_ok else "OPENROUTER_API_KEY not loaded ❌")

    st.write("BASE_URL:", get_openrouter_base_url())

    collection_name = st.text_input("Collection name", value=cfg.collection_name)
    persist_dir = st.text_input("Persist directory", value=cfg.persist_dir)

    st.divider()
    st.subheader("Models")
    model1 = st.text_input("LLM1 (OpenRouter model)", value="openai/gpt-4o-mini")
    model2 = st.text_input("LLM2 (OpenRouter model)", value="meta-llama/llama-3.1-8b-instruct")

    st.divider()
    if st.button("🧹 Reset Vector DB (Collection)", use_container_width=True):
        reset_collection(collection_name, persist_dir)
        reset_index(cfg)
        st.success("Collection + index reset done. Refresh page.")

# Tabs
tab1, tab2 = st.tabs(["📤 Upload & Index", "❓ Ask Questions (RAG)"])

# ---------------- Upload & Index ----------------
with tab1:
    st.subheader("📤 Upload one or multiple PDF resumes")
    chunk_mode = st.selectbox("Chunking mode", ["semantic", "window", "fixed"], index=0)
    colA, colB = st.columns(2)
    with colA:
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=5000, value=900, step=50)
    with colB:
        overlap = st.number_input("Overlap (for window)", min_value=0, max_value=2000, value=150, step=25)

    files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    if st.button("📌 Analyze & Index", disabled=(not files)):
        if not key_ok:
            st.error("Missing OPENROUTER_API_KEY in .env")
            st.stop()

        for f in files:
            # save file
            uid = str(uuid.uuid4())[:8]
            save_path = Path(cfg.uploads_dir) / f"{uid}_{f.name}"
            save_path.write_bytes(f.getbuffer())

            try:
                info = index_resume_pdf(
                    cfg=cfg,
                    pdf_path=str(save_path),
                    original_filename=f.name,
                    mode=chunk_mode,
                    chunk_size=int(chunk_size),
                    overlap=int(overlap),
                )
                st.success(f"Indexed ✅ {info['filename']} | resume_id={info['resume_id']} | chunks={info['num_chunks']}")
                with st.expander("Extracted text preview"):
                    st.write(info["text_preview"])
            except Exception as e:
                st.error(f"Failed indexing {f.name}: {e}")

    st.divider()
    st.subheader("📚 Indexed resumes")
    items = list_resumes(cfg)
    if not items:
        st.info("No indexed resumes yet.")
    else:
        for it in items:
            st.write(f"• **{it['filename']}** — resume_id=`{it['resume_id']}` — chunks={it.get('num_chunks')} — mode={it.get('chunk_mode')}")

# ---------------- Ask Questions ----------------
with tab2:
    st.subheader("❓ Ask Questions (RAG)")

    items = list_resumes(cfg)
    resume_options = ["(All resumes)"] + [f"{it['filename']} | {it['resume_id']}" for it in items]
    pick = st.selectbox("Search scope", resume_options, index=0)

    question = st.text_input("Question", value="Who has experience with FastAPI? Return names + emails.")
    k = st.slider("Top-k retrieved chunks", min_value=2, max_value=20, value=10)

    run = st.button("🚀 Run RAG")
    if run:
        if not key_ok:
            st.error("Missing OPENROUTER_API_KEY in .env")
            st.stop()

        scope = "all"
        resume_id = None
        if pick != "(All resumes)":
            scope = "one"
            resume_id = pick.split("|")[-1].strip()

        try:
            out = compare_two_models(
                collection_name=collection_name,
                persist_dir=persist_dir,
                scope=scope,
                question=question,
                model1=model1,
                model2=model2,
                k=int(k),
                resume_id=resume_id,
            )

            st.success("✅ Answers")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### LLM1: `{model1}`")
                st.json(out["model1"])
            with col2:
                st.markdown(f"### LLM2: `{model2}`")
                st.json(out["model2"])

        except Exception as e:
            st.error(str(e))
            st.exception(e)
