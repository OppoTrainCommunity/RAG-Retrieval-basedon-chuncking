import os
import re
import json
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Optional helpers for ingesting uploaded CVs
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

try:
    # pypdf is the modern successor of PyPDF2
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception:
        PdfReader = None

try:
    from src.chunkers.paragraph_chunker import ParagraphChunker
except Exception:
    ParagraphChunker = None

load_dotenv()


# -----------------------------
# Helpers (UI input handling)
# -----------------------------
def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def load_file_names(cvs_json_path: str) -> List[str]:
    p = Path(cvs_json_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return sorted({d.get("file_name") for d in docs if d.get("file_name")})


def extract_text_from_uploaded_file(file_name: str, file_bytes: bytes) -> str:
    """Extract text from an uploaded CV (PDF/DOCX/TXT).

    Notes:
    - PDF extraction requires pypdf or PyPDF2.
    - DOCX extraction requires python-docx.
    """
    lower = (file_name or "").lower()

    if lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")

    if lower.endswith(".docx"):
        if DocxDocument is None:
            raise RuntimeError("DOCX support is unavailable (python-docx not installed).")
        doc = DocxDocument(BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

    if lower.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError(
                "PDF support is unavailable (install 'pypdf' or 'PyPDF2')."
            )
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages)

    raise RuntimeError("Unsupported file type. Please upload PDF, DOCX, or TXT.")


def simple_clean_text(text: str) -> str:
    # Light cleanup to reduce weird whitespace from PDFs
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text_for_indexing(text: str) -> List[str]:
    """Chunk text using your ParagraphChunker if available, otherwise fallback."""
    text = simple_clean_text(text)

    if ParagraphChunker is not None:
        chunker = ParagraphChunker(tokens_per_chunk=200, chunk_overlap=30)
        chunks, _, _ = chunker.chunk_texts([text], [{"doc_id": "tmp", "file_name": "tmp"}])
        # chunker.chunk_texts returns chunks already filtered/split
        return [c for c in chunks if c and c.strip()]

    # Fallback: split by paragraphs, then merge very small bits
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out = []
    buf = ""
    for p in paras:
        if len(buf) < 400:
            buf = (buf + "\n\n" + p).strip()
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out


def upsert_cv_into_collection(
    collection,
    file_name: str,
    text: str,
    source: str = "uploaded",
) -> int:
    """Chunk a CV and add it to the Chroma collection. Returns #chunks added."""
    doc_id = str(uuid.uuid4())
    chunks = chunk_text_for_indexing(text)
    if not chunks:
        raise RuntimeError("No text could be extracted / chunked from this file.")

    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        metadatas.append(
            {
                "doc_id": doc_id,
                "file_name": file_name,
                "source": source,
                "chunk_id": i,
                "chunk_type": "paragraph",
            }
        )
        ids.append(f"{doc_id}_par_{i}")

    collection.add(documents=chunks, metadatas=metadatas, ids=ids)
    return len(chunks)


def append_to_cvs_json(cvs_json_path: str, file_name: str, text: str) -> None:
    """Optional: keep CVs.json in sync so your file_name filter sees new uploads."""
    p = Path(cvs_json_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            try:
                existing = json.load(f) or []
            except Exception:
                existing = []
    new_item = {
        "id": str(uuid.uuid4()),
        "file_name": file_name,
        "text": text,
        "source": "uploaded",
    }
    existing.append(new_item)
    with p.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# -----------------------------
# Chroma: connect + retrieve
# -----------------------------
@st.cache_resource
def get_chroma_collection(persist_dir: str, collection_name: str, embedding_model: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"description": f"Collection for {collection_name}"},
    )


def retrieve_chunks(
    collection,
    query: str,
    k: int = 40,
    where: Optional[Dict[str, Any]] = None,
):
    return collection.query(query_texts=[query], n_results=k, where=where)


def build_context_with_meta(
    docs: List[str],
    metas: List[Dict[str, Any]],
    max_chars: int = 16000,
) -> str:
    parts = []
    for doc, meta in zip(docs, metas):
        file_name = meta.get("file_name", "unknown")
        chunk_id = meta.get("chunk_id", "unknown")
        parts.append(f'[file_name="{file_name}" chunk_id={chunk_id}]\n{doc}')
    joined = "\n\n---\n\n".join(parts)
    return joined[:max_chars]


# -----------------------------
# Prompt building (Recruitment chatbot)
# -----------------------------

RECRUITMENT_SYSTEM_PROMPT = (
    "You are a technical recruitment helper chatbot.\n\n"
    "Your job:\n"
    "- Answer questions about candidates using ONLY the provided resume evidence (context).\n"
    "- Help with recruiting tasks like:\n"
    "  * finding candidates with specific skills\n"
    "  * recommending the best candidate for a role\n"
    "  * comparing candidates\n\n"
    "Rules (critical):\n"
    "- Use ONLY the provided context. Do NOT invent skills, experience, education, or names.\n"
    "- Every claim MUST be backed by evidence, citing file_name and chunk_id.\n"
    "- If evidence is missing, say: 'not found in the provided resumes'.\n"
    "- Be fair: do not assume years of experience unless explicitly stated.\n"
    "- Prefer clear, structured answers:\n"
    "  * If asked 'best candidate', provide a ranked shortlist and explain why.\n"
    "  * If asked 'who has skill X', list candidates and cite evidence.\n"
    "  * If asked general questions, answer briefly and cite.\n"
    "- Output in normal text (NO JSON required)."
)


def build_recruitment_user_prompt(
    context: str,
    user_question: str,
    conversation_summary: str = "",
) -> str:
    user_prompt = f"""
Conversation summary (may help you keep context):
{conversation_summary}

Question:
{user_question}

Retrieved resume evidence (context):
{context}

Answer now. Remember: cite evidence as (file_name, chunk_id).
""".strip()

    # Keep your conditional prefix behavior exactly.
    if conversation_summary.strip():
        user_prompt = f"Conversation summary:\n{conversation_summary}\n\n" + user_prompt

    return user_prompt


# ChatPromptTemplate: produces messages for the model in a consistent way.
RECRUITMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RECRUITMENT_SYSTEM_PROMPT),
        ("human", "{user_prompt}"),
    ]
)


# -----------------------------
# OpenRouter via LangChain init_chat_model
# -----------------------------
@st.cache_resource
def initialize_models(
    name: str,
    provider: str,
    base_url: str,
    api_key: str,
    default_headers: Dict[str, str],
    temperature: float,
    max_tokens: int,
    timeout: int,
):
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    return init_chat_model(
        model=name,
        model_provider=provider,
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def openrouter_chat_completion(
    prompt: ChatPromptTemplate,
    prompt_inputs: Dict[str, Any],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 60,
) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY environment variable")

    base_url = "https://openrouter.ai/api/v1"
    default_headers = {
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Recruitment-RAG",
    }

    model_instance = initialize_models(
        name=model,
        provider="openai",
        base_url=base_url,
        api_key=api_key,
        default_headers=default_headers,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    chain = prompt | model_instance | StrOutputParser()

    try:
        content = chain.invoke(prompt_inputs)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

    if not content or not str(content).strip():
        raise RuntimeError(
            "Model returned empty output. Try increasing max tokens or reducing context size."
        )

    return content


# -----------------------------
# Streamlit UI (Chatbot)
# -----------------------------
st.set_page_config(page_title="Recruitment RAG Chatbot", layout="wide")
st.title("Recruitment Helper Chatbot (RAG)")

with st.sidebar:
    st.header("Chroma Settings")
    persist_dir = st.text_input("persist_directory", value="./chroma_db")
    collection_name = st.text_input("collection_name", value="paragraph_chunking")
    embedding_model = st.text_input("embedding_model", value="all-mpnet-base-v2")

    st.divider()
    st.header("Retrieval")
    # default is k=25 
    k = st.slider("Top K chunks", 5, 120, 25, 40)
    # default is max_chars=16000
    max_chars = st.slider("Max context chars", 4000, 24000, 16000, 1000)

    st.divider()
    st.header("LLM (tuning for now)")
    model = st.text_input("OpenRouter model", value="openai/gpt-5-mini")
    # default is temperature=0.3
    temperature = st.slider("temperature", 0.0, 1.0, 0.3, 0.05)
    # default is max_tokens=4500
    max_tokens = st.slider("max tokens", 500, 8000, 4500, 2500)
    # default is timeout=60
    timeout = st.slider("timeout (sec)", 10, 180, 60, 5)

    st.divider()
    st.header("Optional CV filter")
    cvs_json_path = st.text_input("CVs.json path", value="data/CVs.json")
    use_filter = st.checkbox("Filter retrieval by one file_name", value=False)

    st.divider()
    st.header("Add a new CV to the index")
    uploaded_cv = st.file_uploader(
        "Upload a CV (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
    )
    save_to_cvs_json = st.checkbox(
        "Also append it to CVs.json (enables file_name filter)",
        value=True,
    )
    ingest_clicked = st.button("Chunk + add to collection")

    st.divider()
    debug_show_chunks = st.checkbox("Show retrieved chunks (debug)", value=False)
    debug_show_context = st.checkbox("Show context text (debug)", value=False)

# optional file filter
where = None
if use_filter:
    file_names = load_file_names(cvs_json_path)
    if file_names:
        selected = st.selectbox("file_name", file_names)
        where = {"file_name": selected}
    else:
        st.warning("No file names found. Check CVs.json path.")
        where = None

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""  # keep it simple for now

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Connect to Chroma collection
collection = get_chroma_collection(persist_dir, collection_name, embedding_model)

# Handle CV ingestion (sidebar button)
if ingest_clicked:
    if uploaded_cv is None:
        st.sidebar.error("Please upload a CV file first.")
    else:
        try:
            file_name = uploaded_cv.name
            file_bytes = uploaded_cv.getvalue()
            raw_text = extract_text_from_uploaded_file(file_name, file_bytes)
            cleaned_text = simple_clean_text(raw_text)

            n_chunks = upsert_cv_into_collection(
                collection=collection,
                file_name=file_name,
                text=cleaned_text,
                source="uploaded",
            )

            if save_to_cvs_json:
                append_to_cvs_json(cvs_json_path, file_name, cleaned_text)

            st.sidebar.success(f"Added {n_chunks} chunks from '{file_name}' to the collection.")
            # Refresh UI elements that depend on CVs.json (like the file_name filter list)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")

# Chat input
user_text = st.chat_input("Ask about candidates, skills, or best fit for a role...")

if user_text:
    # 1) show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    raw_answer = ""
    try:
        # 2) retrieve
        results = retrieve_chunks(collection, user_text, k=k, where=where)

        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []
        dists = results.get("distances", [[]])[0] if results else []

        context = build_context_with_meta(docs, metas, max_chars=max_chars)

        # 3) prompt + llm (ChatPromptTemplate + StrOutputParser)
        user_prompt = build_recruitment_user_prompt(
            context=context,
            user_question=user_text,
            conversation_summary=st.session_state.conversation_summary,
        )

        raw_answer = openrouter_chat_completion(
            prompt=RECRUITMENT_PROMPT,
            prompt_inputs={"user_prompt": user_prompt},
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        # 4) show assistant answer
        st.session_state.messages.append({"role": "assistant", "content": raw_answer})
        with st.chat_message("assistant"):
            st.markdown(raw_answer)

        # 5) debug
        if debug_show_context:
            st.subheader("Context (debug)")
            st.text_area("context", context, height=250)

        if debug_show_chunks:
            st.subheader("Retrieved chunks (debug)")
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                st.markdown(f"**Rank {i}** | distance: `{dist:.4f}`")
                st.caption(
                    f'file_name: {meta.get("file_name")} | chunk_id: {meta.get("chunk_id")}'
                )
                st.write(doc)
                st.divider()

    except Exception as e:
        err_msg = f"Error: {e}"
        st.session_state.messages.append({"role": "assistant", "content": err_msg})
        with st.chat_message("assistant"):
            st.error(err_msg)
            if raw_answer:
                st.text_area("Raw model output (debug)", raw_answer, height=200)
