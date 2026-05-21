import os
import pathlib
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import re
import requests
from typing import List
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ── Config ────────────────────────────────────────────────────────────────────
OS_API_KEY      = os.getenv("OPENROUTER_API_KEY")
PDF_DIR         = os.getenv("PDF_DIR", "pdfs")
CHROMA_PATH     = "/tmp/chroma_db"          # /tmp is writable on Cloud Run
CHUNK_SIZE      = 600
CHUNK_OVERLAP   = 100
TOP_K_RESULTS   = 3
MODEL_NAME      = os.getenv("MODEL_NAME",      "anthropic/claude-3.5-sonnet")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
TEMPERATURE     = 0.0
MAX_TOKENS      = 600

# ── Ensure dirs exist ─────────────────────────────────────────────────────────
pathlib.Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

# ── Global singletons (populated by init_rag) ─────────────────────────────────
_client     = None
_collection = None
_rag_chain  = None

# ── Prompt template ───────────────────────────────────────────────────────────
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a professional Resume Screening Assistant specialized in analyzing candidate qualifications.

Context from Resume(s):
{context}

Question: {question}

Instructions:
- Answer using ONLY the information provided in the context above
- Be specific and cite relevant details from the resume
- If the requested information is not available in the context, explicitly state: "This information is not available in the resume"
- Do not make assumptions or add information not present in the context
- Keep your answer concise and focused on the question

Answer:"""
)

# ── Embeddings ────────────────────────────────────────────────────────────────
def generate_embeddings(texts: List[str], api_key: str, model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    response = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Resume-RAG-System"
        },
        json={"model": model_name, "input": texts},
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        embeddings = [item["embedding"] for item in result.get("data", [])]
        if not embeddings:
            raise ValueError("No embeddings returned from API")
        return embeddings
    raise ValueError(f"Embedding API error {response.status_code}: {response.text}")


def create_embedding_function(api_key: str, model_name: str = EMBEDDING_MODEL):
    class OpenRouterEmbeddings(embedding_functions.EmbeddingFunction):
        def __call__(self, input: List[str]) -> List[List[float]]:
            return generate_embeddings(input, api_key, model_name)
    return OpenRouterEmbeddings()


# ── PDF helpers ───────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text("text", sort=True) + "\n" for page in doc)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""


def preprocess_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


# ── LLM ───────────────────────────────────────────────────────────────────────
def invoke_llm(prompt: str) -> str:
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OS_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Resume-RAG-System"
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"LLM API Error: {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve_documents(query: str, collection, k: int = TOP_K_RESULTS) -> List[Document]:
    results = collection.query(query_texts=[query], n_results=k)
    documents = []
    if results["documents"] and results["documents"][0]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            documents.append(Document(page_content=doc_text, metadata=metadata))
    return documents


def format_docs(docs: List[Document]) -> str:
    if not docs:
        return "No relevant information found in the resumes."
    return "\n\n".join(
        f"[Chunk {i} from {doc.metadata.get('doc_name', 'Unknown')}]:\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


# ── RAG chain ─────────────────────────────────────────────────────────────────
def build_rag_chain(collection, prompt: PromptTemplate, k: int = TOP_K_RESULTS):
    def retriever_fn(query: str) -> List[Document]:
        return retrieve_documents(query, collection, k)

    return (
        {
            "context": RunnableLambda(retriever_fn) | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | RunnableLambda(lambda pv: pv.to_string())
        | RunnableLambda(invoke_llm)
        | StrOutputParser()
    )


# ── Public API ────────────────────────────────────────────────────────────────
def init_rag():
    """Called once at FastAPI startup."""
    global _client, _collection, _rag_chain

    if not OS_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    embedding_fn = create_embedding_function(api_key=OS_API_KEY, model_name=EMBEDDING_MODEL)
    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        _collection = _client.get_collection("resume_index", embedding_function=embedding_fn)
        print(f"✅ Loaded existing collection ({_collection.count()} docs)")
    except Exception:
        _collection = _client.create_collection("resume_index", embedding_function=embedding_fn)
        print("✅ Created new empty collection")

    _rag_chain = build_rag_chain(_collection, prompt_template, k=TOP_K_RESULTS)
    print("✅ RAG chain ready")


def index_pdfs(reset: bool = False):
    """Index all PDFs in PDF_DIR into ChromaDB."""
    global _collection, _rag_chain

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        return {"status": "error", "message": f"No PDFs found in {PDF_DIR}"}

    if reset:
        try:
            _client.delete_collection("resume_index")
        except Exception:
            pass
        embedding_fn = create_embedding_function(api_key=OS_API_KEY, model_name=EMBEDDING_MODEL)
        _collection = _client.create_collection("resume_index", embedding_function=embedding_fn)
        _rag_chain  = build_rag_chain(_collection, prompt_template, k=TOP_K_RESULTS)

    total_chunks = 0
    batch_size   = 10

    for pdf_name in tqdm(pdf_files, desc="Indexing PDFs"):
        pdf_path    = os.path.join(PDF_DIR, pdf_name)
        raw_text    = extract_text_from_pdf(pdf_path)
        if not raw_text:
            continue
        chunks = chunk_text(preprocess_text(raw_text))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                _collection.add(
                    documents=batch,
                    metadatas=[{"doc_name": pdf_name}] * len(batch),
                    ids=[f"{pdf_name}_chunk_{j}" for j in range(i, i + len(batch))]
                )
                total_chunks += len(batch)
            except Exception as e:
                print(f"❌ Batch error {pdf_name}: {e}")

    return {"status": "ok", "chunks": total_chunks, "pdfs": len(pdf_files)}


def ask_question(question: str) -> str:
    if _rag_chain is None:
        init_rag()
    return _rag_chain.invoke(question)
