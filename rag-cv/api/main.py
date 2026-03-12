"""
FastAPI Application — RAG CV Assistant
=======================================

Backend API that serves as the bridge between the frontend
and the RAG pipeline (ChromaDB + LLM).

Usage (local):
    cd rag-cv
    uvicorn api.main:app --reload --port 8000

Usage (GCP / production):
    The entry-point is the `app` object in this module.
"""

import sys, os, re
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()

from rag.config import settings
from rag.logging_utils import get_logger, configure_root_logger
from rag.pdf_loader import extract_pdf
from rag.chunking import chunk_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vector_store_manager
from rag.retriever import create_retriever, format_retrieved_docs
from rag.chain import create_rag_chain
from rag.monitoring import (
    init_app_info,
    record_query,
    record_ingestion,
    update_collection_gauge,
    HEALTH_STATUS,
)

from api.schemas import Query, QueryResponse, HealthResponse, IngestResponse, RootResponse

# ── Logging ─────────────────────────────────────────────────────
configure_root_logger()
logger = get_logger(__name__)

# ── Default embedding model key ────────────────────────────────
DEFAULT_MODEL_KEY = "Model A (MiniLM)"

# ── Lifespan (Startup/Shutdown) ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle app startup and shutdown events.
    Pre-loads heavy resources (embeddings, vector store) to ensure responsiveness.
    """
    logger.info("🚀 Startup: Initializing resources...")
    
    try:
        # Pre-load Vector Store Manager
        manager = get_vector_store_manager()
        
        # Pre-load Embeddings Model
        logger.info(f"Loading embedding model: {DEFAULT_MODEL_KEY}...")
        embeddings = get_embeddings(DEFAULT_MODEL_KEY)
        
        # Ensure collection exists/is loaded
        logger.info("Initializing vector collection...")
        manager.create_or_load_collection(DEFAULT_MODEL_KEY, embeddings)
        
        collection_stats = manager.get_collection_stats(DEFAULT_MODEL_KEY)
        count = collection_stats.get("document_count", 0)
        logger.info(f"✅ Resources ready. Collection contains {count} documents.")

        # Initialize monitoring
        init_app_info(version="1.0.0")
        update_collection_gauge(DEFAULT_MODEL_KEY, count)
        HEALTH_STATUS.set(1)
        
    except Exception as e:
        logger.error(f"⚠️ Startup warning: Failed to pre-load specific resources: {e}")
        HEALTH_STATUS.set(0)
        # We don't raise here to allow the app to start even if DB is empty/broken,
        # but endpoints might fail later.

    yield
    
    logger.info("🛑 Shutdown: Cleaning up resources...")

# ── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="RAG CV Assistant API",
    description="Backend API for the RAG CV Analysis frontend",
    version="1.0.0",
    lifespan=lifespan
)

# ── CORS — add your frontend origins here ──────────────────────
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # add your production domain(s):
    # "https://your-app.web.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics Middleware ──────────────────────────────
from starlette_exporter import PrometheusMiddleware, handle_metrics

app.add_middleware(
    PrometheusMiddleware,
    app_name="rag_cv_assistant",
    prefix="rag_cv",
    group_paths=True,
    filter_unhandled_paths=True,
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)
app.add_route("/metrics", handle_metrics)

# ── Default embedding model key ────────────────────────────────
DEFAULT_MODEL_KEY = "Model A (MiniLM)"

# ── Quick / fun hardcoded responses ────────────────────────────
QUICK_RESPONSES = {
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! Ask me anything about the uploaded CVs.",
    "hey": "Hey! I'm ready to help. What do you want to know?",
    "how are you": "I'm just a program, but I'm doing great! Ask me something about the CVs.",
    "goodbye": "Goodbye! I'll be here when you need me.",
}

# ════════════════════════════════════════════════════════════════
# GET /  —  Root endpoint with API information
# ════════════════════════════════════════════════════════════════

@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG CV Assistant API",
        "version": app.version,
        "endpoints": {
            "synthesize": "/synthesize/ - POST - Query the resume database",
            "ingest": "/ingest - POST - Upload & index PDF CVs",
            "health": "/health - GET - Check API health",
            "docs": "/docs - Interactive API documentation"
        }
    }


# ════════════════════════════════════════════════════════════════
# POST /synthesize/  —  Main RAG endpoint for the frontend
# ════════════════════════════════════════════════════════════════

@app.post("/synthesize/", response_model=QueryResponse)
async def synthesize_response(query: Query):
    """
    Receives a question from the frontend, retrieves relevant CV chunks
    from ChromaDB, feeds them to the LLM, and returns the answer.
    """
    import time
    query_start = time.perf_counter()

    try:
        # ── Quick responses ─────────────────────────────────────
        user_question = query.question.strip()
        if user_question.lower() in QUICK_RESPONSES:
            record_query("quick_response")
            return {"response": QUICK_RESPONSES[user_question.lower()]}

        # ── Load vector store ───────────────────────────────────
        manager = get_vector_store_manager()
        embeddings = get_embeddings(DEFAULT_MODEL_KEY)

        try:
            vectorstore = manager.create_or_load_collection(DEFAULT_MODEL_KEY, embeddings)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            record_query("error")
            raise HTTPException(status_code=500, detail=f"Vector store error: {e}")

        # Check if there is any indexed data
        stats = manager.get_collection_stats(DEFAULT_MODEL_KEY)
        if not stats.get("has_data"):
            record_query("error")
            return {
                "response": "No CVs have been indexed yet. Please upload PDFs first via /ingest."
            }

        # ── Retrieve relevant documents ─────────────────────────
        top_k = query.k if query.k else settings.default_top_k
        retriever = create_retriever(
            vectorstore,
            top_k=top_k,
            search_type="similarity",
        )
        chain = create_rag_chain(retriever, language=query.language)

        # ── Run RAG chain (retrieval + LLM timed internally) ──
        result = chain.invoke(user_question)

        answer = result.get("answer", "").strip()
        if not answer:
            record_query("success", time.perf_counter() - query_start)
            return {
                "response": "Sorry, I couldn't generate a meaningful answer. Please try rephrasing your question.",
                "sources": []
            }

        duration = time.perf_counter() - query_start
        record_query("success", duration)
        logger.info(f"Query answered in {duration:.2f}s")

        return {
            "response": answer,
            "sources": result.get("sources", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        record_query("error", time.perf_counter() - query_start)
        logger.error(f"Error in synthesize_response: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


# ════════════════════════════════════════════════════════════════
# POST /ingest  —  Upload & index PDF CVs
# ════════════════════════════════════════════════════════════════

@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdfs(
    files: List[UploadFile] = File(..., description="PDF files to ingest"),
    model_key: str = Form(default="Model A (MiniLM)"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=150),
):
    """
    Upload one or more PDF CVs, chunk them, and index into ChromaDB.
    """
    if model_key not in settings.embedding_models:
        raise HTTPException(status_code=400, detail=f"Unknown model key: {model_key}")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # 1 — Extract PDFs
    pdf_docs = []
    failed_files = []

    for upload in files:
        file_bytes = await upload.read()
        doc = extract_pdf(file_bytes, upload.filename or "unknown.pdf")
        if doc:
            pdf_docs.append(doc)
        else:
            failed_files.append(upload.filename or "unknown.pdf")

    if not pdf_docs:
        raise HTTPException(status_code=400, detail="No valid PDF content could be extracted")

    # 2 — Chunk
    chunks, chunking_stats = chunk_documents(
        pdf_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_semantic=False,
    )

    if not chunks:
        raise HTTPException(status_code=500, detail="Chunking produced no results")

    # 3 — Embed & store
    embeddings = get_embeddings(model_key)
    manager = get_vector_store_manager()
    manager.create_or_load_collection(model_key, embeddings)
    vs_stats = manager.add_documents(model_key, chunks, embeddings)

    total_chunks = chunking_stats.get("total_chunks", len(chunks))

    logger.info(
        f"Ingested {len(pdf_docs)} PDFs → "
        f"{total_chunks} chunks → "
        f"{vs_stats['new']} new vectors"
    )

    # ── Record ingestion metrics ────────────────────────────────
    record_ingestion(
        status="success",
        num_docs=len(pdf_docs),
        num_chunks=total_chunks,
        new_vectors=vs_stats["new"],
        skipped=vs_stats["skipped"],
    )

    # Update collection gauge
    updated_stats = manager.get_collection_stats(model_key)
    update_collection_gauge(model_key, updated_stats.get("count", 0))

    return {
        "message": f"Successfully ingested {len(pdf_docs)} PDF(s)",
        "new_vectors": vs_stats["new"],
        "skipped_duplicates": vs_stats["skipped"],
        "total_chunks": total_chunks,
        "failed_files": failed_files,
    }


# ════════════════════════════════════════════════════════════════
# GET /health  —  Health Check with DB Status
# ════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Check API health and vector store status.
    """
    try:
        manager = get_vector_store_manager()
        stats = manager.get_collection_stats(DEFAULT_MODEL_KEY)
        count = stats.get("document_count", 0)
        
        status_msg = "healthy"
        if not stats.get("has_data") and count == 0:
            status_msg = "ready_but_empty"

        # Update monitoring gauges
        HEALTH_STATUS.set(1)
        update_collection_gauge(DEFAULT_MODEL_KEY, count)

        return {
            "status": status_msg,
            "embeddings_model": DEFAULT_MODEL_KEY,
            "collection_count": count,
            "version": app.version,
        }
    except Exception as e:
        HEALTH_STATUS.set(0)
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "version": app.version
        }


# ════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)
