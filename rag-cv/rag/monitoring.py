"""
Monitoring Module
=================

Prometheus metrics for the RAG CV Assistant.
Tracks request counts, latencies, vector store stats, and ingestion metrics.

Metrics endpoint: GET /metrics
"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from .logging_utils import get_logger

logger = get_logger(__name__)

# ── Application Info ────────────────────────────────────────────
APP_INFO = Info("rag_cv_app", "RAG CV Assistant application info")

# ── Request Metrics ─────────────────────────────────────────────
# NOTE: General HTTP request count & latency are handled automatically
# by starlette_exporter PrometheusMiddleware. The metrics below track
# RAG-specific application logic only.

# ── RAG Pipeline Metrics ───────────────────────────────────────
QUERY_COUNT = Counter(
    "rag_cv_queries_total",
    "Total number of RAG queries processed",
    ["status"],  # success / error / quick_response
)

QUERY_LATENCY = Histogram(
    "rag_cv_query_latency_seconds",
    "End-to-end RAG query latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_cv_retrieval_latency_seconds",
    "Vector retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

LLM_LATENCY = Histogram(
    "rag_cv_llm_latency_seconds",
    "LLM inference latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

# ── Ingestion Metrics ──────────────────────────────────────────
INGEST_COUNT = Counter(
    "rag_cv_ingestions_total",
    "Total number of ingestion operations",
    ["status"],  # success / error
)

DOCUMENTS_INGESTED = Counter(
    "rag_cv_documents_ingested_total",
    "Total number of PDF documents ingested",
)

CHUNKS_CREATED = Counter(
    "rag_cv_chunks_created_total",
    "Total number of chunks created during ingestion",
)

VECTORS_ADDED = Counter(
    "rag_cv_vectors_added_total",
    "Total number of new vectors added to the store",
)

DUPLICATES_SKIPPED = Counter(
    "rag_cv_duplicates_skipped_total",
    "Total number of duplicate vectors skipped",
)

# ── Vector Store Metrics ───────────────────────────────────────
COLLECTION_DOCUMENT_COUNT = Gauge(
    "rag_cv_collection_documents",
    "Current number of documents in the vector store collection",
    ["model_key"],
)

# ── Health Metrics ─────────────────────────────────────────────
HEALTH_STATUS = Gauge(
    "rag_cv_health_status",
    "Health status of the application (1=healthy, 0=unhealthy)",
)


class MetricsTimer:
    """Context manager for timing operations and recording to a Histogram."""

    def __init__(self, histogram: Histogram, labels: dict = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.perf_counter() - self.start_time
        if self.labels:
            self.histogram.labels(**self.labels).observe(self.duration)
        else:
            self.histogram.observe(self.duration)
        return False  # Don't suppress exceptions


def init_app_info(version: str):
    """Set application info metrics."""
    APP_INFO.info({
        "version": version,
        "app_name": "rag_cv_assistant",
    })


def record_query(status: str, duration: float = None):
    """Record a query event."""
    QUERY_COUNT.labels(status=status).inc()
    if duration is not None:
        QUERY_LATENCY.observe(duration)


def record_ingestion(
    status: str,
    num_docs: int = 0,
    num_chunks: int = 0,
    new_vectors: int = 0,
    skipped: int = 0,
):
    """Record an ingestion event with all sub-metrics."""
    INGEST_COUNT.labels(status=status).inc()
    if status == "success":
        DOCUMENTS_INGESTED.inc(num_docs)
        CHUNKS_CREATED.inc(num_chunks)
        VECTORS_ADDED.inc(new_vectors)
        DUPLICATES_SKIPPED.inc(skipped)


def update_collection_gauge(model_key: str, count: int):
    """Update the gauge for document count in a collection."""
    COLLECTION_DOCUMENT_COUNT.labels(model_key=model_key).set(count)
