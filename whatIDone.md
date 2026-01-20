# RAG Resume Analysis System - Technical Documentation

**Author:** Ahmad F. Obaid  
**Project Type:** Retrieval-Augmented Generation (RAG) Application  
**Domain:** HR/Recruitment - Resume Analysis

---

## 1. Executive Summary

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** designed specifically for analyzing resume documents. The system enables HR professionals and recruiters to perform **semantic search** and ask complex natural language questions about candidate profiles, receiving evidence-backed answers from multiple Large Language Models (LLMs).

The core innovation is moving beyond traditional keyword matching to leverage **vector embeddings** and **LLM reasoning** for deep candidate profile analysis.

---

## 2. System Architecture Overview

### 2.1 High-Level Pipeline

```
[PDF Resumes] → [Document Processor] → [Text Chunks] → [Embeddings] → [ChromaDB Vector Store]
                                                                              ↓
[User Query] → [Query Embedding] → [Semantic Search] → [Retrieved Context] → [LLM] → [Answer + Evaluation]
```

### 2.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Vector Database** | ChromaDB (Persistent) | Embedding storage & similarity search |
| **LLM Framework** | LangChain | RAG pipeline orchestration |
| **LLM Provider** | OpenRouter API | Access to multiple LLMs (GPT-4, Claude, Llama) |
| **Embedding Provider** | OpenAI via OpenRouter | Text vectorization |
| **PDF Processing** | pdfplumber | Text extraction from PDFs |
| **NLP Utilities** | NLTK | Sentence tokenization for semantic chunking |

---

## 3. Project Structure

```
RAG-V4/
├── app.py                      # Main Streamlit application (Presentation Layer)
├── requirements.txt            # Python dependencies
├── data/                       # Directory for PDF resume storage
├── cache/                      # Embedding cache (JSON files)
├── chroma_db/                  # Persistent ChromaDB storage
├── outputs/                    # Generated outputs
└── src/                        # Core Logic Package
    ├── __init__.py             # Package initialization
    ├── config.py               # System configuration & constants
    ├── document_processor.py   # PDF extraction & chunking strategies
    ├── vector_store.py         # Embedding generation & ChromaDB management
    ├── rag_engine.py           # LangChain RAG pipeline & LLM interactions
    └── evaluation.py           # Retrieval metrics calculation
```

---

## 4. Core Modules Detailed Explanation

### 4.1 Configuration Module (`src/config.py`)

**Purpose:** Centralized configuration management for the entire system.

**Key Configuration Elements:**

```python
# Directory paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"           # PDF storage
CACHE_DIR = BASE_DIR / "cache"         # Embedding cache
CHROMA_DIR = BASE_DIR / "chroma_db"    # Vector database

# Available embedding models (via OpenRouter)
EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "openai/text-embedding-ada-002"
]

# Available LLM models for generation
LLM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "mistralai/devstral-2-2512:free",
    "meta-llama/llama-3.2-3b-instruct:free"
]

# Chunking configuration
CHUNKING_STRATEGIES = ["fixed", "semantic", "recursive"]
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50
DEFAULT_TOP_K = 5

# API settings
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3
RETRY_DELAY = 1.0
```

---

### 4.2 Document Processor Module (`src/document_processor.py`)

**Purpose:** Extract text from PDF documents and split into manageable chunks for embedding.

#### 4.2.1 PDF Text Extraction

```python
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Uses pdfplumber to extract text from each page of a PDF.
    Joins all pages with double newlines.
    Returns empty string on failure.
    """
```

#### 4.2.2 Chunking Strategies

The system implements **three distinct chunking strategies**, each with different trade-offs:

**1. Fixed-Length Chunking (`fixed`)**
- Splits text into equal-sized chunks with configurable overlap
- **Parameters:** `chunk_size` (default: 500), `overlap` (default: 50)
- **Use Case:** Simple, predictable chunk sizes; good for uniform content
- **Output Metadata:** `chunk_index`, `start_char`, `end_char`, `strategy`

**2. Semantic Chunking (`semantic`)**
- Uses NLTK's `sent_tokenize()` to split at sentence boundaries
- Aggregates sentences until reaching `target_size`
- **Use Case:** Preserves sentence integrity; better for natural language understanding
- **Output Metadata:** Also includes `sentence_count`

**3. Recursive Chunking (`recursive`)**
- Hierarchical approach with three levels:
  - **Level 0:** Split by paragraphs (`\n\n`)
  - **Level 1:** Split by sentences
  - **Level 2:** Fallback to fixed-length chunking
- Recursively processes chunks that exceed `max_chunk_size`
- **Use Case:** Best for structured documents with clear sections
- **Output Metadata:** Also includes `level` (depth in recursion tree)

#### 4.2.3 Dispatcher Function

```python
def get_chunks(text: str, strategy: str = "fixed", **kwargs) -> List[Dict[str, Any]]:
    """
    Unified interface for all chunking strategies.
    Handles parameter mapping between strategies.
    """
```

---

### 4.3 Vector Store Module (`src/vector_store.py`)

**Purpose:** Manage embeddings, caching, and ChromaDB interactions for efficient similarity search.

#### 4.3.1 ChromaDB Management

```python
# Singleton pattern for database client
def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a singleton ChromaDB client with persistent storage."""

def initialize_collection(collection_name: str) -> chromadb.Collection:
    """Gets or creates a collection with cosine similarity metric."""

def get_collection_name(model: str, strategy: str) -> str:
    """Generates unique collection name like: 'openai_text_embedding_3_small_semantic'"""
```

#### 4.3.2 Embedding Caching System

To minimize API costs and improve performance, embeddings are cached locally:

```python
def get_cache_key(text: str, model: str) -> str:
    """SHA256 hash of model + text for unique cache key."""

def get_embedding(text: str, model: str, api_key: str, use_cache: bool = True) -> List[float]:
    """
    1. Check cache for existing embedding
    2. If not cached, call OpenRouter API
    3. Implements exponential backoff retry (MAX_RETRIES = 3)
    4. Save to cache on success
    """
```

**Cache File Format (JSON):**
```json
{
    "embedding": [0.123, -0.456, ...],
    "model": "openai/text-embedding-3-small",
    "timestamp": "2024-01-15T10:30:00"
}
```

#### 4.3.3 Document Indexing

```python
def index_documents(
    chunks: List[Dict],
    embeddings: List[List[float]],
    source_file: str,
    collection: chromadb.Collection,
    embedding_model: str,
    chunking_strategy: str
) -> int:
    """
    Indexes document chunks into ChromaDB with metadata.
    
    Document ID format: "{filename}_{strategy}_{index}"
    
    Metadata stored:
    - source_file: Original PDF filename
    - chunk_index: Position in original document
    - chunking_strategy: "fixed", "semantic", or "recursive"
    - embedding_model: Which model generated the embedding
    - indexed_at: ISO timestamp
    """
```

#### 4.3.4 Similarity Search

```python
def retrieve_similar_chunks(
    query: str,
    collection: chromadb.Collection,
    embedding_model: str,
    api_key: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    1. Generate query embedding
    2. Perform cosine similarity search in ChromaDB
    3. Return top-k results with similarity scores (1 - distance)
    
    Returns: [
        {
            "chunk_id": "resume.pdf_semantic_3",
            "text": "...",
            "metadata": {...},
            "similarity": 0.92
        },
        ...
    ]
    """
```

---

### 4.4 RAG Engine Module (`src/rag_engine.py`)

**Purpose:** Orchestrate the complete RAG pipeline using LangChain, including response generation and evaluation.

#### 4.4.1 RAGChainManager Class

```python
class RAGChainManager:
    """
    Central manager for RAG operations.
    
    Initialization:
    - Connects to specified ChromaDB collection
    - Stores embedding model and API key references
    - Defines prompt templates for QA and evaluation
    """
```

#### 4.4.2 Custom Retriever

**ChromaRetriever:** A custom LangChain `BaseRetriever` implementation that integrates with the existing `vector_store` module to fetch relevant documents.

#### 4.4.3 Response Generation Pipeline

```python
def generate_response(self, query: str, model_name: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Complete RAG pipeline using LangChain Expression Language (LCEL):
    
    Pipeline Structure:
    {context: Retriever | format, question: Passthrough} 
        -> Prompt 
        -> LLM 
        -> StrOutputParser
    
    1. RETRIEVE: ChromaRetriever fetches similar chunks
    2. FORMAT: Join chunk text
    3. GENERATE: LLM generates answer based on prompt and context
    4. PARSE: Extract string response
    
    Returns:
    {
        "query": "...",
        "answer": "...",
        "context": "...",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "latency_ms": 1234.56
    }
    """
```

#### 4.4.4 Answer Evaluation (LLM-as-a-Judge)

```python
def evaluate_answer(self, query: str, context: str, answer: str, ground_truth: str = None) -> Dict[str, Any]:
    """
    Uses a separate LLM to evaluate answer quality.
    
    Evaluation Dimensions:
    1. RELEVANCE: Does the answer address the question?
    2. FAITHFULNESS: Is the answer grounded in the provided context?
    3. CORRECTNESS: (If ground truth provided) Does answer match expected answer?
    
    Returns:
    {
        "relevance": "YES. The answer directly addresses...",
        "faithfulness": "YES. All claims are from the context...",
        "correctness": 0.0 or 1.0 (if ground_truth provided)
    }
    """
```

---

### 4.5 Evaluation Module (`src/evaluation.py`)

**Purpose:** Calculate information retrieval metrics for assessing retrieval quality.

**Implemented Metrics:**

```python
def calculate_precision(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Precision = |Relevant ∩ Retrieved| / |Retrieved|"""

def calculate_recall(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Recall = |Relevant ∩ Retrieved| / |Relevant|"""

def calculate_f1_score(precision: float, recall: float) -> float:
    """F1 = 2 × (Precision × Recall) / (Precision + Recall)"""
```

---

## 5. Application Layer (`app.py`)

### 5.1 Streamlit Interface Structure

The application provides **two main pages** via sidebar navigation:

#### Page 1: ⚙️ Configuration & Upload
1. **API Configuration:** Input and validate OpenRouter API key
2. **Model Selection:** Choose embedding model and chunking strategy
3. **Advanced Parameters:** Adjust chunk size and overlap
4. **Resume Upload:** Upload PDFs or use existing files from `data/` folder
5. **Processing:** Process and index documents into ChromaDB

#### Page 2: ⛓️ LangChain Comparison
1. **Chain Configuration:** Select collection, choose two LLM models to compare
2. **Query Input:** Enter natural language question about resumes
3. **Ground Truth (Optional):** Provide expected answer for correctness evaluation
4. **Results Display:**
   - Side-by-side comparison of both LLM responses
   - Latency metrics for each model
   - Relevance, Faithfulness, and Correctness scores
   - Retrieved context inspection

### 5.2 Session State Management

```python
# Persisted across Streamlit reruns
st.session_state.api_key = ""
st.session_state.api_valid = False
```

---

## 6. Data Flow Diagrams

### 6.1 Ingestion Flow

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PDF Files  │ ──▶ │ extract_text_    │ ──▶ │   get_chunks()  │
│ (data/*.pdf)│     │ from_pdf()       │     │ (fixed/semantic/│
└─────────────┘     └──────────────────┘     │  recursive)     │
                                              └────────┬────────┘
                                                       │
                    ┌──────────────────┐               ▼
                    │  ChromaDB        │     ┌─────────────────┐
                    │  Collection      │ ◀── │ get_embedding() │
                    │  (persistent)    │     │ (with caching)  │
                    └──────────────────┘     └─────────────────┘
```

### 6.2 Query Flow

```
┌─────────────┐     ┌────────────────────────────────────────────────────────┐
│ User Query  │ ──▶ │                 LangChain RAG Pipeline                 │
└─────────────┘     │  ┌───────────┐   ┌────────┐   ┌─────┐   ┌────────────┐ │
                    │  │ Retriever │──▶│ Prompt │──▶│ LLM │──▶│ OutputParser │ │
                    │  │ (Custom)  │   │        │   │     │   │            │ │
                    │  └─────┬─────┘   └────────┘   └─────┘   └────────────┘ │
                    └────────│──────────────┬──────────────────────────────────┘
                             │              │
                             ▼              ▼
                    ┌──────────────────┐  ┌─────────────┐
                    │ ChromaDB Query   │  │ Final       │
                    │ (cosine search)  │  │ Answer      │
                    └──────────────────┘  └─────────────┘
```

---

## 7. Key Design Decisions

### 7.1 Why OpenRouter?
- Single API endpoint for multiple LLM providers
- Access to free tier models for experimentation
- Unified billing and rate limiting

### 7.2 Why ChromaDB?
- Lightweight, persistent vector database
- No external server required (runs in-process)
- Native Python integration with simple API
- Built-in support for cosine similarity

### 7.3 Why Multiple Chunking Strategies?
- Different resumes have different structures
- Semantic chunking preserves meaning boundaries
- Recursive chunking handles hierarchical documents
- Allows experimentation to find optimal strategy

### 7.4 Why LLM-as-a-Judge Evaluation?
- Automated quality assessment without human annotation
- Measures both relevance (query coverage) and faithfulness (grounding)
- Enables objective comparison between models

---

## 8. API Endpoints & External Services

| Service | Base URL | Purpose |
|---------|----------|---------|
| OpenRouter | `https://openrouter.ai/api/v1` | LLM and Embedding API access |

**Authentication:** Bearer token via `api_key` parameter

---

## 9. Dependencies

```
streamlit         # Web interface
pandas            # Data manipulation
numpy             # Numerical operations
chromadb          # Vector database
openai            # API client for OpenRouter
langchain-openai  # LangChain OpenAI integration
langchain-core    # LangChain core primitives
pdfplumber        # PDF text extraction
nltk              # NLP utilities (sentence tokenization)
```

---

## 10. Usage Instructions

### 10.1 Initial Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 10.2 Workflow
1. Enter OpenRouter API key and validate
2. Select embedding model and chunking strategy
3. Upload PDF resumes or use existing files
4. Click "Process & Index Resumes"
5. Navigate to "LangChain Comparison"
6. Select two LLMs to compare
7. Ask questions and analyze results

---

## 11. Extension Points

For developers looking to extend this system:

1. **Add New LLM Models:** Update `config.LLM_MODELS` list
2. **Add New Embedding Models:** Update `config.EMBEDDING_MODELS` list
3. **Custom Chunking Strategy:** Add function to `document_processor.py` and register in `get_chunks()`
4. **Enhanced Evaluation:** Extend `evaluation.py` with additional metrics (e.g., MRR, NDCG)
5. **Multi-Document QA:** Modify RAG prompts to handle cross-resume comparisons

---

*This documentation is intended to provide comprehensive context for LLM-assisted development and maintenance of this RAG system.*
