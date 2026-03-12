<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
</p>

# 📄 Resume RAG System

> **An intelligent CV analysis platform powered by Retrieval-Augmented Generation.** Upload resumes, ask questions in natural language, and match candidates to job descriptions — all through a sleek, modern interface.

---

## ✨ What It Does

| Feature | Description |
|---------|-------------|
| **Smart Upload** | Drag-and-drop PDF resumes. Each CV is automatically chunked, embedded, and indexed for instant retrieval. |
| **RAG Chat** | Ask anything about your candidates — *"Who has experience with Kubernetes?"*, *"Compare the Python skills of all candidates"* — and get sourced, context-grounded answers. |
| **Job Matching** | Paste a job description and get ranked candidates with match scores, powered by LLM re-ranking with keyword fallback. |
| **Analytics Dashboard** | Visualize skill distributions and experience breakdowns across your entire candidate pool. |
| **Multi-Model Support** | Switch between LLMs on the fly — comes with 4 free OpenRouter models, or type any model ID. |
| **Dark Mode** | Full dark/light theme with glassmorphism UI. |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React Frontend                       │
│         Tailwind CSS · Recharts · React Router          │
├─────────────────────────────────────────────────────────┤
│                    FastAPI Backend                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐     │
│  │  Upload  │  │   Chat   │  │   Job Matching     │     │
│  │  Service │  │  (RAG)   │  │   (LLM Re-rank)    │     │
│  └────┬─────┘  └────┬─────┘  └────────┬───────────┘     │
│       │             │                 │                 │
│  ┌────▼─────────────▼─────────────────▼────────────┐    │
│  │          Hybrid Search Engine                   │    │
│  │     Vector Search + BM25 + RRF Fusion           │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐    │
│  │  ChromaDB           │  FastEmbed (ONNX)         │    │
│  │  Vector Store       │  Local Embeddings         │    │
│  └─────────────────────┴───────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  OpenRouter API → LLM (Nemotron, Mistral, etc.) │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Local Embeddings** — FastEmbed runs `all-MiniLM-L6-v2` via ONNX Runtime directly on the server. Zero API cost, no rate limits, and CV data never leaves the machine.
- **Hybrid Search** — Combines vector similarity (semantic) with BM25 (keyword) using Reciprocal Rank Fusion. This catches both *"machine learning experience"* and exact terms like *"TensorFlow 2.14"*.
- **Graceful Fallbacks** — Every LLM call has a deterministic fallback: regex extraction if metadata parsing fails, keyword scoring if job matching fails. The system never breaks because an API is down.
- **Single Container** — FastAPI serves the React build as static files. One Docker image, one port, no reverse proxy needed.

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- An [OpenRouter](https://openrouter.ai/) API key (free tier works)

### Run with Docker (Recommended)

The image is published on Docker Hub — no build required:

```bash
# Pull the image
docker pull ahamdfobaid/resume-rag-system

# Run the container
docker run -d \
  --name resume-rag \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=sk-or-v1-your-key-here \
  -v resume-data:/app/data \
  ahamdfobaid/resume-rag-system

# Open in browser
open http://localhost:8000
```

### Run with Docker Compose

```bash
# Create your .env file
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

# Start the application
docker compose up -d

# Open http://localhost:8000
```

### Run Locally (Development)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
# Opens at http://localhost:5173 (proxies API to :8000)
```

---

## 🔧 Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key for LLM features |
| `CHROMA_PERSIST_DIR` | No | `./data/chroma_db` | ChromaDB storage directory |
| `UPLOAD_DIR` | No | `./data/uploads` | Uploaded PDF storage directory |
| `LANGCHAIN_TRACING_V2` | No | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | — | LangSmith API key (if tracing enabled) |

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/stats` | System statistics (CV count, top skills) |
| `GET` | `/api/models` | Available LLM models |
| `POST` | `/api/upload` | Upload PDF resume (multipart/form-data) |
| `GET` | `/api/cvs` | List all processed CVs |
| `GET` | `/api/cvs/{cv_id}` | Get a specific CV's details |
| `DELETE` | `/api/cvs/{cv_id}` | Delete a CV and its embeddings |
| `POST` | `/api/chat` | Ask a question about candidates (RAG) |
| `POST` | `/api/match` | Match candidates against a job description |
| `GET` | `/api/analytics/skills` | Skill distribution analytics |
| `GET` | `/api/analytics/experience` | Experience level analytics |
| `POST` | `/api/settings/api-key` | Save OpenRouter API key |
| `POST` | `/api/settings/api-key/validate` | Validate an API key |

---

## 🧰 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Runtime** | Python 3.13 | Backend language |
| **Framework** | FastAPI | Async REST API with auto-docs |
| **RAG** | LangChain | Orchestration, prompt templates, chains |
| **Vector DB** | ChromaDB | Embedded vector storage with HNSW index |
| **Embeddings** | FastEmbed (ONNX) | Local `all-MiniLM-L6-v2` — no GPU needed |
| **Search** | BM25 + RRF | Hybrid retrieval with rank fusion |
| **LLM** | OpenRouter | Gateway to 100+ models (free tier available) |
| **Frontend** | React 18 | Component-based UI |
| **Styling** | Tailwind CSS 3 | Utility-first CSS with dark mode |
| **Charts** | Recharts | Analytics visualizations |
| **Build** | Vite 5 | Lightning-fast frontend tooling |
| **Container** | Docker | Multi-stage build, single image |

---

## 📁 Project Structure

```
Resume-RAG-System/
├── Dockerfile                 # Multi-stage build (Node → Python → Production)
├── docker-compose.yml         # One-command deployment
├── .env.example               # Environment variable template
│
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py            # FastAPI entry point + static file serving
│       ├── config.py          # Pydantic Settings configuration
│       ├── models/
│       │   └── schemas.py     # Request/response Pydantic models
│       ├── services/
│       │   ├── vector_store.py       # ChromaDB + hybrid search (vector+BM25+RRF)
│       │   ├── llm_service.py        # OpenRouter LLM client with model caching
│       │   ├── pdf_processor.py      # PDF extraction (pdfplumber → PyPDF2 fallback)
│       │   ├── metadata_extractor.py # LLM metadata extraction + regex fallback
│       │   ├── rag_service.py        # Full RAG pipeline
│       │   └── job_matcher.py        # LLM re-ranking + keyword scoring fallback
│       └── routes/
│           ├── upload.py      # PDF upload + chunking + embedding
│           ├── cvs.py         # CRUD operations on CVs
│           ├── chat.py        # RAG chat endpoint
│           ├── match.py       # Job matching endpoint
│           ├── models.py      # Available LLM models
│           ├── analytics.py   # Skills & experience analytics
│           ├── health.py      # Health check & system stats
│           └── settings.py    # API key management
│
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx            # Route definitions
        ├── main.jsx           # React entry point
        ├── index.css          # Tailwind + glassmorphism styles
        ├── context/
        │   └── ThemeContext.jsx    # Dark/light mode persistence
        ├── services/
        │   └── api.js             # Axios API client
        ├── components/
        │   ├── Navbar.jsx         # Navigation + theme toggle
        │   ├── ModelSelector.jsx  # LLM model picker
        │   ├── CVCard.jsx         # CV display card
        │   ├── SkillBadge.jsx     # Color-coded skill tags
        │   ├── CircleProgress.jsx # SVG match score ring
        │   └── SourceCitation.jsx # Collapsible RAG sources
        └── pages/
            ├── Upload.jsx     # Drag-drop upload + CV grid
            ├── Chat.jsx       # Chat interface with markdown
            ├── JobMatch.jsx   # Job matching + analytics charts
            └── Settings.jsx   # API key configuration
```

---

## 🔍 How Hybrid Search Works

Traditional vector search is great for semantic meaning but misses exact terms. BM25 is great for keywords but misses meaning. This system combines both:

1. **Vector Search** — Embeds the query with `all-MiniLM-L6-v2` and finds the closest CV chunks by cosine similarity.
2. **BM25 Search** — Tokenizes the query and scores chunks by term frequency–inverse document frequency.
3. **Reciprocal Rank Fusion (RRF)** — Merges both ranked lists using the formula:

$$\text{score}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $k = 60$ and $r(d)$ is the rank of document $d$ in each result set. This produces a unified ranking that captures both semantic relevance and keyword precision.

---

## 🐳 Docker Details

The Dockerfile uses a **3-stage multi-stage build** to keep the image lean:

| Stage | Base Image | Purpose |
|-------|-----------|---------|
| 1 | `node:20-alpine` | Build React frontend → static assets |
| 2 | `python:3.13-slim` | Install Python deps + pre-download ONNX model |
| 3 | `python:3.13-slim` | Production image with everything assembled |

The embedding model is pre-downloaded at build time, so the first request doesn't trigger a download.

```bash
# Build the image
docker build -t cv-rag-system .

# Run with persistent data
docker run -d \
  --name resume-rag \
  -p 8000:8000 \
  -e OPENROUTER_API_KEY=your-key \
  -v resume-data:/app/data \
  --restart unless-stopped \
  cv-rag-system
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ☕ and LLMs that occasionally hallucinate.
</p>
