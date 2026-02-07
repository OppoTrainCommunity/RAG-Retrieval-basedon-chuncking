# 📄 RAG CV System

A **Retrieval-Augmented Generation** system for querying multiple CV/Resume PDFs using semantic search and LLM-powered responses.

## ✨ Features

- 📑 **Multi-PDF Support**: Upload and index multiple CV/Resume PDFs simultaneously
- 🔍 **Dual Embedding Models**: Choose between two HuggingFace embedding models
  - Model A: `sentence-transformers/all-MiniLM-L6-v2` (fast, lightweight)
  - Model B: `intfloat/e5-base-v2` (higher quality, more accurate)
- 💾 **Persistent Vector Store**: ChromaDB with model-specific collections
- 🤖 **LLM Integration**: OpenRouter API for flexible model selection
- 🎯 **Smart Chunking**: Section-based semantic chunking optimized for CVs
- 🖥️ **Streamlit UI**: User-friendly interface with 3 pages (Ingest, Chat, Admin)
- 🛠️ **CLI Tools**: Command-line scripts for batch ingestion

## 📁 Project Structure

```
rag-cv/
├── app/
│   └── streamlit_app.py      # Main Streamlit application
├── rag/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── logging_utils.py      # Logging configuration
│   ├── pdf_loader.py         # PDF text extraction
│   ├── chunking.py           # Text chunking strategies
│   ├── embeddings.py         # Embedding model factory
│   ├── vectorstore.py        # ChromaDB wrapper
│   ├── retriever.py          # Document retrieval
│   ├── prompts.py            # RAG prompt templates
│   ├── chain.py              # LangChain RAG pipeline
│   └── utils.py              # Utility functions
├── scripts/
│   ├── ingest_cli.py         # CLI for batch ingestion
│   └── smoke_test.py         # End-to-end smoke tests
├── tests/
│   └── test_smoke.py         # Pytest test suite
├── data/
│   ├── uploads/              # PDF upload directory
│   └── chroma/               # Vector store persistence
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd rag-cv

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
copy .env.example .env   # Windows
cp .env.example .env     # Linux/Mac

# Edit .env and add your OpenRouter API key
# Get your key from: https://openrouter.ai/keys
```

### 3. Run the Application

```bash
# Start Streamlit app
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Ingest / Index Page

1. Select an embedding model (Model A or B)
2. Upload one or more PDF CV files
3. Click "Process & Index PDFs"
4. Wait for processing to complete

### Chat / Search Page

1. Select the same embedding model used during ingestion
2. Enter your question about the CVs
3. View the AI-generated answer with source citations
4. Expand "Source Documents" to see retrieved chunks

### Admin Page

- View collection statistics (document count, chunk sizes)
- Reset/clear the vector store if needed
- Monitor system health

## 🛠️ CLI Usage

### Batch Ingest PDFs

```bash
# Ingest all PDFs from a directory
python scripts/ingest_cli.py --input-dir ./cvs --model a

# With custom chunk settings
python scripts/ingest_cli.py --input-dir ./cvs --model b --chunk-size 500 --chunk-overlap 100
```

### Run Smoke Tests

```bash
# Quick validation
python scripts/smoke_test.py

# Or using pytest
pytest tests/test_smoke.py -v
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | Required |
| `OPENROUTER_MODEL` | LLM model to use | `deepseek/deepseek-r1:free` |
| `EMBEDDING_MODEL_A` | First embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBEDDING_MODEL_B` | Second embedding model | `intfloat/e5-base-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `data/chroma` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K_RESULTS` | Retrieval results | `5` |

### Recommended Free Models (OpenRouter)

- `deepseek/deepseek-r1:free` - DeepSeek R1 (reasoning)
- `google/gemini-2.0-flash-exp:free` - Google Gemini Flash
- `meta-llama/llama-3.2-3b-instruct:free` - Llama 3.2
- `mistralai/mistral-7b-instruct:free` - Mistral 7B

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. ChromaDB Permission Errors**
```bash
# Create data directories manually
mkdir -p data/chroma data/uploads
```

**3. OpenRouter API Errors (404)**
```
Error: Model not found
```
Solution: Check model name on [OpenRouter Models](https://openrouter.ai/models) and update `.env`

**4. Rate Limiting (429)**
```
Error: Too many requests
```
Solution: Wait a few seconds or switch to a different model

**5. Slow First Run**
```
First startup takes time...
```
This is normal - embedding models are being downloaded (~100-500MB)

### Debug Mode

Enable verbose logging:
```bash
export LOG_LEVEL=DEBUG  # Linux/Mac
set LOG_LEVEL=DEBUG     # Windows
streamlit run app/streamlit_app.py
```

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Ingest    │  │    Chat     │  │    Admin    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │
│  │PDF Loader│─▶│ Chunking │─▶│Embeddings│─▶│ChromaDB│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────┘  │
│                                                │        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │        │
│  │   LLM    │◀─│RAG Chain │◀─│Retriever │◀────┘        │
│  │(OpenRouter)│ │ (LCEL)   │  │          │              │
│  └──────────┘  └──────────┘  └──────────┘              │
└─────────────────────────────────────────────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=rag --cov-report=html

# Run specific test
pytest tests/test_smoke.py::test_pdf_extraction -v
```

## 📝 Example Queries

Once you've indexed some CVs, try these queries:

- "What programming languages does [Name] know?"
- "Who has experience with machine learning?"
- "List all candidates with Python skills"
- "What is [Name]'s work experience?"
- "Compare the education backgrounds of the candidates"
- "Who has the most relevant experience for a data scientist role?"

## 🔐 Security Notes

- Never commit `.env` file with API keys
- API keys in config are for development only
- Use environment variables in production
- Consider using secret management in deployment

## 📄 License

MIT License - Feel free to use and modify.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

**Built with** ❤️ using Streamlit, LangChain, ChromaDB, and OpenRouter
