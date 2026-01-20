# CV RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for querying and analyzing CVs/resumes. Built with Python, LangChain, ChromaDB, and OpenRouter.

## 🚀 Features

- **Semantic CV Chunking**: Automatically segments CVs into meaningful sections (Experience, Education, Skills, etc.)
- **Vector Search**: ChromaDB-powered similarity search for relevant CV content
- **LLM-Powered Answers**: Uses OpenRouter to generate comprehensive answers from retrieved context
- **LLM-as-Judge Evaluation**: Automatic quality assessment of answers (relevance, faithfulness, correctness)
- **Streamlit UI**: Interactive web interface for querying and viewing results
- **Configurable**: YAML-based configuration with environment variable overrides

## 📁 Project Structure

```
.
├── app.py                 # Streamlit main entry point
├── main.ipynb             # End-to-end demonstration notebook
├── requirements.txt       # Python dependencies
├── config.yaml            # Configuration file
├── .env.example           # Environment variables template
├── data/
│   ├── cvs.csv            # Sample CV data
│   └── processed/         # Processed chunks (generated)
├── chroma_db/             # Vector database (generated)
├── outputs/               # Evaluation results (generated)
├── pages/                 # Streamlit multipage app
│   ├── 1_🔍_Query.py
│   ├── 2_📊_Evaluation.py
│   └── 3_🗂️_Index.py
└── src/
    ├── config.py          # Configuration management
    ├── data/
    │   ├── loaders.py     # CSV/Parquet data loading
    │   └── chunking.py    # Semantic CV chunking
    ├── embeddings/
    │   └── factory.py     # Embeddings provider factory
    ├── vectordb/
    │   └── chroma_store.py # ChromaDB wrapper
    ├── prompts/
    │   └── templates.py   # Prompt templates
    ├── rag/
    │   └── chain.py       # RAG chain implementation
    ├── eval/
    │   ├── judge.py       # LLM-as-judge evaluator
    │   └── pipeline.py    # Evaluation pipeline
    └── utils/
        ├── logging.py     # Logging utilities
        └── timing.py      # Timing utilities
```

## 🛠️ Setup

### 1. Clone and Install Dependencies

```bash
# Navigate to project directory
cd RAG

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your OpenRouter API key
# OPENROUTER_API_KEY=your_api_key_here
```

### 3. Configure Models (Optional)

Edit `config.yaml` to customize:
- LLM model for answering queries
- Judge model for evaluation
- Embedding model
- Retrieval settings

```yaml
llm:
  model: "openai/gpt-4o-mini"  # Or any OpenRouter model
  temperature: 0.2

judge:
  model: "openai/gpt-4o-mini"  # Separate model for evaluation

embeddings:
  provider: "openrouter"
  model: "openai/text-embedding-3-small"

retrieval:
  top_k: 6
```

## 📓 Running the Notebook

The main notebook provides a complete walkthrough:

```bash
# Start Jupyter
jupyter notebook main.ipynb
```

**Notebook Sections:**
1. Setup & Configuration
2. Load CV Data
3. Semantic Chunking
4. Build Vector Index
5. Single Query Demo
6. Batch Query Demo
7. Evaluation Pipeline

## 🌐 Running the Streamlit App

```bash
# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser with:
- **Query Page**: Search CVs with natural language
- **Evaluation Page**: View evaluation history and stats
- **Index Page**: Manage the vector index

## 📊 Rebuilding the Index

If you add new CVs or want to rebuild:

1. **Add CVs**: Edit `data/cvs.csv` with new entries
2. **Run the notebook**: Execute all cells in `main.ipynb`
3. **Or use the API**:

```python
from src.config import load_config
from src.data.loaders import load_cvs
from src.data.chunking import chunk_dataframe
from src.embeddings.factory import EmbeddingsFactory
from src.vectordb.chroma_store import ChromaStore

# Load config and data
config = load_config()
df = load_cvs(config.data.cvs_path)

# Chunk CVs
chunks_df = chunk_dataframe(df)

# Build index
embeddings = EmbeddingsFactory.from_config(config)
store = ChromaStore.from_config(config, embeddings)
store.reset_index()  # Clear existing
store.build_index(chunks_df)
```

## 🧪 Evaluation Metrics

The system evaluates answers using LLM-as-judge with three metrics:

| Metric | Description |
|--------|-------------|
| **Relevance** | How well does the answer address the question? |
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Correctness** | Overall quality and accuracy of the answer |

Scores range from 1-5, with explanations provided.

## 📝 Adding Your Own CVs

### Option 1: CSV File

1. Prepare a CSV with columns:
   - `candidate_id` (required): Unique identifier
   - `raw_text` (required): Full CV text
   - Optional: `name`, `email`, `role`, `location`, `years_experience`

2. Place the file at `data/cvs.csv`

3. Rebuild the index using the notebook

### Option 2: Upload PDF (New!)

You can now upload CV PDFs directly through the Streamlit app:

1. **Start the app**: `streamlit run app.py`
2. **Go to "Upload PDF" page** (📄 in sidebar)
3. **Upload your CV PDF** (text-based PDFs only)
4. **Enter a Candidate ID** (e.g., `john_doe_cv`)
5. **Click "Ingest PDF into Vector DB"**
6. **Query immediately** from the Query page

**Note on PDF Support:**
- ✅ **Supported**: Text-based PDFs (created from Word, Google Docs, etc.)
- ❌ **Not Supported**: Scanned/image-based PDFs (requires OCR, not implemented)
- **Tip**: If you can select and copy text in your PDF viewer, it should work!

**Filtering by Candidate:**
After uploading, you can filter search results to a specific candidate:
1. Go to the Query page
2. Enable "Filter by Candidate ID" in the sidebar
3. Enter the exact candidate ID you used during upload

## 🔧 Configuration Reference

### config.yaml

```yaml
# LLM for answering queries
llm:
  model: "openai/gpt-4o-mini"
  temperature: 0.2
  max_tokens: 800

# Judge model for evaluation
judge:
  model: "openai/gpt-4o-mini"
  temperature: 0.0
  max_tokens: 400

# Embeddings configuration
embeddings:
  provider: "openrouter"  # or "local"
  model: "openai/text-embedding-3-small"
  local_model: "all-MiniLM-L6-v2"

# ChromaDB settings
chroma:
  persist_dir: "./chroma_db"
  collection: "cv_rag"

# Retrieval settings
retrieval:
  top_k: 6
  section_filter: null  # e.g., "experience"

# Output paths
outputs:
  dir: "./outputs"
```

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_key

# Optional overrides
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
JUDGE_MODEL=openai/gpt-4o-mini
EMBEDDING_MODEL=openai/text-embedding-3-small
USE_LOCAL_EMBEDDINGS=false
```

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│  Streamlit   │────▶│  RAG Chain  │
│   Query     │     │     UI       │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                 │
                    ┌────────────────────────────┼────────────────────────────┐
                    │                            ▼                            │
                    │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
                    │  │  Retriever  │───▶│   Prompt    │───▶│     LLM     │  │
                    │  │  (Chroma)   │    │  Template   │    │ (OpenRouter)│  │
                    │  └─────────────┘    └─────────────┘    └─────────────┘  │
                    │         │                                      │        │
                    │         ▼                                      ▼        │
                    │  ┌─────────────┐                        ┌───────────┐   │
                    │  │ Embeddings  │                        │  Answer   │   │
                    │  │   Model     │                        │ + Sources │   │
                    │  └─────────────┘                        └───────────┘   │
                    │                                                │        │
                    └────────────────────────────────────────────────┼────────┘
                                                                     │
                                                                     ▼
                                                            ┌─────────────┐
                                                            │ Evaluation  │
                                                            │   (Judge)   │
                                                            └─────────────┘
```

## 📄 License

MIT License - feel free to use and modify for your own projects.

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

---

Built with ❤️ using LangChain, ChromaDB, and OpenRouter
