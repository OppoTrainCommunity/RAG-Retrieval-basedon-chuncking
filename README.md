# CV-Based Career Path Recommendation System – Retrieval Component

## 📋 Overview

RAG system that analyzes resumes/CVs and extracts relevant information using semantic search. The workflow: **Read PDF** → **Chunk text** → **Embed** → **Store in ChromaDB** → **Query semantically** → **Retrieve results**.

---

## 🔧 Installation

### Prerequisites
- **Python** 3.8+
- **pip**

### Setup

```bash
pip install chromadb sentence-transformers pymupdf nltk
python -c "import nltk; nltk.download('punkt')"
```

### Folder Structure

```
Retrieval_Task/
├── retrieval.ipynb
├── questions.txt (one question per line)
├── cvs/ (place your PDF files here)
└── README.md
```

---

## 📊 Chunking Methods (4 Strategies)

| Method | Granularity | Speed | Best For | Trade-offs |
|--------|------------|-------|----------|-----------|
| **1. Sentence-based** | Medium | Fast | QA systems | Variable chunk sizes |
| **2. Paragraph-based** | Coarse | Very Fast | Structured docs | Large variable sizes |
| **3. Semantic (Fixed Window)** | Fine | Fast | General purpose | May split sentences |
| **4. Sliding Window** | Fine | Fast | Production ⭐ | Best all-around |

**Recommended:** Use **Sliding Window (Method 4)** for production and **Semantic (Method 3)** for development.

---

## 🧠 Embedding Models (3 Options)

| Model | Dimensions | Speed | Accuracy | Memory | Best For |
|-------|-----------|-------|----------|--------|----------|
| **all-MiniLM-L6-v2** | 384 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Minimal | Real-time apps |
| **all-mpnet-base-v2** | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~220 MB | Balanced |
| **paraphrase-mpnet-base-v2** | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~220 MB | High accuracy ⭐ |

**Recommended:** Use **paraphrase-mpnet-base-v2** for production quality, **all-MiniLM-L6-v2** for speed.

---

## 🚀 Quick Start

1. Run notebook: `jupyter notebook retrieval.ipynb`
2. Choose chunking method (recommend: Sliding Window)
3. Choose embedding model (recommend: paraphrase-mpnet-base-v2)
4. View results with similarity scores

**Example Output:**
```
🔍 Query: What are my technical skills?
⭐ Result #1 | Similarity: 0.8234 | [Relevant excerpt...]
⭐ Result #2 | Similarity: 0.7891 | [Related content...]
```

---

## 🔍 How It Works

```
PDF → Chunk Text → Embed → Store in ChromaDB → Query → Retrieve Top-K Results
```

**Cosine Similarity:** Measures relevance (0 = different, 1 = identical)

---

## 💡 Recommendations

**Production:** Sliding Window chunking + paraphrase-mpnet-base-v2 + top 3-5 results  
**Development:** Semantic chunking + all-MiniLM-L6-v2 + top 3 results  
**High Accuracy:** Sentence-based chunking + paraphrase-mpnet-base-v2


---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: sentence_transformers` | `pip install sentence-transformers` |
| `FileNotFoundError: cvs` | Create `cvs/` folder and add PDF files |
| `FileNotFoundError: questions.txt` | Create `questions.txt` with queries (one per line) |
| Keras version error | `pip install tf-keras` or set `os.environ["TRANSFORMERS_NO_TF"] = "1"` |

---

## 📚 Next Steps

1. **LLM Integration** - Use retrieved chunks as context for generation
2. **UI Development** - Build Streamlit/Gradio interface
3. **Production Deploy** - Docker + API endpoints + persistent database



