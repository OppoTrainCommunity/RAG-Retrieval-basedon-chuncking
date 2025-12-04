
# 📝 Mini-RAG Resume Retrieval System  
*A lightweight Retrieval-Augmented pipeline for resume understanding and career recommendation.*

This project implements a **Mini RAG (Retrieval-Augmented Generation)** pipeline designed to:

- **Parse resume PDFs or text datasets**
- **Chunk and clean the text (semantic chunking)**
- **Store chunks in a Chroma vector database**
- **Retrieve the most similar resumes using embeddings**
- **Evaluate retrieval performance using IR metrics**
- Prepare data for downstream LLM-powered recommendation or career-matching models

This repository focuses on the **retrieval part**, as required by assignment `<A01>`.

---

## 📂 Project Structure

```
Mini-RAG/
│
├── skills_retreival.py      # Main pipeline (indexing + retrieval + evaluation)
├── data/
│   ├── resumes/             # PDF resumes & Kaggle CSV dataset
│   ├── careers.json         # JSON corpus built from data
│
└── chromadb_data/           # Persisted Chroma DB storage
```

---

## 🚀 Features

### ✔ Resume PDF text extraction  
Using **PyPDF** to parse multi-page resumes.

### ✔ Semantic chunking  
Unlike naive fixed-size chunking, this system:

- Detects headings (SKILLS, EXPERIENCE, PROJECTS…)
- Splits text into meaningful chunks
- Ensures chunk sizes stay between *200–800 chars*
- Greatly improves vector similarity retrieval

### ✔ Supports multiple data sources  
- PDF resumes (your own data)
- Kaggle Resume Dataset  
  https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

### ✔ Vector DB indexing with Chroma  
- Uses **SentenceTransformer: all-MiniLM-L6-v2**
- Stores chunked resumes in `"pdf_docs"` collection
- Metadata saved:
  - base resume id  
  - chunk index  
  - file name  
  - source (pdf/kaggle)

### ✔ High-quality retrieval  
Given a query, retrieves the **top-k most similar resume chunks**.

### ✔ Full evaluation module  
Implements:

- Precision@k
- Recall@k
- MAP (Mean Average Precision)
- nDCG@k

Evaluation is done using Kaggle categories as ground truth.

---

## 🛠 Installation

```bash
pip install chromadb pandas pypdf sentence-transformers
```

---

## 📁 Data Preparation

### 1️⃣ Build corpus from Kaggle dataset

```python
build_kaggle_resume_corpus(
    input_csv_path="data/resumes/UpdatedResumeDataSet.csv",
    output_json_path="data/careers.json",
    text_column="Resume",
)
```

This produces a JSON file:

```json
{
  "id": "kaggle_12",
  "file_name": "kaggle_resume_12.txt",
  "text": "... resume text ...",
  "source": "kaggle_resume"
}
```

### 2️⃣ Build corpus from PDF files (optional)

```python
build_pdf_corpus_json(
    input_dir="data/resumes",
    output_json_path="data/careers.json"
)
```

---

## 🧱 Indexing in ChromaDB

```python
build_pdfs_index_from_json(
    json_path="data/careers.json",
    collection_name="pdf_docs"
)
```

This:

- Applies semantic chunking  
- Creates unique IDs per chunk: `doc_3_chunk_0`  
- Embeds each chunk  
- Saves everything into Chroma persistent storage

---

## 🔍 Retrieval

```python
results = retrieve_from_pdfs(
    query_text="Python backend developer with Flask experience",
    k=5
)
```

Each result contains:

```json
{
  "id": "kaggle_41_chunk_2",
  "file_name": "kaggle_resume_41.txt",
  "source": "kaggle_resume",
  "text": "... retrieved chunk ..."
}
```

---

## 🧪 Evaluation Metrics

The system is evaluated using:

- **Precision@k**
- **Recall@k**
- **MAP**
- **nDCG@k**

Using Kaggle’s Category column as ground truth.

---

## 📊 Example Evaluation Output

```
Evaluation over 100 queries
Precision@10: 0.8670
Recall@10:    0.2269
MAP:          0.8670
nDCG@10:      0.9027
```

---

## 🔮 Next Steps / Roadmap

- Add LLM from TogetherAI/OpenRouter for career recommendation  
- Integrate Streamlit/Gradio UI  
- Improve semantic chunking with sentence transformers  
- Add reranking (e.g. Cohere Rerank)  
- Try multi-modal resume parsing  
- Compare against a classifier baseline  

---

## 🧑‍💻 Author

**Sama Shalabi**  
AI Programmer • Machine Learning & IR Projects
