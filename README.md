# 📝 Mini-RAG Resume Retrieval System  
*A lightweight Retrieval-Augmented Generation (RAG) pipeline for resume understanding and career recommendation.*

This project implements a **Mini RAG pipeline** that:

- Parses resume PDFs or text datasets  
- **Applies multiple chunking strategies (Semantic & Paragraph)**  
- Stores chunks in a Chroma vector database  
- Retrieves the most relevant resumes using embeddings  
- Evaluates retrieval performance using IR metrics  
- Prepares data for downstream LLM-powered recommendation models  

This repository focuses on the **retrieval component**, required for assignment `<A01>`.

---

# 🔥 Chunking Methods (NEW Section)

Chunking is the core of any RAG system. This project implements **two chunking strategies**, both indexed and evaluated independently.

---

## 1️⃣ Semantic Chunking (Structure‑Aware)

Semantic chunking identifies typical resume sections such as:

- SKILLS  
- EXPERIENCE  
- PROJECTS  
- EDUCATION  
- CERTIFICATIONS  

It groups lines under these headings into **coherent semantic units**, ensuring chunk size ranges from **200–800 characters**.

### ✔ Advantages
- Produces higher-quality embeddings  
- Aligns with natural resume structure  
- Fewer chunks = less noise  
- Higher Precision, MAP, and nDCG  

---

## 2️⃣ Paragraph Chunking (Structure‑Free)

Paragraph chunking splits text wherever a blank line appears, merging small paragraphs and splitting large ones.

### ✔ Advantages
- Very simple to apply  
- Works on messy text  
- No need for headings  

### ❗ Drawbacks
- Produces **more chunks** → higher noise  
- Paragraphs may contain mixed topics  
- Slightly lower retrieval performance  

---

# 📊 Experimental Results: Semantic vs Paragraph Chunking

You evaluated both chunkers on the Kaggle dataset using:

- **Precision@10**  
- **Recall@10**  
- **MAP (Mean Average Precision)**  
- **nDCG@10**  

---

### **Semantic Chunking Results**
- Precision@10: **0.8550**  
- Recall@10: **0.2246**  
- MAP: **0.8535**  
- nDCG@10: **0.8941**  
- Total chunks: 4495  

### **Paragraph Chunking Results**
- Precision@10: **0.8540**  
- Recall@10: **0.2236**  
- MAP: **0.8502**  
- nDCG@10: **0.8938**  
- Total chunks: 5183  

---

### ✔ Conclusion  
Semantic chunking outperforms paragraph chunking across all metrics.  
This is expected because semantic chunking creates more meaningful, section‑based chunks, producing clearer embeddings and stronger ranking performance.

---

# 🚀 System Features

### ✔ Resume PDF Text Extraction  
Using **PyPDF** to read multi‑page resumes.

### ✔ Dual Chunking Support  
✓ Semantic chunking  
✓ Paragraph chunking  

### ✔ Vector DB Indexing (Chroma)  
- Uses **SentenceTransformer `all-MiniLM-L6-v2`**  
- Stores chunked documents as embeddings  
- Two separate collections:  
  - `pdf_semantic`  
  - `pdf_paragraph`

### ✔ High‑Quality Retrieval  
Top‑k similarity search over vector embeddings.

### ✔ Full Evaluation Suite  
Includes:  
- Precision@k  
- Recall@k  
- MAP  
- nDCG@k  

---

# 🛠 Installation

```bash
pip install chromadb pandas pypdf sentence-transformers
```

---

# 📁 Data Preparation

### 1️⃣ Build Kaggle Corpus
```python
build_kaggle_resume_corpus(
    input_csv_path="data/resumes/UpdatedResumeDataSet.csv",
    output_json_path="data/careers.json",
    text_column="Resume"
)
```

### 2️⃣ Build PDF Corpus (Optional)
```python
build_pdf_corpus_json(
    input_dir="data/resumes",
    output_json_path="data/careers.json"
)
```

---

# 🧱 Indexing in ChromaDB

### Semantic Chunking
```python
build_pdfs_index_from_json(
    json_path="data/careers.json",
    collection_name="pdf_semantic"
)
```

### Paragraph Chunking
```python
build_pdfs_index_paragraph_from_json(
    json_path="data/careers.json",
    collection_name="pdf_paragraph"
)
```

---

# 🔍 Retrieval Example

```python
results = retrieve_from_pdfs(
    query_text="Python backend developer experience",
    k=5,
    collection_name="pdf_semantic"
)
```

---

# 📊 Evaluation Output

```
[Collection: pdf_semantic]
Precision@10: 0.8550
Recall@10:    0.2246
MAP:          0.8535
nDCG@10:      0.8941

[Collection: pdf_paragraph]
Precision@10: 0.8540
Recall@10:    0.2236
MAP:          0.8502
nDCG@10:      0.8938
```

---

# 👩‍💻 Author

**Sama Shalabi**  
AI Programmer • Machine Learning & IR Projects

