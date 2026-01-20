
# 📄 RAG Retrieval Evaluation — Chunking Strategies  
This repository contains an end-to-end evaluation framework for **Resume-Based RAG (Retrieval-Augmented Generation)** using **two different chunking strategies**.  
The goal of this work is to measure the effect of chunking on retrieval accuracy, latency, and ranking quality.

---

# 🚀 Project Overview
This project implements:

1. **PDF Parsing**  
   - Extract text content from raw resume PDFs.

2. **Chunking Strategies**  
   - **Semantic Chunking** → Based on CV section headers (Education, Experience, Projects… etc.)  
   - **Sliding Window Chunking** → Fixed-size word windows with overlap.

3. **Vector Database (ChromaDB)**  
   - Each chunk is embedded using `all-MiniLM-L6-v2`.  
   - Stored in separate collections for each strategy.

4. **Retrieval Evaluation**  
   - Compare retrieval results against ground-truth for 15 well-defined queries.  
   - Metrics calculated for each strategy:
     - Precision@k  
     - Recall@k  
     - Hit Rate  
     - MRR (Mean Reciprocal Rank)  
     - Latency  

5. **Evaluation Report**  
   - Full results saved automatically into `evaluation_results.csv`

---

# 📦 Directory Structure
```
RAG-Retrieval-basedon-chuncking/
│
├── app.py
├── requirements.txt
├── evaluation_results.csv
│
├── resume_rag/
│   ├── chunker.py
│   ├── llm_chain.py
│   ├── parse_pdf.py
│   ├── pipeline.py
│   ├── evaluate_chunking.py
│   ├──vector_store_chain.py
│   ├── __init__.py
│   │
│   ├── assets/
│   │   ├── ibrahim_cv.pdf
│   │   ├── rama_cv.pdf
│   │   ├── toqa_cv.pdf
│   │   ├── tala_cv.pdf
│   │
│   └── resume_db/
│
└── .gitignore
```

---

# 🧩 Chunking Strategies

## **1. Semantic Chunking**
Divides the resume based on standard CV section headers:
- EDUCATION  
- EXPERIENCE  
- PROJECTS  
- SKILLS  
- CERTIFICATIONS  
- SUMMARY  

Suitable for structured questions such as:
- “What is the candidate’s GPA?”  
- “What university did the candidate attend?”

## **2. Sliding Window Chunking**
A flexible chunking method using:
- **Window = 180 tokens**  
- **Overlap = 40 tokens**

This method increases contextual recall and works better for:
- Complex project descriptions  
- Skills queries  
- Multi-hop inference

---

# 📊 Evaluation Metrics

Each query is executed against both chunking strategies.  
We compute:

| Metric | Meaning |
|--------|---------|
| **Precision@k** | % of retrieved chunks that are correct |
| **Recall@k** | % of all correct chunks retrieved |
| **Hit Rate** | Whether any chunk contained the answer |
| **MRR** | Rank quality of first correct result |
| **Latency** | Retrieval time |

---

# 🔍 Queries Used for Evaluation  
A set of 15 queries were selected to cover:

### ✔ Direct Information Retrieval  
- Email  
- Phone number  
- University  
- GPA  

### ✔ Experience  
- Internships  
- Trainings  
- Certifications  

### ✔ Projects  
- ML classification  
- MERN stack project details  
- ML models used in stock prediction system  

### ✔ Skills  
- Spark  
- React  
- YOLO  

Ground truth for each query is matched using substring comparison.

---

# 🧪 Running the Evaluation

Run the evaluation script:

```bash
python -m resume_rag.evaluate_chunking
```

Or:

```bash
python resume_rag/evaluate_chunking.py
```

Results will appear in:

```
evaluation_results.csv
```

And a printed statistical summary shows which strategy performed better.

---

# 🏆 Final Results Summary

| Strategy | Precision | Recall | Hit Rate | MRR | Latency |
|----------|-----------|--------|----------|-----|---------|
| **Semantic Chunking** | 0.25 | 0.62 | 0.64 | 0.50 | 0.0147s |
| **Sliding Window Chunking** | **0.35** | **0.86** | **1.00** | **0.73** | 0.0143s |

### ✅ **Winner: Sliding Window Chunking**  
Provides:
- Higher recall  
- More accurate retrieval  
- Better ranking quality  
- Stable low latency  

### Hybrid Recommendation  
Use:
- **Semantic** for structured Q&A  
- **Sliding Window** for long-context queries  

---

# 🧷 Usage in RAG Pipeline  
Chunk collections can be plugged directly into any RAG system using the functions in:
- `vector_store.py`
- `pipeline.py`
- `llm_chain.py`

---

# 👩‍💻 Author  
Developed by **Tala Dweikat** as part of **OppoTrain RAG Task**.

---

# 📬 Contact  
📧 tala.nazeeh.dowiekat@gmail.com  
🔗 GitHub: https://github.com/taladowiekat
