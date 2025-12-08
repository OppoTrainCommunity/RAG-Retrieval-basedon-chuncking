"""
Evaluation script for Resume RAG chunking strategies.

- Strategy A: semantic_chunk  -> collection 'resumes_semantic'
- Strategy B: sliding_window_chunk -> collection 'resumes_window'

المخرجات:
- CSV: evaluation_results.csv
- طباعة متوسط المتركس لكل strategy
"""

import csv
import time
from pathlib import Path

from resume_rag.parse_pdf import pdf_to_text
from resume_rag.chunker import semantic_chunk, sliding_window_chunk
from resume_rag.vector_store import (
    reset_collection,
    add_texts,
    search,
    get_all_documents,
)

# ================================
# 1) إعداد الـ Resumes
# ================================
# غيّري المسارات لو أسماء الملفات مختلفة في assets/

RESUMES = [
    {"id": "ibrahim", "path": "resume_rag/assets/ibrahim_cv.pdf"},
    {"id": "rama",    "path": "resume_rag/assets/rama_cv.pdf"},
    {"id": "toqa",    "path": "resume_rag/assets/toqa_cv.pdf"},
    {"id": "tala",    "path": "resume_rag/assets/tala_cv.pdf"},
]

# ================================
# 2) أسئلة التقييم + الإجابات الصحيحة (Ground Truth)
# ================================

EVAL_QUERIES = [
    # ---------- Direct Info ----------
    {
        "id": "rama_email",
        "query": "What is Rama’s email address?",
        "answer_substring": "ramah.sabboubeh",
    },
    {
        "id": "ibrahim_university",
        "query": "Which university does Ibrahim attend?",
        "answer_substring": "an-najah",
    },
    {
        "id": "rama_gpa",
        "query": "What is Rama’s GPA?",
        "answer_substring": "3.68",
    },
    {
        "id": "toqa_email",
        "query": "What is Toqa’s email address?",
        "answer_substring": "toqaasedah",
    },
    {
        "id": "tala_email",
        "query": "What is Tala’s email address?",
        "answer_substring": "tala.nazeeh",
    },

    # ---------- Experience ----------
    {
        "id": "ibrahim_intern",
        "query": "What internship did Ibrahim complete?",
        "answer_substring": "OppoTrain",
    },
    {
        "id": "rama_backend_training",
        "query": "When did Rama complete the Backend Java Training?",
        "answer_substring": "Summer 2025",
    },
    {
        "id": "toqa_internship",
        "query": "Where did Toqa complete her AI internship?",
        "answer_substring": "ASAL",
    },

    # ---------- Projects ----------
    {
        "id": "rama_projects",
        "query": "What machine learning classification projects has Rama worked on?",
        "answer_substring": "Classification",
    },
    {
        "id": "ibrahim_cap_advisor",
        "query": "What technologies were used in Ibrahim’s Cap Advisor project?",
        "answer_substring": "Node.js",
    },
    {
        "id": "tala_stock_prediction",
        "query": "What ML models did Tala use in the Tesla Stock Prediction project?",
        "answer_substring": "Random Forest",
    },

    # ---------- Skills ----------
    {
        "id": "toqa_bigdata",
        "query": "Who has experience with Apache Spark?",
        "answer_substring": "Spark",
    },
    {
        "id": "tala_yolo",
        "query": "Who implemented traffic sign detection using YOLO?",
        "answer_substring": "YOLO",
    },
    {
        "id": "ibrahim_react",
        "query": "Does Ibrahim have experience with React?",
        "answer_substring": "React",
    },
]

TOP_K = 5

# أسماء الكلكشنز لكل strategy
SEMANTIC_COLLECTION = "resumes_semantic"
WINDOW_COLLECTION = "resumes_window"


# ================================
# Helper: بناء الكلكشن لكل Strategy
# ================================
def build_collections():
    """
    - تقرأ كل CV
    - تعمل chunking بطريقتين
    - تخزن النتائج في كلكشنين مختلفين
    """
    print("Building collections for both strategies...")

    # نبدأ بكلكشنز جديدة
    reset_collection(SEMANTIC_COLLECTION)
    reset_collection(WINDOW_COLLECTION)

    for resume in RESUMES:
        rid = resume["id"]
        path = Path(resume["path"])

        if not path.exists():
            raise FileNotFoundError(f"Resume not found: {path}")

        text = pdf_to_text(str(path))

        # --- Strategy A: semantic_chunk ---
        sem_chunks = semantic_chunk(text)
        sem_ids = [f"{rid}_sem_{i}" for i in range(len(sem_chunks))]
        add_texts(SEMANTIC_COLLECTION, sem_ids, sem_chunks)

        # --- Strategy B: sliding_window_chunk ---
        win_chunks = sliding_window_chunk(text, window=180, overlap=40)
        win_ids = [f"{rid}_win_{i}" for i in range(len(win_chunks))]
        add_texts(WINDOW_COLLECTION, win_ids, win_chunks)

    print("Collections built successfully.\n")


# ================================
# Helper: حساب المتركس لستراتيجي معينة
# ================================
def evaluate_strategy(collection_name: str, queries, top_k: int = 5):
    """
    ترجع list من النتائج (row لكل query)
    """
    all_docs = get_all_documents(collection_name)

    results = []

    for q in queries:
        qid = q["id"]
        query = q["query"]
        answer_substring = (q["answer_substring"] or "").lower().strip()

        # latency
        t0 = time.time()
        res = search(query, top_k=top_k, collection_name=collection_name)
        latency = time.time() - t0

        docs_topk = res["documents"][0] if res["documents"] else []

        if not answer_substring:
            precision = None
            recall = None
            hit_rate = None
            mrr = None
        else:
            # total relevant in whole collection (لـ Recall)
            total_relevant = sum(
                1 for d in all_docs if answer_substring in d.lower()
            )

            # relevant in top-k
            relevant_in_topk = sum(
                1 for d in docs_topk if answer_substring in d.lower()
            )

            # Precision@k
            precision = relevant_in_topk / max(len(docs_topk), 1)

            # Recall@k
            recall = (
                relevant_in_topk / total_relevant
                if total_relevant > 0
                else 0.0
            )

            # Hit Rate@k
            hit_rate = 1.0 if relevant_in_topk > 0 else 0.0

            # MRR
            mrr = 0.0
            for rank, d in enumerate(docs_topk, start=1):
                if answer_substring in d.lower():
                    mrr = 1.0 / rank
                    break

        results.append({
            "query_id": qid,
            "query": query,
            "collection": collection_name,
            "precision": precision,
            "recall": recall,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "latency_sec": latency,
        })

    return results


# ================================
# Helper: كتابة النتائج في CSV
# ================================
def write_results_to_csv(rows, out_path="evaluation_results.csv"):
    fieldnames = [
        "query_id",
        "query",
        "collection",
        "precision",
        "recall",
        "hit_rate",
        "mrr",
        "latency_sec",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved results to {out_path}")


# ================================
# Helper: حساب المتوسط لكل Strategy
# ================================
def summarize_by_collection(rows):
    from collections import defaultdict

    agg = defaultdict(lambda: {
        "precision": [],
        "recall": [],
        "hit_rate": [],
        "mrr": [],
        "latency_sec": [],
    })

    for r in rows:
        coll = r["collection"]
        for key in ["precision", "recall", "hit_rate", "mrr", "latency_sec"]:
            val = r[key]
            if val is not None:
                agg[coll][key].append(val)

    print("\n=== Averages per collection ===")
    for coll, metrics in agg.items():
        print(f"\nCollection: {coll}")
        for key, vals in metrics.items():
            if not vals:
                avg = "N/A"
            else:
                avg = sum(vals) / len(vals)
            print(f"  {key}: {avg}")


# ================================
# main
# ================================
def main():
    # 1) Build semantic + window collections
    build_collections()

    # 2) Evaluate both strategies
    rows_semantic = evaluate_strategy(SEMANTIC_COLLECTION, EVAL_QUERIES, top_k=TOP_K)
    rows_window = evaluate_strategy(WINDOW_COLLECTION, EVAL_QUERIES, top_k=TOP_K)

    all_rows = rows_semantic + rows_window

    # 3) Save CSV
    write_results_to_csv(all_rows)

    # 4) Print averages
    summarize_by_collection(all_rows)


if __name__ == "__main__":
    main()
