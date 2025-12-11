import json
import numpy as np
from cv_store import cv_collection

K = 5

y_true = []
y_pred = []
rr_list = []

with open("rag_queries.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)
        cv_id = item["cv_id"]
        question = item["question"]

        results = cv_collection.query(
            query_texts=[question],
            n_results=K
        )

        retrieved_ids = results["ids"][0]

        correct_hit = 0
        hit_rank = None

        for idx, rid in enumerate(retrieved_ids):
            if rid.startswith(cv_id + "_chunk_"):
                correct_hit = 1
                hit_rank = idx
                break

        y_true.append(1)
        y_pred.append(correct_hit)

        if hit_rank is not None:
            rr_list.append(1.0 / (hit_rank + 1))
        else:
            rr_list.append(0.0)

y_true = np.array(y_true, dtype=float)
y_pred = np.array(y_pred, dtype=float)

tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))

precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
mrr = float(np.mean(rr_list)) if rr_list else 0.0

print("RAG Retrieval Evaluation:")
print("Top-K:", K)
print("Num queries:", len(y_true))
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("MRR:", mrr)
