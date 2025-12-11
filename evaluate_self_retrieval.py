import numpy as np
from cv_store import cv_collection

k = 5
max_docs = 300

data = cv_collection.get()
ids = data["ids"]
docs = data["documents"]

if max_docs is not None:
    ids = ids[:max_docs]
    docs = docs[:max_docs]

hits = 0
ranks = []

total = len(ids)

for i, (doc_id, doc_text) in enumerate(zip(ids, docs)):
    res = cv_collection.query(
        query_texts=[doc_text],
        n_results=k
    )
    retrieved_ids = res["ids"][0]

    rank = None
    for idx, rid in enumerate(retrieved_ids):
        if rid == doc_id:
            rank = idx + 1
            break

    if rank is not None:
        hits += 1
        ranks.append(1.0 / rank)
    else:
        ranks.append(0.0)

hit_rate = hits / total
precision = hit_rate
recall = hit_rate
f1 = hit_rate
mrr = float(np.mean(ranks))

print("Total docs:", total)
print("Top-k:", k)
print("Hits:", hits)
print("Hit rate (Recall@k):", hit_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("MRR:", mrr)
