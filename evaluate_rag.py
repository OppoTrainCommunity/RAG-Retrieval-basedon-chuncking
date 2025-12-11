import json
from cv_store import query_single_cv
import numpy as np

eval_file = "eval_data.jsonl"

y_true = []
y_pred = []

with open(eval_file, "r", encoding="utf8") as f:
    for line in f:
        item = json.loads(line)
        cv_id = item["cv_id"]
        question = item["question"]
        true_answer = item["answer"].lower()

        result = query_single_cv(question, cv_id, n_results=5)
        retrieved_chunks = result["documents"][0]
        retrieved_text = " ".join(retrieved_chunks).lower()

        if true_answer in retrieved_text:
            y_pred.append(1)
        else:
            y_pred.append(0)

        y_true.append(1)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

tp = np.sum((y_true == 1) & (y_pred == 1))
fp = np.sum((y_true == 0) & (y_pred == 1))
fn = np.sum((y_true == 1) & (y_pred == 0))

precision = tp / (tp + fp + 1e-10)
recall = tp / (tp + fn + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
