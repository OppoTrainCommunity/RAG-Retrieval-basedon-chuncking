# src/eval_retrieval.py
import csv
import time
import chromadb
from chromadb.utils import embedding_functions
import os

# ---------------------------
# Setup Chroma client
# ---------------------------
client = chromadb.PersistentClient(path="./chroma_db")

# 🔥 استخدمي نفس ال embedding اللي استخدمتيه في build_index.py
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# تأكد وجود الـ collections
try:
    col_A = client.get_collection("cv_collection_fixed", embedding_function=embedding_fn)
    col_B = client.get_collection("cv_collection_section", embedding_function=embedding_fn)
except Exception as e:
    print("❌ ERROR: Collections not found. Did you run build_index.py first?")
    raise e


# ---------------------------
# Load queries
# ---------------------------
def load_queries(path="A01_retrieval/eval/queries.csv"):
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append(row)
    print(f"📥 Loaded {len(queries)} queries.")
    return queries


# ---------------------------
# Load gold answers
# ---------------------------
def load_gold(path="A01_retrieval/eval/gold_answers.csv"):
    gold = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold.append(row)
    print(f"📥 Loaded {len(gold)} gold answers.")
    return gold


# ---------------------------
# Relevance check
# ---------------------------
def is_relevant(document_text, expected_answer):
    return expected_answer.lower() in document_text.lower()


# ---------------------------
# Save detailed retrieval trace per query
# ---------------------------
import json

def save_trace(query_id, query, expected, docs, flags, strategy_name):
    os.makedirs("A01_retrieval/eval/traces", exist_ok=True)

    trace = {
        "query_id": query_id,
        "query": query,
        "expected_answer": expected,
        "strategy": strategy_name,
        "retrieved": []
    }

    for rank, (doc, is_corr) in enumerate(zip(docs, flags), start=1):
        trace["retrieved"].append({
            "rank": rank,
            "snippet": doc[:400],  # جزء صغير من النص
            "is_correct": bool(is_corr)
        })

    out_path = f"A01_retrieval/eval/traces/query_{query_id}_{strategy_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=4, ensure_ascii=False)

    print(f"📄 Trace saved → {out_path}")

# ---------------------------
# Evaluation logic
# ---------------------------
def evaluate_strategy(collection, strategy_name, queries, gold_answers, k=5):
    results = []

    gold_map = {g["query_id"]: g for g in gold_answers}

    print(f"\n🚀 Starting evaluation for Strategy {strategy_name}")

    for q in queries:
        qid = q["query_id"]
        question = q["query_text"]

        expected = gold_map[qid]["expected_answer"]

        # Retrieval
        start = time.time()
        res = collection.query(query_texts=[question], n_results=k)
        latency = time.time() - start

        docs = res["documents"][0]

        print(f"\n🔍 Query {qid}: {question}")
        print(f"➡️ Expected: {expected}")

        if len(docs) == 0:
            print("⚠️ No documents returned!")
        else:
            for i, d in enumerate(docs):
                print(f"   - Doc {i+1}: {d[:80]}...")

        # ---------------------------
        # 🟦 Relevance checking
        # ---------------------------
        relevant_flags = [is_relevant(d, expected) for d in docs]
        relevant_count = sum(relevant_flags)

        # Precision@k
        precision = relevant_count / k

        # Recall: بما إنه لكل سؤال يوجد إجابة صحيحة واحدة
        recall = relevant_count / 1  

        # ---------------------------
        # ⭐ F1 Score
        # ---------------------------
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        # Hit rate
        hit_rate = 1 if relevant_count > 0 else 0

        # Reciprocal Rank
        rr = 1 / (relevant_flags.index(True) + 1) if True in relevant_flags else 0

        # ---------------------------
        # 📌 Save TRACE for this query
        # ---------------------------
        save_trace(qid, question, expected, docs, relevant_flags, strategy_name)

        # ---------------------------
        # 📌 Append metrics
        # ---------------------------
        results.append({
            "query_id": qid,
            "query_text": question,
            "strategy": strategy_name,
            "precision@k": precision,
            "recall@k": recall,
            "f1_score": f1,
            "hit_rate@k": hit_rate,
            "rr": rr,
            "latency_sec": latency,
        })

    return results




# ---------------------------
# Save results to CSV
# ---------------------------
def save_results(path, results):
    if not results:
        print("⚠️ No results to save!")
        return

    fieldnames = results[0].keys()

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"📁 Saved → {path}")


# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    queries = load_queries()
    gold = load_gold()

    results_A = evaluate_strategy(col_A, "A", queries, gold)
    save_results("A01_retrieval/eval/results_strategy_A.csv", results_A)

    results_B = evaluate_strategy(col_B, "B", queries, gold)
    save_results("A01_retrieval/eval/results_strategy_B.csv", results_B)

    print("\n🎉 DONE! Check your eval folder for results.\n")
