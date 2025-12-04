# evaluation.py
# Evaluates retrieval quality using your indexed Chroma vector DB.
# Metrics: Precision@k, Recall@k, MAP, nDCG

import pandas as pd
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------
# 1. Connect to ChromaDB + Embedding
# ---------------------------------------

embedding = SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="resume_chroma_db")

collection = client.get_or_create_collection(
    name="resume_chunks",
    embedding_function=embedding
)

# ---------------------------------------
# 2. Helper: Query top-k chunks
# ---------------------------------------

def retrieve_chunks(query, top_k=10):
    result = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return result["metadatas"][0]   # list of metadata dicts


# ---------------------------------------
# 3. Metric Helpers: DCG, nDCG
# ---------------------------------------

def dcg(relevance):
    return sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(relevance))

def ndcg(relevance):
    ideal = sorted(relevance, reverse=True)
    ideal_dcg = dcg(ideal)
    return (dcg(relevance) / ideal_dcg) if ideal_dcg > 0 else 0


# ---------------------------------------
# 4. Main Evaluation Function
# ---------------------------------------

def evaluate(csv_path, top_k=10):
    df = pd.read_csv(csv_path)

    # total relevant resumes for each category
    category_counts = df["Category"].value_counts().to_dict()

    precision_scores = []
    recall_scores = []
    map_scores = []
    ndcg_scores = []

    for idx, row in df.iterrows():
        true_category = row["Category"]
        query = true_category

        # Retrieve k chunks
        retrieved = retrieve_chunks(query, top_k)

        # 1 = relevant, 0 = not relevant
        relevance = [1 if meta["category"] == true_category else 0
                     for meta in retrieved]

        # ---- Precision@k ----
        precision = sum(relevance) / top_k
        precision_scores.append(precision)

        # ---- Recall@k ----
        recall = sum(relevance) / category_counts[true_category]
        recall_scores.append(recall)

        # ---- MAP ----
        avg_precisions = []
        correct = 0
        for i, rel in enumerate(relevance):
            if rel == 1:
                correct += 1
                avg_precisions.append(correct / (i+1))
        map_score = np.mean(avg_precisions) if avg_precisions else 0
        map_scores.append(map_score)

        # ---- nDCG ----
        ndcg_score = ndcg(relevance)
        ndcg_scores.append(ndcg_score)

    print("\n=========== RETRIEVAL EVALUATION RESULTS ===========")
    print(f"Precision@{top_k}: {np.mean(precision_scores):.4f}")
    print(f"Recall@{top_k}:    {np.mean(recall_scores):.4f}")
    print(f"MAP:               {np.mean(map_scores):.4f}")
    print(f"nDCG:              {np.mean(ndcg_scores):.4f}")
    print("====================================================")


# ---------------------------------------
# 5. Manual Run
# ---------------------------------------

if __name__ == "__main__":
    csv_path = "UpdatedResumeDataSet.csv"   
    evaluate(csv_path, top_k=10)
