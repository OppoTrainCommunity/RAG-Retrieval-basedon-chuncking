import nltk
import numpy as np
import pandas as pd
import chromadb
import networkx as nx

from sklearn.metrics.pairwise import cosine_similarity
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


nltk.download("punkt", quiet=True)

embedding_model = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

client = chromadb.PersistentClient(path="resume_chroma_db")


# =========================================================
# CHUNKING METHODS
# =========================================================

# --- Semantic Sentence Chunking ---
def semantic_chunk(text, max_tokens=250):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current.split()) + len(s.split()) > max_tokens:
            if current.strip():
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s

    if current.strip():
        chunks.append(current.strip())

    return chunks


# --- Graph-Based Chunking ---
def graph_chunk(text, similarity_threshold=0.45):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= 1:
        return [text.strip()]

    embeddings = np.array(embedding_model(sentences))
    sim_matrix = cosine_similarity(embeddings)

    G = nx.Graph()
    G.add_nodes_from(range(len(sentences)))

    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sim_matrix[i][j] >= similarity_threshold:
                G.add_edge(i, j)

    components = list(nx.connected_components(G))
    chunks = []

    for comp in components:
        ordered = sorted(list(comp))
        chunk = " ".join(sentences[i] for i in ordered)
        chunks.append(chunk.strip())

    return chunks


# =========================================================
# BATCH INSERT (avoid Chroma limit)
# =========================================================
def batch_iter(items, size=5000):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# =========================================================
# STORE CHUNKS IN SEPARATE COLLECTIONS
# =========================================================
def save_to_chroma(csv_path, mode="semantic"):
    df = pd.read_csv(csv_path)

    if mode == "semantic":
        collection_name = "resume_chunks_semantic"
    elif mode == "graph":
        collection_name = "resume_chunks_graph"
    else:
        raise ValueError("mode must be 'semantic' or 'graph'")

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model
    )

    ids, docs, metas = [], [], []

    print(f"\n[INFO] Processing {mode} chunking...")

    for idx, row in df.iterrows():
        category = str(row["Category"]).strip()
        text = str(row["Resume"]).strip()

        chunks = (
            semantic_chunk(text)
            if mode == "semantic"
            else graph_chunk(text)
        )

        for i, ch in enumerate(chunks):
            ids.append(f"{idx}_{mode}_{i}")
            docs.append(ch)
            metas.append({"category": category, "resume_id": int(idx)})

    # Batch insertion
    for i_b, d_b, m_b in zip(batch_iter(ids), batch_iter(docs), batch_iter(metas)):
        collection.add(ids=i_b, documents=d_b, metadatas=m_b)

    print(f"[INFO] Saved {len(ids)} chunks into {collection_name}.")


# =========================================================
# SIMPLE RETRIEVAL (NO MMR / NO RERANK)
# =========================================================
def retrieve(collection, query, top_k=10):
    res = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    return res["metadatas"][0]


# =========================================================
# METRICS
# =========================================================
def dcg(rel):
    return sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))


def ndcg(rel):
    ideal = sorted(rel, reverse=True)
    ideal_dcg = dcg(ideal)
    return dcg(rel) / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate(csv_path, mode, top_k=10):
    df = pd.read_csv(csv_path)

    collection_name = (
        "resume_chunks_semantic" if mode == "semantic"
        else "resume_chunks_graph"
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model
    )

    cat_count = df["Category"].value_counts().to_dict()

    P_list, R_list, MAP_list, N_list = [], [], [], []

    for _, row in df.iterrows():
        cat = row["Category"]

        retrieved = retrieve(collection, cat, top_k)
        relevance = [1 if m["category"] == cat else 0 for m in retrieved]

        # Precision
        P_list.append(sum(relevance) / top_k)

        # Recall
        R_list.append(sum(relevance) / cat_count[cat])

        # MAP
        avg_p, correct = [], 0
        for i, r in enumerate(relevance):
            if r == 1:
                correct += 1
                avg_p.append(correct / (i+1))
        MAP_list.append(np.mean(avg_p) if avg_p else 0)

        # nDCG
        N_list.append(ndcg(relevance))

    print(f"\n===== {mode.upper()} CHUNKING RESULTS =====")
    print(f"Precision@{top_k}: {np.mean(P_list):.4f}")
    print(f"Recall@{top_k}:    {np.mean(R_list):.4f}")
    print(f"MAP:               {np.mean(MAP_list):.4f}")
    print(f"nDCG:              {np.mean(N_list):.4f}")
    print("===========================================")

# =========================================================
# QUERY FUNCTIONS FOR TESTING
# =========================================================

def query_semantic(category, top_k=5):
    """
    Query the semantic-only collection.
    """
    collection = client.get_or_create_collection(
        name="resume_chunks_semantic",
        embedding_function=embedding_model
    )

    result = collection.query(
        query_texts=[category],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    print("\n===== SEMANTIC RESULTS =====")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"\n--- Result {i+1} ---")
        print(f"Category: {meta['category']}")
        print(f"Resume ID: {meta['resume_id']}")
        print("Chunk:")
        print(doc[:300], "...")  # preview only


def query_graph(category, top_k=5):
    """
    Query the graph-only collection.
    """
    collection = client.get_or_create_collection(
        name="resume_chunks_graph",
        embedding_function=embedding_model
    )

    result = collection.query(
        query_texts=[category],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    docs = result["documents"][0]
    metas = result["metadatas"][0]

    print("\n===== GRAPH RESULTS =====")
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        print(f"\n--- Result {i+1} ---")
        print(f"Category: {meta['category']}")
        print(f"Resume ID: {meta['resume_id']}")
        print("Chunk:")
        print(doc[:300], "...")  # preview only

def test_query():
    """
    Type a category and see retrieval results for both methods.
    """
    while True:
        category = input("\nEnter a category to query (or 'exit'): ").strip()
        if category.lower() == "exit":
            break

        # Query semantic
        query_semantic(category, top_k=5)

        # Query graph
        query_graph(category, top_k=5)

        print("\n======================================================")

if __name__ == "__main__":
    csv_path = "ResumeData_100.csv"

    # Build separate collections
    save_to_chroma(csv_path, mode="semantic")
    save_to_chroma(csv_path, mode="graph")

    # Evaluate separately
    evaluate(csv_path, mode="semantic", top_k=10)
    evaluate(csv_path, mode="graph", top_k=10)
    #QUERY TESTER
    test_query()
