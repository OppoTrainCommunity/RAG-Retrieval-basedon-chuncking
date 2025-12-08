import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# =========================
# Model + Chroma client
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# قاعدة بيانات ثابتة على القرص
client = chromadb.PersistentClient(path="./resume_db")


def _create_collection(name: str):
    return client.create_collection(
        name,
        metadata={"hnsw:space": "cosine"}
    )


def get_collection(name: str = "resumes"):
    """
    Get or create a collection by name.
    Default collection: 'resumes' (اللي بيستخدمه app.py)
    """
    try:
        return client.get_collection(name)
    except Exception:
        return client.get_or_create_collection(
            name,
            metadata={"hnsw:space": "cosine"}
        )


def reset_collection(name: str):
    """
    يُستخدم في التقييم: نعيد إنشاء الكلكشن من الصفر.
    """
    try:
        client.delete_collection(name)
    except Exception:
        # لو مش موجود، عادي
        pass
    return _create_collection(name)


def add_texts(collection_name: str, ids, texts):
    """
    إضافة مجموعة نصوص دفعة واحدة في كلكشن معيّن.
    ids: list[str]
    texts: list[str]
    """
    embeddings = model.encode(texts).tolist()
    coll = get_collection(collection_name)
    coll.add(ids=ids, documents=texts, embeddings=embeddings)


# =========================
# Backward-compatible API
# (حتى ما نخرب app.py)
# =========================

def save_resume(name, text, collection_name: str = "resumes"):
    add_texts(collection_name, [name], [text])


def save_chunk(chunk_id, text, collection_name: str = "resumes"):
    add_texts(collection_name, [chunk_id], [text])


def search(query_text, top_k: int = 5, collection_name: str = "resumes"):
    embedding = model.encode([query_text]).tolist()
    coll = get_collection(collection_name)

    results = coll.query(
        query_embeddings=embedding,
        n_results=top_k
    )

    return {
        "documents": results.get("documents", []),   # list[list[str]]
        "ids": results.get("ids", []),              # list[list[str]]
        "scores": results.get("distances", [])      # list[list[float]]
    }


def get_all_documents(collection_name: str = "resumes"):
    """
    استرجاع كل الدوكيومنتس في كلكشن معين (لاستخدامها في حساب Recall).
    """
    coll = get_collection(collection_name)
    data = coll.get(include=["documents"])
    # data["documents"] → list[str]
    return data.get("documents", [])
