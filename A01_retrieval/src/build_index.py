import chromadb
from chromadb.utils import embedding_functions
from chunking import chunk_fixed, chunk_by_section
from parse_pdf import load_cvs

client = chromadb.PersistentClient(path="./chroma_db")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def build_collection_strategy_A(docs):
    col = client.get_or_create_collection(
        name="cv_collection_fixed",
        embedding_function=embedding_fn
    )

    ids = []
    texts = []
    metadatas = []

    for doc in docs:
        candidate = doc["candidate"]
        chunks = chunk_fixed(doc["text"])
        for idx, ch in enumerate(chunks):
            ids.append(f"{candidate}_A_{idx}")
            texts.append(ch)
            metadatas.append({
                "candidate": candidate,
                "strategy": "A"
            })

    col.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"🔥 Strategy A inserted {len(ids)} chunks.")


def build_collection_strategy_B(docs):
    col = client.get_or_create_collection(
        name="cv_collection_section",
        embedding_function=embedding_fn
    )

    ids = []
    texts = []
    metadatas = []

    for doc in docs:
        candidate = doc["candidate"]
        sections = chunk_by_section(doc["text"])
        for idx, sec in enumerate(sections):
            ids.append(f"{candidate}_B_{idx}")
            texts.append(sec["text"])
            metadatas.append({
                "candidate": candidate,
                "strategy": "B",
                "section": sec["section"]
            })

    col.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"🔥 Strategy B inserted {len(ids)} chunks.")


if __name__ == "__main__":
    docs = load_cvs()
    print(f"📄 Loaded {len(docs)} CV files.")

    build_collection_strategy_A(docs)
    build_collection_strategy_B(docs)

    print("✅ Finished building index with both chunking strategies.")
