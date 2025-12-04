import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import nltk

# ---------------------------------------
# 1. Setup — Embedding Model + Chroma DB
# ---------------------------------------

# Download tokenizer for chunking
nltk.download('punkt', quiet=True)

# Embedding function (SentenceTransformer inside Chroma)
embedding = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Chroma persistent database
client = chromadb.PersistentClient(path="resume_chroma_db")

# Collection that stores all resume chunks
collection = client.get_or_create_collection(
    name="resume_chunks",
    embedding_function=embedding
)

# ---------------------------------------
# 2. Semantic Chunking
# ---------------------------------------

def semantic_chunk(text, max_tokens=200):
    """
    Splits resume text into semantically meaningful sentence-based chunks.
    """
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) > max_tokens:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ---------------------------------------
# 3. Load CSV → Chunk → Embed → Save
# ---------------------------------------

def save_csv_to_chroma(csv_path):
    """
    Loads your CSV file with columns:
        - Category
        - Resume
    Then chunks, embeds, and stores everything in ChromaDB.
    """
    
    df = pd.read_csv(csv_path)

    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError("CSV must contain 'Category' and 'Resume' columns.")

    all_ids = []
    all_docs = []
    all_metas = []

    for idx, row in df.iterrows():
        category = str(row["Category"]).strip()
        resume_text = str(row["Resume"]).strip()

        # 1. Chunk resume
        chunks = semantic_chunk(resume_text)

        # 2. Save each chunk separately
        for i, chunk in enumerate(chunks):
            chunk_id = f"{idx}_chunk{i}"

            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_metas.append({
                "category": category,
                "resume_id": idx
            })

    # Save in vector DB
    collection.add(
        ids=all_ids,
        documents=all_docs,
        metadatas=all_metas
    )

    print(f"[INFO] Saved {len(all_ids)} chunks from {csv_path} into ChromaDB.")


# ---------------------------------------
# 4. Run Manually
# ---------------------------------------

if __name__ == "__main__":
    csv_path = "UpdatedResumeDataSet.csv"   
    save_csv_to_chroma(csv_path)
