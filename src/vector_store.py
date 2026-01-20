"""
Vector Store Module: Manages Embeddings and ChromaDB interactions.
"""
import time
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from openai import OpenAI
from . import config

# Singleton Client
_chroma_client = None

def get_chroma_client() -> chromadb.PersistentClient:
    """Get or create singleton ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _chroma_client

def initialize_collection(collection_name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"}
    )

def get_collection_name(model: str, strategy: str) -> str:
    """Generate specific collection name based on config."""
    return f"{model.replace('/', '_').replace('-', '_')}_{strategy}"

# ============================================================
# Caching & Embeddings
# ============================================================
def get_cache_key(text: str, model: str) -> str:
    combined = f"{model}:{text}"
    return hashlib.sha256(combined.encode()).hexdigest()

def get_cache_path(cache_key: str) -> config.Path:
    return config.CACHE_DIR / f"{cache_key}.json"

def load_from_cache(cache_key: str) -> Optional[List[float]]:
    cache_path = get_cache_path(cache_key)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                return json.load(f).get('embedding')
        except:
            return None
    return None

def save_to_cache(cache_key: str, embedding: List[float], model: str):
    try:
        with open(get_cache_path(cache_key), 'w') as f:
            json.dump({
                'embedding': embedding, 
                'model': model, 
                'timestamp': datetime.now().isoformat()
            }, f)
    except:
        pass

def get_embedding(text: str, model: str, api_key: str, use_cache: bool = True) -> List[float]:
    """Fetch embedding from OpenRouter with caching and retries."""
    if use_cache:
        cache_key = get_cache_key(text, model)
        cached = load_from_cache(cache_key)
        if cached:
            return cached
    
    client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=api_key)
    for attempt in range(config.MAX_RETRIES):
        try:
            response = client.embeddings.create(model=model, input=text)
            embedding = response.data[0].embedding
            if use_cache:
                save_to_cache(cache_key, embedding, model)
            return embedding
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to get embedding after {config.MAX_RETRIES} attempts: {e}")

# ============================================================
# Indexing & Retrieval
# ============================================================
def index_documents(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    source_file: str,
    collection: chromadb.Collection,
    embedding_model: str,
    chunking_strategy: str
) -> int:
    """Index a batch of documents into ChromaDB."""
    if not chunks or not embeddings:
        return 0
    
    ids = []
    documents = []
    metadatas = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{source_file}_{chunking_strategy}_{i}"
        
        ids.append(chunk_id)
        documents.append(chunk["text"])
        
        metadata = {
            "source_file": source_file,
            "chunk_index": chunk.get("chunk_index", i),
            "chunking_strategy": chunking_strategy,
            "embedding_model": embedding_model,
            "indexed_at": datetime.now().isoformat()
        }
        metadatas.append(metadata)
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    return len(ids)

def retrieve_similar_chunks(
    query: str,
    collection: chromadb.Collection,
    embedding_model: str,
    api_key: str,
    top_k: int = config.DEFAULT_TOP_K
) -> List[Dict[str, Any]]:
    """Retrieve top-k similar chunks for a query."""
    query_embedding = get_embedding(query, embedding_model, api_key)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    formatted = []
    if results and results['ids'] and len(results['ids']) > 0:
        for i in range(len(results['ids'][0])):
            formatted.append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i] if results['documents'] else "",
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "similarity": 1 - results['distances'][0][i] if results['distances'] else 1
            })
    
    return formatted
