
from __future__ import annotations
from typing import List, Dict, Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import torch
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore(collection_name: str, persist_dir: str) -> Chroma:
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

def add_texts(
    collection_name: str,
    persist_dir: str,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
) -> None:
    vs = get_vectorstore(collection_name, persist_dir)
    # add_texts expects list lengths aligned
    vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)

def similarity_search(
    collection_name: str,
    persist_dir: str,
    query: str,
    k: int = 8,
    where: Optional[Dict[str, Any]] = None,
):
    vs = get_vectorstore(collection_name, persist_dir)
    # langchain-chroma supports filter via "filter"
    return vs.similarity_search(query=query, k=k, filter=where)

def reset_collection(collection_name: str, persist_dir: str) -> None:
    """
    Delete collection safely
    """
    vs = get_vectorstore(collection_name, persist_dir)
    # underlying client delete collection
    try:
        vs._client.delete_collection(collection_name)  # type: ignore[attr-defined]
    except Exception:
        # if already gone, ignore
        pass
