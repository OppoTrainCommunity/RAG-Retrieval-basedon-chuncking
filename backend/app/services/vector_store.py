import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastembed import TextEmbedding

from app.config import settings

logger = logging.getLogger("cv_analyzer.vector_store")

# Use a local cache directory to avoid Windows symlink issues with HF Hub
_FASTEMBED_CACHE = str(Path(settings.chroma_persist_dir).parent / "fastembed_models")


class VectorStoreService:
    def __init__(self) -> None:
        logger.info("Initializing ChromaDB at %s", settings.chroma_persist_dir)
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.chunks_collection = self.client.get_or_create_collection(
            name="cv_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        self.metadata_collection = self.client.get_or_create_collection(
            name="cv_metadata",
        )

        logger.info("Loading FastEmbed model (all-MiniLM-L6-v2)...")
        os.makedirs(_FASTEMBED_CACHE, exist_ok=True)
        self.embedding_model = TextEmbedding(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=_FASTEMBED_CACHE,
        )
        logger.info("FastEmbed model loaded")

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self.embedding_model.embed(texts)]

    def add_cv(
        self,
        cv_id: str,
        candidate_name: str,
        full_text: str,
        chunks: list[str],
        metadata: dict,
    ) -> None:
        # Store metadata (JSON-stringify lists)
        meta_flat = {
            "candidate_name": metadata.get("candidate_name", candidate_name),
            "skills_json": json.dumps(metadata.get("skills", [])),
            "years_of_experience": metadata.get("years_of_experience", 0),
            "education_json": json.dumps(metadata.get("education", [])),
            "certifications_json": json.dumps(metadata.get("certifications", [])),
            "email": metadata.get("email", "") or "",
            "phone": metadata.get("phone", "") or "",
            "summary": metadata.get("summary", ""),
            "file_hash": metadata.get("file_hash", ""),
            "filename": metadata.get("filename", ""),
            "upload_date": metadata.get("upload_date", datetime.now(timezone.utc).isoformat()),
        }

        self.metadata_collection.upsert(
            ids=[cv_id],
            documents=[full_text[:5000]],
            metadatas=[meta_flat],
        )

        # Store chunks with embeddings
        if chunks:
            chunk_ids = [f"{cv_id}_chunk_{i}" for i in range(len(chunks))]
            embeddings = self.generate_embeddings(chunks)
            chunk_metas = [
                {"cv_id": cv_id, "candidate_name": candidate_name, "chunk_index": i}
                for i in range(len(chunks))
            ]

            self.chunks_collection.upsert(
                ids=chunk_ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=chunk_metas,
            )

        logger.info("Stored CV %s with %d chunks", cv_id, len(chunks))

    def get_all_cvs(self) -> list[dict]:
        result = self.metadata_collection.get(include=["metadatas", "documents"])
        cvs = []
        for i, cv_id in enumerate(result["ids"]):
            meta = result["metadatas"][i]
            cvs.append({
                "cv_id": cv_id,
                "candidate_name": meta.get("candidate_name", "Unknown"),
                "skills": json.loads(meta.get("skills_json", "[]")),
                "years_of_experience": meta.get("years_of_experience", 0),
                "education": json.loads(meta.get("education_json", "[]")),
                "certifications": json.loads(meta.get("certifications_json", "[]")),
                "email": meta.get("email", ""),
                "phone": meta.get("phone", ""),
                "summary": meta.get("summary", ""),
                "file_hash": meta.get("file_hash", ""),
                "filename": meta.get("filename", ""),
                "upload_date": meta.get("upload_date", ""),
            })
        return cvs

    def get_cv(self, cv_id: str) -> Optional[dict]:
        result = self.metadata_collection.get(ids=[cv_id], include=["metadatas", "documents"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {
            "cv_id": cv_id,
            "candidate_name": meta.get("candidate_name", "Unknown"),
            "skills": json.loads(meta.get("skills_json", "[]")),
            "years_of_experience": meta.get("years_of_experience", 0),
            "education": json.loads(meta.get("education_json", "[]")),
            "certifications": json.loads(meta.get("certifications_json", "[]")),
            "email": meta.get("email", ""),
            "phone": meta.get("phone", ""),
            "summary": meta.get("summary", ""),
            "file_hash": meta.get("file_hash", ""),
            "filename": meta.get("filename", ""),
            "upload_date": meta.get("upload_date", ""),
        }

    def delete_cv(self, cv_id: str) -> bool:
        existing = self.metadata_collection.get(ids=[cv_id])
        if not existing["ids"]:
            return False

        self.metadata_collection.delete(ids=[cv_id])

        # Delete all chunks for this CV
        chunks = self.chunks_collection.get(
            where={"cv_id": cv_id}, include=["metadatas"]
        )
        if chunks["ids"]:
            self.chunks_collection.delete(ids=chunks["ids"])

        logger.info("Deleted CV %s and its chunks", cv_id)
        return True

    def check_hash_exists(self, file_hash: str) -> bool:
        all_cvs = self.metadata_collection.get(include=["metadatas"])
        for meta in all_cvs["metadatas"]:
            if meta.get("file_hash") == file_hash:
                return True
        return False

    def vector_search(self, query: str, n_results: int = 10) -> dict:
        embeddings = self.generate_embeddings([query])
        results = self.chunks_collection.query(
            query_embeddings=embeddings,
            n_results=min(n_results, self.chunks_collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        return results

    def get_all_chunks(self) -> dict:
        count = self.chunks_collection.count()
        if count == 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return self.chunks_collection.get(include=["documents", "metadatas"])

    def get_total_chunks(self) -> int:
        return self.chunks_collection.count()

    def get_total_cvs(self) -> int:
        return self.metadata_collection.count()
