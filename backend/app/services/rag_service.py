import logging
from collections import defaultdict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

logger = logging.getLogger("cv_analyzer.rag_service")

RAG_SYSTEM_PROMPT = """You are an AI assistant that answers questions about job candidates based on their CV/resume data.

RULES:
- Answer ONLY based on the provided CV context below
- Always mention candidates by name
- When comparing candidates, use structured formatting (tables, bullet points)
- If the information is not available in the context, politely state that
- Be precise and factual

CV Context:
{context}

Question: {question}"""


class RAGService:
    def __init__(self, vector_store: VectorStoreService, llm_service: LLMService) -> None:
        self.vector_store = vector_store
        self.llm_service = llm_service

    def _hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Combine vector search + BM25 via Reciprocal Rank Fusion."""
        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks["ids"]:
            return []

        documents = all_chunks["documents"]
        metadatas = all_chunks["metadatas"]
        chunk_ids = all_chunks["ids"]

        # Vector search
        vector_results = self.vector_store.vector_search(query, n_results=top_k)
        vector_ranked = {}
        if vector_results["ids"] and vector_results["ids"][0]:
            for rank, cid in enumerate(vector_results["ids"][0]):
                vector_ranked[cid] = rank + 1

        # BM25 search
        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.lower().split())

        bm25_ranked = {}
        sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        for rank, idx in enumerate(sorted_indices[:top_k]):
            bm25_ranked[chunk_ids[idx]] = rank + 1

        # RRF fusion: score = Σ weight / (k + rank)
        k = 60
        vector_weight = 0.6
        bm25_weight = 0.4
        fused_scores: dict[str, float] = defaultdict(float)

        all_candidate_ids = set(vector_ranked.keys()) | set(bm25_ranked.keys())
        for cid in all_candidate_ids:
            if cid in vector_ranked:
                fused_scores[cid] += vector_weight / (k + vector_ranked[cid])
            if cid in bm25_ranked:
                fused_scores[cid] += bm25_weight / (k + bm25_ranked[cid])

        # Sort by fused score descending
        sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result list
        id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        results = []
        for cid, score in sorted_chunks:
            idx = id_to_idx.get(cid)
            if idx is not None:
                results.append({
                    "chunk_id": cid,
                    "text": documents[idx],
                    "metadata": metadatas[idx],
                    "score": score,
                })

        return results

    def _build_context(self, search_results: list[dict]) -> tuple[str, list[dict]]:
        """Group results by candidate and build context string."""
        candidates: dict[str, list[str]] = defaultdict(list)
        source_map: dict[str, dict] = {}

        for result in search_results:
            name = result["metadata"].get("candidate_name", "Unknown")
            cv_id = result["metadata"].get("cv_id", "")
            candidates[name].append(result["text"])

            if cv_id not in source_map:
                source_map[cv_id] = {
                    "cv_id": cv_id,
                    "name": name,
                    "relevance": result["score"],
                    "snippet": result["text"][:200],
                }

        context_parts = []
        for name, texts in candidates.items():
            context_parts.append(f"[Candidate: {name}]")
            context_parts.extend(texts)
            context_parts.append("---")

        context = "\n".join(context_parts)
        sources = list(source_map.values())
        return context, sources

    async def query(self, question: str, model_id: Optional[str] = None) -> dict:
        """Full RAG pipeline: hybrid search → build context → LLM generate."""
        search_results = self._hybrid_search(question)

        if not search_results:
            return {
                "response": "No CVs found in the database. Please upload some CVs first.",
                "sources": [],
                "model_used": model_id or "none",
            }

        context, sources = self._build_context(search_results)

        try:
            llm = self.llm_service.get_model(model_id)
            prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
            chain = prompt | llm

            response = await chain.ainvoke({"context": context, "question": question})
            return {
                "response": response.content,
                "sources": sources,
                "model_used": model_id or self.llm_service.get_default_models()[0]["id"],
            }
        except Exception as e:
            logger.error("LLM query failed: %s", e)
            return {
                "response": f"Error generating response: {str(e)}",
                "sources": sources,
                "model_used": model_id or "error",
            }
