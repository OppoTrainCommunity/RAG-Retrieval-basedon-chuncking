import json
import logging
import re
from collections import defaultdict
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService

logger = logging.getLogger("cv_analyzer.job_matcher")

MATCHING_PROMPT = """You are an expert HR recruiter. Analyze how well each candidate matches the given job description.

Job Description:
{job_description}

Candidate Profiles:
{candidates_context}

For each candidate, provide a JSON array with this structure:
[
  {{
    "candidate_name": "Name",
    "cv_id": "id",
    "match_score": 85,
    "key_matching_points": ["point1", "point2"],
    "missing_qualifications": ["missing1"],
    "explanation": "Brief explanation of fit"
  }}
]

Score from 0-100 based on skills match, experience relevance, and overall fit.
Return ONLY the JSON array, no other text."""


class JobMatcher:
    def __init__(self, vector_store: VectorStoreService, llm_service: LLMService) -> None:
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.rag_service = RAGService(vector_store, llm_service)

    async def match(self, job_description: str, model_id: Optional[str] = None, top_k: int = 5) -> list[dict]:
        all_cvs = self.vector_store.get_all_cvs()
        if not all_cvs:
            return []

        # Hybrid search for relevant chunks
        search_results = self.rag_service._hybrid_search(job_description, top_k=20)

        # Build per-candidate context
        candidate_chunks: dict[str, list[str]] = defaultdict(list)
        cv_lookup = {cv["cv_id"]: cv for cv in all_cvs}

        for result in search_results:
            cv_id = result["metadata"].get("cv_id", "")
            if cv_id and len(candidate_chunks[cv_id]) < 3:
                candidate_chunks[cv_id].append(result["text"])

        # Build context for LLM
        context_parts = []
        for cv in all_cvs:
            cv_id = cv["cv_id"]
            parts = [f"### Candidate: {cv['candidate_name']} (ID: {cv_id})"]
            parts.append(f"Skills: {', '.join(cv.get('skills', []))}")
            parts.append(f"Experience: {cv.get('years_of_experience', 0)} years")

            education = cv.get("education", [])
            if education:
                edu_strs = [f"{e.get('degree', '')} from {e.get('institution', '')}" for e in education]
                parts.append(f"Education: {'; '.join(edu_strs)}")

            certs = cv.get("certifications", [])
            if certs:
                parts.append(f"Certifications: {', '.join(certs)}")

            # Add relevant chunks
            if cv_id in candidate_chunks:
                parts.append("Relevant excerpts:")
                parts.extend(candidate_chunks[cv_id])

            context_parts.append("\n".join(parts))

        candidates_context = "\n\n---\n\n".join(context_parts)

        try:
            return await self._llm_match(job_description, candidates_context, all_cvs, model_id, top_k)
        except Exception as e:
            logger.warning("LLM matching failed: %s. Using keyword fallback.", e)
            return self._keyword_fallback(job_description, all_cvs, top_k)

    async def _llm_match(
        self, job_description: str, candidates_context: str,
        all_cvs: list[dict], model_id: Optional[str], top_k: int
    ) -> list[dict]:
        llm = self.llm_service.get_model(model_id)
        prompt = ChatPromptTemplate.from_template(MATCHING_PROMPT)
        chain = prompt | llm

        response = await chain.ainvoke({
            "job_description": job_description,
            "candidates_context": candidates_context,
        })

        content = response.content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        results = json.loads(content)

        # Add skills_match
        job_desc_lower = job_description.lower()
        cv_lookup = {cv["cv_id"]: cv for cv in all_cvs}

        for result in results:
            cv_id = result.get("cv_id", "")
            cv = cv_lookup.get(cv_id, {})
            skills = cv.get("skills", [])
            result["skills_match"] = [
                {"skill": s, "matched": s.lower() in job_desc_lower}
                for s in skills
            ]

        # Sort by score, return top_k
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return results[:top_k]

    def _keyword_fallback(self, job_description: str, all_cvs: list[dict], top_k: int) -> list[dict]:
        job_words = set(job_description.lower().split())
        results = []

        for cv in all_cvs:
            skills = cv.get("skills", [])
            matched = [s for s in skills if s.lower() in job_description.lower()]
            score = min(100, int(len(matched) / max(len(skills), 1) * 100)) if skills else 0

            results.append({
                "candidate_name": cv.get("candidate_name", "Unknown"),
                "cv_id": cv.get("cv_id", ""),
                "match_score": score,
                "key_matching_points": [f"Has skill: {s}" for s in matched],
                "missing_qualifications": [],
                "skills_match": [
                    {"skill": s, "matched": s.lower() in job_description.lower()}
                    for s in skills
                ],
                "explanation": f"Keyword-based matching: {len(matched)}/{len(skills)} skills match",
            })

        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:top_k]
