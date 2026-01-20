# resume_rag/llm_chain.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import json

from .vector_store_chain import similarity_search
from .openrouter_llm import call_openrouter_chat

SYSTEM = """You are a strict resume QA assistant.

Rules:
1) Use ONLY the provided Context. Do NOT use outside knowledge.
2) If a detail is not explicitly in the Context, write: Not found in the resume.
3) Do NOT merge multiple resumes into one person.
4) If scope is ALL resumes, answer PER resume_id separately.
5) Always include short evidence quotes (<=20 words) copied from the Context for each claim.
6) Return JSON ONLY (no extra text).
"""

def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        rid = d.metadata.get("resume_id", "unknown")
        fn = d.metadata.get("filename", "unknown")
        cid = d.metadata.get("chunk_id", "?")
        txt = (d.page_content or "").strip()
        parts.append(f"RESUME_ID={rid} | FILE={fn} | CHUNK={cid}\n{txt}")
    return "\n\n---\n\n".join(parts)

def retrieve_docs(
    collection_name: str,
    persist_dir: str,
    query: str,
    k: int,
    resume_id: Optional[str] = None,
):
    where = {"resume_id": resume_id} if resume_id else None
    return similarity_search(collection_name, persist_dir, query=query, k=k, where=where)

def _safe_json(text: str) -> Dict[str, Any]:
    # حاول نقرأ JSON حتى لو model رجع نص + JSON
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
    # find first { last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text)

def answer_one_resume(
    collection_name: str,
    persist_dir: str,
    resume_id: str,
    question: str,
    model: str,
    k: int = 8,
) -> Dict[str, Any]:
    docs = retrieve_docs(collection_name, persist_dir, query=question, k=k, resume_id=resume_id)
    context = _format_docs(docs)

    prompt = f"""Question: {question}

Context:
{context}

Return JSON in exactly this shape:
{{
  "resume_id": "{resume_id}",
  "name": "<name or 'Not found in the resume.'>",
  "email": "<email or 'Not found in the resume.'>",
  "answer": "<direct answer>",
  "evidence": ["<quote1>", "<quote2>"]
}}"""

    out = call_openrouter_chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return _safe_json(out)

def answer_all_resumes(
    collection_name: str,
    persist_dir: str,
    question: str,
    model: str,
    k: int = 10,
    candidate_limit: int = 8,
) -> Dict[str, Any]:
    """
    1) retrieve top docs across ALL resumes
    2) extract distinct resume_ids
    3) for each resume_id do a focused retrieval + answer
    """
    seed_docs = retrieve_docs(collection_name, persist_dir, query=question, k=k, resume_id=None)
    seen = []
    for d in seed_docs:
        rid = d.metadata.get("resume_id")
        if rid and rid not in seen:
            seen.append(rid)
    seen = seen[:candidate_limit]

    results = []
    for rid in seen:
        results.append(answer_one_resume(collection_name, persist_dir, rid, question, model, k=k))

    return {"scope": "all", "question": question, "results": results}

def compare_two_models(
    collection_name: str,
    persist_dir: str,
    scope: str,
    question: str,
    model1: str,
    model2: str,
    k: int = 8,
    resume_id: Optional[str] = None,
) -> Dict[str, Any]:
    scope = (scope or "all").lower()

    if scope == "one":
        if not resume_id:
            raise ValueError("resume_id is required when scope='one'")

        a1 = answer_one_resume(collection_name, persist_dir, resume_id, question, model1, k=k)
        a2 = answer_one_resume(collection_name, persist_dir, resume_id, question, model2, k=k)
        return {"scope": "one", "resume_id": resume_id, "model1": a1, "model2": a2}

    # all
    a1 = answer_all_resumes(collection_name, persist_dir, question, model1, k=k)
    a2 = answer_all_resumes(collection_name, persist_dir, question, model2, k=k)
    return {"scope": "all", "model1": a1, "model2": a2}
