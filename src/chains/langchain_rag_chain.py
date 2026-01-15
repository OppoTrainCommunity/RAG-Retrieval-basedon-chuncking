import os
from typing import Any, Dict, Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_openai import ChatOpenAI

from src.services.retrieval_service import RetrievalService
from .base import BaseRAGChain


def _format_docs(docs: List[str]) -> str:
    return "\n\n---\n\n".join(docs)


class LangChainRAGChain(BaseRAGChain):
    """
    LangChain chain that REUSES your RetrievalService (ChromaIndexer) and uses OpenRouter LLM.

    Chain: Prompt -> retrieval -> LLM -> output
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        collection_name: str = "paragraph_chunking",
        model: str = "meta-llama/llama-3.1-8b-instruct",
        temperature: float = 0.2,
        max_tokens: int = 900,
    ):
        self.retrieval_service = retrieval_service
        self.collection_name = collection_name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY (env var)")

        # OpenRouter is OpenAI-compatible base URL
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a technical recruitment assistant.

                    Goal:
                    Given a job/project requirement and resume evidence retrieved from multiple candidates, rank the best candidates.

                    Rules:
                    - Use ONLY the provided resume evidence (context). Do NOT invent skills, experience, or education.
                    - Every claim MUST be backed by evidence and include file_name + chunk_id.
                    - If something is not found in the evidence, say “not found”.
                    - Be fair: do not assume years of experience unless explicitly stated.
                    - Output ONLY valid JSON (no markdown, no extra text).""",
                ),
                (
                    "user",
                    """
                        Job requirements:
                        Role title: {role_title}
                        Must-have skills: {must_have}
                        Nice-to-have skills: {nice_to_have}
                        Seniority: {seniority}
                        Notes: {job_notes}

                        Recruitment question:
                        {question}

                        Resume evidence (retrieved chunks across multiple candidates):
                        {context}

                        Task:
                        Rank the best candidates for this role.
                        Return top {top_n} candidates (or fewer if evidence is weak).

                        Every score and claim MUST cite evidence with file_name and chunk_id.

                        Return ONLY JSON with this schema:
                        {{
                            "job": {{
                            "role_title": "string",
                            "must_have": ["string"],
                            "nice_to_have": ["string"],
                            "seniority": "intern/junior/mid/senior/any",
                            "notes": "string"
                            }},
                            "ranked_candidates": [
                            {{
                                "file_name": "string",
                                "overall_score": 0,
                                "score_breakdown": {{
                                "skills_match": 0,
                                "project_relevance": 0,
                                "experience_level": 0,
                                "communication_clarity": 0
                                }},
                                "highlights": ["string"],
                                "gaps": ["string"],
                                "evidence": [
                                {{
                                    "file_name": "string",
                                    "chunk_id": 0,
                                    "snippet": "string",
                                    "why_it_matters": "string"
                                }}
                                ],
                                "recommendation": "strong_interview / interview / maybe / no",
                                "targeted_interview_questions": ["string"]
                            }}
                            ],
                            "final_notes": {{
                            "missing_info": ["string"],
                            "tie_breakers_used": ["string"]
                            }}
                        }}
                        """.strip(),
                ),
            ]
        )

        # Runnable retrieval step (uses your RetrievalService.search)
        def retrieval_step(inputs: Dict[str, Any]) -> str:
            question = inputs["question"]
            where = inputs.get("where")
            k = int(inputs.get("k", 40))

            results = self.retrieval_service.search(
                collection_name=self.collection_name,
                query=question,
                k=k,
                where=where,
            )

            docs = results.get("documents", [[]])[0] if results else []
            metas = results.get("metadatas", [[]])[0] if results else []

            parts = []
            for doc, meta in zip(docs, metas):
                parts.append(
                    f'[file_name="{meta.get("file_name","unknown")}" chunk_id={meta.get("chunk_id","unknown")}]\n{doc}'
                )
            return "\n\n---\n\n".join(parts)

        self.chain = (
            {
                "context": RunnableLambda(retrieval_step),
                "question": RunnableLambda(lambda x: x["question"]),
                "role_title": RunnableLambda(lambda x: x["role_title"]),
                "must_have": RunnableLambda(lambda x: x["must_have"]),
                "nice_to_have": RunnableLambda(lambda x: x["nice_to_have"]),
                "seniority": RunnableLambda(lambda x: x.get("seniority", "any")),
                "job_notes": RunnableLambda(lambda x: x.get("job_notes", "")),
                "top_n": RunnableLambda(lambda x: x.get("top_n", 5)),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(
        self,
        question: str,
        role_title: str,
        must_have: list,
        nice_to_have: list,
        seniority: str = "any",
        job_notes: str = "",
        top_n: int = 5,
        where: Optional[Dict[str, Any]] = None,
        k: int = 40,
    ) -> str:
        payload = {
            "question": question,
            "role_title": role_title,
            "must_have": must_have,
            "nice_to_have": nice_to_have,
            "seniority": seniority,
            "job_notes": job_notes,
            "top_n": top_n,
            "where": where,
            "k": k,
        }
        return self.chain.invoke(payload)
