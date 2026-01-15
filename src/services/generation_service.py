from typing import Dict, Any, Optional, List

from src.services.retrieval_service import RetrievalService
from src.generation import PromptBuilder, JsonUtils
from src.LLM import OpenRouterClient, TogetherClient, BaseLLMClient


class GenerationService:
    """
    High-level service:
    - retrieve top-k chunks
    - send to LLM
    - return structured JSON
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        provider: str = "openrouter",  # "openrouter" or "together"
        model: str = "meta-llama/llama-3.1-8b-instruct",
        temperature: float = 0.2,
        max_tokens: int = 900,
    ):
        self.retrieval_service = retrieval_service
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm: BaseLLMClient = self._init_llm()

    def _init_llm(self) -> BaseLLMClient:
        if self.provider == "openrouter":
            return OpenRouterClient()
        if self.provider == "together":
            return TogetherClient()
        raise ValueError("provider must be 'openrouter' or 'together'")

    def generate_candidate_ranking(
        self,
        collection_name: str,
        question: str,
        role_title: str,
        must_have: list,
        nice_to_have: list,
        seniority: str = "any",
        job_notes: str = "",
        k: int = 40,
        top_n: int = 5,
        where=None,
    ) -> Dict[str, Any]:

        # 1) Retrieve
        results = self.retrieval_service.search(
            collection_name=collection_name,
            query=question,
            k=k,
            where=where,
        )

        docs = results.get("documents", [[]])[0] if results else []
        metas = results.get("metadatas", [[]])[0] if results else []

        context = PromptBuilder.build_context_with_meta(docs, metas)

        # 2) Build prompt
        messages = PromptBuilder.build_messages(
            context=context,
            question=question,
            role_title=role_title,
            must_have=must_have,
            nice_to_have=nice_to_have,
            seniority=seniority,
            job_notes=job_notes,
            top_n=top_n,
        )

        # 3) Call LLM
        raw = self.llm.chat_completion(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # 4) Parse JSON
        data = JsonUtils.extract_json(raw)

        # Optional: attach trace info
        data["_retrieval_trace"] = {
            "k": k,
            "query": question,
            "sources": list({m.get("file_name") for m in metas if m.get("file_name")}),
        }
        return data
