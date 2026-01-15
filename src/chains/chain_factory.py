from dataclasses import dataclass
from typing import Dict

from src.services.retrieval_service import RetrievalService
from .langchain_rag_chain import LangChainRAGChain


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str


class ChainFactory:
    """
    Builds chains consistently (same retrieval settings, different LLMs).
    """

    # Lowest cost baseline
    MODEL_A = ModelConfig("llama_3_1_8b", "meta-llama/llama-3.1-8b-instruct")

    # Comparison model (still reasonable cost)
    MODEL_B = ModelConfig("qwen_2_5_7b", "qwen/qwen-2.5-7b-instruct")

    @staticmethod
    def build_two_chains(
        retrieval_service: RetrievalService,
        collection_name: str = "paragraph_chunking",
    ) -> Dict[str, LangChainRAGChain]:
        return {
            ChainFactory.MODEL_A.name: LangChainRAGChain(
                retrieval_service=retrieval_service,
                collection_name=collection_name,
                model=ChainFactory.MODEL_A.model_id,
                temperature=0.2,
                max_tokens=900,
            ),
            ChainFactory.MODEL_B.name: LangChainRAGChain(
                retrieval_service=retrieval_service,
                collection_name=collection_name,
                model=ChainFactory.MODEL_B.model_id,
                temperature=0.2,
                max_tokens=900,
            ),
        }
