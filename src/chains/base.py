from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseRAGChain(ABC):
    """
    Abstract RAG Chain: input -> (retrieval) -> LLM -> output

    We keep the interface stable so you can swap chain implementations later.
    """

    @abstractmethod
    def invoke(
        self,
        question: str,
        user_goal: str = "",
        where: Optional[Dict[str, Any]] = None,
        k: int = 8,
    ) -> str:
        """
        Returns model raw output (string). Parsing is handled separately in evaluation/service.
        """
        raise NotImplementedError
