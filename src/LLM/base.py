from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM providers (OpenRouter / Together).
    """

    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute a chat completion request and return the assistant text response.
        """
        raise NotImplementedError
