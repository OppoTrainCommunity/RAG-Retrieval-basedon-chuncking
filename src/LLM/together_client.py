import os
import json
import requests
from typing import Dict, Any, List, Optional

from .base import BaseLLMClient


class TogetherClient(BaseLLMClient):
    """
    Together.ai client using /v1/chat/completions
    Docs: https://docs.together.ai/reference/chat-completions-1
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.together.ai/v1",
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing TOGETHER_API_KEY (env var)")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 800,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Together doesn’t use OpenRouter-style response_format.
        # We'll enforce JSON via prompt and then parse it.
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()

        data = r.json()
        return data["choices"][0]["message"]["content"]
