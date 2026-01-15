import os
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
load_dotenv()


from .base import BaseLLMClient

class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter client using the Chat Completions endpoint.
    Docs: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY (env var)")

        self.base_url = base_url.rstrip("/")
        self.site_url = site_url
        self.app_name = app_name
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

        # Optional headers recommended by OpenRouter
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # If your model/provider supports structured outputs on OpenRouter,
        # you can pass response_format (JSON schema). Otherwise we still parse JSON from text.
        if response_format:
            payload["response_format"] = response_format

        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        r.raise_for_status()

        data = r.json()
        # Standard: choices[0].message.content
        return data["choices"][0]["message"]["content"]
