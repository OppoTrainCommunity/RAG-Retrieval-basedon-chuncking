# resume_rag/openrouter_llm.py
from __future__ import annotations
from typing import List, Dict, Any
import requests
import os

def get_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing OPENROUTER_API_KEY in .env or environment variables.")
    return key

def get_base_url() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

def call_openrouter_chat(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 700,
) -> str:
    api_key = get_key()
    base_url = get_base_url()

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended:
        "HTTP-Referer": "http://localhost",
        "X-Title": "Resume RAG",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

    data = r.json()
    return data["choices"][0]["message"]["content"]
