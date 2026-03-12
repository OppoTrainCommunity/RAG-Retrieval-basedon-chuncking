import logging
from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import settings

logger = logging.getLogger("cv_analyzer.llm_service")

DEFAULT_MODELS = [
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "name": "Nemotron 3 Super 120B (Free)"},
    {"id": "mistralai/mistral-small-3.1-24b-instruct:free", "name": "Mistral Small 3.1 24B (Free)"},
    {"id": "qwen/qwen3-coder:free", "name": "Qwen3 Coder 480B (Free)"},
    {"id": "google/gemma-3n-e4b-it:free", "name": "Gemma 3n 4B (Free)"},
]


class LLMService:
    def __init__(self) -> None:
        self._model_cache: dict[str, ChatOpenAI] = {}
        self._api_key: str = settings.openrouter_api_key
        logger.info("LLM service initialized")

    @property
    def api_key(self) -> str:
        return self._api_key

    def update_api_key(self, new_key: str) -> None:
        self._api_key = new_key
        self.clear_cache()
        logger.info("API key updated and model cache cleared")

    def clear_cache(self) -> None:
        self._model_cache.clear()

    def get_model(self, model_id: Optional[str] = None) -> ChatOpenAI:
        model_id = model_id or DEFAULT_MODELS[0]["id"]

        if model_id not in self._model_cache:
            self._model_cache[model_id] = ChatOpenAI(
                model=model_id,
                openai_api_key=self._api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.3,
                max_tokens=2000,
            )

        return self._model_cache[model_id]

    def get_default_models(self) -> list[dict]:
        return DEFAULT_MODELS
