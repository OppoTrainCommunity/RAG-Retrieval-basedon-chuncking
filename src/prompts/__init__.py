# Prompts module
from .templates import (
    RAG_PROMPT_TEMPLATE,
    JUDGE_RELEVANCE_TEMPLATE,
    JUDGE_FAITHFULNESS_TEMPLATE,
    JUDGE_CORRECTNESS_TEMPLATE,
    get_rag_prompt,
    get_judge_prompts,
)

__all__ = [
    "RAG_PROMPT_TEMPLATE",
    "JUDGE_RELEVANCE_TEMPLATE", 
    "JUDGE_FAITHFULNESS_TEMPLATE",
    "JUDGE_CORRECTNESS_TEMPLATE",
    "get_rag_prompt",
    "get_judge_prompts",
]
