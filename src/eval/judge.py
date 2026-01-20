"""
LLM-as-Judge evaluator for CV RAG System.
Evaluates answer quality using a separate judge model.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI

from src.prompts.templates import get_judge_prompts

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from a single judge evaluation."""
    metric: str
    score: int
    explanation: str
    raw_response: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "metric": self.metric,
            "score": self.score,
            "explanation": self.explanation,
            "raw_response": self.raw_response,
        }


def parse_judge_response(response: str) -> tuple:
    """
    Parse judge response to extract score and explanation.
    
    Args:
        response: Raw judge response.
    
    Returns:
        Tuple of (score, explanation).
    """
    # Extract score
    score_match = re.search(r"SCORE:\s*(\d+)", response, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        score = max(1, min(5, score))  # Clamp to 1-5
    else:
        score = 3  # Default to middle score
    
    # Extract explanation
    explanation_match = re.search(
        r"EXPLANATION:\s*(.+?)(?:\n\n|$)",
        response,
        re.IGNORECASE | re.DOTALL
    )
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    else:
        # Fallback: use everything after SCORE as explanation
        parts = response.split("SCORE:")
        if len(parts) > 1:
            explanation = parts[1].strip()
            # Remove the score number
            explanation = re.sub(r"^\d+\s*", "", explanation).strip()
        else:
            explanation = response.strip()
    
    return score, explanation


class LLMJudge:
    """
    LLM-as-Judge for evaluating RAG responses.
    Uses a separate judge model to score answers.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 400,
    ):
        """
        Initialize LLM Judge.
        
        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter base URL.
            model: Judge model name.
            temperature: Sampling temperature (0 for consistent scoring).
            max_tokens: Maximum tokens for judge response.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        self.prompts = get_judge_prompts()
    
    def _call_judge(self, prompt: str) -> str:
        """
        Call the judge model.
        
        Args:
            prompt: Formatted prompt.
        
        Returns:
            Judge response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def evaluate_relevance(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """
        Evaluate answer relevance to the question.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
        
        Returns:
            JudgeResult with score and explanation.
        """
        prompt = self.prompts["relevance"].format(
            question=question,
            answer=answer,
            context=context,
        )
        
        raw_response = self._call_judge(prompt)
        score, explanation = parse_judge_response(raw_response)
        
        return JudgeResult(
            metric="relevance",
            score=score,
            explanation=explanation,
            raw_response=raw_response,
        )
    
    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """
        Evaluate answer faithfulness to the context.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
        
        Returns:
            JudgeResult with score and explanation.
        """
        prompt = self.prompts["faithfulness"].format(
            question=question,
            answer=answer,
            context=context,
        )
        
        raw_response = self._call_judge(prompt)
        score, explanation = parse_judge_response(raw_response)
        
        return JudgeResult(
            metric="faithfulness",
            score=score,
            explanation=explanation,
            raw_response=raw_response,
        )
    
    def evaluate_correctness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgeResult:
        """
        Evaluate answer correctness and quality.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
        
        Returns:
            JudgeResult with score and explanation.
        """
        prompt = self.prompts["correctness"].format(
            question=question,
            answer=answer,
            context=context,
        )
        
        raw_response = self._call_judge(prompt)
        score, explanation = parse_judge_response(raw_response)
        
        return JudgeResult(
            metric="correctness",
            score=score,
            explanation=explanation,
            raw_response=raw_response,
        )
    
    def evaluate_all(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, JudgeResult]:
        """
        Run all evaluation metrics.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
        
        Returns:
            Dictionary of metric name to JudgeResult.
        """
        results = {}
        
        results["relevance"] = self.evaluate_relevance(question, answer, context)
        results["faithfulness"] = self.evaluate_faithfulness(question, answer, context)
        results["correctness"] = self.evaluate_correctness(question, answer, context)
        
        return results
    
    @classmethod
    def from_config(cls, config) -> "LLMJudge":
        """
        Create LLMJudge from configuration.
        
        Args:
            config: Config object.
        
        Returns:
            LLMJudge instance.
        """
        return cls(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            model=config.judge.model,
            temperature=config.judge.temperature,
            max_tokens=config.judge.max_tokens,
        )
