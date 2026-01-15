import json
import time
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.generation.json_utils import JsonUtils
from src.LLM.openrouter_client import OpenRouterClient


@dataclass
class EvaluationResult:
    model_name: str
    num_samples: int
    json_valid_rate: float
    answer_match_rate: float
    avg_latency_ms: float
    details: List[Dict[str, Any]]


class ChainEvaluator:
    """
    End-to-end chain evaluator.

    Metrics:
    - JSON valid rate
    - Answer match rate: whether expected answer appears in output (substring, case-insensitive)
    - latency

    Optional:
    - LLM judge scoring (faithfulness + relevance) using cheap OpenRouter model
    """

    def __init__(self, use_llm_judge: bool = False):
        self.use_llm_judge = use_llm_judge
        self.judge_llm = OpenRouterClient() if use_llm_judge else None
        self.judge_model = "meta-llama/llama-3.1-8b-instruct"

    @staticmethod
    def load_ground_truth(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    @staticmethod
    def _answer_in_output(expected_answer: str, output_text: str) -> bool:
        a = ChainEvaluator._normalize(expected_answer)
        o = ChainEvaluator._normalize(output_text)
        return a in o if a else False

    def _judge(self, question: str, expected_answer: str, model_output: str) -> Dict[str, Any]:
        """
        LLM-as-judge prompt (optional). Cheap but useful for “quality beyond substring match”.
        Scores 0-5.
        """
        if not self.judge_llm:
            return {}

        prompt = f"""
You are evaluating a RAG system output.

Question:
{question}

Ground-truth answer:
{expected_answer}

Model output:
{model_output}

Rate the output:
- correctness (0-5): Does it correctly answer the question?
- relevance (0-5): Does it stay on-topic?
- format (0-5): Is it valid JSON and matches the required schema?

Return ONLY JSON:
{{
  "correctness": 0,
  "relevance": 0,
  "format": 0,
  "notes": "short reason"
}}
""".strip()

        raw = self.judge_llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.judge_model,
            temperature=0.0,
            max_tokens=300,
        )

        try:
            return JsonUtils.extract_json(raw)
        except Exception:
            return {"error": "judge_parse_failed", "raw": raw[:500]}

    def evaluate(
        self,
        chain,
        model_name: str,
        qa_items: List[Dict[str, Any]],
        k: int = 8,
        where_by_source_file: bool = True,
        limit: Optional[int] = None,
    ) -> EvaluationResult:
        items = qa_items[:limit] if limit else qa_items

        json_ok = 0
        ans_ok = 0
        total_latency = 0.0
        details: List[Dict[str, Any]] = []

        for item in items:
            q = item["question"]
            expected = item.get("answer", "")
            source_file = item.get("source_file")

            where = {"file_name": source_file} if (where_by_source_file and source_file) else None

            start = time.time()
            out_text = chain.invoke(question=q, user_goal="", where=where, k=k)
            latency_ms = (time.time() - start) * 1000.0
            total_latency += latency_ms

            # JSON validity
            parsed = None
            json_valid = False
            try:
                parsed = JsonUtils.extract_json(out_text)
                json_valid = True
                json_ok += 1
            except Exception:
                pass

            # Answer match (substring)
            matched = self._answer_in_output(expected, out_text)
            if matched:
                ans_ok += 1

            row: Dict[str, Any] = {
                "question": q,
                "expected_answer": expected,
                "source_file": source_file,
                "latency_ms": round(latency_ms, 2),
                "json_valid": json_valid,
                "answer_matched": matched,
                "raw_output_preview": out_text[:400],
            }

            if self.use_llm_judge:
                row["judge"] = self._judge(q, expected, out_text)

            details.append(row)

        n = len(items) if items else 1
        return EvaluationResult(
            model_name=model_name,
            num_samples=len(items),
            json_valid_rate=json_ok / n,
            answer_match_rate=ans_ok / n,
            avg_latency_ms=total_latency / n,
            details=details,
        )
