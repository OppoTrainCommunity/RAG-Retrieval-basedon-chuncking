"""
Evaluation pipeline for CV RAG System.
Runs LLM-as-judge evaluation on RAG responses.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.eval.judge import LLMJudge, JudgeResult
from src.prompts.templates import format_context_for_prompt
from src.rag.chain import RAGResponse

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""
    query: str
    answer: str
    relevance_score: int
    relevance_explanation: str
    faithfulness_score: int
    faithfulness_explanation: str
    correctness_score: int
    correctness_explanation: str
    average_score: float
    num_sources: int
    retrieval_time: float
    generation_time: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "relevance_score": self.relevance_score,
            "relevance_explanation": self.relevance_explanation,
            "faithfulness_score": self.faithfulness_score,
            "faithfulness_explanation": self.faithfulness_explanation,
            "correctness_score": self.correctness_score,
            "correctness_explanation": self.correctness_explanation,
            "average_score": self.average_score,
            "num_sources": self.num_sources,
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp,
        }


class EvaluationPipeline:
    """
    Pipeline for evaluating RAG responses using LLM-as-judge.
    """
    
    def __init__(
        self,
        judge: LLMJudge,
        output_dir: str = "./outputs",
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            judge: LLMJudge instance.
            output_dir: Directory to save evaluation results.
        """
        self.judge = judge
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[EvaluationResult] = []
    
    def evaluate_response(
        self,
        rag_response: RAGResponse,
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            rag_response: RAGResponse to evaluate.
        
        Returns:
            EvaluationResult with all scores.
        """
        # Format context from sources
        from langchain_core.documents import Document
        
        docs = [
            Document(
                page_content=source.content,
                metadata={
                    "candidate_id": source.candidate_id,
                    "section_name": source.section_name,
                    "chunk_id": source.chunk_id,
                    **source.metadata,
                }
            )
            for source in rag_response.sources
        ]
        
        context = format_context_for_prompt(docs)
        
        # Run all evaluations
        logger.info(f"Evaluating response for query: {rag_response.query[:50]}...")
        
        judge_results = self.judge.evaluate_all(
            question=rag_response.query,
            answer=rag_response.answer,
            context=context,
        )
        
        # Calculate average score
        scores = [
            judge_results["relevance"].score,
            judge_results["faithfulness"].score,
            judge_results["correctness"].score,
        ]
        average_score = sum(scores) / len(scores)
        
        # Build evaluation result
        result = EvaluationResult(
            query=rag_response.query,
            answer=rag_response.answer,
            relevance_score=judge_results["relevance"].score,
            relevance_explanation=judge_results["relevance"].explanation,
            faithfulness_score=judge_results["faithfulness"].score,
            faithfulness_explanation=judge_results["faithfulness"].explanation,
            correctness_score=judge_results["correctness"].score,
            correctness_explanation=judge_results["correctness"].explanation,
            average_score=round(average_score, 2),
            num_sources=rag_response.num_sources,
            retrieval_time=rag_response.retrieval_time,
            generation_time=rag_response.generation_time,
        )
        
        self.results.append(result)
        
        logger.info(
            f"Evaluation complete - Relevance: {result.relevance_score}, "
            f"Faithfulness: {result.faithfulness_score}, "
            f"Correctness: {result.correctness_score}, "
            f"Average: {result.average_score}"
        )
        
        return result
    
    def evaluate_batch(
        self,
        rag_responses: List[RAGResponse],
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses.
        
        Args:
            rag_responses: List of RAGResponse objects.
        
        Returns:
            List of EvaluationResult objects.
        """
        results = []
        
        for i, response in enumerate(rag_responses, 1):
            logger.info(f"Evaluating response {i}/{len(rag_responses)}...")
            result = self.evaluate_response(response)
            results.append(result)
        
        return results
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Returns:
            DataFrame with evaluation results.
        """
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def save_results(
        self,
        csv_filename: str = "eval_results.csv",
        json_filename: str = "eval_results.json",
    ) -> None:
        """
        Save evaluation results to files.
        
        Args:
            csv_filename: Name of CSV file.
            json_filename: Name of JSON file.
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save to CSV
        df = self.to_dataframe()
        csv_path = self.output_dir / csv_filename
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved evaluation results to {csv_path}")
        
        # Save to JSON
        json_path = self.output_dir / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Saved evaluation results to {json_path}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for evaluation results.
        
        Returns:
            Dictionary with summary statistics.
        """
        if not self.results:
            return {}
        
        df = self.to_dataframe()
        
        return {
            "num_evaluations": len(self.results),
            "avg_relevance": round(df["relevance_score"].mean(), 2),
            "avg_faithfulness": round(df["faithfulness_score"].mean(), 2),
            "avg_correctness": round(df["correctness_score"].mean(), 2),
            "avg_overall": round(df["average_score"].mean(), 2),
            "avg_retrieval_time": round(df["retrieval_time"].mean(), 3),
            "avg_generation_time": round(df["generation_time"].mean(), 3),
            "avg_sources": round(df["num_sources"].mean(), 1),
        }
    
    def clear_results(self) -> None:
        """Clear stored results."""
        self.results = []
    
    @classmethod
    def from_config(cls, config) -> "EvaluationPipeline":
        """
        Create EvaluationPipeline from configuration.
        
        Args:
            config: Config object.
        
        Returns:
            EvaluationPipeline instance.
        """
        judge = LLMJudge.from_config(config)
        
        return cls(
            judge=judge,
            output_dir=config.outputs.dir,
        )
