"""
Retrieval Evaluation Pipeline for CV RAG System.
Evaluates retrieval quality without LLM generation.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEvalQuery:
    """A single retrieval evaluation query with ground truth."""
    query: str
    expected_chunk_ids: Optional[List[str]] = None
    expected_candidate_ids: Optional[List[str]] = None
    expected_section_names: Optional[List[str]] = None
    notes: Optional[str] = None
    
    def has_ground_truth(self) -> bool:
        """Check if any ground truth is provided."""
        return bool(
            self.expected_chunk_ids or 
            self.expected_candidate_ids or 
            self.expected_section_names
        )
    
    def get_expected_count(self) -> int:
        """Get the number of expected relevant items (for recall calculation)."""
        if self.expected_chunk_ids:
            return len(self.expected_chunk_ids)
        if self.expected_candidate_ids:
            return len(self.expected_candidate_ids)
        if self.expected_section_names:
            return len(self.expected_section_names)
        return 0


@dataclass
class RetrievalEvalResult:
    """Results from evaluating a single retrieval query."""
    query: str
    top_k: int
    retrieved_docs: List[Dict]  # List of doc metadata
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: Optional[float] = None
    num_relevant_retrieved: int = 0
    first_relevant_rank: Optional[int] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "top_k": self.top_k,
            "hit_at_k": self.hit_at_k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "num_relevant_retrieved": self.num_relevant_retrieved,
            "first_relevant_rank": self.first_relevant_rank,
            "num_docs_retrieved": len(self.retrieved_docs),
            "notes": self.notes,
        }


def load_retrieval_eval_data(path: str) -> List[RetrievalEvalQuery]:
    """
    Load retrieval evaluation dataset from JSONL or CSV file.
    
    Args:
        path: Path to the evaluation data file (.jsonl or .csv)
    
    Returns:
        List of RetrievalEvalQuery objects.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Evaluation data file not found: {path}")
    
    queries = []
    
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    queries.append(_parse_eval_query(data))
    
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            queries.append(_parse_eval_query(row.to_dict()))
    
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    queries.append(_parse_eval_query(item))
            else:
                queries.append(_parse_eval_query(data))
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Loaded {len(queries)} evaluation queries from {path}")
    return queries


def _parse_eval_query(data: Dict) -> RetrievalEvalQuery:
    """Parse a dictionary into a RetrievalEvalQuery."""
    # Handle list fields that might be stored as strings
    def parse_list_field(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # Try JSON parsing first
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                # Fall back to comma-separated
                return [v.strip() for v in value.split(",") if v.strip()]
        return [str(value)]
    
    return RetrievalEvalQuery(
        query=str(data.get("query", "")),
        expected_chunk_ids=parse_list_field(data.get("expected_chunk_ids")),
        expected_candidate_ids=parse_list_field(data.get("expected_candidate_ids")),
        expected_section_names=parse_list_field(data.get("expected_section_names")),
        notes=data.get("notes"),
    )


def run_retrieval(
    chroma_store,
    query: str,
    top_k: int = 6,
    filter_dict: Optional[Dict] = None,
) -> List[Document]:
    """
    Run retrieval using the ChromaStore.
    
    Args:
        chroma_store: ChromaStore instance
        query: Query string
        top_k: Number of documents to retrieve
        filter_dict: Optional metadata filter
    
    Returns:
        List of retrieved Document objects
    """
    return chroma_store.similarity_search(query, k=top_k, filter_dict=filter_dict)


def is_doc_relevant(
    doc: Document,
    eval_query: RetrievalEvalQuery,
) -> bool:
    """
    Check if a retrieved document is relevant based on ground truth.
    
    Relevance priority: chunk_id > candidate_id > section_name
    
    Args:
        doc: Retrieved document
        eval_query: Evaluation query with ground truth
    
    Returns:
        True if document is relevant, False otherwise
    """
    metadata = doc.metadata
    
    # Priority 1: Match by chunk_id
    if eval_query.expected_chunk_ids:
        chunk_id = metadata.get("chunk_id", "")
        return chunk_id in eval_query.expected_chunk_ids
    
    # Priority 2: Match by candidate_id
    if eval_query.expected_candidate_ids:
        candidate_id = metadata.get("candidate_id", "")
        return candidate_id in eval_query.expected_candidate_ids
    
    # Priority 3: Match by section_name
    if eval_query.expected_section_names:
        section_name = metadata.get("section_name", "")
        return section_name in eval_query.expected_section_names
    
    # No ground truth provided
    return False


def compute_metrics(
    retrieved_docs: List[Document],
    eval_query: RetrievalEvalQuery,
    top_k: int,
) -> RetrievalEvalResult:
    """
    Compute retrieval evaluation metrics for a single query.
    
    Args:
        retrieved_docs: List of retrieved documents
        eval_query: Evaluation query with ground truth
        top_k: Number of documents retrieved
    
    Returns:
        RetrievalEvalResult with computed metrics
    """
    # Track relevance for each retrieved doc
    relevance_flags = [is_doc_relevant(doc, eval_query) for doc in retrieved_docs]
    
    # Count relevant documents
    num_relevant = sum(relevance_flags)
    
    # Find first relevant rank (1-indexed)
    first_relevant_rank = None
    for i, is_rel in enumerate(relevance_flags):
        if is_rel:
            first_relevant_rank = i + 1
            break
    
    # Get expected count for recall
    expected_count = eval_query.get_expected_count()
    
    # Compute Hit@k: 1 if any relevant doc in top-k, else 0
    hit_at_k = 1.0 if num_relevant > 0 else 0.0
    
    # Compute Precision@k: (# relevant retrieved) / k
    precision_at_k = num_relevant / top_k if top_k > 0 else 0.0
    
    # Compute Recall@k: (# relevant retrieved) / (# expected relevant)
    recall_at_k = num_relevant / expected_count if expected_count > 0 else 0.0
    
    # Compute MRR@k: 1/rank of first relevant hit, else 0
    mrr_at_k = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    
    # Compute nDCG@k
    ndcg_at_k = compute_ndcg(relevance_flags, expected_count)
    
    # Extract doc metadata for logging
    retrieved_docs_meta = [
        {
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "candidate_id": doc.metadata.get("candidate_id", ""),
            "section_name": doc.metadata.get("section_name", ""),
            "relevant": relevance_flags[i],
        }
        for i, doc in enumerate(retrieved_docs)
    ]
    
    return RetrievalEvalResult(
        query=eval_query.query,
        top_k=top_k,
        retrieved_docs=retrieved_docs_meta,
        hit_at_k=hit_at_k,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        mrr_at_k=mrr_at_k,
        ndcg_at_k=ndcg_at_k,
        num_relevant_retrieved=num_relevant,
        first_relevant_rank=first_relevant_rank,
        notes=eval_query.notes,
    )


def compute_ndcg(relevance_flags: List[bool], num_expected: int) -> float:
    """
    Compute normalized Discounted Cumulative Gain (nDCG).
    
    Uses binary relevance (1 for relevant, 0 for not relevant).
    
    Args:
        relevance_flags: List of boolean relevance flags for retrieved docs
        num_expected: Number of expected relevant documents
    
    Returns:
        nDCG score between 0 and 1
    """
    import math
    
    if not relevance_flags or num_expected == 0:
        return 0.0
    
    # DCG: sum of rel_i / log2(i + 1) for i from 1 to k
    dcg = 0.0
    for i, is_rel in enumerate(relevance_flags):
        if is_rel:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed, formula uses 1-indexed
    
    # Ideal DCG: all relevant docs at top positions
    idcg = 0.0
    for i in range(min(num_expected, len(relevance_flags))):
        idcg += 1.0 / math.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def evaluate_retriever(
    chroma_store,
    dataset: List[RetrievalEvalQuery],
    top_k: int = 6,
) -> pd.DataFrame:
    """
    Evaluate retriever on a dataset of queries.
    
    Args:
        chroma_store: ChromaStore instance
        dataset: List of RetrievalEvalQuery objects
        top_k: Number of documents to retrieve per query
    
    Returns:
        DataFrame with evaluation results per query
    """
    results = []
    
    for eval_query in dataset:
        if not eval_query.query:
            continue
        
        if not eval_query.has_ground_truth():
            logger.warning(f"Skipping query without ground truth: {eval_query.query[:50]}...")
            continue
        
        # Run retrieval
        retrieved_docs = run_retrieval(chroma_store, eval_query.query, top_k)
        
        # Compute metrics
        result = compute_metrics(retrieved_docs, eval_query, top_k)
        results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame([r.to_dict() for r in results])
    
    return df


def get_aggregate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute aggregate metrics from evaluation results.
    
    Args:
        results_df: DataFrame with per-query metrics
    
    Returns:
        Dictionary with aggregate metrics
    """
    if results_df.empty:
        return {}
    
    return {
        "num_queries": len(results_df),
        "avg_hit_at_k": results_df["hit_at_k"].mean(),
        "avg_precision_at_k": results_df["precision_at_k"].mean(),
        "avg_recall_at_k": results_df["recall_at_k"].mean(),
        "avg_mrr_at_k": results_df["mrr_at_k"].mean(),
        "avg_ndcg_at_k": results_df["ndcg_at_k"].mean() if "ndcg_at_k" in results_df else None,
        "total_relevant_retrieved": results_df["num_relevant_retrieved"].sum(),
    }


def save_retrieval_eval_results(
    results_df: pd.DataFrame,
    output_dir: str = "./outputs",
    csv_filename: str = "retrieval_eval_results.csv",
    json_filename: str = "retrieval_eval_results.json",
) -> None:
    """
    Save retrieval evaluation results to CSV and JSON.
    
    Args:
        results_df: DataFrame with evaluation results
        output_dir: Output directory
        csv_filename: CSV output filename
        json_filename: JSON output filename
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_path / csv_filename
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Saved retrieval eval results to {csv_path}")
    
    # Save JSON with aggregate metrics
    json_path = output_path / json_filename
    
    output_data = {
        "aggregate_metrics": get_aggregate_metrics(results_df),
        "per_query_results": results_df.to_dict(orient="records"),
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved retrieval eval results to {json_path}")


class RetrievalEvaluator:
    """
    High-level class for running retrieval evaluation.
    """
    
    def __init__(
        self,
        chroma_store,
        top_k: int = 6,
    ):
        """
        Initialize the retrieval evaluator.
        
        Args:
            chroma_store: ChromaStore instance
            top_k: Number of documents to retrieve per query
        """
        self.chroma_store = chroma_store
        self.top_k = top_k
        self.results_df: Optional[pd.DataFrame] = None
        self.dataset: Optional[List[RetrievalEvalQuery]] = None
    
    def load_dataset(self, path: str) -> "RetrievalEvaluator":
        """Load evaluation dataset from file."""
        self.dataset = load_retrieval_eval_data(path)
        return self
    
    def evaluate(
        self,
        dataset: Optional[List[RetrievalEvalQuery]] = None,
    ) -> pd.DataFrame:
        """
        Run evaluation on the dataset.
        
        Args:
            dataset: Optional dataset to use (overrides loaded dataset)
        
        Returns:
            DataFrame with evaluation results
        """
        dataset = dataset or self.dataset
        if not dataset:
            raise ValueError("No dataset provided. Load a dataset first.")
        
        self.results_df = evaluate_retriever(
            self.chroma_store,
            dataset,
            self.top_k,
        )
        return self.results_df
    
    def get_summary(self) -> Dict[str, float]:
        """Get aggregate metrics summary."""
        if self.results_df is None:
            raise ValueError("No results available. Run evaluate() first.")
        return get_aggregate_metrics(self.results_df)
    
    def save_results(
        self,
        output_dir: str = "./outputs",
        csv_filename: str = "retrieval_eval_results.csv",
        json_filename: str = "retrieval_eval_results.json",
    ) -> None:
        """Save results to files."""
        if self.results_df is None:
            raise ValueError("No results available. Run evaluate() first.")
        save_retrieval_eval_results(
            self.results_df,
            output_dir,
            csv_filename,
            json_filename,
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        if self.results_df is None:
            raise ValueError("No results available. Run evaluate() first.")
        return self.results_df
