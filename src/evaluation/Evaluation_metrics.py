from typing import Dict
from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    """
    Container for evaluation metrics.
    
    Attributes:
        precision_at_k (Dict[int, float]): Precision at different k values
        recall_at_k (Dict[int, float]): Recall at different k values
        mrr (float): Mean Reciprocal Rank
        ndcg_at_k (Dict[int, float]): Normalized Discounted Cumulative Gain
        hit_rate_at_k (Dict[int, float]): Hit rate at different k values
    """
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    hit_rate_at_k: Dict[int, float]

