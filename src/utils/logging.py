"""
Logging utilities for CV RAG System.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def setup_logger(
    name: str = "cv_rag",
    level: str = "INFO",
    console: bool = True,
    file_path: Optional[str] = None,
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Logger name.
        level: Logging level.
        console: Whether to log to console.
        file_path: Optional path to log file.
    
    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_path:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class RunLogger:
    """
    Logger for structured run logs in JSONL format.
    """
    
    def __init__(
        self,
        output_path: str = "./outputs/run_logs.jsonl",
    ):
        """
        Initialize run logger.
        
        Args:
            output_path: Path to the JSONL log file.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def log(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Log an event to the JSONL file.
        
        Args:
            event_type: Type of event.
            data: Event data dictionary.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **data,
        }
        
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def log_query(
        self,
        query: str,
        num_sources: int,
        retrieval_time: float,
        generation_time: float,
        answer_length: int,
    ) -> None:
        """
        Log a query event.
        
        Args:
            query: User query.
            num_sources: Number of retrieved sources.
            retrieval_time: Time for retrieval in seconds.
            generation_time: Time for generation in seconds.
            answer_length: Length of generated answer.
        """
        self.log("query", {
            "query": query,
            "num_sources": num_sources,
            "retrieval_time_s": retrieval_time,
            "generation_time_s": generation_time,
            "total_time_s": retrieval_time + generation_time,
            "answer_length": answer_length,
        })
    
    def log_evaluation(
        self,
        query: str,
        relevance_score: int,
        faithfulness_score: int,
        correctness_score: int,
        average_score: float,
    ) -> None:
        """
        Log an evaluation event.
        
        Args:
            query: Evaluated query.
            relevance_score: Relevance score.
            faithfulness_score: Faithfulness score.
            correctness_score: Correctness score.
            average_score: Average score.
        """
        self.log("evaluation", {
            "query": query,
            "relevance_score": relevance_score,
            "faithfulness_score": faithfulness_score,
            "correctness_score": correctness_score,
            "average_score": average_score,
        })
    
    def log_index_build(
        self,
        num_chunks: int,
        build_time: float,
        collection_name: str,
    ) -> None:
        """
        Log an index build event.
        
        Args:
            num_chunks: Number of chunks indexed.
            build_time: Time to build index in seconds.
            collection_name: Name of the collection.
        """
        self.log("index_build", {
            "num_chunks": num_chunks,
            "build_time_s": build_time,
            "collection_name": collection_name,
        })


def log_run(
    output_path: str,
    event_type: str,
    data: Dict[str, Any],
) -> None:
    """
    Convenience function to log a single event.
    
    Args:
        output_path: Path to log file.
        event_type: Type of event.
        data: Event data.
    """
    logger = RunLogger(output_path)
    logger.log(event_type, data)
