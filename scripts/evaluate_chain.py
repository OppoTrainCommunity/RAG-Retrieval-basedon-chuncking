import json
from pathlib import Path

from src.services.retrieval_service import RetrievalService
from src.chains import ChainFactory
from src.chains.evaluation import ChainEvaluator


def main():
    persist_dir = "./chroma_db"
    collection_name = "paragraph_chunking"
    embedding_model = "all-mpnet-base-v2"

    qa_path = "data/qa_processed.json"

    retrieval = RetrievalService(persist_dir=persist_dir, embedding_model=embedding_model)
    chains = ChainFactory.build_two_chains(retrieval, collection_name=collection_name)

    evaluator = ChainEvaluator(use_llm_judge=False)  # set True if you want LLM judge scores

    qa_items = evaluator.load_ground_truth(qa_path)

    reports = {}
    for model_name, chain in chains.items():
        res = evaluator.evaluate(
            chain=chain,
            model_name=model_name,
            qa_items=qa_items,
            k=8,
            where_by_source_file=True,  
            limit=20,                  
        )

        reports[model_name] = {
            "model_name": res.model_name,
            "num_samples": res.num_samples,
            "json_valid_rate": res.json_valid_rate,
            "answer_match_rate": res.answer_match_rate,
            "avg_latency_ms": res.avg_latency_ms,
            "details": res.details,
        }

        print(f"\n=== {model_name} ===")
        print(f"JSON valid rate: {res.json_valid_rate:.3f}")
        print(f"Answer match rate: {res.answer_match_rate:.3f}")
        print(f"Avg latency (ms): {res.avg_latency_ms:.2f}")

    Path("chain_evaluation_report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print("\nSaved: chain_evaluation_report.json")


if __name__ == "__main__":
    main()
