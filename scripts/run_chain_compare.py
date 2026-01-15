import json
from pathlib import Path

from src.services.retrieval_service import RetrievalService
from src.chains import ChainFactory

def main():
    persist_dir = "./chroma_db"
    collection_name = "paragraph_chunking"
    embedding_model = "all-mpnet-base-v2"

    retrieval = RetrievalService(persist_dir=persist_dir, embedding_model=embedding_model)
    chains = ChainFactory.build_two_chains(retrieval, collection_name=collection_name)

    question = "Suggest career paths for this candidate."
    user_goal = "AI/Backend, building real projects"

    # Example: choose one CV file to avoid mixing sources
    where = {"file_name": "sama shalabi CV(AI) latest.pdf"}
    k = 8

    results = {}
    for name, chain in chains.items():
        out = chain.invoke(question=question, user_goal=user_goal, where=where, k=k)
        results[name] = out
        print(f"\n\n===== {name} =====\n{out}\n")

    Path("comparison_outputs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved: comparison_outputs.json")


if __name__ == "__main__":
    main()
