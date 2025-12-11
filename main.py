# main.py
import nltk
nltk.download('punkt', quiet=True)

from cv_store import (
    index_cv_pdf_in_chroma,
    query_all_cvs,
    query_single_cv
)

def print_results(res):
    if not res["documents"] or not res["documents"][0]:
        print("No results.")
        return

    for i, doc in enumerate(res["documents"][0]):
        meta = res["metadatas"][0][i]
        print(f"\n=== Result {i+1} ===")
        print(f"CV File   : {meta.get('file_name')}")
        print(f"CV ID     : {meta.get('cv_id')}")
        print(f"Chunk idx : {meta.get('chunk_index')}")
        print(f"Content   :\n{doc[:600]}...")
        print("-" * 40)


def main():
    print("=== CV RAG DEMO ===")
    print("1) Index new CV PDF")
    print("2) Ask question over ALL CVs")
    print("3) Ask question over ONE CV (by cv_id)")
    print("0) Exit")

    while True:
        choice = input("\nChoose an option (0-3): ").strip()

        if choice == "1":
            path = input("Enter path to CV PDF (e.g., data/sample_cv.pdf): ").strip()
            cv_id = index_cv_pdf_in_chroma(path)
            if cv_id:
                print(f"[OK] CV indexed with cv_id = {cv_id}")
        elif choice == "2":
            q = input("Your question: ").strip()
            res = query_all_cvs(q)
            print_results(res)
        elif choice == "3":
            cv_id = input("Enter cv_id: ").strip()
            q = input("Your question: ").strip()
            res = query_single_cv(q, cv_id)
            print_results(res)
        elif choice == "0":
            print("Bye!")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
