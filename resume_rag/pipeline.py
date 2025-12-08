from resume_rag.parse_pdf import pdf_to_text
from resume_rag.vector_store import save_resume, save_chunk, search
from resume_rag.llm_chain import suggest_career, evaluate_resume
from resume_rag.chunker import semantic_chunk


def analyze_resume(path: str):
    """
    Complete RAG + Evaluation Pipeline:
    1. Extract text from PDF
    2. Semantic Chunking
    3. Store resume + chunks in vector DB
    4. RAG Search
    5. Career Suggestions (LLM)
    6. Resume Evaluation (LLM)
    """

    # 1) Extract resume text
    resume_text = pdf_to_text(path)

    # 2) Save full resume
    save_resume("full_resume", resume_text)

    # 3) Create chunks
    chunks = semantic_chunk(resume_text)
    for i, ch in enumerate(chunks):
        save_chunk(f"chunk_{i}", ch)

    # 4) RAG search
    rag_results = search(resume_text)

    # 5) Career suggestions
    career = suggest_career(resume_text)

    # 6) Evaluation
    evaluation = evaluate_resume(resume_text)

    # Final structured output
    return {
        "parsed_text": resume_text,
        "chunks": chunks,
        "similar_profiles": rag_results,
        "career_recommendations": career,
        "evaluation": evaluation
    }
