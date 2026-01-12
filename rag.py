import os
import pathlib
import pandas as pd
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
from typing import List, Dict, Any
import re
import requests
import json

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    print("Please run: pip install langchain-text-splitters")
    raise

# --- CONFIGURATION & SETUP ---
OS_API_KEY = "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"
PDF_DIR = "pdfs"
QUERY_FILE = "queries.txt"
RESULTS_DIR = "./results"
CHROMA_PATH = "./chroma_db"

pathlib.Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# 1. Initialize STRONGER Embedding Function (3072 dimensions)
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OS_API_KEY,
    model_name="openai/text-embedding-3-large", 
    api_base="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost", 
        "X-Title": "Strong-Resume-RAG"
    }
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

# --- ENHANCED TEXT EXTRACTION ---

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def preprocess_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    return text.strip()

# --- CHUNKING STRATEGIES ---

def get_fixed_chunks(text: str, size: int = 500) -> List[str]:
    chunks = []
    overlap = size // 4 
    for i in range(0, len(text), size - overlap):
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def get_recursive_chunks(text: str, size: int = 500, overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)

def get_semantic_chunks(text: str, size: int = 600, overlap: int = 100) -> List[str]:
    section_patterns = [
        r'\n(?:EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|PROFESSIONAL EXPERIENCE)\s*\n',
        r'\n(?:EDUCATION|ACADEMIC BACKGROUND)\s*\n',
        r'\n(?:SKILLS|TECHNICAL SKILLS|CORE COMPETENCIES)\s*\n',
        r'\n(?:PROJECTS|KEY PROJECTS)\s*\n',
        r'\n(?:CERTIFICATIONS|CERTIFICATES)\s*\n',
        r'\n(?:SUMMARY|PROFESSIONAL SUMMARY|PROFILE)\s*\n',
    ]
    sections = []
    last_pos = 0
    for pattern in section_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.start() > last_pos:
                sections.append(text[last_pos:match.start()])
            last_pos = match.start()
    if last_pos < len(text):
        sections.append(text[last_pos:])
    
    if len(sections) <= 1:
        return get_recursive_chunks(text, size, overlap)
    
    chunks = []
    for section in sections:
        if len(section) <= size:
            chunks.append(section)
        else:
            sub_chunks = get_recursive_chunks(section, size, overlap)
            chunks.extend(sub_chunks)
    return [c for c in chunks if c.strip()]

# --- HYBRID RETRIEVAL WITH RERANKING ---

def hybrid_query(question: str, collection, top_k: int = 10, rerank_top: int = 3):
    expanded_queries = [question]
    if any(word in question.lower() for word in ['skill', 'experience', 'expertise', 'proficient']):
        expanded_queries.append(f"technical skills and expertise: {question}")
    if any(word in question.lower() for word in ['degree', 'education', 'university', 'graduate']):
        expanded_queries.append(f"educational background: {question}")
    
    all_results = []
    seen_ids = set()
    for query in expanded_queries:
        res = collection.query(query_texts=[query], n_results=top_k)
        for i, doc_id in enumerate(res['ids'][0]):
            if doc_id not in seen_ids:
                all_results.append({
                    'id': doc_id,
                    'document': res['documents'][0][i],
                    'metadata': res['metadatas'][0][i],
                    'distance': res['distances'][0][i] if 'distances' in res else 0,
                })
                seen_ids.add(doc_id)
    
    def rerank_score(result):
        distance_score = 1 / (1 + result['distance'])
        query_terms = set(question.lower().split())
        doc_terms = set(result['document'].lower().split())
        keyword_overlap = len(query_terms.intersection(doc_terms)) / (len(query_terms) + 1e-6)
        return distance_score * 0.7 + keyword_overlap * 0.3
    
    all_results.sort(key=rerank_score, reverse=True)
    top_results = all_results[:rerank_top]
    
    if not top_results:
        return [], "N/A"
    
    source = top_results[0]['metadata']['doc_name']
    return top_results, source

# --- LLM ANSWER GENERATION ---

def generate_answer(question: str, context_chunks: List[dict], source: str) -> str:
    # Format chunks with XML tags for better LLM reasoning
    formatted_context = ""
    for i, item in enumerate(context_chunks):
        formatted_context += f"<document_chunk id='{i+1}'>\n{item['document']}\n</document_chunk>\n"

    prompt = f"""You are a professional Resume Screening Assistant. 
Analyze the provided resume segments to answer the user's question accurately.

<context>
Resume Source: {source}
{formatted_context}
</context>

User Question: {question}

STRICT INSTRUCTIONS:
1. Use ONLY the information provided in the <context> tags.
2. Provide a concise but detailed response.
3. If the answer is not in the context, say "I could not find this information in the resume."
4. DO NOT hallucinate dates or skills.
5. Briefly cite chunks used, e.g., [1].

Answer:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OS_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Strong-Resume-RAG"
            },
            json={
                "model": "anthropic/claude-3.5-sonnet",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.0
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return f"API Error: {response.text}"
    except Exception as e:
        return f"System Error: {str(e)}"

# --- EVALUATION UTILS ---

def calculate_word_iou(retrieved_text: str, ground_truth: str) -> float:
    def get_words(text):
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return set(text.split())
    words_ret = get_words(retrieved_text)
    words_gt = get_words(ground_truth)
    if not words_ret or not words_gt: return 0.0
    return len(words_ret.intersection(words_gt)) / len(words_ret.union(words_gt))

def is_contained(retrieved_text: str, ground_truth: str) -> bool:
    ret_norm = re.sub(r'[^\w\s]', ' ', retrieved_text.lower())
    gt_norm = re.sub(r'[^\w\s]', ' ', ground_truth.lower())
    if gt_norm.strip() in ret_norm: return True
    gt_words = set(gt_norm.split())
    ret_words = set(ret_norm.split())
    if not gt_words: return False
    return (len(gt_words.intersection(ret_words)) / len(gt_words)) > 0.8

def load_queries():
    queries = []
    if not os.path.exists(QUERY_FILE): return []
    with open(QUERY_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "|||" not in line: continue
            parts = [p.strip() for p in line.split("|||")]
            if len(parts) == 3:
                queries.append({"id": str(i), "query": parts[0], "gt_doc": parts[1], "gt_excerpt": parts[2]})
    return queries

# --- INTERACTIVE MODE ---

def simple_interactive_mode(collection):
    print("\n🔍 Strong RAG Search (type 'exit' to quit)")
    print("=" * 60)
    while True:
        question = input("\n❓ Ask a question: ").strip()
        if question.lower() in {"exit", "quit"}: break

        print("\n🔄 Searching and generating...")
        context_list, source = hybrid_query(question, collection)

        if not context_list or source == "N/A":
            print("❌ No information found.")
            continue

        answer = generate_answer(question, context_list, source)
        print(f"\n📄 Source: {source}\n" + "="*60 + f"\n💡 ANSWER:\n{answer}\n" + "="*60)
        
        if input("\nShow raw context? (y/n): ").lower() == 'y':
            for i, c in enumerate(context_list):
                print(f"\n[Chunk {i+1}]: {c['document'][:300]}...")

# --- MAIN ---

def main():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files: return

    # Reset collections to handle new 3072-dim embeddings
    for name in ["fixed_index", "recursive_index", "semantic_index"]:
        try: client.delete_collection(name)
        except: pass
    
    col_fixed = client.create_collection("fixed_index", embedding_function=openai_ef)
    col_recursive = client.create_collection("recursive_index", embedding_function=openai_ef)
    col_semantic = client.create_collection("semantic_index", embedding_function=openai_ef)

    print(f"Indexing {len(pdf_files)} PDFs with 3072-dim embeddings...")
    for pdf_name in tqdm(pdf_files):
        raw_text = extract_text_from_pdf(os.path.join(PDF_DIR, pdf_name))
        text = preprocess_text(raw_text)
        
        # Indexing logic
        for strategy, col, func in [("fixed", col_fixed, get_fixed_chunks), 
                                     ("rec", col_recursive, get_recursive_chunks), 
                                     ("sem", col_semantic, get_semantic_chunks)]:
            chunks = func(text)
            if chunks:
                col.add(
                    documents=chunks,
                    metadatas=[{"doc_name": pdf_name} for _ in chunks],
                    ids=[f"{strategy}_{pdf_name}_{i}" for i in range(len(chunks))]
                )

    print("\n1 - Evaluation Mode\n2 - Interactive Mode")
    if input("Choice: ") == "2":
        simple_interactive_mode(col_semantic)
    else:
        print("Evaluation stats would be calculated here based on queries.txt...")

if __name__ == "__main__":
    main()