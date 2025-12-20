"""
RAG Resume Analysis - Streamlit Application
Ground-truth based evaluation for comparing embedding models and chunking strategies
"""

import streamlit as st
import json
import hashlib
import time
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import pdfplumber
import chromadb
from openai import OpenAI
import plotly.express as px

# NLTK for sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    NLTK_AVAILABLE = True
except:
    NLTK_AVAILABLE = False

# ===========================================
# Configuration
# ===========================================
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"

for d in [DATA_DIR, CACHE_DIR, OUTPUT_DIR, CHROMA_DIR]:
    d.mkdir(exist_ok=True)

EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "openai/text-embedding-ada-002",
]
CHUNKING_STRATEGIES = ["fixed", "semantic", "recursive"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ===========================================
# Core Functions
# ===========================================

def get_cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

def load_from_cache(cache_key: str) -> Optional[List[float]]:
    path = CACHE_DIR / f"{cache_key}.json"
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f).get('embedding')
        except:
            pass
    return None

def save_to_cache(cache_key: str, embedding: List[float], model: str):
    try:
        path = CACHE_DIR / f"{cache_key}.json"
        with open(path, 'w') as f:
            json.dump({'embedding': embedding, 'model': model}, f)
    except:
        pass

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    if not api_key or len(api_key) < 10:
        return False, "API key too short"
    try:
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        client.embeddings.create(model="openai/text-embedding-3-small", input="test")
        return True, "Valid ✓"
    except Exception as e:
        if "401" in str(e):
            return False, "Invalid key"
        return False, str(e)[:50]

def get_embedding(text: str, model: str, api_key: str) -> List[float]:
    cache_key = get_cache_key(text, model)
    cached = load_from_cache(cache_key)
    if cached:
        return cached
    
    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    for attempt in range(3):
        try:
            response = client.embeddings.create(model=model, input=text)
            embedding = response.data[0].embedding
            save_to_cache(cache_key, embedding, model)
            return embedding
        except Exception as e:
            if attempt < 2:
                time.sleep(1 * (2 ** attempt))
            else:
                raise e

def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n\n".join([p.extract_text() or "" for p in pdf.pages]).strip()
    except:
        return ""

# ===========================================
# Chunking Functions
# ===========================================

def fixed_chunking(text: str, size: int = 500, overlap: int = 50) -> List[Dict]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({"text": text[start:end], "index": len(chunks)})
        start += size - overlap
    return chunks

def semantic_chunking(text: str, target_size: int = 500) -> List[Dict]:
    if not text:
        return []
    
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current = []
    current_len = 0
    
    for sent in sentences:
        if current_len + len(sent) > target_size and current:
            chunks.append({"text": " ".join(current), "index": len(chunks)})
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent) + 1
    
    if current:
        chunks.append({"text": " ".join(current), "index": len(chunks)})
    return chunks

def recursive_chunking(text: str, max_size: int = 500) -> List[Dict]:
    if not text:
        return []
    if len(text) <= max_size:
        return [{"text": text, "index": 0}]
    
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    
    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append({"text": para, "index": len(chunks)})
        else:
            for sub in fixed_chunking(para, max_size, 50):
                sub["index"] = len(chunks)
                chunks.append(sub)
    return chunks

def get_chunks(text: str, strategy: str, size: int = 500, overlap: int = 50) -> List[Dict]:
    if strategy == "fixed":
        return fixed_chunking(text, size, overlap)
    elif strategy == "semantic":
        return semantic_chunking(text, size)
    elif strategy == "recursive":
        return recursive_chunking(text, size)
    return fixed_chunking(text, size, overlap)

# ===========================================
# ChromaDB Functions
# ===========================================

_client = None

def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client

def get_collection_name(model: str, strategy: str) -> str:
    return f"{model.replace('/', '_').replace('-', '_')}_{strategy}"

def get_collection(name: str):
    return get_client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

def index_chunks(chunks: List[Dict], embeddings: List, source: str, collection, model: str, strategy: str) -> int:
    if not chunks:
        return 0
    
    ids = [f"{source}_{strategy}_{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [{"source_file": source, "chunk_index": i, "model": model, "strategy": strategy} for i in range(len(chunks))]
    
    collection.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(ids)

def search(query: str, collection, model: str, api_key: str, top_k: int = 5) -> List[Dict]:
    emb = get_embedding(query, model, api_key)
    results = collection.query(query_embeddings=[emb], n_results=top_k, include=["documents", "metadatas", "distances"])
    
    if not results or not results['ids'] or not results['ids'][0]:
        return []
    
    return [{
        "id": results['ids'][0][i],
        "text": results['documents'][0][i],
        "source": results['metadatas'][0][i].get('source_file', ''),
        "similarity": 1 - results['distances'][0][i]
    } for i in range(len(results['ids'][0]))]

def list_collections() -> List[str]:
    return [c.name for c in get_client().list_collections()]

def delete_collection(name: str):
    try:
        get_client().delete_collection(name)
        return True
    except:
        return False

def get_collection_count(name: str) -> int:
    try:
        return get_client().get_collection(name).count()
    except:
        return 0

# ===========================================
# Ground Truth Evaluation
# ===========================================

def load_ground_truth() -> Dict:
    """Load ground truth from data/ground_truth.json"""
    path = DATA_DIR / "ground_truth.json"
    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {}

def check_relevance(retrieved_text: str, expected_values: List[Dict]) -> Tuple[bool, str]:
    """
    Check if retrieved text contains any of the expected values.
    Returns (is_relevant, matched_value)
    """
    text_lower = retrieved_text.lower()
    
    for item in expected_values:
        value = item.get("value", "") if isinstance(item, dict) else str(item)
        if not value:
            continue
        
        value_lower = value.lower()
        
        # Exact substring match
        if value_lower in text_lower:
            return True, value
        
        # Word-based fuzzy match (at least 70% of words present)
        value_words = set(value_lower.split())
        if value_words:
            text_words = set(text_lower.split())
            overlap = len(value_words & text_words) / len(value_words)
            if overlap >= 0.7:
                return True, value
    
    return False, ""

def evaluate_query(query: str, query_id: str, ground_truth: Dict, 
                   collection, model: str, api_key: str, top_k: int) -> Dict:
    """Evaluate a single query against ground truth."""
    
    start = time.time()
    results = search(query, collection, model, api_key, top_k)
    latency = (time.time() - start) * 1000
    
    # Get expected values from ground truth
    gt_entry = ground_truth.get(query_id, {})
    expected = gt_entry.get("relevant", [])
    
    if not expected:
        return {
            "query": query,
            "query_id": query_id,
            "retrieved": results,
            "hits": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "mrr": 0,
            "latency_ms": latency,
            "has_ground_truth": False
        }
    
    # Count hits and find first hit position
    hits = 0
    first_hit_rank = 0
    matched_values = set()
    
    for rank, r in enumerate(results, 1):
        is_relevant, matched = check_relevance(r["text"], expected)
        if is_relevant:
            hits += 1
            matched_values.add(matched)
            if first_hit_rank == 0:
                first_hit_rank = rank
    
    # Calculate metrics
    total_relevant = len(expected)
    precision = hits / len(results) if results else 0
    recall = len(matched_values) / total_relevant if total_relevant else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mrr = 1 / first_hit_rank if first_hit_rank > 0 else 0
    
    return {
        "query": query,
        "query_id": query_id,
        "retrieved": results,
        "hits": hits,
        "matched_values": list(matched_values),
        "expected_count": total_relevant,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": mrr,
        "latency_ms": latency,
        "has_ground_truth": True
    }

# ===========================================
# Streamlit App
# ===========================================

st.set_page_config(page_title="RAG Resume Analysis", page_icon="📄", layout="wide")

# Session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "api_valid" not in st.session_state:
    st.session_state.api_valid = False

# Sidebar
st.sidebar.title("📄 RAG Resume Analysis")
page = st.sidebar.radio("", ["⚙️ Config & Index", "❓ Ask & Add GT", "📊 Evaluate"])

# ===========================================
# Page 1: Configuration & Indexing
# ===========================================
if page == "⚙️ Config & Index":
    st.title("⚙️ Configuration & Indexing")
    
    # API Key
    st.subheader("🔑 API Key")
    col1, col2 = st.columns([4, 1])
    with col1:
        api_key = st.text_input("OpenRouter API Key", type="password", value=st.session_state.api_key)
        st.session_state.api_key = api_key
    with col2:
        st.write("")
        st.write("")
        if st.button("Validate"):
            valid, msg = validate_api_key(api_key)
            st.session_state.api_valid = valid
            if valid:
                st.success(msg)
            else:
                st.error(msg)
    
    if st.session_state.api_valid:
        st.success("✓ API Key validated")
    
    st.divider()
    
    # Model & Strategy
    st.subheader("🤖 Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Embedding Model", EMBEDDING_MODELS)
    with col2:
        strategy = st.selectbox("Chunking Strategy", CHUNKING_STRATEGIES)
    with col3:
        chunk_size = st.number_input("Chunk Size", 100, 2000, 500)
    
    st.divider()
    
    # Index
    st.subheader("📁 Index Resumes")
    
    pdfs = list(DATA_DIR.glob("*.pdf"))
    st.info(f"Found {len(pdfs)} PDFs in data/ folder")
    
    if st.button("🚀 Index All", type="primary", disabled=not st.session_state.api_valid):
        if not pdfs:
            st.error("No PDFs found")
        else:
            coll_name = get_collection_name(model, strategy)
            collection = get_collection(coll_name)
            
            progress = st.progress(0)
            status = st.empty()
            total = 0
            
            for i, pdf in enumerate(pdfs):
                status.text(f"Processing: {pdf.name}")
                
                text = extract_text_from_pdf(str(pdf))
                if not text:
                    continue
                
                chunks = get_chunks(text, strategy, chunk_size)
                if not chunks:
                    continue
                
                embeddings = [get_embedding(c["text"], model, api_key) for c in chunks]
                indexed = index_chunks(chunks, embeddings, pdf.name, collection, model, strategy)
                total += indexed
                
                progress.progress((i + 1) / len(pdfs))
            
            status.text("Done!")
            st.success(f"✅ Indexed {total} chunks into: {coll_name}")
    
    st.divider()
    
    # Collections
    st.subheader("🗄️ Collections")
    collections = list_collections()
    
    if collections:
        for name in collections:
            col1, col2, col3 = st.columns([4, 1, 1])
            col1.write(f"📁 {name}")
            col2.write(f"{get_collection_count(name)} chunks")
            if col3.button("Delete", key=f"del_{name}"):
                delete_collection(name)
                st.rerun()
    else:
        st.info("No collections yet")

# ===========================================
# Page 2: Ask Questions & Add Ground Truth
# ===========================================
elif page == "❓ Ask & Add GT":
    st.title("❓ Ask Questions & Add Ground Truth")
    
    if not st.session_state.api_valid:
        st.warning("⚠️ Validate API key first")
        st.stop()
    
    collections = list_collections()
    if not collections:
        st.warning("No collections. Index resumes first.")
        st.stop()
    
    # Settings
    col1, col2 = st.columns(2)
    with col1:
        selected_coll = st.selectbox("Collection", collections, key="ask_coll")
    with col2:
        top_k = st.number_input("Top-K Results", 1, 20, 5, key="ask_topk")
    
    # Find model from collection name
    ask_model = EMBEDDING_MODELS[0]
    for m in EMBEDDING_MODELS:
        if m.replace("/", "_").replace("-", "_") in selected_coll:
            ask_model = m
            break
    
    st.caption(f"Using model: {ask_model}")
    
    st.divider()
    
    # Ask Question
    st.subheader("🔍 Ask a Question")
    query = st.text_area("Enter your question:", height=80, placeholder="e.g., Who has Python experience?")
    
    if st.button("🔎 Search", type="primary") and query:
        collection = get_collection(selected_coll)
        
        with st.spinner("Searching..."):
            start = time.time()
            results = search(query, collection, ask_model, st.session_state.api_key, top_k)
            latency = (time.time() - start) * 1000
        
        st.success(f"Found {len(results)} results in {latency:.1f}ms")
        
        # Store in session for ground truth adding
        st.session_state.last_query = query
        st.session_state.last_results = results
        
        # Display results
        for i, r in enumerate(results):
            with st.expander(f"#{i+1} - {r['source']} (similarity: {r['similarity']:.3f})", expanded=(i==0)):
                st.text(r["text"])
    
    st.divider()
    
    # Add Ground Truth
    st.subheader("➕ Add to Ground Truth")
    
    # Load existing ground truth
    ground_truth = load_ground_truth()
    st.info(f"Current ground truth has {len(ground_truth)} queries")
    
    with st.form("add_gt_form"):
        st.write("Add a new ground truth entry:")
        
        # Pre-fill with last query if available
        default_query = st.session_state.get("last_query", "")
        
        gt_query = st.text_input("Query Text", value=default_query)
        gt_expected = st.text_area(
            "Expected Values (one per line)", 
            height=100,
            help="Enter the expected values/answers that should be found in relevant chunks"
        )
        gt_source = st.text_input("Source File (optional)", placeholder="e.g., john_doe_cv.pdf")
        
        submitted = st.form_submit_button("➕ Add Ground Truth")
        
        if submitted and gt_query and gt_expected:
            # Generate query ID
            existing_ids = [k for k in ground_truth.keys() if k.startswith("q_")]
            max_id = max([int(k.split("_")[1]) for k in existing_ids]) if existing_ids else -1
            new_id = f"q_{max_id + 1}"
            
            # Parse expected values
            expected_list = [v.strip() for v in gt_expected.strip().split("\n") if v.strip()]
            relevant = [{"type": "text", "value": v, "source_file": gt_source} for v in expected_list]
            
            # Add to ground truth
            ground_truth[new_id] = {
                "query_text": gt_query,
                "relevant": relevant
            }
            
            # Save to file
            gt_path = DATA_DIR / "ground_truth.json"
            with open(gt_path, 'w', encoding='utf-8') as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)
            
            st.success(f"✅ Added ground truth entry: {new_id}")
            st.rerun()
    
    st.divider()
    
    # View/Edit Ground Truth
    st.subheader("📋 Current Ground Truth")
    
    if ground_truth:
        for qid, entry in ground_truth.items():
            query_text = entry.get("query_text", qid)
            relevant = entry.get("relevant", [])
            
            with st.expander(f"{qid}: {query_text[:60]}..."):
                st.write(f"**Query:** {query_text}")
                st.write(f"**Expected Values ({len(relevant)}):**")
                for r in relevant:
                    value = r.get("value", r) if isinstance(r, dict) else r
                    source = r.get("source_file", "") if isinstance(r, dict) else ""
                    st.write(f"  - {value}" + (f" ({source})" if source else ""))
                
                # Delete button
                if st.button(f"🗑️ Delete {qid}", key=f"del_gt_{qid}"):
                    del ground_truth[qid]
                    gt_path = DATA_DIR / "ground_truth.json"
                    with open(gt_path, 'w', encoding='utf-8') as f:
                        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
                    st.success(f"Deleted {qid}")
                    st.rerun()
    else:
        st.info("No ground truth entries yet. Add one above!")
    
    st.divider()
    
    # Quick Test Against Ground Truth
    st.subheader("🧪 Quick Test")
    
    if ground_truth and st.session_state.get("last_query") and st.session_state.get("last_results"):
        st.write(f"**Last Query:** {st.session_state.last_query}")
        
        # Find if this query matches any ground truth
        matched_gt = None
        for qid, entry in ground_truth.items():
            if entry.get("query_text", "").lower() == st.session_state.last_query.lower():
                matched_gt = (qid, entry)
                break
        
        if matched_gt:
            qid, entry = matched_gt
            expected = entry.get("relevant", [])
            
            st.write(f"**Matching Ground Truth:** {qid}")
            
            # Check each result
            hits = 0
            matched_values = set()
            for r in st.session_state.last_results:
                is_relevant, matched = check_relevance(r["text"], expected)
                if is_relevant:
                    hits += 1
                    matched_values.add(matched)
            
            precision = hits / len(st.session_state.last_results) if st.session_state.last_results else 0
            recall = len(matched_values) / len(expected) if expected else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Hits", hits)
            col2.metric("Precision", f"{precision:.3f}")
            col3.metric("Recall", f"{recall:.3f}")
            col4.metric("F1", f"{f1:.3f}")
            
            if matched_values:
                st.write(f"**Matched:** {', '.join(matched_values)}")
        else:
            st.info("No matching ground truth for this query. Add one above!")

# ===========================================
# Page 3: Evaluation
# ===========================================
elif page == "📊 Evaluate":
    st.title("📊 Ground Truth Evaluation")
    
    if not st.session_state.api_valid:
        st.warning("⚠️ Validate API key first")
        st.stop()
    
    collections = list_collections()
    if not collections:
        st.warning("No collections. Index resumes first.")
        st.stop()
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    if not ground_truth:
        st.error("❌ No ground truth found. Add data/ground_truth.json")
        st.stop()
    
    st.success(f"✅ Loaded {len(ground_truth)} queries from ground_truth.json")
    
    # Settings
    st.subheader("🎯 Settings")
    col1, col2 = st.columns(2)
    with col1:
        selected_coll = st.selectbox("Collection", collections)
    with col2:
        top_k = st.number_input("Top-K", 1, 20, 5)
    
    # Find model from collection name
    eval_model = EMBEDDING_MODELS[0]
    for m in EMBEDDING_MODELS:
        if m.replace("/", "_").replace("-", "_") in selected_coll:
            eval_model = m
            break
    
    st.caption(f"Using model: {eval_model}")
    
    st.divider()
    
    # Run Evaluation
    if st.button("🚀 Run Evaluation", type="primary"):
        collection = get_collection(selected_coll)
        
        results = []
        progress = st.progress(0)
        
        query_ids = list(ground_truth.keys())
        
        for i, qid in enumerate(query_ids):
            query_text = ground_truth[qid].get("query_text", qid)
            result = evaluate_query(query_text, qid, ground_truth, collection, eval_model, st.session_state.api_key, top_k)
            results.append(result)
            progress.progress((i + 1) / len(query_ids))
        
        # Summary metrics
        st.subheader("📈 Summary")
        
        valid_results = [r for r in results if r["has_ground_truth"]]
        
        if valid_results:
            avg_precision = np.mean([r["precision"] for r in valid_results])
            avg_recall = np.mean([r["recall"] for r in valid_results])
            avg_f1 = np.mean([r["f1"] for r in valid_results])
            avg_mrr = np.mean([r["mrr"] for r in valid_results])
            hit_rate = sum(1 for r in valid_results if r["hits"] > 0) / len(valid_results)
            avg_latency = np.mean([r["latency_ms"] for r in valid_results])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{avg_precision:.3f}")
            col2.metric("Recall", f"{avg_recall:.3f}")
            col3.metric("F1 Score", f"{avg_f1:.3f}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MRR", f"{avg_mrr:.3f}")
            col2.metric("Hit Rate", f"{hit_rate:.1%}")
            col3.metric("Avg Latency", f"{avg_latency:.1f} ms")
        
        st.divider()
        
        # Per-query table
        st.subheader("📋 Per-Query Results")
        
        df = pd.DataFrame([{
            "Query": r["query"][:50] + "..." if len(r["query"]) > 50 else r["query"],
            "Hits": r["hits"],
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1": f"{r['f1']:.3f}",
            "MRR": f"{r['mrr']:.3f}",
            "Latency": f"{r['latency_ms']:.1f}ms"
        } for r in results])
        
        st.dataframe(df, use_container_width=True)
        
        # Chart
        fig = px.bar(
            x=[r["query"][:30] for r in results],
            y=[r["f1"] for r in results],
            labels={"x": "Query", "y": "F1 Score"},
            title="F1 Score by Query"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Details
        st.subheader("🔍 Details")
        
        for r in results:
            with st.expander(f"{r['query_id']}: {r['query'][:60]}..."):
                st.write(f"**Hits:** {r['hits']} | **Precision:** {r['precision']:.3f} | **Recall:** {r['recall']:.3f}")
                
                if r.get("matched_values"):
                    st.write(f"**Matched:** {', '.join(r['matched_values'])}")
                
                st.write("**Retrieved chunks:**")
                for j, chunk in enumerate(r["retrieved"][:3]):
                    st.markdown(f"**#{j+1}** (sim: {chunk['similarity']:.3f}) - {chunk['source']}")
                    st.text(chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"])
        
        # Export
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "evaluation.csv", "text/csv")
        with col2:
            export = {"collection": selected_coll, "top_k": top_k, "results": results}
            st.download_button("📥 Download JSON", json.dumps(export, indent=2, default=str), "evaluation.json")
    
    st.divider()
    
    # Compare collections
    st.subheader("🔬 Compare Collections")
    
    if len(collections) > 1:
        selected = st.multiselect("Select to compare", collections)
        
        if st.button("Compare") and len(selected) > 1:
            comparison = []
            
            for coll_name in selected:
                coll = get_collection(coll_name)
                
                # Find model
                coll_model = EMBEDDING_MODELS[0]
                for m in EMBEDDING_MODELS:
                    if m.replace("/", "_").replace("-", "_") in coll_name:
                        coll_model = m
                        break
                
                results = []
                for qid in ground_truth.keys():
                    query_text = ground_truth[qid].get("query_text", qid)
                    r = evaluate_query(query_text, qid, ground_truth, coll, coll_model, st.session_state.api_key, top_k)
                    results.append(r)
                
                valid = [r for r in results if r["has_ground_truth"]]
                comparison.append({
                    "Collection": coll_name,
                    "Precision": np.mean([r["precision"] for r in valid]),
                    "Recall": np.mean([r["recall"] for r in valid]),
                    "F1": np.mean([r["f1"] for r in valid]),
                    "MRR": np.mean([r["mrr"] for r in valid]),
                    "Hit Rate": sum(1 for r in valid if r["hits"] > 0) / len(valid) if valid else 0
                })
            
            comp_df = pd.DataFrame(comparison)
            st.dataframe(comp_df, use_container_width=True)
            
            fig = px.bar(comp_df, x="Collection", y="F1", title="F1 Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
