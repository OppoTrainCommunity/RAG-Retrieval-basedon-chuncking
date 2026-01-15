import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st

from src.services.retrieval_service import RetrievalService
from src.chains.chain_factory import ChainFactory  # builds two chains (2 models)


# -----------------------------
# Helpers
# -----------------------------
def load_file_names_from_cvs_json(cvs_json_path: str) -> List[str]:
    p = Path(cvs_json_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        docs = json.load(f)
    return sorted({d.get("file_name") for d in docs if d.get("file_name")})


def parse_csv_list(s: str) -> List[str]:
    # "Python, Django, REST" -> ["Python", "Django", "REST"]
    items = [x.strip() for x in (s or "").split(",")]
    return [x for x in items if x]


def show_model_output(raw_text: str):
    """
    Tries to show JSON nicely; falls back to raw text.
    """
    try:
        data = json.loads(raw_text)
        st.json(data)
    except Exception:
        st.text_area("Model output (raw)", value=raw_text, height=350)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Recruitment RAG – LangChain", layout="wide")
st.title("Recruitment RAG – Candidate Ranking (LangChain + OpenRouter)")

with st.sidebar:
    st.header("Vector DB (already indexed)")
    persist_dir = st.text_input("persist_directory", value="./chroma_db")
    collection_name = st.text_input("collection_name", value="paragraph_chunking")
    embedding_model = st.text_input("embedding_model", value="all-mpnet-base-v2")

    st.divider()
    st.header("Retrieval")
    k = st.slider("Top K retrieved chunks", 5, 120, 40, 5)

    st.divider()
    st.header("Candidate scope (optional)")
    cvs_json_path = st.text_input("CVs.json path (for file_name filter)", value="data/CVs.json")
    use_filter = st.checkbox("Filter retrieval to ONE CV file", value=False)

    st.divider()
    st.header("Output")
    top_n = st.slider("Return top N candidates", 1, 10, 5, 1)


# Init retrieval service + chains
retrieval_service = RetrievalService(persist_dir=persist_dir, embedding_model=embedding_model)
chains = ChainFactory.build_two_chains(retrieval_service, collection_name=collection_name)

# Filter (optional)
file_names = load_file_names_from_cvs_json(cvs_json_path) if use_filter else []
selected_file_name: Optional[str] = None
if use_filter:
    if file_names:
        selected_file_name = st.selectbox("Select file_name", file_names)
    else:
        st.warning("No file names found. Check CVs.json path.")
where = {"file_name": selected_file_name} if (use_filter and selected_file_name) else None

# Main form inputs
st.subheader("Job Requirements")

col1, col2 = st.columns(2)
with col1:
    role_title = st.text_input("Role title", value="Backend Developer")
    seniority = st.selectbox("Seniority", ["any", "intern", "junior", "mid", "senior"], index=0)
with col2:
    must_have_str = st.text_input("Must-have skills (comma-separated)", value="Python, Django, REST APIs")
    nice_to_have_str = st.text_input("Nice-to-have skills (comma-separated)", value="PostgreSQL, Docker, CI/CD")

job_notes = st.text_area("Job notes (optional)", value="We prefer candidates with real projects and clean API design.", height=80)

st.subheader("Recruitment Question")
question = st.text_input(
    "Ask something like: 'Who is the best candidate for this role?'",
    value="Who is the best candidate for this backend role?"
)

must_have = parse_csv_list(must_have_str)
nice_to_have = parse_csv_list(nice_to_have_str)

st.divider()

# Two-model comparison UI
st.subheader("Run LangChain RAG (2 models comparison)")

model_a_name, model_b_name = list(chains.keys())[0], list(chains.keys())[1]
chain_a = chains[model_a_name]
chain_b = chains[model_b_name]

run = st.button("✨ Rank Candidates (run both models)", type="primary")

if run:
    if not must_have:
        st.error("Must-have skills list is empty. Add at least 1 skill.")
        st.stop()

    payload_common = dict(
        question=question,
        role_title=role_title,
        must_have=must_have,
        nice_to_have=nice_to_have,
        seniority=seniority,
        job_notes=job_notes,
        top_n=top_n,
        where=where,
        k=k,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"### Model A: `{model_a_name}`")
        try:
            out_a = chain_a.invoke(**payload_common)
            show_model_output(out_a)
        except Exception as e:
            st.error("Model A failed.")
            st.code(str(e))

    with c2:
        st.markdown(f"### Model B: `{model_b_name}`")
        try:
            out_b = chain_b.invoke(**payload_common)
            show_model_output(out_b)
        except Exception as e:
            st.error("Model B failed.")
            st.code(str(e))

    st.divider()
    st.info(
        "Tip: For true recruitment ranking across all candidates, keep the filter OFF. "
        "Turn filter ON only for debugging a single CV."
    )
