import json
import tempfile
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

from resume_rag.pipeline import analyze_resume

st.set_page_config(
    page_title="Resume → Career Path Advisor",
    layout="wide",
)

st.title("📄 Resume → Career Path Advisor")
st.markdown(
    "Upload a PDF resume and the system will extract text, create semantic chunks, "
    "store them in a vector DB, perform RAG search, and generate AI-powered career suggestions."
)

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    st.write(f"**File name:** {uploaded_file.name}")
    st.write(f"**Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")

    if st.button("Analyze Resume"):
        with st.spinner("Processing resume..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            result = analyze_resume(tmp_path)

        # ===============================
        st.subheader("🧾 Extracted Resume Text")
        st.text(result["parsed_text"])

        # ===============================
        st.subheader("🧩 Semantic Chunks")
        for i, ch in enumerate(result["chunks"]):
            st.markdown(f"### Chunk {i+1}")
            st.text(ch)
            st.markdown("---")

        # ===============================
        st.subheader("📚 RAG Similarity Search")
        st.write(result["similar_profiles"])

        # ===============================
        st.subheader("💼 AI Career Recommendations")
        st.text(result["career_recommendations"])

        # ===============================
        st.subheader("📊 Resume Evaluation Score")

        try:
            eval_json = json.loads(result["evaluation"])

            st.metric("Overall Score", eval_json["overall_score"])

            st.markdown("### Section Scores")
            st.write(eval_json["section_scores"])

            st.markdown("### Strengths")
            for s in eval_json["strengths"]:
                st.markdown(f"✅ {s}")

            st.markdown("### Weaknesses")
            for w in eval_json["weaknesses"]:
                st.markdown(f"⚠️ {w}")

            st.markdown("### Recommendations")
            for r in eval_json["recommendations"]:
                st.markdown(f"💡 {r}")

        except Exception as e:
            st.error(f"JSON Parse Error: {e}")
            st.text(result["evaluation"])

else:
    st.info("Upload a PDF resume to get started.")
