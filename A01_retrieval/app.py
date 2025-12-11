import streamlit as st
import pandas as pd
import json
import csv
import os

# Paths
RESULT_A = "A01_retrieval/eval/results_strategy_A.csv"
RESULT_B = "A01_retrieval/eval/results_strategy_B.csv"
QUERIES = "A01_retrieval/eval/queries.csv"
GOLD = "A01_retrieval/eval/gold_answers.csv"

# ---------------------
# Load CSV files
# ---------------------
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)

    # detect if this file is a results CSV
    is_results_file = ("precision@k" in df.columns) and ("recall@k" in df.columns)

    if is_results_file:
        # Add F1 score safely
        eps = 1e-9  # avoid division by zero
        df["f1_score"] = 2 * (df["precision@k"] * df["recall@k"]) / (
            df["precision@k"] + df["recall@k"] + eps
        )

    return df


df_a = load_csv(RESULT_A)
df_b = load_csv(RESULT_B)
queries = load_csv(QUERIES)
gold = load_csv(GOLD)

st.title("📊 Retrieval Evaluation Dashboard – CV RAG System")
st.write("This dashboard helps you compare Strategy A vs Strategy B clearly and visually.")

# ---------------------
# Summary Section
# ---------------------
st.header("📌 Overall Metrics Comparison")

metrics = ["precision@k", "recall@k", "f1_score", "hit_rate@k", "rr", "latency_sec"]

summary = pd.DataFrame({
    "Metric": metrics,
    "Strategy A (avg)": [df_a[m].mean() for m in metrics],
    "Strategy B (avg)": [df_b[m].mean() for m in metrics]
})

st.dataframe(summary)

st.subheader("📊 Metrics Bar Chart")
st.bar_chart(summary.set_index("Metric"))

# ---------------------
# Per Query Analysis
# ---------------------
st.header("🔍 Per-Query Results")

query_id = st.selectbox("Select Query ID", df_a["query_id"].unique())

row_a = df_a[df_a["query_id"] == query_id].iloc[0]
row_b = df_b[df_b["query_id"] == query_id].iloc[0]

q_text = row_a["query_text"]
expected = gold[gold["query_id"] == query_id]["expected_answer"].values[0]

st.subheader(f"📝 Query: {q_text}")
st.info(f"**Expected Answer:** {expected}")

col1, col2 = st.columns(2)

with col1:
    st.write("### Strategy A")
    st.json(row_a.to_dict())

with col2:
    st.write("### Strategy B")
    st.json(row_b.to_dict())

# ---------------------
# Compare A vs B for this query
# ---------------------
st.header("⚔️ A vs B – Metric Comparison for this Query")

comparison_df = pd.DataFrame({
    "metric": metrics,
    "A": [row_a[m] for m in metrics],
    "B": [row_b[m] for m in metrics],
})

st.dataframe(comparison_df)
st.bar_chart(comparison_df.set_index("metric"))

# ---------------------
# Show Raw Results
# ---------------------
st.header("📂 Raw Evaluation Data")
tabA, tabB = st.tabs(["Strategy A Results", "Strategy B Results"])

with tabA:
    st.dataframe(df_a)

with tabB:
    st.dataframe(df_b)

st.success("Dashboard loaded successfully!")
