"""
CV RAG System - Evaluation History Page
View and export evaluation results.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Evaluation - CV RAG System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Evaluation History")
st.markdown("View and analyze evaluation results from previous queries.")

# Check for saved evaluation results
from src.config import load_config

config = load_config("config.yaml")
eval_csv_path = Path(config.outputs.dir) / config.outputs.eval_csv
eval_json_path = Path(config.outputs.dir) / config.outputs.eval_json

if eval_csv_path.exists():
    # Load saved results
    eval_df = pd.read_csv(eval_csv_path)
    
    st.subheader("📈 Summary Statistics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Evaluations", len(eval_df))
    col2.metric("Avg Relevance", f"{eval_df['relevance_score'].mean():.2f}/5")
    col3.metric("Avg Faithfulness", f"{eval_df['faithfulness_score'].mean():.2f}/5")
    col4.metric("Avg Correctness", f"{eval_df['correctness_score'].mean():.2f}/5")
    col5.metric("Overall Average", f"{eval_df['average_score'].mean():.2f}/5")
    
    st.divider()
    
    # Results table
    st.subheader("📋 All Evaluations")
    
    # Select columns to display
    display_cols = [
        "query", "relevance_score", "faithfulness_score",
        "correctness_score", "average_score"
    ]
    available_cols = [c for c in display_cols if c in eval_df.columns]
    
    st.dataframe(eval_df[available_cols], use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Visualization
    st.subheader("📊 Score Distribution")
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    ax1 = axes[0]
    metrics = ["relevance_score", "faithfulness_score", "correctness_score"]
    x = range(len(eval_df))
    width = 0.25
    
    colors = ["#3498db", "#2ecc71", "#9b59b6"]
    labels = ["Relevance", "Faithfulness", "Correctness"]
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        offset = (i - 1) * width
        ax1.bar([xi + offset for xi in x], eval_df[metric], width, label=label, color=color)
    
    ax1.set_xlabel("Query Index")
    ax1.set_ylabel("Score (1-5)")
    ax1.set_title("Scores by Query")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{i+1}" for i in x])
    ax1.legend()
    ax1.set_ylim(0, 5.5)
    
    # Histogram
    ax2 = axes[1]
    ax2.hist(eval_df["average_score"], bins=5, range=(1, 5), color="#3498db", edgecolor="white")
    ax2.set_xlabel("Average Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Average Scores")
    ax2.axvline(x=eval_df["average_score"].mean(), color="red", linestyle="--", label=f'Mean: {eval_df["average_score"].mean():.2f}')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.divider()
    
    # Download buttons
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = eval_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="cv_rag_eval_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col2:
        if eval_json_path.exists():
            with open(eval_json_path, 'r') as f:
                json_data = f.read()
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="cv_rag_eval_results.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    st.info("""
    📝 **No evaluation results found.**
    
    Run queries with evaluation enabled to generate results:
    1. Go to the Query page
    2. Enable "Run Evaluation" in the sidebar
    3. Submit some queries
    
    Or run the main notebook to generate batch evaluations.
    """)
    
    # Show expected file location
    st.code(f"Expected file: {eval_csv_path}")
