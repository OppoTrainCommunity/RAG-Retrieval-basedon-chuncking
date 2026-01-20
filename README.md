# RAG Resume Analysis System

**Author:** Ahmad F.Obaid

## Overview
The **RAG Resume Analysis System** is a production-grade Retrieval-Augmented Generation (RAG) application designed to process, index, and analyze resume documents. Built with a modular Python architecture, it leverages advanced NLP techniques, vector search, and Large Language Models (LLMs) to provide deep insights into candidate profiles.

This system moves beyond simple keyword matching by implementing semantic search, allowing HR professionals and recruiters to ask complex questions about candidates and receive evidence-backed answers.

## Key Features

### 🧠 Advanced RAG Architecture
*   **Modular Design**: Rebuilt from the ground up with a clean separation of concerns (`src/` package) for scalability.
*   **Multi-Model Support**: Seamlessly switch between state-of-the-art LLMs via OpenRouter (GPT-4, Claude 3.5 Sonnet, Llama 3).
*   **Vector Search**: Utilizes **ChromaDB** for efficient, persistent embedding storage and retrieval.

### 📄 Intelligent Document Processing
*   **PDF Ingestion**: Robust text extraction from PDF resumes using `pdfplumber`.
*   **Smart Chunking**: Includes three configurable strategies to optimize context retrieval:
    *   **Fixed**: Standard overlapping windows.
    *   **Semantic**: Sentence-boundary aware splitting using NLTK.
    *   **Recursive**: Hierarchical structure preservation.

### ⛓️ LangChain Integration
*   **RAG Chains**: sophisticated retrieval chains (Retriever → Prompt → LLM).
*   **LLM-as-a-Judge**: Built-in evaluation framework using a superior LLM to grade answer quality on:
    *   **Relevance**: Does the answer address the user's question?
    *   **Faithfulness**: Is the answer derived *only* from the resume context?
    *   **Correctness**: (Optional) Checks against provided ground truth.

### 📊 Comprehensive Evaluation
*   **Retrieval Metrics**: Precision, Recall, F1 Score, and Hit Rate analysis.
*   **Comparison UI**: dedicated interface to compare generation quality between different LLMs side-by-side.

## Project Structure

```text
.
├── app.py                  # Main Streamlit Application (Presentation Layer)
├── requirements.txt        # Project Dependencies
├── data/                   # Directory for PDF Resumes
├── src/                    # Core Logic Package
│   ├── config.py           # System Configuration & Constants
│   ├── document_processor.py # Text Extraction & Chunking Logic
│   ├── vector_store.py     # Embedding Generation & ChromaDB Management
│   ├── rag_engine.py       # LangChain & RAG Pipeline
│   └── evaluation.py       # Metrics & scoring
└── README.md               # Project Documentation
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   An **OpenRouter API Key** (for accessing embeddings and LLMs).

### Installation

1.  Clone the repository or navigate to the project directory.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Launch the Streamlit interface:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage Guide

### 1. Configuration & Ingestion
*   Navigate to the **⚙️ Configuration & Upload** tab.
*   Enter your **OpenRouter API Key**.
*   Select your desired **Embedding Model** (e.g., `text-embedding-3-small`) and **Chunking Strategy**.
*   Upload PDF resumes or use existing files in the `data/` folder.
*   Click **Process & Index Resumes**.

### 2. Multi-LLM Comparison
*   Navigate to the **⛓️ LangChain Comparison** tab.
*   Select the collection you created in step 1.
*   Choose two different LLMs (e.g., **GPT-4** vs **Claude 3.5 Sonnet**) to compare.
*   Ask a question (e.g., *"Which candidate has the most experience with Python?"*).
*   View generated answers, retrieval context, and automated quality scores.

---
*Built for the Advanced Agentic Coding Project.*
