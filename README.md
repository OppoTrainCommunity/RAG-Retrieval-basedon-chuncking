# RAG Resume Analysis

## Overview
This project is a Retrieval-Augmented Generation (RAG) application designed for Resume Analysis. It allows users to index PDF resumes, perform semantic searches, and evaluate the performance of different embedding models and chunking strategies against a ground truth dataset.

**Author:** Ahmad F.Obaid

## Features
*   **Indexing & Configuration:**
    *   Support for multiple embedding models (OpenAI/OpenRouter).
    *   Various chunking strategies: Fixed, Semantic, and Recursive.
    *   PDF text extraction and indexing into ChromaDB.
*   **Query & Ground Truth:**
    *   Semantic search functionality to query indexed resumes.
    *   Ability to add and manage ground truth data for evaluation.
*   **Evaluation:**
    *   Comprehensive evaluation metrics: Precision, Recall, F1 Score, MRR.
    *   Detailed per-query analysis and visualization.
    *   Comparison tools for different collections/strategies.

## Setup & Running

### Prerequisites
*   Python 3.8+
*   An OpenRouter API Key

### Installation
1.  Navigate to the project directory.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To start the Streamlit application, run the following command in your terminal:

```bash
streamlit run app.py
```

Once running, access the application in your web browser (typically at `http://localhost:8501`).

1.  **Configure**: Enter your API key and select your preferred model and chunking strategy.
2.  **Index**: Place your PDF resumes in the `data/` directory and click "Index All".
3.  **Evaluate**: Use the query interface to search or switch to the evaluation tab to benchmark performance.
