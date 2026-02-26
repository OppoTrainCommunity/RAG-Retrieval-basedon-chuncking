# Converted from Rag.ipynb
# Auto-generated .py version of the notebook


# --- Markdown cell 1 ---
# # 📄 PDF RAG Pipeline


# --- Code cell 2 ---
import os
import pathlib
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import re
import requests
from typing import List
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

print("✅ All libraries imported successfully")


# --- Code cell 3 ---
"""
Configuration Settings
"""

# API Configuration
OS_API_KEY = os.getenv("OPENROUTER_API_KEY")
PDF_DIR = "pdfs"
CHROMA_PATH = "./chroma_db"

# Text Processing Parameters
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_RESULTS = 3

# LLM Parameters
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic/claude-3.5-sonnet")
TEMPERATURE = 0.0
MAX_TOKENS = 600

# Embedding Parameters
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Create necessary directories
pathlib.Path(PDF_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)

print(f"📁 PDF Directory: {PDF_DIR}")
print(f"💾 ChromaDB Path: {CHROMA_PATH}")


# --- Code cell 4 ---
def generate_embeddings(texts: List[str], api_key: str, model_name: str = EMBEDDING_MODEL) -> List[List[float]]:
    """
    Generate embeddings for input texts using OpenRouter API.
    
    Args:
        texts: List of text strings to embed
        api_key: OpenRouter API key
        model_name: Model identifier on OpenRouter
        
    Returns:
        List of embedding vectors
    """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Resume-RAG-System"
            },
            json={
                "model": model_name,
                "input": texts
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            embeddings = [item['embedding'] for item in result.get('data', [])]
            
            if not embeddings:
                raise ValueError("No embeddings returned from API")
            
            return embeddings
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
            
    except requests.exceptions.Timeout:
        raise ValueError("Request timed out while generating embeddings")
    except Exception as e:
        raise ValueError(f"Embedding generation failed: {str(e)}")


# Create embedding function wrapper for ChromaDB
def create_embedding_function(api_key: str, model_name: str = EMBEDDING_MODEL):
    """Create a ChromaDB-compatible embedding function."""
    
    class OpenRouterEmbeddings(embedding_functions.EmbeddingFunction):
        def __call__(self, input: List[str]) -> List[List[float]]:
            return generate_embeddings(input, api_key, model_name)
    
    return OpenRouterEmbeddings()


print("✅ embedding function defined")


# --- Code cell 5 ---
"""
PDF Processing Utilities
"""

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text", sort=True) + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """
    Clean and normalize extracted text.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval.
    
    Args:
        text: Text to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)


print("✅ Text processing functions defined")


# --- Code cell 6 ---
"""
LLM Invocation Function
"""

def invoke_llm(prompt: str, model_name: str = MODEL_NAME, 
               temperature: float = TEMPERATURE, 
               max_tokens: int = MAX_TOKENS) -> str:
    """
    Generate a response from the LLM using OpenRouter API.
    
    Args:
        prompt: Input prompt string
        model_name: Model identifier on OpenRouter
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        
    Returns:
        Generated text response
    """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OS_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Resume-RAG-System"
            },
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {str(e)}"


print("✅ LLM function defined")


# --- Code cell 7 ---
"""
ChromaDB Retrieval Function
"""

def retrieve_documents(query: str, collection, k: int = TOP_K_RESULTS) -> List[Document]:
    """
    Retrieve relevant documents for a query from ChromaDB.
    
    Args:
        query: Search query string
        collection: ChromaDB collection instance
        k: Number of documents to retrieve
        
    Returns:
        List of LangChain Document objects
    """
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    
    documents = []
    if results['documents'] and results['documents'][0]:
        for i, doc_text in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            documents.append(
                Document(page_content=doc_text, metadata=metadata)
            )
    
    return documents


print("✅ Retriever function defined")


# --- Code cell 8 ---
"""
RAG Prompt Template
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a professional Resume Screening Assistant specialized in analyzing candidate qualifications.

Context from Resume(s):
{context}

Question: {question}

Instructions:
- Answer using ONLY the information provided in the context above
- Be specific and cite relevant details from the resume
- If the requested information is not available in the context, explicitly state: "This information is not available in the resume"
- Do not make assumptions or add information not present in the context
- Keep your answer concise and focused on the question

Answer:"""
)

print("✅ Prompt template created")
print(f"\nTemplate variables: {prompt_template.input_variables}")


# --- Code cell 9 ---
"""
Helper function to format retrieved documents
"""

def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved documents into a context string.
    
    Args:
        docs: List of LangChain Document objects
        
    Returns:
        Formatted context string
    """
    if not docs:
        return "No relevant information found in the resumes."
    
    formatted_chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('doc_name', 'Unknown')
        formatted_chunks.append(
            f"[Chunk {i} from {source}]:\n{doc.page_content}"
        )
    
    return "\n\n".join(formatted_chunks)


print("✅ Document formatter defined")


# --- Code cell 10 ---
"""
Complete RAG Pipeline using LangChain LCEL
"""

def build_rag_chain(collection, prompt: PromptTemplate, k: int = TOP_K_RESULTS):
    """
    Construct a complete RAG chain using LangChain Expression Language.
    
    """
    
    # Create retriever function
    def retriever_fn(query: str) -> List[Document]:
        return retrieve_documents(query, collection, k)
    
    # Convert PromptValue to string before passing to LLM
    def prompt_to_string(prompt_value):
        """Convert LangChain PromptValue to plain string."""
        return prompt_value.to_string()
    
    # Wrap functions in RunnableLambda to make them LCEL-compatible
    retriever_runnable = RunnableLambda(retriever_fn)
    format_docs_runnable = RunnableLambda(format_docs)
    prompt_string_runnable = RunnableLambda(prompt_to_string)
    llm_runnable = RunnableLambda(invoke_llm)
    
    # Build the chain using LCEL syntax
    rag_chain = (
        {
            "context": retriever_runnable | format_docs_runnable,
            "question": RunnablePassthrough()
        }
        | prompt
        | prompt_string_runnable
        | llm_runnable
        | StrOutputParser()
    )
    
    return rag_chain


print("✅ RAG chain builder defined")


# --- Code cell 11 ---
"""
Setup ChromaDB with Custom OpenRouter Embeddings
"""

# Initialize custom embedding function
embedding_function = create_embedding_function(
    api_key=OS_API_KEY,
    model_name=EMBEDDING_MODEL
)

# Initialize ChromaDB client

client = chromadb.PersistentClient(path=CHROMA_PATH)

print("✅ Custom embedding function and ChromaDB client initialized")


# --- Code cell 12 ---
"""
Load and Index Resume PDFs
"""

# Check for PDF files
pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

if not pdf_files:
    print(f"❌ No PDFs found in {PDF_DIR}!")
    print(f"Please add resume PDFs to the '{PDF_DIR}' directory.")
else:
    # Reset collection
    try:
        client.delete_collection("resume_index")
        print("🗑️  Deleted existing collection")
    except:
        pass
    
    # Create new collection with custom embedding function
    collection = client.create_collection(
        name="resume_index", 
        embedding_function=embedding_function
    )
    
    # Index PDFs with batch processing
    print(f"\n📄 Indexing {len(pdf_files)} PDF(s)...")
    
    total_chunks = 0
    batch_size = 10  # Process chunks in batches to avoid API limits
    
    for pdf_name in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf_name)
        
        # Extract and process text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"⚠️  Skipping {pdf_name} - no text extracted")
            continue
            
        cleaned_text = preprocess_text(raw_text)
        chunks = chunk_text(cleaned_text)
        
        if chunks:
            # Add chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = [f"{pdf_name}_chunk_{j}" for j in range(i, i + len(batch_chunks))]
                batch_metadata = [{"doc_name": pdf_name} for _ in batch_chunks]
                
                try:
                    collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    total_chunks += len(batch_chunks)
                except Exception as e:
                    print(f"\n❌ Error adding batch from {pdf_name}: {e}")
                    continue
    
    print(f"\n✅ Successfully indexed {total_chunks} chunks from {len(pdf_files)} PDF(s)")
    print(f"📊 Collection size: {collection.count()} documents")


# --- Code cell 13 ---
"""
Assemble All Components into RAG Chain
"""

# Build the chain
rag_chain = build_rag_chain(collection, prompt_template, k=TOP_K_RESULTS)

print("✅ RAG chain assembled successfully!")
print("\n🔗 Chain Components:")
print("  1. Retriever: ChromaDB vector search")
print("  2. Formatter: Document context builder")
print("  3. Prompt: Resume screening template")
print("  4. LLM: Claude 3.5 Sonnet via OpenRouter")
print("  5. Parser: String output parser")


_client = None
_collection = None
_rag_chain = None

def init_rag():
    global _client, _collection, _rag_chain

    embedding_function = create_embedding_function(
        api_key=OS_API_KEY,
        model_name=EMBEDDING_MODEL
    )

    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        _collection = _client.get_collection(
            "resume_index",
            embedding_function=embedding_function
        )
    except:
        _collection = _client.create_collection(
            name="resume_index",
            embedding_function=embedding_function
        )

    _rag_chain = _rag_chain = build_rag_chain(_collection, prompt_template, k=TOP_K_RESULTS)
    return True


def index_pdfs(reset: bool = False):
    if pdf_files := [
        f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")
    ]:
        _extracted_from_index_pdfs_10(pdf_files)
    else:
        print(f"❌ No PDFs found in {PDF_DIR}!")
        print(f"Please add resume PDFs to the '{PDF_DIR}' directory.")
    pass


# TODO Rename this here and in `index_pdfs`
def _extracted_from_index_pdfs_10(pdf_files):
    # Reset collection
    try:
        client.delete_collection("resume_index")
        print("🗑️  Deleted existing collection")
    except:
        pass

    # Create new collection with custom embedding function
    collection = client.create_collection(
        name="resume_index", 
        embedding_function=embedding_function
    )

    # Index PDFs with batch processing
    print(f"\n📄 Indexing {len(pdf_files)} PDF(s)...")

    total_chunks = 0
    batch_size = 10  # Process chunks in batches to avoid API limits

    for pdf_name in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(PDF_DIR, pdf_name)

        # Extract and process text
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"⚠️  Skipping {pdf_name} - no text extracted")
            continue

        cleaned_text = preprocess_text(raw_text)
        if chunks := chunk_text(cleaned_text):
                # Add chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = [f"{pdf_name}_chunk_{j}" for j in range(i, i + len(batch_chunks))]
                batch_metadata = [{"doc_name": pdf_name} for _ in batch_chunks]

                try:
                    collection.add(
                        documents=batch_chunks,
                        metadatas=batch_metadata,
                        ids=batch_ids
                    )
                    total_chunks += len(batch_chunks)
                except Exception as e:
                    print(f"\n❌ Error adding batch from {pdf_name}: {e}")
    print(f"\n✅ Successfully indexed {total_chunks} chunks from {len(pdf_files)} PDF(s)")
    print(f"📊 Collection size: {collection.count()} documents")


def ask_question(question: str):
    if _rag_chain is None:
        init_rag()
    return _rag_chain.invoke(question)
