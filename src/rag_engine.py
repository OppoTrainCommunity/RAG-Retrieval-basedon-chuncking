"""
RAG Engine Module: Handles LangChain integration and LLM interactions.
"""
import time
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from . import config
from . import vector_store

class ChromaRetriever(BaseRetriever):
    """Custom Retriever using the existing vector store implementation."""
    collection: Any
    embedding_model: str
    api_key: str
    top_k: int = 5

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        results = vector_store.retrieve_similar_chunks(
            query, self.collection, self.embedding_model, self.api_key, self.top_k
        )
        documents = []
        for r in results:
            documents.append(Document(
                page_content=r['text'],
                metadata=r['metadata']
            ))
        return documents

class RAGChainManager:
    """Manages LangChain RAG chains for multiple LLMs."""
    
    def __init__(self, collection_name: str, embedding_model: str, api_key: str):
        self.api_key = api_key
        self.collection = vector_store.initialize_collection(collection_name)
        self.embedding_model = embedding_model
        
        # Define prompts
        self.qa_template = """You are a helpful assistant for analyzing resumes.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
        self.qa_prompt = PromptTemplate.from_template(self.qa_template)
        
        self.eval_relevance_template = """Does the following answer directly address the question?
Answer 'YES' or 'NO' and explain why.

Question: {question}
Answer: {answer}
"""
        self.eval_faithfulness_template = """Is the following answer based ONLY on the provided context?
Answer 'YES' or 'NO' and explain why.

Context: {context}
Answer: {answer}
"""

    def get_llm(self, model_name: str, temperature: float = 0):
        """Get ChatOpenAI instance for OpenRouter."""
        return ChatOpenAI(
            base_url=config.OPENROUTER_BASE_URL,
            api_key=self.api_key,
            model=model_name,
            temperature=temperature
        )

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Manual retrieval to pass to chain."""
        results = vector_store.retrieve_similar_chunks(
            query, self.collection, self.embedding_model, self.api_key, top_k
        )
        return "\n\n".join([r['text'] for r in results])

    def generate_response(self, query: str, model_name: str, top_k: int = 5) -> Dict[str, Any]:
        """Generate answer using RAG chain."""
        start_time = time.perf_counter()
        
        try:
            # 1. Setup Components
            llm = self.get_llm(model_name)
            retriever = ChromaRetriever(
                collection=self.collection,
                embedding_model=self.embedding_model,
                api_key=self.api_key,
                top_k=top_k
            )
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # 2. Define Chain
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | self.qa_prompt
                | llm
                | StrOutputParser()
            )
            
            # 3. Generate
            answer = chain.invoke(query)
            
            # Retrieve context separately just for reporting/logging purposes (optional, but good for UI)
            # The chain run already retrieved it, but standard LCEL doesn't easily return intermediate steps 
            # without using retrieval chains or callbacks. 
            # For simplicity in this specific return format, we can re-retrieve or use a custom chain that returns inputs.
            # To avoid double retrieval cost, we can use the manual retrieval we had, OR trust the chain.
            # But the UI expects 'context' string.
            
            # Improved approach: Use assign to keep context
            chain_with_context = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | RunnablePassthrough.assign(answer= self.qa_prompt | llm | StrOutputParser())
            )
            
            result = chain_with_context.invoke(query)
            answer = result["answer"]
            context = result["context"]
            
        except Exception as e:
            return {"error": str(e), "latency_ms": 0, "answer": f"Error: {str(e)}", "context": ""}
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            "query": query,
            "answer": answer,
            "context": context,
            "model": model_name,
            "latency_ms": latency_ms
        }

    def evaluate_answer(self, query: str, context: str, answer: str, ground_truth: str = None) -> Dict[str, Any]:
        """Evaluate answer using LLM-as-a-Judge."""
        # Use a free model for evaluation
        eval_llm = self.get_llm("meta-llama/llama-3.3-70b-instruct:free") 
        
        try:
            # Relevance
            rel_chain = PromptTemplate.from_template(self.eval_relevance_template) | eval_llm | StrOutputParser()
            relevance_eval = rel_chain.invoke({"question": query, "answer": answer})
            
            # Faithfulness
            faith_chain = PromptTemplate.from_template(self.eval_faithfulness_template) | eval_llm | StrOutputParser()
            faithfulness_eval = faith_chain.invoke({"context": context, "answer": answer})
            
            # Correctness (if GT)
            correctness_score = 0.0
            if ground_truth:
                # Simple presence check as placeholder for semantic eval
                correctness_score = 1.0 if ground_truth.lower() in answer.lower() else 0.0
                
            return {
                "relevance": relevance_eval,
                "faithfulness": faithfulness_eval,
                "correctness": correctness_score if ground_truth else None
            }
        except Exception as e:
             return {"error": str(e)}
