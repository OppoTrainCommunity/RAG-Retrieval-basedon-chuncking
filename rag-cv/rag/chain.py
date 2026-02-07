"""
RAG Chain Module
================

LangChain RAG chain with support for OpenRouter and Ollama LLMs.
"""

from typing import Dict, Any, Optional, List

from .logging_utils import get_logger
from .config import settings, OPENROUTER_HEADERS
from .prompts import get_rag_prompt
from .retriever import format_retrieved_docs, get_sources_summary

logger = get_logger(__name__)

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

try:
    from langchain_ollama import ChatOllama
except ImportError:
    logger.warning("langchain-ollama not installed. Ollama support won't work.")
    ChatOllama = None

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_core.runnables import RunnableLambda


class LLMFactory:
    """Factory for creating LLM instances based on configuration."""
    
    @staticmethod
    def create_llm():
        """Create an LLM instance based on settings.llm_provider."""
        provider = settings.llm_provider.lower()
        
        if provider == "ollama":
            return OllamaLLM()
        else:
            return OpenRouterLLM()


class OllamaLLM:
    """Wrapper for local Ollama LLM."""
    
    def __init__(self):
        self.model = settings.ollama_model
        self.base_url = settings.ollama_base_url
        self._llm = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        try:
            if ChatOllama:
                self._llm = ChatOllama(
                    model=self.model,
                    base_url=self.base_url,
                    temperature=settings.llm_temperature
                )
                logger.info(f"Initialized Ollama LLM: {self.model}")
            else:
                logger.error("ChatOllama not available. Install langchain-ollama.")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            
    def invoke(self, messages):
        if self._llm:
            return self._llm.invoke(messages)
        
        # return detailed error if possible
        if not ChatOllama:
             return "Error: langchain-ollama library not found. Please install it."
        return "Error: LLM not initialized. Check logs for details."


class OpenRouterLLM:
    """
    Wrapper for OpenRouter LLM using OpenAI-compatible API.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize the OpenRouter LLM.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o-mini")
            api_key: OpenRouter API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model or settings.openrouter_model
        self.api_key = api_key or settings.openrouter_api_key
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        
        self._llm = None
        
        if self.api_key:
            self._initialize_llm()
        else:
            logger.warning("OpenRouter API key not set. LLM will not be available.")
    
    def _initialize_llm(self):
        """Initialize the LangChain ChatOpenAI instance."""
        try:
            self._llm = ChatOpenAI(
                model=self.model,
                openai_api_key=self.api_key,
                openai_api_base=settings.openrouter_base_url,
                default_headers=OPENROUTER_HEADERS,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                request_timeout=settings.llm_request_timeout
            )
            logger.info(f"Initialized OpenRouter LLM: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self._llm = None
    
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        return self._llm is not None
    
    def get_llm(self):
        """Get the underlying LLM instance."""
        return self._llm
    
    def invoke(self, messages) -> str:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: Chat messages
            
        Returns:
            Response content
        """
        if not self.is_available():
            raise RuntimeError("LLM not available. Check API key.")
        
        response = self._llm.invoke(messages)
        return response.content


def create_rag_chain(retriever) -> "RAGChain":
    """
    Factory function to create a RAG chain based on configuration.
    
    Args:
        retriever: Configuration retriever instance
        
    Returns:
        Configured RAGChain
    """
    # Use LLMFactory to get the correct LLM
    llm = LLMFactory.create_llm()
    
    return RAGChain(
        llm=llm,
        retriever=retriever
    )

class RAGChain:
    """
    RAG Chain for question answering over CVs.
    """
    
    def __init__(
        self,
        llm,
        retriever
    ):
        """
        Initialize the RAG chain.
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt = get_rag_prompt()
        
        self._chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain."""
        def retrieve_and_format(question: str) -> Dict[str, Any]:
            """Retrieve documents and format context."""
            docs = self.retriever.retrieve(question)
            return {
                "context": format_retrieved_docs(docs),
                "question": question,
                "source_documents": docs
            }
        
        def generate_answer(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Generate answer using LLM."""
            messages = self.prompt.invoke({
                "context": inputs["context"],
                "question": inputs["question"]
            })
            
            # Handle both string response (Ollama wrapper) and AIMessage (LangChain)
            response = self.llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": content,
                "source_documents": inputs["source_documents"],
                "context": inputs["context"]
            }
        
        chain = RunnableLambda(retrieve_and_format) | RunnableLambda(generate_answer)
        return chain
    
    def invoke(self, question: str) -> Dict[str, Any]:
        """
        Invoke the RAG chain with a question.
        
        Args:
            question: User question
            
        Returns:
            Dict with answer, sources, and context
        """
        try:
            result = self._chain.invoke(question)
            
            # Add sources summary
            result["sources"] = get_sources_summary(result["source_documents"])
            
            logger.info(f"Generated answer for: {question[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "source_documents": [],
                "sources": [],
                "context": ""
            }
    
    def ask(self, question: str) -> str:
        """
        Simple interface to ask a question and get an answer.
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        result = self.invoke(question)
        return result["answer"]





def get_openrouter_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None
) -> Optional[OpenRouterLLM]:
    """
    Get an OpenRouter LLM instance.
    
    Args:
        model: Optional model override
        api_key: Optional API key override
        
    Returns:
        OpenRouterLLM instance or None if not available
    """
    llm = OpenRouterLLM(model=model, api_key=api_key)
    
    if llm.is_available():
        return llm
    
    return None
