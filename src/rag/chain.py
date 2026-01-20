"""
RAG Chain for CV RAG System.
Implements the full retrieval-augmented generation pipeline.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from openai import OpenAI

from src.prompts.templates import (
    get_rag_prompt,
    format_context_for_prompt,
)
from src.vectordb.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    """Represents a source document with metadata."""
    chunk_id: str
    candidate_id: str
    section_name: str
    content: str
    score: Optional[float] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "candidate_id": self.candidate_id,
            "section_name": self.section_name,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class RAGResponse:
    """Response from RAG chain."""
    answer: str
    sources: List[SourceDocument]
    query: str
    retrieval_time: float
    generation_time: float
    num_sources: int
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "query": self.query,
            "retrieval_time": self.retrieval_time,
            "generation_time": self.generation_time,
            "num_sources": self.num_sources,
        }


class OpenRouterLLM:
    """
    OpenRouter LLM wrapper using OpenAI-compatible API.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 800,
    ):
        """
        Initialize OpenRouter LLM.
        
        Args:
            api_key: OpenRouter API key.
            base_url: OpenRouter base URL.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> str:
        """
        Invoke the LLM with messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters.
        
        Returns:
            Generated text.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content


class RAGChain:
    """
    RAG Chain for CV question answering.
    Composes: Retriever → Prompt → LLM → Output Parser
    """
    
    def __init__(
        self,
        chroma_store: ChromaStore,
        llm: OpenRouterLLM,
        top_k: int = 6,
        section_filter: Optional[str] = None,
        json_output: bool = False,
    ):
        """
        Initialize RAG Chain.
        
        Args:
            chroma_store: ChromaStore instance for retrieval.
            llm: OpenRouterLLM instance for generation.
            top_k: Number of documents to retrieve.
            section_filter: Optional section name filter.
            json_output: Whether to use JSON output format.
        """
        self.chroma_store = chroma_store
        self.llm = llm
        self.top_k = top_k
        self.section_filter = section_filter
        self.json_output = json_output
        
        # Get prompt template
        self.prompt = get_rag_prompt(json_output=json_output)
        
        # Output parser
        if json_output:
            self.output_parser = JsonOutputParser()
        else:
            self.output_parser = StrOutputParser()
    
    def _retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        candidate_filter: Optional[str] = None,
    ) -> tuple:
        """
        Retrieve relevant documents.
        
        Args:
            query: Query string.
            top_k: Override top_k.
            section_filter: Override section filter.
            candidate_filter: Optional candidate ID to filter by.
        
        Returns:
            Tuple of (documents, retrieval_time, sources).
        """
        k = top_k or self.top_k
        filter_section = section_filter or self.section_filter
        
        # Build filter - Chroma uses $and for multiple conditions
        filter_dict = None
        filters = []
        
        if filter_section:
            filters.append({"section_name": filter_section})
        
        if candidate_filter:
            filters.append({"candidate_id": candidate_filter})
        
        if len(filters) == 1:
            filter_dict = filters[0]
        elif len(filters) > 1:
            filter_dict = {"$and": filters}
        
        start_time = time.time()
        
        # Retrieve with scores
        results = self.chroma_store.similarity_search_with_score(
            query, k=k, filter_dict=filter_dict
        )
        
        retrieval_time = time.time() - start_time
        
        # Convert to source documents
        documents = []
        sources = []
        
        for doc, score in results:
            documents.append(doc)
            
            source = SourceDocument(
                chunk_id=doc.metadata.get("chunk_id", ""),
                candidate_id=doc.metadata.get("candidate_id", ""),
                section_name=doc.metadata.get("section_name", ""),
                content=doc.page_content,
                score=float(score) if score else None,
                metadata={
                    k: v for k, v in doc.metadata.items()
                    if k not in ["chunk_id", "candidate_id", "section_name"]
                },
            )
            sources.append(source)
        
        return documents, retrieval_time, sources
    
    def _generate(
        self,
        query: str,
        context: str,
    ) -> tuple:
        """
        Generate answer using LLM.
        
        Args:
            query: User query.
            context: Formatted context string.
        
        Returns:
            Tuple of (answer, generation_time, raw_response).
        """
        # Format messages
        messages = [
            {
                "role": "system",
                "content": self.prompt.messages[0].prompt.template.format(context=context),
            },
            {
                "role": "user",
                "content": query,
            },
        ]
        
        start_time = time.time()
        
        raw_response = self.llm.invoke(messages)
        
        generation_time = time.time() - start_time
        
        # Parse output
        if self.json_output:
            try:
                answer = json.loads(raw_response)
            except json.JSONDecodeError:
                answer = raw_response
        else:
            answer = raw_response
        
        return answer, generation_time, raw_response
    
    def invoke(
        self,
        query: str,
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
        candidate_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline.
        
        Args:
            query: User query.
            top_k: Override top_k.
            section_filter: Override section filter.
            candidate_filter: Optional candidate ID to filter results.
        
        Returns:
            RAGResponse with answer and sources.
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Retrieve
        documents, retrieval_time, sources = self._retrieve(
            query, top_k, section_filter, candidate_filter
        )
        
        logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}s")
        
        # Step 2: Format context
        context = format_context_for_prompt(documents)
        
        # Step 3: Generate
        answer, generation_time, raw_response = self._generate(query, context)
        
        logger.info(f"Generated answer in {generation_time:.2f}s")
        
        # Build response
        response = RAGResponse(
            answer=answer if isinstance(answer, str) else json.dumps(answer),
            sources=sources,
            query=query,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            num_sources=len(sources),
            raw_response=raw_response,
        )
        
        return response
    
    def batch_invoke(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        section_filter: Optional[str] = None,
    ) -> List[RAGResponse]:
        """
        Run RAG pipeline on multiple queries.
        
        Args:
            queries: List of queries.
            top_k: Override top_k.
            section_filter: Override section filter.
        
        Returns:
            List of RAGResponse objects.
        """
        responses = []
        
        for query in queries:
            response = self.invoke(query, top_k, section_filter)
            responses.append(response)
        
        return responses
    
    @classmethod
    def from_config(
        cls,
        config,
        chroma_store: ChromaStore,
        json_output: bool = False,
    ) -> "RAGChain":
        """
        Create RAGChain from configuration.
        
        Args:
            config: Config object.
            chroma_store: ChromaStore instance.
            json_output: Whether to use JSON output.
        
        Returns:
            RAGChain instance.
        """
        llm = OpenRouterLLM(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        
        return cls(
            chroma_store=chroma_store,
            llm=llm,
            top_k=config.retrieval.top_k,
            section_filter=config.retrieval.section_filter,
            json_output=json_output,
        )
