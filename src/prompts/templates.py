"""
Prompt templates for CV RAG System.
Includes RAG prompts and LLM-as-judge evaluation prompts.
"""

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# =============================================================================
# RAG Prompt Template
# =============================================================================

RAG_SYSTEM_PROMPT = """You are an expert HR assistant specialized in analyzing CVs and candidate profiles. 
Your task is to answer questions about candidates based ONLY on the provided context from their CVs.

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context below.
2. If the context doesn't contain enough information to answer the question, clearly state that.
3. Always cite your sources by referencing the candidate ID and section name.
4. Be specific and provide concrete details from the CVs when available.
5. If asked to compare candidates, only compare based on information present in the context.

CONTEXT FROM CVs:
{context}

---

When answering, format your response as follows:
- Provide a clear, direct answer to the question
- Include relevant quotes or details from the CVs
- Cite sources using format: [Candidate: {{candidate_id}}, Section: {{section_name}}]
"""

RAG_HUMAN_PROMPT = "{question}"

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", RAG_HUMAN_PROMPT),
])


# =============================================================================
# Judge Prompt Templates (for LLM-as-Judge evaluation)
# =============================================================================

JUDGE_RELEVANCE_TEMPLATE = """You are evaluating the relevance of an AI assistant's answer to a user's question.

USER QUESTION:
{question}

AI ASSISTANT'S ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}

---

Evaluate how relevant the answer is to the user's question.
Consider:
- Does the answer directly address what the user asked?
- Is the answer focused and on-topic?
- Does it provide useful information for the user's query?

Provide your evaluation in the following format:
SCORE: [1-5] (1=Not relevant at all, 5=Highly relevant and directly answers the question)
EXPLANATION: [Brief explanation of your score in 2-3 sentences]
"""

JUDGE_FAITHFULNESS_TEMPLATE = """You are evaluating whether an AI assistant's answer is faithful to the provided context.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

AI ASSISTANT'S ANSWER:
{answer}

---

Evaluate whether the answer is faithful to the retrieved context.
Consider:
- Does the answer only contain information from the context?
- Are there any claims that are not supported by the context?
- Does it avoid hallucinating or making up information?

Provide your evaluation in the following format:
SCORE: [1-5] (1=Contains significant hallucinations, 5=Completely faithful to context)
EXPLANATION: [Brief explanation of your score in 2-3 sentences]
"""

JUDGE_CORRECTNESS_TEMPLATE = """You are evaluating the overall correctness and quality of an AI assistant's answer.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

AI ASSISTANT'S ANSWER:
{answer}

---

Evaluate the overall correctness and quality of the answer.
Consider:
- Is the answer accurate based on the context provided?
- Is it well-structured and easy to understand?
- Does it properly cite sources?
- Is it comprehensive without being unnecessarily verbose?

Provide your evaluation in the following format:
SCORE: [1-5] (1=Poor quality/incorrect, 5=Excellent quality and correct)
EXPLANATION: [Brief explanation of your score in 2-3 sentences]
"""


# =============================================================================
# JSON Output Format Template
# =============================================================================

RAG_JSON_SYSTEM_PROMPT = """You are an expert HR assistant specialized in analyzing CVs and candidate profiles.
Your task is to answer questions about candidates based ONLY on the provided context from their CVs.

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context below.
2. If the context doesn't contain enough information to answer the question, clearly state that.
3. Be specific and provide concrete details from the CVs when available.

CONTEXT FROM CVs:
{context}

---

You MUST respond in valid JSON format with the following structure:
{{
    "answer": "Your detailed answer to the question",
    "confidence": "high|medium|low",
    "sources": [
        {{"candidate_id": "...", "section": "...", "relevant_info": "..."}}
    ],
    "limitations": "Any limitations or missing information (or null if none)"
}}
"""

RAG_JSON_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", RAG_JSON_SYSTEM_PROMPT),
    ("human", RAG_HUMAN_PROMPT),
])


# =============================================================================
# Helper Functions
# =============================================================================

def get_rag_prompt(json_output: bool = False) -> ChatPromptTemplate:
    """
    Get the RAG prompt template.
    
    Args:
        json_output: Whether to use JSON output format.
    
    Returns:
        ChatPromptTemplate for RAG.
    """
    if json_output:
        return RAG_JSON_PROMPT_TEMPLATE
    return RAG_PROMPT_TEMPLATE


def get_judge_prompts() -> dict:
    """
    Get all judge prompt templates.
    
    Returns:
        Dictionary of judge prompts.
    """
    return {
        "relevance": PromptTemplate.from_template(JUDGE_RELEVANCE_TEMPLATE),
        "faithfulness": PromptTemplate.from_template(JUDGE_FAITHFULNESS_TEMPLATE),
        "correctness": PromptTemplate.from_template(JUDGE_CORRECTNESS_TEMPLATE),
    }


def format_context_for_prompt(documents: list) -> str:
    """
    Format retrieved documents into context string for prompts.
    
    Args:
        documents: List of LangChain Document objects.
    
    Returns:
        Formatted context string.
    """
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        candidate_id = metadata.get("candidate_id", "Unknown")
        section_name = metadata.get("section_name", "Unknown")
        chunk_id = metadata.get("chunk_id", "N/A")
        
        # Add additional metadata if available
        extra_info = []
        if metadata.get("name"):
            extra_info.append(f"Name: {metadata['name']}")
        if metadata.get("role"):
            extra_info.append(f"Role: {metadata['role']}")
        
        extra_str = f" ({', '.join(extra_info)})" if extra_info else ""
        
        context_parts.append(
            f"[Source {i}]\n"
            f"Candidate ID: {candidate_id}{extra_str}\n"
            f"Section: {section_name}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)
