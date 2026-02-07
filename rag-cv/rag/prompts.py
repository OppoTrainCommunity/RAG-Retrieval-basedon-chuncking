"""
Prompts Module
==============

Prompt templates for the RAG pipeline.
"""

from .config import RAG_PROMPT_TEMPLATE

# LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate


def get_rag_prompt() -> ChatPromptTemplate:
    """
    Get the RAG prompt template.
    
    Returns:
        ChatPromptTemplate instance
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def get_simple_prompt() -> PromptTemplate:
    """
    Get a simple prompt template for basic QA.
    
    Returns:
        PromptTemplate instance
    """
    template = """Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


# Prompt for summarizing a CV
CV_SUMMARY_PROMPT = """Analyze the following CV/resume content and provide a brief professional summary.

CV Content:
{content}

Provide:
1. Candidate's key skills
2. Years of experience (if mentioned)
3. Notable achievements
4. Education highlights

Summary:"""


def get_cv_summary_prompt() -> PromptTemplate:
    """
    Get a prompt for summarizing CVs.
    
    Returns:
        PromptTemplate instance
    """
    return PromptTemplate(
        template=CV_SUMMARY_PROMPT,
        input_variables=["content"]
    )


# Prompt for comparing candidates
COMPARE_CANDIDATES_PROMPT = """Based on the provided CV information, compare the candidates for the following criteria:

Context from CVs:
{context}

Comparison Criteria: {criteria}

Provide a structured comparison with:
1. Each candidate's strengths
2. Each candidate's relevant experience
3. Overall recommendation

Comparison:"""


def get_comparison_prompt() -> PromptTemplate:
    """
    Get a prompt for comparing candidates.
    
    Returns:
        PromptTemplate instance
    """
    return PromptTemplate(
        template=COMPARE_CANDIDATES_PROMPT,
        input_variables=["context", "criteria"]
    )
