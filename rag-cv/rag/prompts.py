"""
Prompts Module
==============

Prompt templates for the RAG pipeline.
Now powered by the Template Parser for multi-language support.
"""

from .template_parser import template_parser

# LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate, PromptTemplate


def get_rag_prompt(language: str = None) -> ChatPromptTemplate:
    """
    Get the RAG prompt template.

    Args:
        language: Optional language override ('en', 'ar')

    Returns:
        ChatPromptTemplate instance
    """
    if language:
        template_parser.set_language(language)

    template = template_parser.get("rag", "rag_prompt")

    if template is None:
        template = _FALLBACK_RAG_PROMPT
    else:
        # Convert $-style placeholders to {}-style for LangChain
        template = template.replace("${context}", "{context}").replace("${question}", "{question}")

    return ChatPromptTemplate.from_template(template)


def get_simple_prompt(language: str = None) -> PromptTemplate:
    """
    Get a simple prompt template for basic QA.

    Args:
        language: Optional language override ('en', 'ar')

    Returns:
        PromptTemplate instance
    """
    if language:
        template_parser.set_language(language)

    template = template_parser.get("rag", "simple_qa_prompt")

    if template is None:
        template = _FALLBACK_SIMPLE_PROMPT
    else:
        template = template.replace("${context}", "{context}").replace("${question}", "{question}")

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def get_cv_summary_prompt(language: str = None) -> PromptTemplate:
    """
    Get a prompt for summarizing CVs.

    Args:
        language: Optional language override ('en', 'ar')

    Returns:
        PromptTemplate instance
    """
    if language:
        template_parser.set_language(language)

    template = template_parser.get("summary", "summary_prompt")

    if template is None:
        template = _FALLBACK_SUMMARY_PROMPT
    else:
        template = template.replace("${content}", "{content}")

    return PromptTemplate(
        template=template,
        input_variables=["content"]
    )


def get_comparison_prompt(language: str = None) -> PromptTemplate:
    """
    Get a prompt for comparing candidates.

    Args:
        language: Optional language override ('en', 'ar')

    Returns:
        PromptTemplate instance
    """
    if language:
        template_parser.set_language(language)

    template = template_parser.get("compare", "compare_prompt")

    if template is None:
        template = _FALLBACK_COMPARE_PROMPT
    else:
        template = template.replace("${context}", "{context}").replace("${criteria}", "{criteria}")

    return PromptTemplate(
        template=template,
        input_variables=["context", "criteria"]
    )


def get_response_text(key: str, language: str = None, vars: dict = None) -> str:
    """
    Get a response text template (e.g., no_data_response, fallback_response).

    Args:
        key: Template key in the 'rag' group
        language: Optional language override
        vars: Optional variables to substitute

    Returns:
        Response text string
    """
    if language:
        template_parser.set_language(language)

    result = template_parser.get("rag", key, vars)
    return result if result else key


# ── Fallback prompts (in case template files are missing) ──────

_FALLBACK_RAG_PROMPT = """You are a professional HR assistant specialized in analyzing CVs/resumes.
Answer the question based ONLY on the provided context from uploaded CVs.

IMPORTANT RULES:
1. Only use information explicitly present in the context below.
2. If the information is not in the context, say "I could not find this information in the uploaded CVs".
3. Provide concise, professional answers using bullet points when appropriate.
4. Always cite your sources using the format: (Source: filename, Page: X, Chunk: Y)
5. Do not make assumptions or hallucinate information not present in the context.
6. Provide a brief summary at the end of your answer.

CONTEXT FROM CVs:
{context}

QUESTION: {question}

ANSWER (with citations):"""

_FALLBACK_SIMPLE_PROMPT = """You are an expert HR and Technical Recruitment Consultant.
Answer the query based strictly on the provided context.

Context:
{context}

Question: {question}

Professional Answer:"""

_FALLBACK_SUMMARY_PROMPT = """Analyze the provided CV content and generate a professional executive summary.

CV Content:
{content}

Executive Summary:"""

_FALLBACK_COMPARE_PROMPT = """Compare the candidates based on the provided CV excerpts.

Context from CVs:
{context}

Comparison Criteria: {criteria}

Comparative Analysis:"""
