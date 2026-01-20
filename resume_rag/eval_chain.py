import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .openrouter_llm import get_chat_model

EVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a strict evaluator. Return ONLY a JSON object with keys: "
         "groundedness, relevance, correctness, completeness, notes. "
         "Each score 1-5."),
        ("user",
         "Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}\n\nReturn JSON only.")
    ]
)

def evaluate_answer(question: str, context: str, answer: str, model: str = "openai/gpt-4o-mini"):
    llm = get_chat_model(model=model, temperature=0.0)
    chain = EVAL_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": question, "context": context, "answer": answer})
    return json.loads(raw)
