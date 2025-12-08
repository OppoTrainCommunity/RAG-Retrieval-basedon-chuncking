import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def _client():
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in .env")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def suggest_career(text: str):
    client = _client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a career expert AI."},
            {"role": "user",
             "content":
             f"Analyze this resume:\n{text}\n\nProvide career paths, skills to improve, certifications, and job titles."}
        ]
    )

    return response.choices[0].message.content


def evaluate_resume(text: str):
    client = _client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an HR resume evaluation AI."},
            {"role": "user",
             "content":
             f"""Evaluate the resume and return this JSON ONLY:

{{
  "overall_score": <1-100>,
  "section_scores": {{
      "education": <1-100>,
      "projects": <1-100>,
      "skills": <1-100>,
      "experience": <1-100>
  }},
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "recommendations": ["...", "..."]
}}

Resume:
{text}
"""
            }
        ]
    )

    return response.choices[0].message.content
