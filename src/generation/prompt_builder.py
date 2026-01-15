from typing import List, Dict, Any

SYSTEM_PROMPT = f"""You are a technical recruitment assistant.

Goal:
Given a job/project requirement and resume evidence retrieved from multiple candidates, rank the best candidates.

Rules:
- Use ONLY the provided resume evidence (context). Do NOT invent skills, experience, or education.
- Every claim MUST be backed by evidence and include file_name + chunk_id.
- If something is not found in the evidence, say “not found”.
- Be fair: do not assume years of experience unless explicitly stated.
- Output ONLY valid JSON (no markdown, no extra text)."""


class PromptBuilder:
  """
  Builds prompts for career-path suggestions using retrieved chunks.
  """


  @staticmethod
  def build_context(chunks: List[str]) -> str:
      # Keep it clean and readable for the LLM
      return "\n\n---\n\n".join(chunks)

  @staticmethod
  def build_messages(
      context: str,
      question: str,
      role_title: str,
      must_have: list,
      nice_to_have: list,
      seniority: str = "any",
      job_notes: str = "",
      top_n: int = 5,
  ):
      user_prompt = f"""
  Job requirements:
  Role title: {role_title}
  Must-have skills: {must_have}
  Nice-to-have skills: {nice_to_have}
  Seniority: {seniority}
  Notes: {job_notes}

  Recruitment question:
  {question}

  Resume evidence (retrieved chunks across multiple candidates):
  {context}

  Task:
  Rank the best candidates for this role.
  Return top {top_n} candidates (or fewer if evidence is weak).

  Every score and claim MUST cite evidence with file_name and chunk_id.

  Return ONLY JSON with this schema:
  {{
    "job": {{
      "role_title": "string",
      "must_have": ["string"],
      "nice_to_have": ["string"],
      "seniority": "intern/junior/mid/senior/any",
      "notes": "string"
    }},
    "ranked_candidates": [
      {{
        "file_name": "string",
        "overall_score": 0,
        "score_breakdown": {{
          "skills_match": 0,
          "project_relevance": 0,
          "experience_level": 0,
          "communication_clarity": 0
        }},
        "highlights": ["string"],
        "gaps": ["string"],
        "evidence": [
          {{
            "file_name": "string",
            "chunk_id": 0,
            "snippet": "string",
            "why_it_matters": "string"
          }}
        ],
        "recommendation": "strong_interview / interview / maybe / no",
        "targeted_interview_questions": ["string"]
      }}
    ],
    "final_notes": {{
      "missing_info": ["string"],
      "tie_breakers_used": ["string"]
    }}
  }}
  """.strip()


      return [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": user_prompt},
      ]

  @staticmethod
  def build_context_with_meta(docs: list, metas: list, max_chars: int = 16000) -> str:
      parts = []
      for doc, meta in zip(docs, metas):
          file_name = meta.get("file_name", "unknown")
          chunk_id = meta.get("chunk_id", "unknown")
          parts.append(
              f'[file_name="{file_name}" chunk_id={chunk_id}]\n{doc}'
          )

      joined = "\n\n---\n\n".join(parts)
      return joined[:max_chars]

