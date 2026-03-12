import json
import logging
import re
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger("cv_analyzer.metadata_extractor")

EXTRACTION_PROMPT = """You are a CV/resume parser. Extract structured information from the following CV text.

Return ONLY a valid JSON object with these exact fields:
{{
  "candidate_name": "Full Name",
  "skills": ["skill1", "skill2"],
  "years_of_experience": 0,
  "education": [{{"degree": "...", "institution": "...", "year": "..."}}],
  "certifications": ["cert1"],
  "email": "email@example.com",
  "phone": "+1234567890",
  "summary": "Brief professional summary in 1-2 sentences"
}}

CV Text:
{cv_text}

Return ONLY the JSON object, no other text."""

COMMON_TECH_SKILLS = [
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust", "ruby", "php", "swift", "kotlin",
    "react", "angular", "vue", "next.js", "node.js", "express", "django", "flask", "fastapi", "spring",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd",
    "git", "linux", "agile", "scrum", "rest", "graphql", "microservices",
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "html", "css", "tailwind", "sass", "bootstrap",
    "sql", "nosql", "kafka", "rabbitmq", "nginx",
]


class MetadataExtractor:
    def __init__(self, llm_service) -> None:
        self.llm_service = llm_service

    async def extract(self, text: str, model_id: Optional[str] = None) -> dict:
        try:
            return await self._llm_extract(text, model_id)
        except Exception as e:
            logger.warning("LLM extraction failed: %s. Falling back to regex.", e)
            return self._regex_extract(text)

    async def _llm_extract(self, text: str, model_id: Optional[str] = None) -> dict:
        llm = self.llm_service.get_model(model_id)
        prompt = ChatPromptTemplate.from_template(EXTRACTION_PROMPT)
        chain = prompt | llm

        truncated_text = text[:6000]
        response = await chain.ainvoke({"cv_text": truncated_text})
        content = response.content.strip()

        # Strip markdown code fences
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)

        # Validate and normalize
        return {
            "candidate_name": data.get("candidate_name", "Unknown"),
            "skills": data.get("skills", []),
            "years_of_experience": int(data.get("years_of_experience", 0)),
            "education": data.get("education", []),
            "certifications": data.get("certifications", []),
            "email": data.get("email"),
            "phone": data.get("phone"),
            "summary": data.get("summary", ""),
        }

    def _regex_extract(self, text: str) -> dict:
        text_lower = text.lower()

        # Extract email
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        email = email_match.group(0) if email_match else None

        # Extract phone
        phone_match = re.search(r"[\+]?[\d\s\-\(\)]{7,15}", text)
        phone = phone_match.group(0).strip() if phone_match else None

        # Extract skills by matching against common list
        skills = [s for s in COMMON_TECH_SKILLS if s in text_lower]

        # Try to extract name from first line
        lines = text.strip().split("\n")
        candidate_name = lines[0].strip() if lines else "Unknown"
        # Clean up name (remove if too long or has special chars)
        if len(candidate_name) > 50 or re.search(r"[{}\[\]@]", candidate_name):
            candidate_name = "Unknown"

        return {
            "candidate_name": candidate_name,
            "skills": skills,
            "years_of_experience": 0,
            "education": [],
            "certifications": [],
            "email": email,
            "phone": phone,
            "summary": "",
        }
