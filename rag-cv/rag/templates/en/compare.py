"""
English Candidate Comparison Prompt Templates
===============================================

Prompts for comparing multiple candidates.
Variables: {context}, {criteria}
"""

from string import Template

# ── Compare Candidates Prompt ──────────────────────────────────
compare_prompt = Template(
    """You are a Hiring Manager evaluating multiple candidates. Compare the candidates based on the provided CV excerpts and the specified criteria.

Context from CVs:
${context}

Comparison Criteria: ${criteria}

Please provide a detailed comparative analysis:
1. **Strengths Analysis**: Highlight the specific advantages of each candidate regarding the criteria.
2. **Gap Analysis**: Identify any missing skills or weaknesses relative to the criteria.
3. **Experience Match**: Compare the relevance and depth of their experience.
4. **Final Recommendation**: Suggest the most suitable candidate for the requirement, with justification.

Comparative Analysis:"""
)
