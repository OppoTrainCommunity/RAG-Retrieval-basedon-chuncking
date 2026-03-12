"""
English CV Summary Prompt Templates
=====================================

Prompts for generating executive summaries of CVs.
Variables: {content}
"""

from string import Template

# ── CV Summary Prompt ──────────────────────────────────────────
summary_prompt = Template(
    """You are a Senior Talent Acquisition Specialist. Analyze the provided CV content and generate a comprehensive professional executive summary.

CV Content:
${content}

Please provide a structured summary including:
1. **Professional Profile**: A brief statement of who the candidate is.
2. **Key Competencies & Skills**: Technical and soft skills extracted from the document.
3. **Experience Overview**: Total years of experience and key roles held.
4. **Notable Achievements**: Specific accomplishments, metrics, or successful projects.
5. **Education & Certifications**: Academic background and relevant certifications.

Executive Summary:"""
)
