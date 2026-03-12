import logging
from collections import Counter

from fastapi import APIRouter, Request

from app.models.schemas import SkillsAnalyticsResponse, SkillCount, ExperienceAnalyticsResponse, ExperienceBucket

logger = logging.getLogger("cv_analyzer.routes.analytics")
router = APIRouter()


@router.get("/analytics/skills", response_model=SkillsAnalyticsResponse)
async def get_skills_analytics(request: Request):
    vector_store = request.app.state.vector_store
    all_cvs = vector_store.get_all_cvs()

    skill_counter: Counter = Counter()
    for cv in all_cvs:
        for skill in cv.get("skills", []):
            skill_counter[skill.lower()] += 1

    skills = [SkillCount(skill=s, count=c) for s, c in skill_counter.most_common(30)]
    return SkillsAnalyticsResponse(skills=skills)


@router.get("/analytics/experience", response_model=ExperienceAnalyticsResponse)
async def get_experience_analytics(request: Request):
    vector_store = request.app.state.vector_store
    all_cvs = vector_store.get_all_cvs()

    buckets = {"0-2 years": 0, "3-5 years": 0, "6-10 years": 0, "10+ years": 0}
    for cv in all_cvs:
        years = cv.get("years_of_experience", 0)
        if years <= 2:
            buckets["0-2 years"] += 1
        elif years <= 5:
            buckets["3-5 years"] += 1
        elif years <= 10:
            buckets["6-10 years"] += 1
        else:
            buckets["10+ years"] += 1

    distribution = [ExperienceBucket(range=r, count=c) for r, c in buckets.items()]
    return ExperienceAnalyticsResponse(distribution=distribution)
