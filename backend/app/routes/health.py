import json
import logging
from collections import Counter

from fastapi import APIRouter, Request

from app.config import settings
from app.models.schemas import HealthResponse, StatsResponse

logger = logging.getLogger("cv_analyzer.routes.health")
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version=settings.app_version)


@router.get("/stats", response_model=StatsResponse)
async def get_stats(request: Request):
    vector_store = request.app.state.vector_store
    total_cvs = vector_store.get_total_cvs()
    total_chunks = vector_store.get_total_chunks()

    # Get top skills
    top_skills = []
    if total_cvs > 0:
        all_cvs = vector_store.get_all_cvs()
        skill_counter: Counter = Counter()
        for cv in all_cvs:
            for skill in cv.get("skills", []):
                skill_counter[skill.lower()] += 1
        top_skills = [{"skill": s, "count": c} for s, c in skill_counter.most_common(10)]

    return StatsResponse(
        total_cvs=total_cvs,
        total_chunks=total_chunks,
        top_skills=top_skills,
        status="active",
    )
