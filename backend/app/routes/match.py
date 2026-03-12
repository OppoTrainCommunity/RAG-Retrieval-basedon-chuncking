import logging

from fastapi import APIRouter, Request

from app.models.schemas import MatchRequest, MatchResponse, CandidateMatch, SkillMatch
from app.services.job_matcher import JobMatcher

logger = logging.getLogger("cv_analyzer.routes.match")
router = APIRouter()


@router.post("/match", response_model=MatchResponse)
async def match_candidates(request: Request, body: MatchRequest):
    vector_store = request.app.state.vector_store
    llm_service = request.app.state.llm_service
    matcher = JobMatcher(vector_store, llm_service)

    try:
        results = await matcher.match(body.job_description, body.model, body.top_k)
        candidates = []
        for r in results:
            candidates.append(CandidateMatch(
                candidate_name=r.get("candidate_name", "Unknown"),
                cv_id=r.get("cv_id", ""),
                match_score=r.get("match_score", 0),
                key_matching_points=r.get("key_matching_points", []),
                missing_qualifications=r.get("missing_qualifications", []),
                skills_match=[SkillMatch(**s) for s in r.get("skills_match", [])],
                explanation=r.get("explanation", ""),
            ))
        return MatchResponse(success=True, candidates=candidates)
    except Exception as e:
        logger.error("Match error: %s", e, exc_info=True)
        return MatchResponse(success=False, error=str(e))
