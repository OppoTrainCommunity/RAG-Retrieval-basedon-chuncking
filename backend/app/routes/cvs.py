import logging

from fastapi import APIRouter, Request, HTTPException

from app.models.schemas import CVListResponse, CVDetailResponse, CVMetadata

logger = logging.getLogger("cv_analyzer.routes.cvs")
router = APIRouter()


@router.get("/cvs", response_model=CVListResponse)
async def list_cvs(request: Request):
    vector_store = request.app.state.vector_store
    cvs_data = vector_store.get_all_cvs()
    cvs = [CVMetadata(**cv) for cv in cvs_data]
    return CVListResponse(success=True, cvs=cvs)


@router.get("/cvs/{cv_id}", response_model=CVDetailResponse)
async def get_cv(request: Request, cv_id: str):
    vector_store = request.app.state.vector_store
    cv_data = vector_store.get_cv(cv_id)
    if not cv_data:
        raise HTTPException(status_code=404, detail="CV not found")
    return CVDetailResponse(success=True, cv=CVMetadata(**cv_data))


@router.delete("/cvs/{cv_id}")
async def delete_cv(request: Request, cv_id: str):
    vector_store = request.app.state.vector_store
    deleted = vector_store.delete_cv(cv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="CV not found")
    return {"success": True, "message": "CV deleted successfully"}
