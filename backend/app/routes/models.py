from fastapi import APIRouter, Request
from app.models.schemas import ModelsResponse, ModelInfo

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request):
    llm_service = request.app.state.llm_service
    models = [ModelInfo(**m) for m in llm_service.get_default_models()]
    return ModelsResponse(models=models)
