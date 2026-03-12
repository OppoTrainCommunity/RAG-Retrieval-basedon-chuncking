import logging

import httpx
from fastapi import APIRouter, Request

from app.models.schemas import ApiKeyRequest, ApiKeyResponse
from app.config import settings

logger = logging.getLogger("cv_analyzer.routes.settings")
router = APIRouter()


async def _validate_api_key(api_key: str) -> tuple[bool, str]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code == 200:
                return True, "API key is valid"
            else:
                return False, f"Invalid API key (status {response.status_code})"
    except httpx.TimeoutException:
        return False, "Validation request timed out"
    except Exception as e:
        logger.error("API key validation error: %s", e)
        return False, f"Validation error: {str(e)}"


@router.post("/settings/api-key", response_model=ApiKeyResponse)
async def save_api_key(request: Request, body: ApiKeyRequest):
    valid, message = await _validate_api_key(body.api_key)

    if valid:
        settings.openrouter_api_key = body.api_key
        request.app.state.llm_service.update_api_key(body.api_key)
        return ApiKeyResponse(success=True, status="active", message="API key validated and saved")
    else:
        return ApiKeyResponse(success=False, status="invalid", message=message)


@router.post("/settings/api-key/validate", response_model=ApiKeyResponse)
async def validate_api_key(request: ApiKeyRequest):
    valid, message = await _validate_api_key(request.api_key)
    status = "active" if valid else "invalid"
    return ApiKeyResponse(success=valid, status=status, message=message)
