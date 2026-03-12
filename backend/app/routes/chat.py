import logging

from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse, SourceInfo
from app.services.rag_service import RAGService

logger = logging.getLogger("cv_analyzer.routes.chat")
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    vector_store = request.app.state.vector_store
    llm_service = request.app.state.llm_service
    rag_service = RAGService(vector_store, llm_service)

    try:
        result = await rag_service.query(body.query, body.model)
        sources = [SourceInfo(**s) for s in result.get("sources", [])]
        return ChatResponse(
            success=True,
            response=result["response"],
            sources=sources,
            model_used=result["model_used"],
        )
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return ChatResponse(success=False, error=str(e))
