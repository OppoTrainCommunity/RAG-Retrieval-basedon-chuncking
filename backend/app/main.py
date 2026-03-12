import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.services.vector_store import VectorStoreService
from app.services.llm_service import LLMService

logger = logging.getLogger("cv_analyzer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize services
    logger.info("Starting CV Analysis RAG System v%s", settings.app_version)

    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)

    app.state.vector_store = VectorStoreService()
    app.state.llm_service = LLMService()

    logger.info("All services initialized")
    yield
    # Shutdown
    logger.info("Shutting down")


app = FastAPI(
    title="CV Analysis RAG System",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# Import and include routers
from app.routes import health, upload, cvs, chat, models, match, analytics, settings as settings_routes

app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(cvs.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(match.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(settings_routes.router, prefix="/api")

# Serve frontend static files in production
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
