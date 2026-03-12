"""
Pydantic Schemas
================

Request/Response models for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Any, Optional, List


class Query(BaseModel):
    """Request body for the synthesize endpoint."""
    question: str = Field(..., description="Question to ask about the resumes", min_length=1)
    k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    language: Optional[str] = Field("en", description="Prompt language: 'en' for English, 'ar' for Arabic")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Who has experience in machine learning?",
                "k": 5,
                "language": "en"
            }
        }


class SourceInfo(BaseModel):
    """Source document information."""
    source: str = Field(default="Unknown", description="Source file name")
    page: Any = Field(default=None, description="Page number")
    chunk_id: Any = Field(default=None, description="Chunk identifier")
    candidate_name: Optional[str] = Field(default=None, description="Candidate name")
    preview: Optional[str] = Field(default=None, description="Content preview")


class QueryResponse(BaseModel):
    """Response model for the synthesize endpoint."""
    response: str = Field(..., description="Generated answer from the RAG system")
    sources: List[SourceInfo] = Field(default_factory=list, description="Source documents used")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Based on the resumes, John Doe has 3 years of experience in machine learning...",
                "sources": [{"source": "resume_john.pdf", "page": 1, "chunk_id": "1", "candidate_name": "John Doe"}]
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status of the API")
    embeddings_model: Optional[str] = None
    collection_count: Optional[int] = None
    version: Optional[str] = None
    error: Optional[str] = None


class IngestResponse(BaseModel):
    """Response model for the ingest endpoint."""
    message: str = Field(..., description="Summary message")
    new_vectors: int = Field(..., description="Number of new vectors added")
    skipped_duplicates: int = Field(..., description="Number of duplicate vectors skipped")
    total_chunks: int = Field(..., description="Total chunks created")
    failed_files: List[str] = Field(default_factory=list, description="Files that failed to process")


class RootResponse(BaseModel):
    """Response model for the root endpoint."""
    message: str
    version: str
    endpoints: dict
