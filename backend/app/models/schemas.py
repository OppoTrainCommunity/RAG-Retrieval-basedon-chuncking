from pydantic import BaseModel
from typing import Optional


class CVMetadata(BaseModel):
    cv_id: str
    candidate_name: str
    skills: list[str] = []
    years_of_experience: int = 0
    education: list[dict] = []
    certifications: list[str] = []
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: str = ""
    file_hash: str = ""
    filename: str = ""
    upload_date: str = ""


class UploadResult(BaseModel):
    filename: str
    status: str  # "success" | "duplicate" | "error"
    cv_id: Optional[str] = None
    message: str = ""


class UploadResponse(BaseModel):
    success: bool
    results: list[UploadResult]


class CVListResponse(BaseModel):
    success: bool
    cvs: list[CVMetadata]


class CVDetailResponse(BaseModel):
    success: bool
    cv: CVMetadata


class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None


class SourceInfo(BaseModel):
    cv_id: str
    name: str
    relevance: float = 0.0
    snippet: str = ""


class ChatResponse(BaseModel):
    success: bool
    response: str = ""
    sources: list[SourceInfo] = []
    model_used: str = ""
    error: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    name: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class MatchRequest(BaseModel):
    job_description: str
    model: Optional[str] = None
    top_k: int = 5


class SkillMatch(BaseModel):
    skill: str
    matched: bool


class CandidateMatch(BaseModel):
    candidate_name: str
    cv_id: str
    match_score: int = 0
    key_matching_points: list[str] = []
    missing_qualifications: list[str] = []
    skills_match: list[SkillMatch] = []
    explanation: str = ""


class MatchResponse(BaseModel):
    success: bool
    candidates: list[CandidateMatch] = []
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str


class StatsResponse(BaseModel):
    total_cvs: int = 0
    total_chunks: int = 0
    top_skills: list[dict] = []
    status: str = "active"


class SkillCount(BaseModel):
    skill: str
    count: int


class SkillsAnalyticsResponse(BaseModel):
    skills: list[SkillCount] = []


class ExperienceBucket(BaseModel):
    range: str
    count: int


class ExperienceAnalyticsResponse(BaseModel):
    distribution: list[ExperienceBucket] = []


class ApiKeyRequest(BaseModel):
    api_key: str


class ApiKeyResponse(BaseModel):
    success: bool
    status: str = "unknown"
    message: str = ""


class ErrorResponse(BaseModel):
    success: bool = False
    error: str = ""
