# ============================================================
# Dockerfile — RAG CV Assistant
# ============================================================
# Multi-stage build for a lean production image.
#
# Build:
#   docker build -t rag-cv-api .
#
# Run:
#   docker run -p 8000:8000 --env-file .env rag-cv-api
# ============================================================

# ---------- Stage 1: Base ----------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------- Stage 2: Dependencies ----------
FROM base AS dependencies

# Install pip (standard Python package manager) instead of uv due to memory constraints
RUN pip install --upgrade pip

# Install system dependencies required by some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# Install torch CPU-only first to avoid pulling CUDA (~2GB savings)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt

# ---------- Stage 3: Production ----------
FROM base AS production

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/chroma data/uploads

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
