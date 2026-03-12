# Stage 1: Build frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Install Python dependencies
FROM python:3.13-slim AS python-deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# Pre-download FastEmbed model to a known directory
RUN PYTHONPATH=/install/lib/python3.13/site-packages mkdir -p /fastembed_models && PYTHONPATH=/install/lib/python3.13/site-packages python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/fastembed_models')"

# Stage 3: Production
FROM python:3.13-slim
WORKDIR /app/backend

# Copy Python deps
COPY --from=python-deps /install /usr/local
COPY --from=python-deps /fastembed_models /app/data/fastembed_models

# Copy backend code
COPY backend/ ./

# Copy built frontend
COPY --from=frontend-build /app/frontend/dist ./static

# Create data directories
RUN mkdir -p /app/data/chroma /app/data/uploads

ENV PYTHONUNBUFFERED=1
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV UPLOAD_DIR=/app/data/uploads

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
