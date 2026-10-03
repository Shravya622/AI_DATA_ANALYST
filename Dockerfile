# Multi-stage build: stage 1 builds the React frontend, stage 2 runs the backend

# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend ───────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY tests/ ./tests/
COPY pyproject.toml .

# Copy built frontend static files
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Expose port
EXPOSE 8000

# Set non-sensitive defaults
ENV APP_ENV=production
ENV MAX_UPLOAD_SIZE_MB=50
ENV MAX_HISTORY_EXCHANGES=20

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
