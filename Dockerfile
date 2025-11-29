# ============================================================================
# XTTS-v2 TTS API - Dockerfile for Hugging Face Spaces
# ============================================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# HF Spaces uses /data for persistent storage
ENV DATA_DIR=/data

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create data directories
RUN mkdir -p /data/voices /data/cache /data/jobs /data/models

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user for security (HF Spaces requirement)
RUN useradd -m -u 1000 user
RUN chown -R user:user /app /data

USER user

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health')" || exit 1

# Start the server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
