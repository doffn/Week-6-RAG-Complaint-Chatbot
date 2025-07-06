# Multi-stage build for CrediTrust RAG System
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    pytest \
    pytest-cov \
    black \
    flake8 \
    isort

# Copy source code
COPY --chown=app:app . .

# Switch to app user
USER app

# Expose ports
EXPOSE 7860 8888

# Default command for development
CMD ["python", "app.py"]

# Production stage
FROM base as production

# Copy only necessary files
COPY --chown=app:app src/ ./src/
COPY --chown=app:app app.py .
COPY --chown=app:app README.md .

# Create necessary directories
RUN mkdir -p data vector_store results && \
    chown -R app:app data vector_store results

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Expose port
EXPOSE 7860

# Production command
CMD ["python", "app.py"]
