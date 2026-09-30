# ==============================================================================
# Production Dockerfile for Unofficial Rotax 912 Simulation FastAPI Backend
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    APP_ENV=production \
    LOG_LEVEL=INFO \
    ENABLE_API_DOCS=false

WORKDIR /app

# Install security updates and build requirements layer first for caching
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root system user for secure container execution
RUN useradd -m -u 1000 -s /bin/bash appuser

# Copy application code and configuration
COPY app/ ./app/
COPY config/ ./config/
COPY main.py .

# Grant ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Native Python healthcheck against /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, os; sys_port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{sys_port}/health')" || exit 1

# Production server execution (delegates to uvicorn)
CMD ["python", "main.py"]
