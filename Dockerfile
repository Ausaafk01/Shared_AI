# =============================================================================
# Production Dockerfile for AI Brain API & Worker Fleet (M5.7)
# =============================================================================
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (curl for healthchecks, tini for signal forwarding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Create non-root system user for runtime execution
RUN groupadd -g 1000 aibrain && \
    useradd -u 1000 -g aibrain -m -s /bin/bash aibrain

# Copy application source code and migration scripts
COPY ai_brain /app/ai_brain
COPY alembic /app/alembic
COPY alembic.ini /app/alembic.ini

# Ensure non-root user owns the application directory and runtime data directory
RUN mkdir -p /app/data /tmp/aibrain && \
    chown -R aibrain:aibrain /app /tmp/aibrain

USER aibrain

# Expose REST API & Prometheus metrics port
EXPOSE 8000

# Health check configuration for API container
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Process supervisor / signal handler
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: start Uvicorn ASGI server with production factory
CMD ["uvicorn", "ai_brain.server.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
