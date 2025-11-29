# Dockerfile for DeepBoner
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (curl needed for HEALTHCHECK)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv==0.5.4

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY src/ src/
COPY README.md .

# Install runtime dependencies only (no dev/test tools)
RUN uv sync --frozen --no-dev --extra embeddings --extra magentic

# Create non-root user BEFORE downloading models
RUN useradd --create-home --shell /bin/bash appuser

# Set cache directory for HuggingFace models (must be writable by appuser)
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Create cache dir with correct ownership
RUN mkdir -p /app/.cache && chown -R appuser:appuser /app/.cache

# Pre-download the embedding model during build (as appuser to set correct ownership)
USER appuser
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app

# Run the app
CMD ["uv", "run", "python", "-m", "src.app"]
