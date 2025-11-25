# Dockerfile for DeepCritical
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv==0.5.4

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY src/ src/
COPY README.md .

# Install dependencies
RUN uv sync --frozen --no-dev --all-extras

# Set cache directory for HuggingFace models
ENV HF_HOME=/app/.cache
ENV TRANSFORMERS_CACHE=/app/.cache

# Pre-download the embedding model during build to speed up cold starts
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

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
