# Dockerfile for CausalOps OpenEnv deployment
# Compatible with Hugging Face Spaces (Docker SDK) and OpenEnv from_docker_image
# Constraints: 2 vCPU, 8GB RAM, <20 min runtime

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# PORT: 7860 for HF Spaces, overridable for OpenEnv (which expects 8000)
ENV PORT=7860

WORKDIR /app

# Install dependencies
COPY . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Expose both ports (HF Spaces=7860, OpenEnv default=8000)
EXPOSE 7860 8000

# Health check (uses PORT env var)
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"7860\")}/health')" || exit 1

# Run FastAPI via uvicorn — use $PORT so both HF Spaces and OpenEnv work
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT} --workers 1
