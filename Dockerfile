# Dockerfile for CausalOps OpenEnv deployment
# Compatible with Hugging Face Spaces (Docker SDK)
# Constraints: 2 vCPU, 8GB RAM, <20 min runtime

FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
COPY . .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# HF Spaces expects port 7860 by default
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run FastAPI via uvicorn
# --workers 1 keeps memory low (single-process, stateful env)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
