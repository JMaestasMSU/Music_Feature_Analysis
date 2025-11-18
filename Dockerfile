# Multi-stage Dockerfile for the Music Feature Analysis inference service
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy dependency specs first for better caching
COPY requirements.txt pyproject.toml ./

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    find /root/.cache -type f -delete || true

FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . /app

RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# For development use uvicorn. For production use gunicorn with uvicorn workers:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Example prod command (not executed by default):
# gunicorn -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000 --log-level info
