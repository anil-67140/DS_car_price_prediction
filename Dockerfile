# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    CARPRICE_MODELS_DIR=/app/artifacts/models PORT=8000
WORKDIR /app

# libgomp is required by lightgbm/xgboost/catboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY configs ./configs
COPY api ./api
COPY app ./app
COPY artifacts/models ./artifacts/models
ENV PYTHONPATH=/app/src:/app

# run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:${PORT}/health || exit 1

# default: REST API.  Override the command to run the Gradio UI:  python -m app.app
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
