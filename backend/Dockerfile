# Build context: the repository root (docker build -f backend/Dockerfile .)
FROM python:3.10-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv

# The API imports the variable registry from the ML package.
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/app backend/app
RUN pip install ./ml "./backend[s3]"

RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /srv/backend
# Outputs are read from RELVAL_DATA_ROOT / RELVAL_REPORTS_ROOT (a mounted folder or s3:// URIs).
ENV RELVAL_DATA_ROOT=/data RELVAL_REPORTS_ROOT=/reports
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
