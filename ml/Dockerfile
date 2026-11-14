# Training image of the replication pipeline. Build context: the repository root
#   docker build -f ml/Dockerfile -t relval-training .
# The image contains code only. Data are mounted (local) or synchronised from S3 (AWS Batch).
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && pip install awscli

WORKDIR /workspace
ENV RELVAL_ROOT=/workspace
COPY ml/pyproject.toml ml/pyproject.toml
COPY ml/src ml/src
RUN pip install ./ml
# configs, paper targets (tracked transcriptions) and the Batch entry point
COPY ml/configs ml/configs
COPY docs/replication/paper_targets docs/replication/paper_targets
COPY scripts/batch_entrypoint.sh /usr/local/bin/batch_entrypoint.sh
RUN chmod +x /usr/local/bin/batch_entrypoint.sh && mkdir -p data reports

ENTRYPOINT []
CMD ["relval", "--help"]
