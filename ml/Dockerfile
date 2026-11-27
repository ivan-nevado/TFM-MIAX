# Training image of the replication pipeline. Build context: the repository root
#   docker build -f ml/Dockerfile -t relval-training \
#     --build-arg GIT_COMMIT=$(git rev-parse HEAD) --build-arg IMAGE_TAG=<tag> .
# The image contains code only. Data are mounted (local) or synchronised from S3 (AWS Batch).
# GIT_COMMIT and IMAGE_TAG are written into every run manifest (the image has no .git).
FROM python:3.10-slim
ARG GIT_COMMIT=unknown
ARG IMAGE_TAG=unknown
ENV RELVAL_GIT_COMMIT=${GIT_COMMIT} RELVAL_IMAGE_TAG=${IMAGE_TAG}
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
