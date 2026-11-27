#!/bin/sh
# Entry point of the AWS Batch job (terraform/modules/training_batch).
#
#   batch_entrypoint.sh <stage> <group> <first_year> <chunk> [config]
#
# stage  panel    build the panel from the raw bucket and store it in the processed bucket
#        train    train the models of <group> for one chunk of years (array job: child i handles
#                 the years first_year + i*chunk .. first_year + (i+1)*chunk - 1)
#        report   peer weights, tables, figures and the report of the run from its predictions
#        compare  every experiment (ml/configs/experiments) against the replication
# config ml/configs file of the run, e.g. replication.yaml or experiments/e1_l1.yaml
#        (default: RELVAL_CONFIG, then replication.yaml). An experiment config trains only the
#        models of its `experiment.models`; its outputs live under its own run name
#        (s3://<artifacts>/predictions/experiments/<id>/...), never mixed with the replication.
#
# Buckets come from RELVAL_RAW_BUCKET, RELVAL_PROCESSED_BUCKET and RELVAL_ARTIFACTS_BUCKET.
# The job never uploads raw data anywhere: it only reads the raw bucket.
set -eu

STAGE="${1:-train}"
GROUP="${2:-all}"
FIRST_YEAR="${3:-1990}"
CHUNK="${4:-1}"
CONFIG="${5:-${RELVAL_CONFIG:-replication.yaml}}"
INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"

cd "${RELVAL_ROOT:-/workspace}"
sync_down() { aws s3 sync "$1" "$2" --only-show-errors; }
sync_up() { aws s3 sync "$1" "$2" --only-show-errors; }

RUN="$(relval run-name --config "$CONFIG")"
ART="s3://${RELVAL_ARTIFACTS_BUCKET}"
echo "stage=${STAGE} config=${CONFIG} run=${RUN} group=${GROUP} image=${RELVAL_IMAGE_TAG:-?} commit=${RELVAL_GIT_COMMIT:-?}"

sync_down "s3://${RELVAL_PROCESSED_BUCKET}/processed" data/processed
sync_down "s3://${RELVAL_PROCESSED_BUCKET}/external" data/external

case "$STAGE" in
  panel)
    sync_down "s3://${RELVAL_RAW_BUCKET}/raw" data/raw
    relval external --config "$CONFIG"
    relval ingest --config "$CONFIG"
    relval panel --config "$CONFIG"
    sync_up data/processed "s3://${RELVAL_PROCESSED_BUCKET}/processed"
    sync_up data/external "s3://${RELVAL_PROCESSED_BUCKET}/external"
    # the API reads the firm table next to the other outputs
    aws s3 cp data/processed/ "${ART}/processed/" --recursive --exclude "*" --include "firms_*.parquet" --only-show-errors
    ;;
  train)
    START=$((FIRST_YEAR + INDEX * CHUNK))
    END=$((START + CHUNK - 1))
    # only this run's predictions: finished years are skipped (resume after a Spot interruption)
    sync_down "${ART}/predictions/${RUN}" "data/predictions/${RUN}"
    relval train --config "$CONFIG" --group "$GROUP" --years "${START}-${END}"
    sync_up "data/predictions/${RUN}" "${ART}/predictions/${RUN}"
    sync_up "data/artifacts/${RUN}" "${ART}/artifacts/${RUN}"
    ;;
  report)
    sync_down "${ART}/predictions/${RUN}" "data/predictions/${RUN}"
    sync_down "${ART}/artifacts/${RUN}" "data/artifacts/${RUN}"
    relval peers --config "$CONFIG"      # skipped for L1/Huber losses and training windows
    relval report --config "$CONFIG"
    sync_up "data/artifacts/${RUN}" "${ART}/artifacts/${RUN}"
    sync_up "reports/${RUN}" "${ART}/reports/${RUN}"
    if [ "$RUN" = "replication" ]; then
      # relval report writes the headline report next to the docs, outside reports/
      aws s3 cp docs/replication/REPLICATION_REPORT.md "${ART}/reports/REPLICATION_REPORT.md" --only-show-errors
    fi
    ;;
  compare)
    # baseline: the three base GBMs of the replication (all seeds, for the seed-noise level)
    aws s3 sync "${ART}/predictions/replication" data/predictions/replication --only-show-errors \
      --exclude "*" --include "model=ML_lnm2b/*" --include "model=ML_lnv2a/*" --include "model=ML_lnv2s/*"
    sync_down "${ART}/predictions/experiments" data/predictions/experiments
    aws s3 sync "${ART}/artifacts/experiments" data/artifacts/experiments --only-show-errors \
      --exclude "*" --include "*/manifests/*"
    relval compare --config replication.yaml
    sync_up reports/experiments "${ART}/reports/experiments"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
