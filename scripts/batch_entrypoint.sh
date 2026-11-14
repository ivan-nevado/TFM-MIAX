#!/bin/sh
# Entry point of the AWS Batch job (terraform/modules/training_batch).
#
#   batch_entrypoint.sh <stage> <group> <first_year> <chunk>
#
# stage  panel   build the panel from the raw bucket and store it in the processed bucket
#        train   train the models of <group> for one chunk of years (array job: child i handles
#                the years first_year + i*chunk .. first_year + (i+1)*chunk - 1)
#        report  peer weights, tables, figures and the report from all stored predictions
#
# Buckets come from RELVAL_RAW_BUCKET, RELVAL_PROCESSED_BUCKET and RELVAL_ARTIFACTS_BUCKET.
# The job never uploads raw data anywhere: it only reads the raw bucket.
set -eu

STAGE="${1:-train}"
GROUP="${2:-all}"
FIRST_YEAR="${3:-1990}"
CHUNK="${4:-1}"
CONFIG="${RELVAL_CONFIG:-replication.yaml}"
INDEX="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"

cd "${RELVAL_ROOT:-/workspace}"
sync_down() { aws s3 sync "$1" "$2" --only-show-errors; }
sync_up() { aws s3 sync "$1" "$2" --only-show-errors; }

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
    aws s3 cp data/processed/ "s3://${RELVAL_ARTIFACTS_BUCKET}/processed/" --recursive --exclude "*" --include "firms_*.parquet" --only-show-errors
    ;;
  train)
    START=$((FIRST_YEAR + INDEX * CHUNK))
    END=$((START + CHUNK - 1))
    # models that depend on the base run (80:20 split) read its predictions
    sync_down "s3://${RELVAL_ARTIFACTS_BUCKET}/predictions" data/predictions
    relval train --config "$CONFIG" --group "$GROUP" --years "${START}-${END}"
    sync_up data/predictions "s3://${RELVAL_ARTIFACTS_BUCKET}/predictions"
    sync_up data/artifacts "s3://${RELVAL_ARTIFACTS_BUCKET}/artifacts"
    ;;
  report)
    sync_down "s3://${RELVAL_ARTIFACTS_BUCKET}/predictions" data/predictions
    sync_down "s3://${RELVAL_ARTIFACTS_BUCKET}/artifacts" data/artifacts
    relval peers --config "$CONFIG"
    relval report --config "$CONFIG"
    sync_up data/artifacts "s3://${RELVAL_ARTIFACTS_BUCKET}/artifacts"
    sync_up reports "s3://${RELVAL_ARTIFACTS_BUCKET}/reports"
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
