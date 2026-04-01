#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit DC motor fleet generation to GCP Dataproc Serverless (custom container)
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT
#   gcloud services enable dataproc.googleapis.com artifactregistry.googleapis.com
#
# Usage:
#   ./dataproc_submit_motor.sh                                    # full run (6000 devices)
#   NUM_DEVICES=100 DURATION_DAYS=7 ./dataproc_submit_motor.sh    # quick test
#   SKIP_BUILD=1 ./dataproc_submit_motor.sh                       # reuse existing image
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration (override via env vars) ────────────────────────────────────
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT}-motor-data}"
NUM_DEVICES="${NUM_DEVICES:-6000}"
DURATION_DAYS="${DURATION_DAYS:-365}"
SAVE_FREQ_S="${SAVE_FREQ_S:-60}"
BASE_SEED="${BASE_SEED:-12345}"
BATCH_ID="${BATCH_ID:-motor-fleet-$(date +%Y%m%d-%H%M%S)}"
SUBNET="${SUBNET:-default}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
SKIP_BUILD="${SKIP_BUILD:-0}"

# Dataproc Serverless resources
DRIVER_CORES="${DRIVER_CORES:-4}"
DRIVER_MEMORY="${DRIVER_MEMORY:-16g}"
EXECUTOR_CORES="${EXECUTOR_CORES:-4}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-16g}"
NUM_EXECUTORS="${NUM_EXECUTORS:-16}"

# Container image
AR_REPO="${AR_REPO:-motor-fleet}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/spark:${IMAGE_TAG}"

# GCS paths
GCS_CODE="gs://${BUCKET}/code"
GCS_OUTPUT="gs://${BUCKET}/fleet_output/${BATCH_ID}"

echo "═══════════════════════════════════════════════════"
echo "  Motor Fleet Generator — Dataproc Serverless"
echo "  Project:     ${PROJECT}"
echo "  Region:      ${REGION}"
echo "  Bucket:      ${BUCKET}"
echo "  Image:       ${IMAGE_URI}"
echo "  Batch ID:    ${BATCH_ID}"
echo "  Devices:     ${NUM_DEVICES}"
echo "  Duration:    ${DURATION_DAYS} days"
echo "  Executors:   ${NUM_EXECUTORS} × ${EXECUTOR_CORES} cores"
echo "  Output:      ${GCS_OUTPUT}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Create GCS bucket if needed ──────────────────────────────────────
if ! gsutil ls "gs://${BUCKET}" &>/dev/null; then
    echo "Creating bucket gs://${BUCKET}..."
    gsutil mb -l "${REGION}" "gs://${BUCKET}"
fi

# ── Step 2: Build and push custom container image ────────────────────────────
if [ "${SKIP_BUILD}" = "0" ]; then
    echo ""
    echo "Building custom Spark container image..."

    # Create Artifact Registry repo if needed
    if ! gcloud artifacts repositories describe "${AR_REPO}" \
        --location="${REGION}" --project="${PROJECT}" &>/dev/null; then
        echo "Creating Artifact Registry repo: ${AR_REPO}"
        gcloud artifacts repositories create "${AR_REPO}" \
            --repository-format=docker \
            --location="${REGION}" \
            --project="${PROJECT}"
    fi

    # Configure docker auth for AR
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

    # Build and push
    docker build -f Dockerfile.dataproc.motor -t "${IMAGE_URI}" .
    docker push "${IMAGE_URI}"
    echo "Image pushed: ${IMAGE_URI}"
else
    echo "Skipping image build (SKIP_BUILD=1), using: ${IMAGE_URI}"
fi

# ── Step 3: Upload PySpark driver to GCS ─────────────────────────────────────
echo ""
echo "Uploading driver script to ${GCS_CODE}/..."

cat > /tmp/spark_motor_fleet_driver.py << 'PYSPARK_EOF'
"""
PySpark driver for Dataproc Serverless motor fleet generation.

Runs inside the custom container where dcmotor_generator and dcmotor_fleet_generator
are pre-installed at /opt/motor-fleet/ (on PYTHONPATH).

Each Spark executor generates a partition of devices, writing parquet to GCS.

Memory profile per executor:
  - Calibration grid: ~90 progpy sims × 2s each → ~5 MB cached dict (one-time)
  - Per device: ~50 MB peak (signals array + DataFrame), freed after parquet write
  - With output_path mode, DataFrame is written and freed inside generate_long_series
  - Safe at 4 GB executor memory for sequential per-partition generation
"""
import argparse
import json
import time

from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-devices', type=int, required=True)
    parser.add_argument('--duration-days', type=float, required=True)
    parser.add_argument('--save-freq-s', type=int, default=60)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"MotorFleet-{args.num_devices}dev") \
        .getOrCreate()

    sc = spark.sparkContext
    output_path = args.output_path

    # ── Sample device configs on driver ──────────────────────────────────
    from dcmotor_fleet_generator import FleetConfig, DeviceConfigSampler
    from dataclasses import asdict

    fleet_cfg = FleetConfig(
        num_devices=args.num_devices,
        duration_days=args.duration_days,
        save_freq_s=args.save_freq_s,
        base_seed=args.base_seed,
    )

    sampler = DeviceConfigSampler(fleet_cfg)
    all_configs = sampler.sample_all()

    # Save pipeline config as JSON
    config_json = json.dumps(asdict(fleet_cfg), indent=2)
    sc.parallelize([config_json], 1) \
        .saveAsTextFile(f"{output_path}/pipeline_config")

    # Save device manifest as CSV via Spark
    import pandas as pd
    manifest_rows = []
    for cfg in all_configs:
        row = asdict(cfg)
        row['decoy_types'] = '|'.join(row['decoy_types'])
        manifest_rows.append(row)
    manifest_df = spark.createDataFrame(pd.DataFrame(manifest_rows))
    manifest_df.coalesce(1).write.mode('overwrite') \
        .csv(f"{output_path}/device_manifest", header=True)

    n_healthy = sum(1 for c in all_configs if c.failure_type is None)
    n_winding = sum(1 for c in all_configs if c.failure_type == 'winding')
    n_bearing = sum(1 for c in all_configs if c.failure_type == 'bearing')
    n_demag = sum(1 for c in all_configs if c.failure_type == 'demag')
    print(f"Device configs: {len(all_configs)} total "
          f"({n_healthy} healthy, {n_winding} winding, "
          f"{n_bearing} bearing, {n_demag} demag)")

    # ── Distribute generation across executors ───────────────────────────
    config_dicts = [asdict(c) for c in all_configs]

    # ~50 devices per partition for good parallelism
    num_partitions = max(1, args.num_devices // 50)
    config_rdd = sc.parallelize(config_dicts, num_partitions)

    # Broadcast output_path so executors can use it
    bc_output = sc.broadcast(output_path)

    def generate_device_partition(configs_iter):
        """
        Runs on each executor — builds calibration grid once, generates all
        assigned devices sequentially.

        Memory: calibration grid is cached in _GRID_CACHE (~5 MB). Each device
        uses ~50 MB peak (signals array → DataFrame → parquet), freed after
        write via output_path mode.
        """
        import warnings
        warnings.filterwarnings('ignore')

        from dcmotor_fleet_generator import DeviceConfig, generate_single_device, _worker_init
        _worker_init()

        results = []
        for cfg_dict in configs_iter:
            cfg_dict['decoy_types'] = (
                cfg_dict['decoy_types'] if isinstance(cfg_dict['decoy_types'], list)
                else cfg_dict['decoy_types'].split('|')
            )
            cfg = DeviceConfig(**cfg_dict)
            result = generate_single_device(cfg, bc_output.value)
            results.append(result)

        return iter(results)

    t0 = time.time()
    results = config_rdd.mapPartitions(generate_device_partition).collect()
    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    completed = [r for r in results if r['status'] == 'completed']
    failed = [r for r in results if r['status'] != 'completed']
    total_bytes = sum(r['file_size_bytes'] for r in completed)
    total_rows = sum(r['num_rows'] for r in completed)

    print(f"\n{'='*60}")
    print(f"Motor fleet generation complete")
    print(f"  Devices: {len(completed)} completed, {len(failed)} failed")
    print(f"  Total rows:  {total_rows:,}")
    print(f"  Total size:  {total_bytes / 1e9:.2f} GB")
    print(f"  Wall time:   {elapsed:.0f}s ({elapsed/60:.1f}m)")
    if elapsed > 0:
        print(f"  Throughput:  {len(completed)/elapsed:.1f} devices/s")
    print(f"  Output:      {output_path}")
    print(f"{'='*60}")

    if failed:
        print(f"\nFailed devices ({len(failed)}):")
        for r in failed[:20]:
            print(f"  {r['device_id']}: {r['status']}")
        if len(failed) > 20:
            print(f"  ... and {len(failed)-20} more")

    # Save generation results
    results_df = spark.createDataFrame(pd.DataFrame(results))
    results_df.coalesce(1).write.mode('overwrite') \
        .csv(f"{output_path}/generation_results", header=True)

    spark.stop()


if __name__ == '__main__':
    main()
PYSPARK_EOF

gsutil -q cp /tmp/spark_motor_fleet_driver.py "${GCS_CODE}/spark_motor_fleet_driver.py"
echo "Driver uploaded."

# ── Step 4: Submit Dataproc Serverless batch ─────────────────────────────────
echo ""
echo "Submitting batch: ${BATCH_ID}"
echo ""

SA_FLAG=""
if [ -n "${SERVICE_ACCOUNT}" ]; then
    SA_FLAG="--service-account=${SERVICE_ACCOUNT}"
fi

gcloud dataproc batches submit pyspark \
    "${GCS_CODE}/spark_motor_fleet_driver.py" \
    --batch="${BATCH_ID}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --subnet="${SUBNET}" \
    --container-image="${IMAGE_URI}" \
    ${SA_FLAG} \
    --properties="\
spark.executor.instances=${NUM_EXECUTORS},\
spark.executor.cores=${EXECUTOR_CORES},\
spark.executor.memory=${EXECUTOR_MEMORY},\
spark.driver.cores=${DRIVER_CORES},\
spark.driver.memory=${DRIVER_MEMORY},\
spark.dynamicAllocation.enabled=true,\
spark.dynamicAllocation.maxExecutors=$((NUM_EXECUTORS * 2)),\
spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem" \
    -- \
    --num-devices "${NUM_DEVICES}" \
    --duration-days "${DURATION_DAYS}" \
    --save-freq-s "${SAVE_FREQ_S}" \
    --base-seed "${BASE_SEED}" \
    --output-path "${GCS_OUTPUT}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Batch submitted: ${BATCH_ID}"
echo ""
echo "  Monitor:"
echo "    gcloud dataproc batches describe ${BATCH_ID} --region=${REGION}"
echo "    gcloud dataproc batches wait ${BATCH_ID} --region=${REGION}"
echo ""
echo "  Output:"
echo "    gsutil ls ${GCS_OUTPUT}/devices/"
echo "    gsutil du -sh ${GCS_OUTPUT}/"
echo ""
echo "  Cancel:"
echo "    gcloud dataproc batches cancel ${BATCH_ID} --region=${REGION}"
echo "═══════════════════════════════════════════════════"
