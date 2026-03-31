#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit fleet generation to GCP Dataproc Serverless
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT
#   gsutil mb gs://YOUR_BUCKET  (if not exists)
#
# Usage:
#   ./dataproc_submit.sh                          # defaults (6000 devices, 365 days)
#   ./dataproc_submit.sh --num-devices 100        # quick test
#   NUM_DEVICES=6000 BUCKET=my-bucket ./dataproc_submit.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration (override via env vars or CLI) ─────────────────────────────
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-${PROJECT}-pump-data}"
NUM_DEVICES="${NUM_DEVICES:-6000}"
DURATION_DAYS="${DURATION_DAYS:-365}"
SAVE_FREQ_S="${SAVE_FREQ_S:-60}"
BASE_SEED="${BASE_SEED:-12345}"
BATCH_ID="${BATCH_ID:-pump-fleet-$(date +%Y%m%d-%H%M%S)}"
SUBNET="${SUBNET:-default}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"

# Dataproc Serverless resources
DRIVER_CORES="${DRIVER_CORES:-4}"
DRIVER_MEMORY="${DRIVER_MEMORY:-16g}"
EXECUTOR_CORES="${EXECUTOR_CORES:-4}"
EXECUTOR_MEMORY="${EXECUTOR_MEMORY:-16g}"
NUM_EXECUTORS="${NUM_EXECUTORS:-16}"

# GCS paths
GCS_CODE="gs://${BUCKET}/code"
GCS_OUTPUT="gs://${BUCKET}/fleet_output/${BATCH_ID}"
GCS_DEPS="gs://${BUCKET}/deps"

echo "═══════════════════════════════════════════════════"
echo "  Pump Fleet Generator — Dataproc Serverless"
echo "  Project:     ${PROJECT}"
echo "  Region:      ${REGION}"
echo "  Bucket:      ${BUCKET}"
echo "  Batch ID:    ${BATCH_ID}"
echo "  Devices:     ${NUM_DEVICES}"
echo "  Duration:    ${DURATION_DAYS} days"
echo "  Executors:   ${NUM_EXECUTORS} × ${EXECUTOR_CORES} cores"
echo "  Output:      ${GCS_OUTPUT}"
echo "═══════════════════════════════════════════════════"

# ── Step 1: Create bucket if needed ──────────────────────────────────────────
if ! gsutil ls "gs://${BUCKET}" &>/dev/null; then
    echo "Creating bucket gs://${BUCKET}..."
    gsutil mb -l "${REGION}" "gs://${BUCKET}"
fi

# ── Step 2: Package and upload code ──────────────────────────────────────────
echo "Uploading code to ${GCS_CODE}/..."

# Upload the generator modules
gsutil -q cp sample_generator.py "${GCS_CODE}/sample_generator.py"
gsutil -q cp fleet_generator.py "${GCS_CODE}/fleet_generator.py"

# Create the PySpark driver script
cat > /tmp/spark_fleet_driver.py << 'PYSPARK_EOF'
"""
PySpark driver for Dataproc Serverless fleet generation.

Distributes device generation across Spark executors.
Each executor runs generate_single_device() for its partition of devices.
Output is written directly to GCS as parquet.
"""
import argparse
import json
import os
import sys
import time

from pyspark.sql import SparkSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-devices', type=int, required=True)
    parser.add_argument('--duration-days', type=float, required=True)
    parser.add_argument('--save-freq-s', type=int, default=60)
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--code-path', type=str, required=True)
    args = parser.parse_args()

    spark = SparkSession.builder \
        .appName(f"PumpFleet-{args.num_devices}dev") \
        .getOrCreate()

    sc = spark.sparkContext

    # Add the generator modules so executors can import them
    sc.addPyFile(f"{args.code_path}/sample_generator.py")
    sc.addPyFile(f"{args.code_path}/fleet_generator.py")

    output_path = args.output_path
    num_devices = args.num_devices
    duration_days = args.duration_days
    save_freq_s = args.save_freq_s
    base_seed = args.base_seed

    # ── Sample device configs on driver ──────────────────────────────────
    from fleet_generator import FleetConfig, DeviceConfigSampler
    from dataclasses import asdict

    fleet_cfg = FleetConfig(
        num_devices=num_devices,
        duration_days=duration_days,
        save_freq_s=save_freq_s,
        base_seed=base_seed,
    )

    sampler = DeviceConfigSampler(fleet_cfg)
    all_configs = sampler.sample_all()

    # Save pipeline config
    config_dict = asdict(fleet_cfg)
    config_json = json.dumps(config_dict, indent=2)
    spark.sparkContext.parallelize([config_json], 1) \
        .saveAsTextFile(f"{output_path}/pipeline_config")

    # Save device manifest
    import pandas as pd
    manifest_rows = []
    for cfg in all_configs:
        row = asdict(cfg)
        row['decoy_types'] = '|'.join(row['decoy_types'])
        manifest_rows.append(row)
    manifest_df = spark.createDataFrame(pd.DataFrame(manifest_rows))
    manifest_df.coalesce(1).write.mode('overwrite') \
        .csv(f"{output_path}/device_manifest", header=True)

    print(f"Generated {len(all_configs)} device configs")
    print(f"Failure distribution: "
          f"{sum(1 for c in all_configs if c.failure_type is None)} healthy, "
          f"{sum(1 for c in all_configs if c.failure_type == 'bearing')} bearing, "
          f"{sum(1 for c in all_configs if c.failure_type == 'impeller')} impeller, "
          f"{sum(1 for c in all_configs if c.failure_type == 'radial')} radial")

    # ── Distribute generation across executors ───────────────────────────
    # Serialize configs as dicts for Spark distribution
    config_dicts = [asdict(c) for c in all_configs]

    # Partition: ~50 devices per partition for good parallelism
    num_partitions = max(1, num_devices // 50)
    config_rdd = sc.parallelize(config_dicts, num_partitions)

    def generate_device_partition(configs_iter):
        """
        Runs on each executor. Generates devices and writes parquet to GCS.
        Builds template + decoy cache once per executor (not per device).
        """
        import warnings
        warnings.filterwarnings('ignore')

        from fleet_generator import DeviceConfig, generate_single_device, _worker_init

        # Build caches once for this executor
        _worker_init()

        results = []
        for cfg_dict in configs_iter:
            # Reconstruct DeviceConfig
            cfg_dict['decoy_types'] = cfg_dict['decoy_types'] if isinstance(
                cfg_dict['decoy_types'], list) else cfg_dict['decoy_types'].split('|')
            cfg = DeviceConfig(**cfg_dict)

            # Generate and write parquet
            result = generate_single_device(cfg, output_path)
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
    print(f"Fleet generation complete")
    print(f"  Devices: {len(completed)} completed, {len(failed)} failed")
    print(f"  Total rows:  {total_rows:,}")
    print(f"  Total size:  {total_bytes / 1e9:.2f} GB")
    print(f"  Wall time:   {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Throughput:  {len(completed)/elapsed:.1f} devices/s")
    print(f"  Output:      {output_path}")
    print(f"{'='*60}")

    if failed:
        print(f"\nFailed devices:")
        for r in failed[:10]:
            print(f"  {r['device_id']}: {r['status']}")
        if len(failed) > 10:
            print(f"  ... and {len(failed)-10} more")

    # Save results summary
    results_df = spark.createDataFrame(pd.DataFrame(results))
    results_df.coalesce(1).write.mode('overwrite') \
        .csv(f"{output_path}/generation_results", header=True)

    spark.stop()


if __name__ == '__main__':
    main()
PYSPARK_EOF

gsutil -q cp /tmp/spark_fleet_driver.py "${GCS_CODE}/spark_fleet_driver.py"
echo "Code uploaded."

# ── Step 3: Build dependencies archive ───────────────────────────────────────
# progpy and its deps need to be available on executors.
# Create a zip of the installed packages.
echo "Packaging Python dependencies..."

DEPS_ZIP="/tmp/pump_deps.zip"
rm -f "${DEPS_ZIP}"

VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo ".venv/lib/python3.11/site-packages")
if [ ! -d "${VENV_SITE}" ]; then
    VENV_SITE=".venv/lib/python3.11/site-packages"
fi

# Package only the needed libraries (progpy + its deps)
(cd "${VENV_SITE}" && zip -qr "${DEPS_ZIP}" \
    progpy/ numpy/ pandas/ pyarrow/ scipy/ chaospy/ numpoly/ filterpy/ fastdtw/ \
    2>/dev/null || true)

gsutil -q cp "${DEPS_ZIP}" "${GCS_DEPS}/pump_deps.zip"
echo "Dependencies packaged and uploaded ($(du -h ${DEPS_ZIP} | cut -f1))."

# ── Step 4: Submit Dataproc Serverless batch ─────────────────────────────────
echo ""
echo "Submitting Dataproc Serverless batch: ${BATCH_ID}"
echo ""

SA_FLAG=""
if [ -n "${SERVICE_ACCOUNT}" ]; then
    SA_FLAG="--service-account=${SERVICE_ACCOUNT}"
fi

gcloud dataproc batches submit pyspark \
    "${GCS_CODE}/spark_fleet_driver.py" \
    --batch="${BATCH_ID}" \
    --project="${PROJECT}" \
    --region="${REGION}" \
    --subnet="${SUBNET}" \
    ${SA_FLAG} \
    --deps-bucket="gs://${BUCKET}" \
    --py-files="${GCS_CODE}/sample_generator.py,${GCS_CODE}/fleet_generator.py,${GCS_DEPS}/pump_deps.zip" \
    --properties="spark.executor.instances=${NUM_EXECUTORS},spark.executor.cores=${EXECUTOR_CORES},spark.executor.memory=${EXECUTOR_MEMORY},spark.driver.cores=${DRIVER_CORES},spark.driver.memory=${DRIVER_MEMORY},spark.dynamicAllocation.enabled=true,spark.dynamicAllocation.maxExecutors=$((NUM_EXECUTORS * 2))" \
    -- \
    --num-devices "${NUM_DEVICES}" \
    --duration-days "${DURATION_DAYS}" \
    --save-freq-s "${SAVE_FREQ_S}" \
    --base-seed "${BASE_SEED}" \
    --output-path "${GCS_OUTPUT}" \
    --code-path "${GCS_CODE}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Batch submitted: ${BATCH_ID}"
echo "  Monitor: gcloud dataproc batches describe ${BATCH_ID} --region=${REGION}"
echo "  Logs:    gcloud dataproc batches wait ${BATCH_ID} --region=${REGION}"
echo "  Output:  ${GCS_OUTPUT}/devices/"
echo "  Cancel:  gcloud dataproc batches cancel ${BATCH_ID} --region=${REGION}"
echo "═══════════════════════════════════════════════════"
