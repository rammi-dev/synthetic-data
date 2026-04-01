# Fleet Generation — Architecture & Memory Profile

This document describes how the pump and DC motor fleet generators work,
where files are created, and where memory is consumed at each stage.

## Overview

Both fleet generators follow the same pattern:

```
FleetConfig → DeviceConfigSampler → [DeviceConfig, ...] → ProcessPoolExecutor → parquet files
```

1. **Driver** samples all device configs (deterministic from seed)
2. **Workers** (local) or **executors** (Dataproc) each build a one-time cache, then generate devices sequentially
3. Each device is written to an individual parquet file immediately after generation

## File layout

### Local (fleet_generator.py / dcmotor_fleet_generator.py)

```
{output_dir}/
  pipeline_config.json          # FleetConfig as JSON (reproducibility)
  device_manifest.csv           # one row per device: all params + status
  devices/
    device_000000.parquet       # pump: ~38 MB (365d @ 60s), motor: ~25 MB
    device_000001.parquet
    ...
```

### Dataproc Serverless (dataproc_submit.sh / dataproc_submit_motor.sh)

```
gs://{BUCKET}/fleet_output/{BATCH_ID}/
  pipeline_config/              # FleetConfig JSON (Spark text output)
  device_manifest/              # CSV with all device parameters
  devices/
    device_000000.parquet       # same format as local
    ...
  generation_results/           # CSV with per-device status + timing
```

## Memory profile — Pump fleet

### Per-worker one-time setup (_worker_init)

| Allocation | Size | Lifetime |
|-----------|------|----------|
| Steady-state template (1 progpy 6h run) | ~2 MB | process lifetime |
| 4 decoy templates (4 progpy 6h runs) | ~8 MB | process lifetime |
| **Total cached per worker** | **~10 MB** | |

### Per-device generation (generate_long_series)

| Stage | Allocation | Size (365d @ 60s) | Freed when |
|-------|-----------|-------------------|------------|
| `signals = np.zeros((525001, 10))` | numpy float64 | 40 MB | `del signals` after DataFrame creation |
| `labels`, `event_types` (object arrays) | numpy object | ~20 MB | `del df` after parquet write |
| `df = pd.DataFrame(signals)` | pandas DataFrame | ~60 MB | `del df` after parquet write |
| `df.to_parquet(output_path)` | pyarrow buffer | ~38 MB (transient) | after write completes |
| **Peak per device** | | **~120 MB** | |
| **Resident after write** | | **~0 MB** | (output_path mode) |

Without `output_path` (notebook mode), the DataFrame is returned to the caller
and stays in memory. With `output_path` (fleet mode), `del df` frees it
immediately after the parquet write.

### Total memory per worker process

```
Worker RSS ≈ Python base (~50 MB)
           + cached templates (~10 MB)
           + peak per-device (~120 MB)
           ≈ 180 MB peak per worker
```

With `--max-workers 8`: peak ~1.4 GB total.

## Memory profile — DC Motor fleet

### Per-worker one-time setup (_worker_init → _build_calibration_grid)

| Allocation | Size | Lifetime |
|-----------|------|----------|
| Calibration grid: ~90 short progpy Powertrain sims (2s @ dt=2e-5) | ~5 MB (cached dicts) | process lifetime |
| `_SS_CACHE` (steady-state result cache) | ~1 MB | process lifetime |
| **Total cached per worker** | **~6 MB** | |

The calibration grid takes ~10 minutes to build on first call (90 Powertrain
sims at dt=2e-5). Subsequent calls return the cached result instantly via
`_GRID_CACHE`.

### Per-device generation (generate_long_series)

| Stage | Allocation | Size (365d @ 60s) | Freed when |
|-------|-----------|-------------------|------------|
| `signals = np.zeros((525001, 8))` | numpy float64 | 32 MB | `del signals` after DataFrame creation |
| `labels`, `event_types`, `ttf` | numpy arrays | ~20 MB | `del df` after parquet write |
| `df = pd.DataFrame(signals)` | pandas DataFrame | ~50 MB | `del df` after parquet write |
| `df.to_parquet(output_path)` | pyarrow buffer | ~25 MB (transient) | after write completes |
| **Peak per device** | | **~100 MB** | |
| **Resident after write** | | **~0 MB** | (output_path mode) |

### Total memory per worker process

```
Worker RSS ≈ Python base (~50 MB)
           + calibration grid (~6 MB)
           + peak per-device (~100 MB)
           ≈ 156 MB peak per worker
```

With `--max-workers 8`: peak ~1.2 GB total.

## Generation time profile

### Pump

| Phase | Time | Notes |
|-------|------|-------|
| Worker init (template + decoy cache) | ~30s | 5 progpy CentrifugalPump sims (6h each) |
| Per device (365 days) | ~10-15s | Tiling + noise + splicing (no progpy per row) |
| Per device (7 days) | ~1-2s | |

### DC Motor

| Phase | Time | Notes |
|-------|------|-------|
| Worker init (calibration grid) | ~10 min | 90 Powertrain sims (2s @ dt=2e-5 each) |
| Per device (365 days) | ~5-10s | Grid interpolation + noise (no progpy per row) |
| Per device (7 days) | ~1s | |

The motor's init is slower because the Powertrain model requires dt=2e-5 for
PWM fidelity. Once the grid is cached, per-device generation is actually
faster than pump (no failure splicing requires additional progpy runs).

## Dataproc Serverless specifics

### Resource recommendations

| Scale | Executors | Cores | Memory | Est. wall time |
|-------|-----------|-------|--------|---------------|
| 100 devices / 7d | 4 | 4 each | 4g each | ~5 min (pump), ~15 min (motor) |
| 1000 devices / 365d | 8 | 4 each | 8g each | ~30 min (pump), ~45 min (motor) |
| 6000 devices / 365d | 16 | 4 each | 16g each | ~60 min (pump), ~90 min (motor) |

Motor is slower due to calibration grid init per executor. Each executor
builds its own grid independently (no cross-executor sharing).

### Executor memory breakdown

```
Executor JVM heap: spark.executor.memory (e.g. 16g)
  └─ PySpark Python worker:
       Python base:        ~50 MB
       Cached grid/templates: ~6-10 MB
       Per-device peak:    ~100-120 MB
       Parquet write buffer: ~25-38 MB
       ────────────────────
       Total:              ~200 MB peak
```

With 4g executor memory, you can safely run sequential device generation.
The JVM overhead is minimal since all work happens in Python.

### Partition strategy

Devices are distributed across `num_devices // 50` partitions (~50 devices
per partition). Each partition is processed by one executor core sequentially.
This keeps memory bounded: only one device's data is in memory at a time per
core.

## Resumability

### Local fleet

The local generators support resumption:
- Device manifest tracks `status` (pending / completed / error)
- On re-run, completed devices are skipped
- Manifest is updated incrementally

### Dataproc

Dataproc batches are not resumable. If a batch fails:
1. Check which devices were written: `gsutil ls gs://BUCKET/fleet_output/BATCH_ID/devices/`
2. Re-submit with the same config — files are overwritten
3. Or adjust `--num-devices` and `--base-seed` to generate missing devices

## Kubernetes Spark Operator

An alternative to Dataproc Serverless for running fleet generation on any
Kubernetes cluster with the [Spark Operator](https://github.com/kubeflow/spark-operator) installed.

### Prerequisites

```bash
# Install Spark Operator via Helm
helm repo add spark-operator https://kubeflow.github.io/spark-operator
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator --create-namespace

# Create service account for Spark pods
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role \
  --clusterrole=edit --serviceaccount=default:spark

# Create PVC for output (or use cloud storage paths instead)
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: fleet-output-pvc
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 250Gi
EOF
```

### Build and push images

```bash
# Pump fleet image
docker build -f k8s/Dockerfile.spark --target pump \
  -t ${REGISTRY}/pump-fleet-spark:latest .
docker push ${REGISTRY}/pump-fleet-spark:latest

# Motor fleet image
docker build -f k8s/Dockerfile.spark --target motor \
  -t ${REGISTRY}/motor-fleet-spark:latest .
docker push ${REGISTRY}/motor-fleet-spark:latest
```

### Submit via SparkApplication CRD

```bash
# Quick test — 5 devices, 7 days
export REGISTRY=my-registry.io NUM_DEVICES=5 DURATION_DAYS=7 NUM_EXECUTORS=2
envsubst < k8s/pump-fleet-sparkapplication.yaml | kubectl apply -f -
envsubst < k8s/motor-fleet-sparkapplication.yaml | kubectl apply -f -

# Full run — 6000 devices, 365 days
export NUM_DEVICES=6000 DURATION_DAYS=365 NUM_EXECUTORS=16
envsubst < k8s/pump-fleet-sparkapplication.yaml | kubectl apply -f -
```

### Monitor and manage

```bash
# Status
kubectl get sparkapplication
kubectl describe sparkapplication pump-fleet

# Driver logs
kubectl logs pump-fleet-driver

# Executor logs (find executor pod names first)
kubectl get pods -l spark-role=executor,app=synthetic-data
kubectl logs pump-fleet-<executor-id>

# Delete
kubectl delete sparkapplication pump-fleet
```

### Resource mapping

The SparkApplication manifests map to the same resources as Dataproc:

| Dataproc param | SparkApplication field | Default |
|----------------|----------------------|---------|
| `EXECUTOR_CORES` | `spec.executor.cores` | 4 |
| `EXECUTOR_MEMORY` | `spec.executor.memory` | 8g |
| `NUM_EXECUTORS` | `spec.executor.instances` | 4 |
| `DRIVER_CORES` | `spec.driver.cores` | 2 |
| `DRIVER_MEMORY` | `spec.driver.memory` | 4g |

### Cloud storage options

The manifests support three output modes:

1. **PVC (default)** — output to a shared PersistentVolumeClaim mounted at `/data/output/`
2. **GCS** — uncomment `hadoopConf` and `secrets` sections, set `--output-path=gs://bucket/path`
3. **S3** — uncomment S3 `hadoopConf`, set `--output-path=s3a://bucket/path`

### Memory per executor pod

```
Pod memory request = executor.memory + overhead (30%)
                   = 8g + 2.4g = 10.4g

Within the Python worker:
  Calibration grid (motor) or templates (pump): ~6-10 MB
  Per-device peak: ~100-120 MB
  Actual Python usage: ~200 MB

The remaining ~8 GB is JVM overhead + Spark shuffle buffers.
For fleet generation (no Spark shuffle), 4g executor memory is sufficient.
```

## Key design decisions

1. **One parquet file per device** — enables parallel reads, partial downloads,
   and device-level data management (delete, resample, etc.)

2. **output_path mode** — fleet generators pass a file path into
   `generate_long_series`, which writes parquet and frees the DataFrame
   immediately. This halves peak memory vs returning the DataFrame to the caller.

3. **Process-level caching** — templates (pump) and calibration grids (motor)
   are built once per worker process and reused across all devices in that
   worker. This is the biggest performance win for the motor generator.

4. **Sequential per-partition** — devices within a partition are generated one
   at a time, not in parallel. This keeps memory bounded and predictable.
