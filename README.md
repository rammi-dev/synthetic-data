# Synthetic Pump Failure Data

Synthetic time-series data for **load-testing and benchmarking ML pipelines** from a performance perspective. The data simulates centrifugal pump sensor readings using NASA's [progpy](https://nasa.github.io/progpy/) CentrifugalPump model.

**This is not production training data.** The purpose is to generate realistic-looking, correctly-structured time series at scale (up to 200GB+) so you can:
- Stress-test data ingestion, feature engineering, and model training pipelines
- Benchmark throughput and resource usage under realistic data volumes
- Validate that pipeline code handles failure labels, decoy events, multi-device partitioning, and mixed event types correctly
- Develop and debug ML workflows before real sensor data is available

Two modes of operation:
1. **Sample notebook** — interactive exploration of failure signatures (4 series, ~150MB)
2. **Fleet generator** — production-scale dataset from 1000s of unique devices (~200GB)

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv sync
```

---

## 1. Sample notebook

Open `sample_visualiser.ipynb` in VS Code or Jupyter and **Run All**.

If data files already exist in `sample_data/`, the notebook loads them directly.
Delete `sample_data/` to force regeneration.

### Running with papermill

```bash
# Quick test — 7 days, long series only
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 7 \
  -p run_short_scenarios false \
  -p downsample 50 \
  --progress-bar

# Full year with diversity
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 365 \
  -p failure_severity 0 \
  -p ambient_var_K 5.0 \
  -p duty_cycle_var 0.5 \
  -p downsample 50 \
  --progress-bar

# No variation (original constant behaviour)
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p ambient_var_K 0 \
  -p duty_cycle_var 0 \
  -p failure_severity 1.0 \
  --progress-bar
```

### Notebook parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_days` | 365 | Length of long time series in days |
| `save_freq_s` | 60 | Seconds between rows (60 = 1-min resolution) |
| `decoy_freq_per_day` | 2.0 | Average decoy events per day per series |
| `seed` | 42 | Random seed |
| `run_short_scenarios` | true | Generate 6h comparison plots (10 scenarios) |
| `run_long_series` | true | Generate year-long time series (4 series) |
| `downsample` | 50 | Downsample for long series plots (1 = full, 50 = fast) |
| `failure_severity` | 0 | Wear rate multiplier. 0 = random 0.5–2.0x, 1.0 = normal ~6h, 2.0 = fast ~3h |
| `maintenance_freq_per_month` | 1.0 | Planned maintenance shutdowns per month. Looks like failure (speed→0) but labelled NORMAL. 0 = none |
| `ambient_var_K` | 5.0 | Ambient temp variation (K). Daily ±5K cycle + seasonal ±2.5K. 0 = constant |
| `duty_cycle_var` | 0.5 | Duty cycle variation (0–1). Day/night + weekend load patterns. 0 = constant |

---

## 2. Fleet generator (production scale)

`fleet_generator.py` creates thousands of unique pump devices as individual parquet files, parallelized across CPU cores.

### Quick test

```bash
uv run python fleet_generator.py \
  --num-devices 5 \
  --duration-days 7 \
  --output-dir test_fleet \
  --max-workers 2
```

### Full 200GB generation

```bash
uv run python fleet_generator.py \
  --num-devices 6000 \
  --duration-days 365 \
  --output-dir /data/pump_fleet \
  --max-workers 31
```

Estimated ~85 minutes on 32 cores. **Resumable** — if interrupted, rerun the same command and it skips completed devices.

### Testing before full run

```bash
# Smoke test — 5 devices, 1 day each (~10s)
uv run python fleet_generator.py \
  --num-devices 5 --duration-days 1 --output-dir test_smoke

# All failure types present, short time — verify diversity (~30s)
# 10 devices guarantees at least 1 of each type (40/20/20/20 split)
uv run python fleet_generator.py \
  --num-devices 10 --duration-days 7 --output-dir test_diversity

# Few devices, full year — verify long series quality (~2min)
uv run python fleet_generator.py \
  --num-devices 4 --duration-days 365 --output-dir test_full_year

# Medium scale — 100 devices, 30 days (~3min)
uv run python fleet_generator.py \
  --num-devices 100 --duration-days 30 --output-dir test_medium
```

After each test, inspect the manifest to confirm device variety:
```bash
# Check failure type distribution
cut -d',' -f4 test_diversity/device_manifest.csv | sort | uniq -c

# Check parameter ranges
head -3 test_diversity/device_manifest.csv | column -t -s','
```

### Fleet parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-devices` | 6000 | Number of unique pump devices |
| `--duration-days` | 365 | Time series length per device |
| `--save-freq-s` | 60 | Seconds between rows |
| `--max-workers` | 0 | Parallel workers (0 = cpu_count - 1) |
| `--base-seed` | 12345 | Base random seed (each device gets seed + index) |
| `--output-dir` | fleet_data | Output directory |
| `--healthy-fraction` | 0.4 | Fraction of devices with no failure |
| `--bearing-fraction` | 0.2 | Fraction with bearing failure |
| `--impeller-fraction` | 0.2 | Fraction with impeller failure |
| `--radial-fraction` | 0.2 | Fraction with radial failure |
| `--config` | — | JSON config file (overrides CLI defaults) |

### What makes each device unique

Every device gets a unique parameter set sampled from distributions:

| Parameter | Distribution | Effect |
|-----------|-------------|--------|
| `failure_type` | Categorical (40/20/20/20%) | Some healthy, some fail differently |
| `failure_start_day` | Uniform(30%–95% of duration) | Some fail early, some late |
| `failure_severity` | LogNormal(0, 0.4), clipped 0.3–3.0x | Failures take 3–10h, not always 6h |
| `ambient_var_K` | Uniform(2–12 K) | Indoor site vs desert vs arctic |
| `duty_cycle_var` | Uniform(0.1–0.8) | 24/7 process plant vs 1-shift shop |
| `decoy_freq_per_day` | Exponential(λ=2), clipped 0–8 | Frequent vs rare operational disturbances |
| `decoy_types` | Random 1–4 subset | Not all sites see all load changes |
| `noise_scale` | Uniform(0.5–2.0) | Sensor quality varies by installation |
| `v_offset` | Normal(0, ±2% V_NOM) | Grid voltage varies by location |
| `p_offset` | Normal(0, ±3% PDISCH_NOM) | System pressure varies by installation |

### Fleet output layout

```
fleet_data/
  pipeline_config.json        # full config for reproducibility
  device_manifest.csv         # one row per device, all params + status
  devices/
    device_000000.parquet     # ~38MB each (365 days at 60s)
    device_000001.parquet
    ...
    device_005999.parquet
```

### Device manifest columns

| Column | Description |
|--------|-------------|
| `device_id` | Unique identifier (device_000000 format) |
| `seed` | Per-device random seed |
| `failure_type` | bearing, impeller, radial, or empty (healthy) |
| `failure_start_day` | Day failure begins (empty if healthy) |
| `failure_severity` | Wear rate multiplier |
| `ambient_var_K` | Ambient temperature variation amplitude |
| `duty_cycle_var` | Duty cycle variation amplitude |
| `decoy_freq_per_day` | Decoy event frequency |
| `decoy_types` | Pipe-separated list of decoy types |
| `noise_scale` | Sensor noise multiplier |
| `v_offset` | Voltage offset from nominal |
| `p_offset` | Pressure offset from nominal |
| `status` | completed, failed, or pending |
| `num_rows` | Actual row count |
| `file_size_bytes` | Parquet file size |
| `generation_time_s` | Wall time to generate |

### Sizing reference

| Devices | Duration | Rows/device | File size/device | Total |
|---------|----------|-------------|-----------------|-------|
| 5 | 7 days | 10K | ~0.9 MB | ~4 MB |
| 100 | 30 days | 43K | ~3.7 MB | ~370 MB |
| 1,000 | 365 days | 525K | ~38 MB | ~38 GB |
| 6,000 | 365 days | 525K | ~38 MB | ~228 GB |

---

## Column reference

**Signal columns** (sensor-observable):

| Column | Unit | Description |
|--------|------|-------------|
| `shaft_speed_rads` | rad/s | Shaft rotational speed |
| `flow_out_m3s` | m³/s | Outlet volumetric flow |
| `flow_in_m3s` | m³/s | Inlet volumetric flow |
| `thrust_bearing_K` | K | Thrust bearing temperature |
| `radial_bearing_K` | K | Radial bearing temperature |
| `fluid_temp_K` | K | Oil/fluid temperature |
| `pump_speed_rads` | rad/s | Pump speed (= shaft speed) |

**State columns** (model-internal, not directly measurable):

| Column | Description |
|--------|-------------|
| `impeller_area_A` | Impeller head coefficient — degrades with wA wear |
| `r_thrust` | Thrust bearing friction coefficient — grows with wThrust |
| `r_radial` | Radial bearing friction coefficient — grows with wRadial |

**Label / metadata columns:**

| Column | Values | Description |
|--------|--------|-------------|
| `label` | `NORMAL`, `PRE_FAILURE`, `FAILURE` | Ground truth |
| `time_to_failure_s` | float | Seconds until failure (-1 if healthy) |
| `event_type` | `normal`, `decoy_*`, `failure_*`, `post_failure`, `startup`, `maintenance` | Current event |
| `scenario_id` / `device_id` | int / str | Series identifier |

## Failure modes

Based on [progpy CentrifugalPump](https://nasa.github.io/progpy/) physics:

| Mode | Wear parameter | State equation | Threshold |
|------|---------------|----------------|-----------|
| Thrust bearing overheat | `wThrust = 1e-10` | rThrust' = wThrust · rThrust · w² | Tt ≥ 370 K |
| Radial bearing overheat | `wRadial = 1e-10` | rRadial' = wRadial · rRadial · w² | Tr ≥ 370 K |
| Impeller wear | `wA = 0.01` | A' = -wA · Q² | A ≤ 9.5 |

### Post-failure behaviour

After failure the pump stops for 6 hours:
- Speed and flow drop to zero immediately
- Temperatures decay exponentially to ambient (290 K) over ~2 hours
- After 6h downtime, pump restarts with a 4h smooth ramp back to normal

### Decoy events (hard negatives)

| Decoy | What it does | Mimics |
|-------|-------------|--------|
| `highload_step` | Step voltage up 15% → speed/temp rise | Bearing overheat |
| `highload_ramp` | Ramp voltage up gradually | Bearing overheat (gradual) |
| `bp_step` | Step discharge pressure up 30% → flow drops | Impeller wear |
| `bp_ramp` | Ramp discharge pressure up gradually | Impeller wear (gradual) |

Each instance has **random amplitude** (0.4–1.6×) and **random duration** (1–6h).

### Discrimination challenge

A simple threshold on any single signal will false-alarm on decoys:

- **Bearing wear vs high-load:** both show Tt rising. Discriminator: Tt vs speed relationship
- **Impeller wear vs back-pressure:** both show flow dropping. Discriminator: flow vs temperature relationship

---

## 3. GCP Dataproc Serverless (cloud scale)

Uses a **custom container image** with progpy + dependencies baked in, submitted as a PySpark batch to Dataproc Serverless.

### Prerequisites

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT
gcloud services enable dataproc.googleapis.com artifactregistry.googleapis.com
```

Docker must be available locally (for building the image).

### Testing before full run

```bash
# Smoke test — 5 devices, 1 day, 2 executors (~2min including image build)
NUM_DEVICES=5 DURATION_DAYS=1 NUM_EXECUTORS=2 ./dataproc_submit.sh

# All failure types, short time — verify diversity
NUM_DEVICES=10 DURATION_DAYS=7 NUM_EXECUTORS=4 ./dataproc_submit.sh

# Few devices, full year — verify long series quality
NUM_DEVICES=4 DURATION_DAYS=365 NUM_EXECUTORS=4 ./dataproc_submit.sh

# Medium scale — 100 devices, 30 days
NUM_DEVICES=100 DURATION_DAYS=30 NUM_EXECUTORS=8 ./dataproc_submit.sh
```

After each test, check the output:
```bash
# List generated files
gsutil ls gs://BUCKET/fleet_output/BATCH_ID/devices/ | head -10

# Check total size
gsutil du -sh gs://BUCKET/fleet_output/BATCH_ID/

# Download manifest to inspect device variety
gsutil cat gs://BUCKET/fleet_output/BATCH_ID/device_manifest/*.csv | head -5
```

### Full 200GB generation (6000 devices, 365 days)

```bash
./dataproc_submit.sh
```

Or with explicit config:

```bash
PROJECT=my-project \
BUCKET=my-pump-data \
NUM_DEVICES=6000 \
DURATION_DAYS=365 \
NUM_EXECUTORS=16 \
./dataproc_submit.sh
```

### Reuse existing image (skip rebuild)

```bash
SKIP_BUILD=1 ./dataproc_submit.sh
SKIP_BUILD=1 NUM_DEVICES=100 DURATION_DAYS=7 ./dataproc_submit.sh
```

### How it works

1. **Builds a custom Spark container** (`Dockerfile.dataproc`) based on `gcr.io/dataproc-images/dataproc_serverless_pyspark:2.2-debian12` with progpy, numpy, pandas, pyarrow, etc. pre-installed
2. **Pushes** the image to Artifact Registry (`${REGION}-docker.pkg.dev/${PROJECT}/pump-fleet/spark:latest`)
3. **Uploads** the PySpark driver script to GCS
4. **Submits** a Dataproc Serverless batch with `--container-image`
5. The **driver** samples all device configs and distributes them as an RDD (~50 devices per partition)
6. Each **executor** calls `_worker_init()` once (builds progpy template + decoy cache), then generates its assigned devices, writing parquet directly to GCS
7. Dynamic allocation scales up to 2× requested executors if available

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT` | gcloud default | GCP project ID |
| `REGION` | us-central1 | Dataproc region |
| `BUCKET` | `${PROJECT}-pump-data` | GCS bucket for output |
| `NUM_DEVICES` | 6000 | Number of pump devices |
| `DURATION_DAYS` | 365 | Days per device |
| `SAVE_FREQ_S` | 60 | Seconds between rows |
| `BASE_SEED` | 12345 | Random seed |
| `NUM_EXECUTORS` | 16 | Spark executor count |
| `EXECUTOR_CORES` | 4 | Cores per executor |
| `EXECUTOR_MEMORY` | 16g | Memory per executor |
| `DRIVER_CORES` | 4 | Driver cores |
| `DRIVER_MEMORY` | 16g | Driver memory |
| `SUBNET` | default | VPC subnet |
| `SERVICE_ACCOUNT` | — | SA for the batch (optional) |
| `AR_REPO` | pump-fleet | Artifact Registry repo name |
| `IMAGE_TAG` | latest | Container image tag |
| `SKIP_BUILD` | 0 | Set to 1 to skip docker build/push |

### Monitoring

```bash
# Watch logs live
gcloud dataproc batches wait pump-fleet-20260331-120000 --region=us-central1

# Check status
gcloud dataproc batches describe pump-fleet-20260331-120000 --region=us-central1

# Check output size
gsutil du -sh gs://BUCKET/fleet_output/pump-fleet-20260331-120000/

# Cancel
gcloud dataproc batches cancel pump-fleet-20260331-120000 --region=us-central1
```

### Output on GCS

```
gs://BUCKET/fleet_output/BATCH_ID/
  pipeline_config/            # FleetConfig as JSON
  device_manifest/            # CSV with all device parameters
  devices/
    device_000000.parquet     # ~38MB each
    ...
    device_005999.parquet
  generation_results/         # CSV with per-device status + timing
```

### Estimated cost

| Scale | Executors | vCPUs | Wall time | Cost (approx) |
|-------|-----------|-------|-----------|---------------|
| 100 devices / 7 days | 4 | 16 | ~5 min | ~$0.15 |
| 1000 devices / 365 days | 8 | 32 | ~30 min | ~$3 |
| 6000 devices / 365 days | 16 | 64 | ~60 min | ~$10 |

Based on Dataproc Serverless pricing (~$0.06/vCPU-hour + ~$0.01/GB-hour).

---

## Architecture

```
sample_generator.py          # progpy simulation + long series tiling engine
sample_visualiser.ipynb       # interactive notebook (papermill-compatible)
fleet_generator.py            # production-scale parallel generator (local)
dataproc_submit.sh            # GCP Dataproc Serverless submission script
sample_data/                  # notebook output (gitignored)
fleet_data/                   # local fleet output (gitignored)
```

### How the tiling engine works

Instead of running progpy for 365 days (would take hours), the generator:
1. Runs progpy **once** for 6h → extracts a 1-hour steady-state template
2. **Tiles** the template for the full duration with fresh noise per cycle
3. Overlays **ambient variation** (daily + seasonal) and **duty cycle** (day/night + weekends)
4. **Splices in** failure episodes (actual progpy runs, ~6h each)
5. **Splices in** decoy events (pre-simulated templates, scaled per event)
6. After failure: 6h **downtime** (cooling) then **restart ramp** back to normal

Total progpy calls: ~8 regardless of duration. Generation is O(1) in time series length.
