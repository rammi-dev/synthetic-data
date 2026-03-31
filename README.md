# Synthetic Pump Failure Data

Synthetic time-series data for centrifugal pump failure detection, generated using NASA's [progpy](https://nasa.github.io/progpy/) CentrifugalPump model.

## Setup

```bash
uv venv .venv && source .venv/bin/activate
uv sync
```

## Quick start

Open `sample_visualiser.ipynb` in VS Code or Jupyter and **Run All**.

If data files already exist in `sample_data/`, the notebook loads them directly (no re-generation).
To regenerate, delete `sample_data/` and run again.

## Running with papermill

Papermill lets you run the notebook from the command line with custom parameters and monitor progress:

```bash
# Quick test — 7 days, long series only
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 7 \
  -p run_short_scenarios false \
  --progress-bar

# Full year — all plots
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 365 \
  -p decoy_freq_per_day 2.0 \
  --progress-bar

# High decoy frequency, 90 days
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 90 \
  -p decoy_freq_per_day 5.0 \
  -p run_short_scenarios false \
  --progress-bar
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_days` | 365 | Length of long time series (days) |
| `save_freq_s` | 60 | Seconds between rows (60 = 1-min resolution) |
| `decoy_freq_per_day` | 2.0 | Average decoy (load-change) events per day |
| `seed` | 42 | Random seed for reproducibility |
| `run_short_scenarios` | true | Generate 6h comparison plots (10 scenarios) |
| `run_long_series` | true | Generate year-long time series (4 series) |

## Output files

### Short scenarios (`sample_data/*.csv`)

10 independent 6-hour simulations for visual comparison of failure vs decoy signatures:

| File | Type | Description |
|------|------|-------------|
| `normal.csv` | healthy | Baseline — all signals stable |
| `pump_bearing_wear.csv` | failure | Thrust bearing overheat (wThrust) |
| `pump_impeller_wear.csv` | failure | Impeller area degradation (wA) |
| `pump_radial_wear.csv` | failure | Radial bearing overheat (wRadial) |
| `decoy_highload_*.csv` | decoy | Higher voltage/speed — temp rises from load |
| `decoy_back_pressure_*.csv` | decoy | Higher discharge pressure — flow drops |
| `scenarios.csv` | lookup | Scenario metadata (join on `scenario_id`) |

### Long series (`sample_data/*.parquet`)

4 year-long time series with decoy events mixed throughout:

| File | Description |
|------|-------------|
| `normal_long.parquet` | Healthy pump + decoys (no failure) |
| `bearing_failure_long.parquet` | Normal + decoys + bearing failure + post-failure flatline |
| `impeller_failure_long.parquet` | Normal + decoys + impeller failure + post-failure flatline |
| `radial_failure_long.parquet` | Normal + decoys + radial failure + post-failure flatline |
| `scenarios_long.csv` | Lookup table for long series |

### Column reference

**Signal columns** (sensor-observable):

| Column | Unit | Description |
|--------|------|-------------|
| `shaft_speed_rads` | rad/s | Shaft rotational speed |
| `flow_out_m3s` | m3/s | Outlet volumetric flow |
| `flow_in_m3s` | m3/s | Inlet volumetric flow |
| `thrust_bearing_K` | K | Thrust bearing temperature |
| `radial_bearing_K` | K | Radial bearing temperature |
| `fluid_temp_K` | K | Oil/fluid temperature |
| `pump_speed_rads` | rad/s | Pump speed (= shaft speed) |

**State columns** (model-internal, not directly measurable):

| Column | Description |
|--------|-------------|
| `impeller_area_A` | Impeller head coefficient (degrades with wA wear) |
| `r_thrust` | Thrust bearing friction coefficient |
| `r_radial` | Radial bearing friction coefficient |

**Label columns:**

| Column | Values |
|--------|--------|
| `label` | `NORMAL`, `PRE_FAILURE`, `FAILURE` |
| `time_to_failure_s` | Seconds until failure (-1 if healthy) |
| `event_type` | `normal`, `decoy_*`, `failure_*`, `post_failure` |
| `scenario_id` | Foreign key to `scenarios.csv` |

## Failure modes (progpy CentrifugalPump physics)

| Mode | Wear param | State equation | Threshold |
|------|-----------|----------------|-----------|
| Thrust bearing overheat | `wThrust=1e-10` | `rThrustdot = wThrust * rThrust * w^2` | Tt >= 370 K |
| Radial bearing overheat | `wRadial=1e-10` | `rRadialdot = wRadial * rRadial * w^2` | Tr >= 370 K |
| Impeller wear | `wA=0.01` | `Adot = -wA * Q^2` | A <= 9.5 |

### Discrimination challenge

Simple thresholds on individual signals don't work because **decoys produce similar symptoms**:

- Bearing wear: Tt rises — but higher load also raises Tt. Discriminator: Tt vs speed relationship.
- Impeller wear: flow drops — but higher back-pressure also drops flow. Discriminator: flow vs temperature relationship.

After failure, the pump stops: speed and flow go to zero, temperatures decay exponentially to ambient (290 K).

## Architecture

```
sample_generator.py        # progpy simulation + long series generator
sample_visualiser.ipynb     # parameterized notebook (papermill-compatible)
```

The long series generator tiles a steady-state template (1 progpy run) instead of simulating for 365 days. Only failure/decoy windows use actual progpy simulation (~8 calls total regardless of duration).
