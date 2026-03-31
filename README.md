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

Papermill runs the notebook from the command line with custom parameters and a progress bar:

```bash
# Quick test — 7 days, long series only, fast plots
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 7 \
  -p run_short_scenarios false \
  -p downsample 50 \
  --progress-bar

# Full year — all plots
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 365 \
  -p decoy_freq_per_day 2.0 \
  --progress-bar

# High decoy frequency, 90 days, full-detail plots
uv run papermill sample_visualiser.ipynb output.ipynb \
  -p duration_days 90 \
  -p decoy_freq_per_day 5.0 \
  -p run_short_scenarios false \
  -p downsample 10 \
  --progress-bar

# Regenerate data (delete cache first)
rm -rf sample_data/
uv run papermill sample_visualiser.ipynb output.ipynb --progress-bar
```

## Parameters

All parameters can be overridden via papermill `-p` flags or by editing the first code cell in the notebook.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `duration_days` | 365 | Length of long time series in days |
| `save_freq_s` | 60 | Seconds between rows (60 = 1-min resolution) |
| `decoy_freq_per_day` | 2.0 | Average decoy (load-change) events per day per series |
| `seed` | 42 | Random seed for reproducibility |
| `run_short_scenarios` | true | Generate/load 6h comparison plots (10 scenarios) |
| `run_long_series` | true | Generate/load year-long time series (4 series) |
| `downsample` | 50 | Downsample factor for long series plots (1 = full detail but slow, 50 = fast) |

### Caching behaviour

The notebook checks for existing data before generating:
- **Short scenarios:** looks for `sample_data/normal.csv` + `sample_data/scenarios.csv`
- **Long series:** looks for `sample_data/*_long.parquet` (all 4 files)

If found, data is loaded from disk. Delete `sample_data/` to force regeneration with new parameters.

## Output files

### Short scenarios (`sample_data/*.csv`)

10 independent 6-hour simulations for visual comparison of failure vs decoy signatures:

| File | Type | Description |
|------|------|-------------|
| `normal.csv` | healthy | Baseline — all signals stable |
| `pump_bearing_wear.csv` | failure | Thrust bearing overheat (`wThrust`) |
| `pump_impeller_wear.csv` | failure | Impeller area degradation (`wA`) |
| `pump_radial_wear.csv` | failure | Radial bearing overheat (`wRadial`) |
| `decoy_highload_step.csv` | decoy | Voltage step up — speed and temp rise from load |
| `decoy_highload_ramp.csv` | decoy | Voltage ramp up — gradual speed/temp rise |
| `decoy_back_pressure_step.csv` | decoy | Discharge pressure step — flow drops |
| `decoy_back_pressure_ramp.csv` | decoy | Discharge pressure ramp — gradual flow drop |
| `decoy_radial_highload_step.csv` | decoy | Same as highload step (radial group) |
| `decoy_radial_highload_ramp.csv` | decoy | Same as highload ramp (radial group) |
| `scenarios.csv` | lookup | Scenario metadata (join on `scenario_id`) |

### Long series (`sample_data/*.parquet`)

4 time series (default 365 days) with decoy events mixed throughout at configurable frequency. Each decoy has random amplitude (0.4–1.6x) and duration (1–6h).

| File | Description |
|------|-------------|
| `normal_long.parquet` | Healthy pump + decoys, no failure |
| `bearing_failure_long.parquet` | Normal + decoys + bearing failure + post-failure flatline |
| `impeller_failure_long.parquet` | Normal + decoys + impeller failure + post-failure flatline |
| `radial_failure_long.parquet` | Normal + decoys + radial failure + post-failure flatline |
| `scenarios_long.csv` | Lookup table for long series |

### Column reference

**Signal columns** (sensor-observable):

| Column | Unit | Description |
|--------|------|-------------|
| `shaft_speed_rads` | rad/s | Shaft rotational speed |
| `flow_out_m3s` | m³/s | Outlet volumetric flow |
| `flow_in_m3s` | m³/s | Inlet volumetric flow |
| `thrust_bearing_K` | K | Thrust bearing temperature |
| `radial_bearing_K` | K | Radial bearing temperature |
| `fluid_temp_K` | K | Oil/fluid temperature |
| `pump_speed_rads` | rad/s | Pump speed (= shaft speed in this model) |

**State columns** (model-internal, not directly measurable in production):

| Column | Description |
|--------|-------------|
| `impeller_area_A` | Impeller head coefficient — degrades with `wA` wear |
| `r_thrust` | Thrust bearing friction coefficient — grows with `wThrust` wear |
| `r_radial` | Radial bearing friction coefficient — grows with `wRadial` wear |

**Label / metadata columns:**

| Column | Values | Description |
|--------|--------|-------------|
| `label` | `NORMAL`, `PRE_FAILURE`, `FAILURE` | Ground truth for classification |
| `time_to_failure_s` | float | Seconds until failure (-1 if no failure pending) |
| `event_type` | `normal`, `decoy_*`, `failure_*`, `post_failure` | What is happening at this timestep |
| `scenario_id` | int | Foreign key to `scenarios.csv` / `scenarios_long.csv` |

## Failure modes

Based on [progpy CentrifugalPump](https://nasa.github.io/progpy/) physics:

| Mode | Wear parameter | State equation | Failure threshold |
|------|---------------|----------------|-------------------|
| Thrust bearing overheat | `wThrust = 1e-10` | `rThrust' = wThrust · rThrust · w²` | Tt ≥ 370 K |
| Radial bearing overheat | `wRadial = 1e-10` | `rRadial' = wRadial · rRadial · w²` | Tr ≥ 370 K |
| Impeller wear | `wA = 0.01` | `A' = -wA · Q²` | A ≤ 9.5 |

### Post-failure behaviour

After failure the pump stops operating:
- Shaft speed and flow drop to zero
- Temperatures decay exponentially to ambient (290 K) over ~2 hours
- State variables (A, rThrust, rRadial) freeze at their failure values
- All subsequent rows are labelled `FAILURE` with `event_type = post_failure`

### Decoy events (hard negatives)

Decoys are operational load changes that produce **similar symptoms** to failures:

| Decoy | What it does | Mimics |
|-------|-------------|--------|
| `highload_step` | Step voltage up 15% → speed and temp rise | Bearing overheat (temp rises) |
| `highload_ramp` | Ramp voltage up gradually | Bearing overheat (gradual temp rise) |
| `bp_step` | Step discharge pressure up 30% → flow drops | Impeller wear (flow drops) |
| `bp_ramp` | Ramp discharge pressure up gradually | Impeller wear (gradual flow drop) |

Each decoy instance has **random amplitude** (0.4–1.6× template) and **random duration** (1–6 hours), so no two events look identical.

### Discrimination challenge

A simple threshold on any single signal will false-alarm on decoys:

- **Bearing wear vs high-load:** both show Tt rising. Discriminator: is Tt higher than expected *for the current speed*? (scatter plot of speed vs Tt separates them)
- **Impeller wear vs back-pressure:** both show flow dropping. Discriminator: is temperature higher than expected *for the current flow*? (scatter plot of flow vs To separates them)

## Architecture

```
sample_generator.py         # progpy simulation engine + long series tiling generator
sample_visualiser.ipynb      # parameterized notebook (papermill-compatible, cached data loading)
sample_data/                 # generated data (gitignored)
  *.csv                      #   short 6h scenarios
  *_long.parquet             #   long time series
  scenarios.csv              #   lookup tables
  scenarios_long.csv
```

### Performance

The long series generator tiles a **steady-state template** (single 6h progpy run) instead of simulating the full duration. Only failure and decoy windows use actual progpy simulation.

| Duration | progpy calls | Wall time |
|----------|-------------|-----------|
| 7 days | ~8 | ~2 min |
| 30 days | ~8 | ~2 min |
| 365 days | ~8 | ~2 min |

Generation time is constant regardless of duration — only plotting time scales with data size (controlled by `downsample` parameter).
