"""
Pump Scenario Sample Generator & Visualiser
===========================================
Runs all 9 pump scenarios serially (no Ray) on short windows,
then produces three grouped plots showing what makes each failure
distinct from its hard negatives.

Setup:
  # Install uv if not present
  pip install uv

  # Create environment and install dependencies
  uv venv .venv
  source .venv/bin/activate        # Linux / Mac
  # .venv\\Scripts\\activate       # Windows

  uv pip install progpy numpy matplotlib pyarrow pandas

Run:
  python sample_generator.py

Output:
  sample_data/   — one CSV per scenario
  plots/         — three PNG files, one per failure group
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
if matplotlib.get_backend() == "agg" or not hasattr(matplotlib, "_pylab_helpers"):
    matplotlib.use("Agg")      # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Output directories ─────────────────────────────────────────────────────────
Path("sample_data").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)

# ── Simulation config ──────────────────────────────────────────────────────────
SAVE_FREQ     = 60          # seconds between saved rows
HORIZON_S     = 6 * 3600    # max 6-hour window
CHANGE_AT_S   = 2 * 3600    # decoy changes happen at 2-hour mark
PRE_FAIL_S    = 3600         # 1-hour pre-failure label window
CYCLE_TIME    = 3600         # 1-hour voltage cycle (per NASA example)

# Nominal operating point (from NASA progpy sim_pump example)
V_NOM         = 471.2389
V_HIGH        = 571.2389
TAMB          = 290
PDISCH_NOM    = 928654
PSUC          = 239179
PDISCH_HIGH   = 1_200_000

# Wear rates via x0 initial state — tuned so each failure takes ~6h
#
# Physics (from centrifugal_pump.py next_state):
#   rThrustdot = wThrust * rThrust * w^2   → friction heat → Tt rises → ThrustBearingOverheat
#   rRadialdot = wRadial * rRadial * w^2   → friction heat → Tr rises → RadialBearingOverheat
#   Adot       = -wA * Q^2                 → impeller area shrinks  → ImpellerWearFailure
#   QLeak is recalculated from pressure each step — no seal leak wear mode exists.
#
WEAR_X0_BEARING  = dict(wA=0, wRadial=0, wThrust=1e-10)    # bearing friction → Tt rises ~6.9h
WEAR_X0_IMPELLER = dict(wA=0.01, wRadial=0, wThrust=0)     # impeller area shrinks → ~6.6h
WEAR_X0_SEAL     = dict(wA=0, wRadial=1e-10, wThrust=0)    # radial bearing → Tr rises ~6h
WEAR_X0_HEALTHY  = dict(wA=0, wRadial=0, wThrust=0)

# Per-signal noise (dicts keyed by state/output name)
PROCESS_NOISE = {
    'w': 0.5, 'Q': 1e-5, 'Tt': 0.1, 'Tr': 0.1, 'To': 0.05,
    'A': 0, 'rRadial': 0, 'rThrust': 0, 'QLeak': 0,
    'wA': 0, 'wRadial': 0, 'wThrust': 0,
}
MEASUREMENT_NOISE = {'w': 0.3, 'Qout': 5e-5, 'Tt': 0.2, 'Tr': 0.2, 'To': 0.1}


# ── Load functions ────────────────────────────────────────────────────────────
# Each returns an InputContainer; pump instance is created once and captured
# in closure. Cycling voltage gives natural signal variation (per NASA example).

def _make_pump():
    """Create a bare CentrifugalPump for InputContainer only (no sim)."""
    from progpy.models import CentrifugalPump
    return CentrifugalPump()

_IC_PUMP = None  # lazy singleton for InputContainer creation

def _ic_pump():
    global _IC_PUMP
    if _IC_PUMP is None:
        _IC_PUMP = _make_pump()
    return _IC_PUMP


def _cycling_v(t):
    """Voltage cycling: 1-hour period with ramp transitions (NASA example)."""
    t_cyc = t % CYCLE_TIME
    if t_cyc < CYCLE_TIME / 2.0:
        return V_NOM
    elif t_cyc < CYCLE_TIME / 2 + 100:
        return V_NOM + (t_cyc - CYCLE_TIME / 2)
    elif t_cyc < CYCLE_TIME - 100:
        return V_HIGH
    else:
        return V_NOM - (t_cyc - CYCLE_TIME)


def base_load(t, x=None):
    V = _cycling_v(t)
    return _ic_pump().InputContainer(
        {'Tamb': TAMB, 'V': V, 'pdisch': PDISCH_NOM,
         'psuc': PSUC, 'wsync': V * 0.8})


def highload_step_load(t, x=None):
    """At CHANGE_AT_S, step voltage up → higher speed → Tt rises from load, not wear."""
    V = _cycling_v(t)
    if t >= CHANGE_AT_S:
        V = V * 1.15  # 15% voltage increase → higher speed → more heat
    wsync = V * 0.8
    return _ic_pump().InputContainer(
        {'Tamb': TAMB, 'V': V, 'pdisch': PDISCH_NOM,
         'psuc': PSUC, 'wsync': wsync})


def highload_ramp_load(t, x=None):
    """Ramp voltage up 15% over hours 2–4 → gradual speed increase → Tt rises."""
    V = _cycling_v(t)
    ramp_end = CHANGE_AT_S + 2 * 3600
    if t >= CHANGE_AT_S:
        if t < ramp_end:
            frac = (t - CHANGE_AT_S) / (ramp_end - CHANGE_AT_S)
            V = V * (1.0 + 0.15 * frac)  # 100% → 115%
        else:
            V = V * 1.15
    wsync = V * 0.8
    return _ic_pump().InputContainer(
        {'Tamb': TAMB, 'V': V, 'pdisch': PDISCH_NOM,
         'psuc': PSUC, 'wsync': wsync})


def bp_step_load(t, x=None):
    """At CHANGE_AT_S, step discharge pressure up 30%."""
    V = _cycling_v(t)
    pdisch = PDISCH_NOM
    if t >= CHANGE_AT_S:
        pdisch = PDISCH_HIGH
    return _ic_pump().InputContainer(
        {'Tamb': TAMB, 'V': V, 'pdisch': pdisch,
         'psuc': PSUC, 'wsync': V * 0.8})


def bp_ramp_load(t, x=None):
    """Ramp discharge pressure up over hours 2–4."""
    V = _cycling_v(t)
    pdisch = PDISCH_NOM
    ramp_end = CHANGE_AT_S + 2 * 3600
    if t >= CHANGE_AT_S:
        if t < ramp_end:
            frac = (t - CHANGE_AT_S) / (ramp_end - CHANGE_AT_S)
            pdisch = PDISCH_NOM + frac * (PDISCH_HIGH - PDISCH_NOM)
        else:
            pdisch = PDISCH_HIGH
    return _ic_pump().InputContainer(
        {'Tamb': TAMB, 'V': V, 'pdisch': pdisch,
         'psuc': PSUC, 'wsync': V * 0.8})


# ── progpy runner ──────────────────────────────────────────────────────────────

def run_progpy(load_fn, wear_x0: dict, faulty: bool) -> pd.DataFrame:
    """
    Run one pump simulation using NASA progpy CentrifugalPump.

    - Wear rates set via x0 initial state (not parameters)
    - Per-signal process_noise and measurement_noise for realism
    - dt left to progpy default for numerical stability
    - Results extracted via .to_simresult().frame
    """
    import warnings
    warnings.filterwarnings('ignore')
    from progpy.models import CentrifugalPump

    pump = CentrifugalPump(
        process_noise=PROCESS_NOISE,
        measurement_noise=MEASUREMENT_NOISE,
    )
    # Set wear rates in initial state (correct progpy API)
    for k, v in wear_x0.items():
        pump.parameters['x0'][k] = v

    first_output = pump.output(pump.initialize(load_fn(0), {}))

    if faulty:
        r = pump.simulate_to_threshold(
            load_fn, first_output,
            events=list(pump.events),
            horizon=HORIZON_S,
            save_freq=SAVE_FREQ,
        )
    else:
        r = pump.simulate_to(
            HORIZON_S, load_fn,
            save_freq=SAVE_FREQ,
        )

    # Extract outputs → DataFrame
    out_df = r.outputs.to_simresult().frame
    out_df.index.name = 'time_s'
    out_df = out_df.reset_index()

    # Extract states for derived columns (Q, QLeak)
    st_df = r.states.frame
    st_df.index.name = 'time_s'
    st_df = st_df.reset_index()

    # Rename model outputs → friendly names
    df = out_df.rename(columns={
        'w':    'shaft_speed_rads',
        'Qout': 'flow_out_m3s',
        'To':   'fluid_temp_K',
        'Tt':   'thrust_bearing_K',
        'Tr':   'radial_bearing_K',
    })

    # Derived columns from states
    df['flow_in_m3s']     = st_df['Q'].values
    df['pump_speed_rads'] = df['shaft_speed_rads']
    df['impeller_area_A'] = st_df['A'].values
    df['r_thrust']        = st_df['rThrust'].values
    df['r_radial']        = st_df['rRadial'].values
    df['time_h']          = df['time_s'] / 3600.0

    return df


def make_label(df: pd.DataFrame, faulty: bool) -> pd.DataFrame:
    n = len(df)
    df['label'] = 'NORMAL'
    df['time_to_failure_s'] = -1.0
    if faulty:
        fi = n - 1
        pre_steps = int(PRE_FAIL_S / SAVE_FREQ)
        pi = max(0, fi - pre_steps)
        df.loc[pi:fi-1, 'label'] = 'PRE_FAILURE'
        df.loc[fi:,      'label'] = 'FAILURE'
        for i in range(pi, n):
            df.loc[i, 'time_to_failure_s'] = max(0, (fi - i) * SAVE_FREQ)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# All 9 scenarios
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:        str
    group:       str       # "bearing" | "impeller" | "seal"
    kind:        str       # "failure" | "decoy" | "normal"
    wear_x0:    dict       # initial-state wear rates
    faulty:      bool
    description: str
    load_fn:     object = field(default=None, repr=False)


def build_scenarios() -> list[Scenario]:
    return [
        # ── Normal ────────────────────────────────────────────────────────
        Scenario(
            name='normal',
            group='all',
            kind='normal',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='Healthy pump — all sensors stable',
            load_fn=base_load,
        ),

        # ── Thrust bearing overheat (wThrust) ────────────────────────────
        # rThrust grows → friction heat → Tt rises while w stays motor-controlled
        # Decoys: wsync change drops w but Tt doesn't diverge from normal
        Scenario(
            name='pump_bearing_wear',
            group='bearing',
            kind='failure',
            wear_x0=WEAR_X0_BEARING,
            faulty=True,
            description='Thrust bearing wear — Tt rises above healthy baseline',
            load_fn=base_load,
        ),
        Scenario(
            name='decoy_highload_step',
            group='bearing',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='High-load step — speed ↑, Tt rises from load (not wear)',
            load_fn=highload_step_load,
        ),
        Scenario(
            name='decoy_highload_ramp',
            group='bearing',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='High-load ramp — speed ↑ gradually, Tt rises proportionally',
            load_fn=highload_ramp_load,
        ),

        # ── Impeller wear (wA) ───────────────────────────────────────────
        # A shrinks → pump head drops → flow capacity degrades
        # Decoys: back pressure change also reduces flow but A stays constant
        Scenario(
            name='pump_impeller_wear',
            group='impeller',
            kind='failure',
            wear_x0=WEAR_X0_IMPELLER,
            faulty=True,
            description='Impeller wear — A shrinks, flow capacity degrades',
            load_fn=base_load,
        ),
        Scenario(
            name='decoy_back_pressure_step',
            group='impeller',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='Back pressure step — flow drops but A stays constant',
            load_fn=bp_step_load,
        ),
        Scenario(
            name='decoy_back_pressure_ramp',
            group='impeller',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='Back pressure ramp — flow drops gradually, A constant',
            load_fn=bp_ramp_load,
        ),

        # ── Radial bearing overheat (wRadial) ────────────────────────────
        # rRadial grows → friction heat → Tr rises
        # Decoys: same wsync changes — w drops but Tr stays healthy
        Scenario(
            name='pump_radial_wear',
            group='radial',
            kind='failure',
            wear_x0=WEAR_X0_SEAL,   # uses wRadial
            faulty=True,
            description='Radial bearing wear — Tr rises above healthy baseline',
            load_fn=base_load,
        ),
        Scenario(
            name='decoy_radial_highload_step',
            group='radial',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='High-load step — speed ↑, Tr rises from load (not wear)',
            load_fn=highload_step_load,
        ),
        Scenario(
            name='decoy_radial_highload_ramp',
            group='radial',
            kind='decoy',
            wear_x0=WEAR_X0_HEALTHY,
            faulty=False,
            description='High-load ramp — speed ↑ gradually, Tr rises proportionally',
            load_fn=highload_ramp_load,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Run all scenarios
# ─────────────────────────────────────────────────────────────────────────────

def run_all(scenarios: list[Scenario]) -> dict[str, pd.DataFrame]:
    results = {}
    scenario_meta = []

    for sc_id, sc in enumerate(scenarios):
        print(f"  Simulating: {sc.name:<35}", end="", flush=True)
        df = run_progpy(sc.load_fn, sc.wear_x0, sc.faulty)
        df = make_label(df, sc.faulty)
        df["scenario_id"] = sc_id

        # Derived column (flow gap — useful for analysis)
        if "flow_in_m3s" in df.columns and "flow_out_m3s" in df.columns:
            df["flow_gap_m3s"] = (df["flow_in_m3s"] - df["flow_out_m3s"]).clip(lower=0)

        # Save per-scenario CSV (data only, no denormalized text columns)
        df.to_csv(f"sample_data/{sc.name}.csv", index=False)

        scenario_meta.append({
            "scenario_id":  sc_id,
            "scenario":     sc.name,
            "group":        sc.group,
            "kind":         sc.kind,
            "faulty":       sc.faulty,
            "description":  sc.description,
        })
        results[sc.name] = df
        print(f" {len(df)} rows  [{df['label'].value_counts().to_dict()}]")

    # Save scenario lookup table
    meta_df = pd.DataFrame(scenario_meta)
    meta_df.to_csv("sample_data/scenarios.csv", index=False)
    print(f"\n  Scenario lookup table: sample_data/scenarios.csv ({len(meta_df)} rows)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Long time-series generator (days → year scale)
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy: run progpy ONCE for 6h to get a steady-state template (1 voltage
# cycle = 1h at 60s save_freq = 60 rows). Tile that template for the full
# duration with fresh noise. Splice in failure episodes and decoy events.
# Total progpy calls: ~8 regardless of duration.

# Signal columns that get tiled / spliced (excludes metadata)
_SIGNAL_COLS = [
    'shaft_speed_rads', 'flow_out_m3s', 'thrust_bearing_K',
    'radial_bearing_K', 'fluid_temp_K', 'flow_in_m3s',
    'pump_speed_rads', 'impeller_area_A', 'r_thrust', 'r_radial',
]

# Noise std per signal for tiling (derived from MEASUREMENT_NOISE mapping)
_TILE_NOISE_STD = {
    'shaft_speed_rads': 0.3,
    'flow_out_m3s':     5e-5,
    'thrust_bearing_K': 0.2,
    'radial_bearing_K': 0.2,
    'fluid_temp_K':     0.1,
    'flow_in_m3s':      5e-5,
    'pump_speed_rads':  0.3,
    'impeller_area_A':  0.0,
    'r_thrust':         0.0,
    'r_radial':         0.0,
}

# Map failure type → wear_x0 dict
_FAILURE_WEAR = {
    'bearing':  WEAR_X0_BEARING,
    'impeller': WEAR_X0_IMPELLER,
    'radial':   WEAR_X0_SEAL,     # uses wRadial
}

# Available decoy types → load functions
_DECOY_LOAD_FNS = {
    'highload_step': highload_step_load,
    'highload_ramp': highload_ramp_load,
    'bp_step':       bp_step_load,
    'bp_ramp':       bp_ramp_load,
}

# Cache for the steady-state template
_TEMPLATE_CACHE: dict | None = None


def _build_template(save_freq_s: int = SAVE_FREQ) -> dict:
    """Run a 6h healthy sim, return startup transient + 1-cycle template."""
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is not None and _TEMPLATE_CACHE['save_freq_s'] == save_freq_s:
        return _TEMPLATE_CACHE

    print("  Building steady-state template (one-time progpy run)...", flush=True)
    df = run_progpy(base_load, WEAR_X0_HEALTHY, faulty=False)

    cycle_rows = CYCLE_TIME // save_freq_s  # rows per voltage cycle
    # Startup = first 4 cycles; template = last cycle (steady state)
    startup = df.iloc[:4 * cycle_rows].copy()
    template = df.iloc[-cycle_rows:].copy()

    _TEMPLATE_CACHE = {
        'startup':     startup[_SIGNAL_COLS].values,
        'template':    template[_SIGNAL_COLS].values,
        'cycle_rows':  cycle_rows,
        'save_freq_s': save_freq_s,
    }
    return _TEMPLATE_CACHE


def _tile_signals(total_rows: int, rng: np.random.Generator,
                  save_freq_s: int = SAVE_FREQ,
                  noise_scale: float = 1.0) -> np.ndarray:
    """Build a (total_rows, n_signals) array by tiling the steady-state template."""
    tmpl = _build_template(save_freq_s)
    startup = tmpl['startup']
    template = tmpl['template']
    cycle_rows = tmpl['cycle_rows']
    n_signals = template.shape[1]

    out = np.empty((total_rows, n_signals), dtype=np.float64)

    # Fill startup
    n_startup = min(len(startup), total_rows)
    out[:n_startup] = startup[:n_startup]

    # Tile remaining
    pos = n_startup
    while pos < total_rows:
        chunk = min(cycle_rows, total_rows - pos)
        out[pos:pos + chunk] = template[:chunk]
        pos += chunk

    # Add fresh noise to each row (except state columns with 0 noise)
    noise_stds = np.array([_TILE_NOISE_STD[c] for c in _SIGNAL_COLS])
    noise = rng.normal(0, 1, size=out.shape) * noise_stds * noise_scale
    out += noise

    # Add slow random-walk drift on temperature columns (ambient drift)
    for i, col in enumerate(_SIGNAL_COLS):
        if col.endswith('_K'):
            drift = np.cumsum(rng.normal(0, 0.001, size=total_rows))
            out[:, i] += drift

    return out


def _splice_window(series: np.ndarray, splice_data: np.ndarray,
                   start_row: int, blend_rows: int = 30) -> None:
    """
    Splice data into series at start_row.

    The splice is shifted so its starting values match the series at the
    insertion point (preserving the degradation *trend* while eliminating
    the jump from different initial conditions). A cosine crossfade over
    blend_rows smooths the entry transition.
    """
    n = min(len(splice_data), len(series) - start_row)
    if n <= 0:
        return
    end_row = start_row + n
    splice = splice_data[:n].copy()

    # Shift splice to match series at insertion point
    # offset = series_value_at_splice_start - splice_value_at_start
    offset = series[start_row] - splice[0]
    splice += offset

    # Cosine crossfade entry (smoother than linear)
    b = min(blend_rows, n)
    for i in range(b):
        alpha = 0.5 * (1 - np.cos(np.pi * i / b))  # 0→1 smooth
        series[start_row + i] = (1 - alpha) * series[start_row + i] + alpha * splice[i]

    # Bulk copy the rest
    if b < n:
        series[start_row + b:end_row] = splice[b:n]


def generate_long_series(
    name: str,
    duration_days: float = 365.0,
    save_freq_s: int = SAVE_FREQ,
    failure_type: str | None = None,
    failure_start_day: float | None = None,
    decoy_types: list[str] | None = None,
    decoy_freq_per_day: float = 0.0,
    decoy_duration_h: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a long pump time series (days to years).

    Parameters
    ----------
    name : str             Series identifier.
    duration_days : float  Total duration in days.
    save_freq_s : int      Seconds between saved rows.
    failure_type : str     'bearing', 'impeller', 'radial', or None for healthy.
    failure_start_day : float  Day when failure degradation begins (None = random 70-90%).
    decoy_types : list     Subset of ['highload_step','highload_ramp','bp_step','bp_ramp'].
    decoy_freq_per_day : float  Average decoy events per day (Poisson rate).
    decoy_duration_h : float    Duration of each decoy episode in hours.
    seed : int             Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with signal columns, time_s, time_h, label, scenario_id, event_type.
    """
    rng = np.random.default_rng(seed)
    total_s = duration_days * 86400
    total_rows = int(total_s / save_freq_s) + 1

    print(f"  Generating {name}: {duration_days:.0f} days, {total_rows} rows", flush=True)

    # ── 1. Tile steady-state template ──────────────────────────────────────────
    signals = _tile_signals(total_rows, rng, save_freq_s)
    time_s = np.arange(total_rows) * save_freq_s
    time_h = time_s / 3600.0

    # Track events for labeling
    labels = np.full(total_rows, 'NORMAL', dtype=object)
    event_types = np.full(total_rows, 'normal', dtype=object)
    ttf = np.full(total_rows, -1.0)

    # ── 2. Schedule and splice decoy events ────────────────────────────────────
    # Each decoy gets random duration (1–6h) and amplitude (0.4–1.6× the
    # template's deviation from baseline), so no two decoys look the same.
    if decoy_types and decoy_freq_per_day > 0:
        n_decoys = rng.poisson(decoy_freq_per_day * duration_days)
        decoy_times_s = np.sort(rng.uniform(0, total_s, size=n_decoys))

        # Pre-simulate each decoy type once (full 6h template)
        decoy_cache = {}
        for dt_name in decoy_types:
            if dt_name in _DECOY_LOAD_FNS and dt_name not in decoy_cache:
                print(f"    Simulating decoy template: {dt_name}...", flush=True)
                ddf = run_progpy(_DECOY_LOAD_FNS[dt_name], WEAR_X0_HEALTHY, faulty=False)
                change_row = CHANGE_AT_S // save_freq_s
                decoy_raw = ddf[_SIGNAL_COLS].values[change_row:]
                # Store deviation from the tiled baseline at those rows
                # so we can scale amplitude per event
                decoy_cache[dt_name] = decoy_raw

        # Get baseline values (mean of steady-state template) for amplitude scaling
        tmpl_baseline = _build_template(save_freq_s)['template'].mean(axis=0)

        used_windows = []

        for t_s in decoy_times_s:
            # Random duration: 1–6 hours
            this_dur_h = rng.uniform(1.0, 6.0)
            this_dur_rows = int(this_dur_h * 3600 / save_freq_s)

            start_row = int(t_s / save_freq_s)
            end_row = start_row + this_dur_rows
            if end_row >= total_rows:
                continue

            overlap = any(not (end_row <= ws or start_row >= we)
                          for ws, we in used_windows)
            if overlap:
                continue

            dt_name = rng.choice(decoy_types)
            if dt_name not in decoy_cache:
                continue

            # Random amplitude scale: 0.4–1.6× the deviation from baseline
            amplitude = rng.uniform(0.4, 1.6)

            full_splice = decoy_cache[dt_name]
            # Trim or pad to this_dur_rows
            n_avail = min(len(full_splice), this_dur_rows)
            splice = full_splice[:n_avail].copy()

            # Scale the deviation: splice = baseline + amplitude * (splice - baseline)
            splice = tmpl_baseline + amplitude * (splice - tmpl_baseline)

            # Add per-event noise so identical types look different
            evt_noise = rng.normal(0, 1, size=splice.shape)
            noise_stds = np.array([_TILE_NOISE_STD[c] for c in _SIGNAL_COLS])
            splice += evt_noise * noise_stds * 2.0

            _splice_window(signals, splice, start_row)
            event_types[start_row:start_row + n_avail] = f'decoy_{dt_name}'
            used_windows.append((start_row, start_row + n_avail))

        print(f"    Spliced {len(used_windows)} decoy events (varied amplitude & duration)",
              flush=True)
    else:
        used_windows = []

    # ── 3. Splice failure episode ──────────────────────────────────────────────
    fail_end_row = None
    if failure_type and failure_type in _FAILURE_WEAR:
        if failure_start_day is None:
            failure_start_day = duration_days * rng.uniform(0.7, 0.9)

        fail_start_s = failure_start_day * 86400
        fail_start_row = int(fail_start_s / save_freq_s)

        print(f"    Simulating {failure_type} failure (starts day {failure_start_day:.1f})...",
              flush=True)
        fdf = run_progpy(base_load, _FAILURE_WEAR[failure_type], faulty=True)
        fail_signals = fdf[_SIGNAL_COLS].values
        n_fail = min(len(fail_signals), total_rows - fail_start_row)

        if n_fail > 0:
            _splice_window(signals, fail_signals[:n_fail], fail_start_row)

            fail_end_row = fail_start_row + n_fail
            pre_fail_rows = PRE_FAIL_S // save_freq_s
            pre_start = max(fail_start_row, fail_end_row - pre_fail_rows - 1)
            labels[pre_start:fail_end_row - 1] = 'PRE_FAILURE'
            labels[fail_end_row - 1] = 'FAILURE'
            event_types[fail_start_row:fail_end_row] = f'failure_{failure_type}'

            for i in range(pre_start, fail_end_row):
                ttf[i] = max(0, (fail_end_row - 1 - i) * save_freq_s)

            print(f"    Failure spliced: rows {fail_start_row}-{fail_end_row} "
                  f"({n_fail} rows, {n_fail * save_freq_s / 3600:.1f}h)", flush=True)

    # ── 3b. Post-failure flatline — pump is dead ──────────────────────────────
    # After failure the device stops: speed→0, flow→0, temps→ambient + sensor noise
    if fail_end_row is not None and fail_end_row < total_rows:
        dead_rows = total_rows - fail_end_row
        dead_values = {
            'shaft_speed_rads': 0.0,
            'flow_out_m3s':     0.0,
            'thrust_bearing_K': TAMB,
            'radial_bearing_K': TAMB,
            'fluid_temp_K':     TAMB,
            'flow_in_m3s':      0.0,
            'pump_speed_rads':  0.0,
            'impeller_area_A':  signals[fail_end_row - 1, _SIGNAL_COLS.index('impeller_area_A')],
            'r_thrust':         signals[fail_end_row - 1, _SIGNAL_COLS.index('r_thrust')],
            'r_radial':         signals[fail_end_row - 1, _SIGNAL_COLS.index('r_radial')],
        }
        for i, col in enumerate(_SIGNAL_COLS):
            base_val = dead_values[col]
            noise_std = _TILE_NOISE_STD[col] * 0.5  # quieter sensor noise on dead pump
            signals[fail_end_row:, i] = base_val + rng.normal(0, noise_std, size=dead_rows)

        # Temperatures cool down gradually (exponential decay to ambient over ~2h)
        cool_tau = 2 * 3600 / save_freq_s  # 2-hour time constant in rows
        for col_name in ['thrust_bearing_K', 'radial_bearing_K', 'fluid_temp_K']:
            ci = _SIGNAL_COLS.index(col_name)
            hot_val = signals[fail_end_row - 1, ci]
            for r in range(dead_rows):
                decay = (hot_val - TAMB) * np.exp(-r / cool_tau)
                signals[fail_end_row + r, ci] = TAMB + decay + rng.normal(0, _TILE_NOISE_STD[col_name] * 0.3)

        labels[fail_end_row:] = 'FAILURE'
        event_types[fail_end_row:] = 'post_failure'
        print(f"    Post-failure flatline: {dead_rows} rows ({dead_rows * save_freq_s / 86400:.1f} days)",
              flush=True)

    # ── 4. Assemble DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame(signals, columns=_SIGNAL_COLS)
    df['time_s'] = time_s
    df['time_h'] = time_h
    df['label'] = labels
    df['time_to_failure_s'] = ttf
    df['event_type'] = event_types

    if 'flow_in_m3s' in df.columns and 'flow_out_m3s' in df.columns:
        df['flow_gap_m3s'] = (df['flow_in_m3s'] - df['flow_out_m3s']).clip(lower=0)

    return df


def generate_dataset(
    duration_days: float = 365.0,
    save_freq_s: int = SAVE_FREQ,
    decoy_freq_per_day: float = 2.0,
    decoy_types: list[str] | None = None,
    seed: int = 42,
    output_dir: str = 'sample_data',
) -> dict[str, pd.DataFrame]:
    """
    Generate a full dataset: one healthy series + one per failure type,
    all with decoy events mixed in.

    Parameters
    ----------
    duration_days : float        Duration of each series in days.
    save_freq_s : int            Seconds between rows.
    decoy_freq_per_day : float   Average decoy events per day in each series.
    decoy_types : list           Decoy types to use.
    seed : int                   Base seed (each series gets seed + offset).
    output_dir : str             Directory for output files.

    Returns
    -------
    dict mapping series name → DataFrame
    """
    if decoy_types is None:
        decoy_types = list(_DECOY_LOAD_FNS.keys())

    Path(output_dir).mkdir(exist_ok=True)

    series_specs = [
        ('normal_long',           None),
        ('bearing_failure_long',  'bearing'),
        ('impeller_failure_long', 'impeller'),
        ('radial_failure_long',   'radial'),
    ]

    results = {}
    meta_rows = []

    for i, (name, fail_type) in enumerate(series_specs):
        df = generate_long_series(
            name=name,
            duration_days=duration_days,
            save_freq_s=save_freq_s,
            failure_type=fail_type,
            decoy_types=decoy_types,
            decoy_freq_per_day=decoy_freq_per_day,
            seed=seed + i,
        )
        df['scenario_id'] = i

        # Save as parquet for large files, CSV for small
        out_path = f"{output_dir}/{name}"
        if len(df) > 50000:
            df.to_parquet(f"{out_path}.parquet", index=False)
            print(f"    → {out_path}.parquet ({len(df)} rows)")
        else:
            df.to_csv(f"{out_path}.csv", index=False)
            print(f"    → {out_path}.csv ({len(df)} rows)")

        meta_rows.append({
            'scenario_id': i,
            'scenario': name,
            'group': fail_type or 'all',
            'kind': 'failure' if fail_type else 'normal',
            'faulty': fail_type is not None,
            'duration_days': duration_days,
            'decoy_freq_per_day': decoy_freq_per_day,
            'description': f"{'Healthy' if not fail_type else fail_type.title() + ' failure'}"
                           f" — {duration_days:.0f} days, ~{decoy_freq_per_day:.1f} decoys/day",
        })
        results[name] = df

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(f"{output_dir}/scenarios_long.csv", index=False)
    print(f"\n  Lookup table: {output_dir}/scenarios_long.csv ({len(meta_df)} rows)")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "failure": "#D85A30",
    "decoy":   "#1D9E75",
    "normal":  "#534AB7",
}

LABEL_ALPHA = {
    "NORMAL":      0.08,
    "PRE_FAILURE": 0.18,
    "FAILURE":     0.25,
}
LABEL_COLOUR = {
    "NORMAL":      "#1D9E75",
    "PRE_FAILURE": "#BA7517",
    "FAILURE":     "#D85A30",
}


def _shade_labels(ax, df, col="label"):
    """Shade background by label."""
    labels = df[col].values
    times  = df["time_h"].values
    for i in range(len(labels)):
        if i < len(labels) - 1:
            ax.axvspan(times[i], times[i+1],
                       alpha=LABEL_ALPHA.get(labels[i], 0.05),
                       color=LABEL_COLOUR.get(labels[i], "#888"),
                       linewidth=0)


def _plot_line(ax, df, col, colour, label, lw=1.5):
    if col in df.columns:
        ax.plot(df["time_h"], df[col], color=colour,
                linewidth=lw, label=label)


def _finish_ax(ax, ylabel, title, show_legend=True):
    ax.set_xlabel("Time (hours)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=7, framealpha=0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Thrust bearing overheat group
#   Key signals: thrust_bearing_K (Tt), shaft_speed_rads
#   Failure: rThrust grows → friction heat → Tt diverges above healthy baseline
#   Decoys: wsync change drops speed but Tt stays normal
# ─────────────────────────────────────────────────────────────────────────────

def plot_bearing_group(results: dict):
    all_scenarios = [
        ("normal",              "Healthy",          COLOURS["normal"]),
        ("pump_bearing_wear",   "Bearing wear",     COLOURS["failure"]),
        ("decoy_highload_step", "High-load step",   COLOURS["decoy"]),
        ("decoy_highload_ramp", "High-load ramp",   "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Thrust bearing wear vs high-load decoys\n"
        "All show Tt rising — the discriminator is Tt vs speed relationship",
        fontsize=11, fontweight="bold", y=1.02
    )

    # Panel 1: Tt over time (all overlaid)
    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["thrust_bearing_K"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Thrust bearing Tt (K)", "Tt over time (all scenarios)")

    # Panel 2: shaft speed over time (all overlaid)
    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["shaft_speed_rads"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Shaft speed (rad/s)", "Speed over time (all scenarios)")

    # Panel 3: scatter speed vs Tt — reveals the discrimination
    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["shaft_speed_rads"], df["thrust_bearing_K"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Shaft speed (rad/s)", fontsize=8)
    ax.set_ylabel("Thrust bearing Tt (K)", fontsize=8)
    ax.set_title("Speed vs Tt  (the discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: Tt is HIGHER than expected for its speed (points shift up)  |  "
             "Decoys: Tt tracks speed proportionally (same band as healthy)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("plots/1_bearing_wear_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → plots/1_bearing_wear_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Impeller wear group
#   Observable signals only: flow_out, fluid_temp (To), shaft_speed
#   Failure: flow drops + temp rises at SAME speed → hidden A degradation
#   Decoys: flow drops because pressure changed (observable cause)
# ─────────────────────────────────────────────────────────────────────────────

def plot_impeller_group(results: dict):
    all_scenarios = [
        ("normal",                   "Healthy",           COLOURS["normal"]),
        ("pump_impeller_wear",       "Impeller wear",     COLOURS["failure"]),
        ("decoy_back_pressure_step", "Back-press step",   COLOURS["decoy"]),
        ("decoy_back_pressure_ramp", "Back-press ramp",   "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Impeller wear vs back-pressure decoys  (observable signals only)\n"
        "Both show flow dropping — the discriminator is flow vs temperature relationship",
        fontsize=11, fontweight="bold", y=1.02
    )

    # Panel 1: flow_out over time
    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["flow_out_m3s"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Flow out (m³/s)", "Flow over time (all scenarios)")

    # Panel 2: oil temp over time
    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["fluid_temp_K"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Oil temperature To (K)", "Oil temp over time (all scenarios)")

    # Panel 3: scatter flow vs temp — reveals discrimination
    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["flow_out_m3s"], df["fluid_temp_K"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Flow out (m³/s)", fontsize=8)
    ax.set_ylabel("Oil temperature To (K)", fontsize=8)
    ax.set_title("Flow vs To  (the discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: temp is HIGHER than expected for its flow (points shift up)  |  "
             "Decoys: temp tracks flow proportionally (same band as healthy)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("plots/2_impeller_wear_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → plots/2_impeller_wear_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Radial bearing overheat group
#   Key signals: radial_bearing_K (Tr), shaft_speed_rads
#   Failure: rRadial grows → friction heat → Tr rises
#   Decoys: wsync change drops speed but Tr stays normal
# ─────────────────────────────────────────────────────────────────────────────

def plot_seal_group(results: dict):
    all_scenarios = [
        ("normal",                       "Healthy",          COLOURS["normal"]),
        ("pump_radial_wear",             "Radial wear",      COLOURS["failure"]),
        ("decoy_radial_highload_step",   "High-load step",   COLOURS["decoy"]),
        ("decoy_radial_highload_ramp",   "High-load ramp",   "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Radial bearing wear vs high-load decoys\n"
        "All show Tr rising — the discriminator is Tr vs speed relationship",
        fontsize=11, fontweight="bold", y=1.02
    )

    # Panel 1: Tr over time
    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["radial_bearing_K"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Radial bearing Tr (K)", "Tr over time (all scenarios)")

    # Panel 2: shaft speed over time
    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["shaft_speed_rads"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Shaft speed (rad/s)", "Speed over time (all scenarios)")

    # Panel 3: scatter speed vs Tr
    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["shaft_speed_rads"], df["radial_bearing_K"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Shaft speed (rad/s)", fontsize=8)
    ax.set_ylabel("Radial bearing Tr (K)", fontsize=8)
    ax.set_title("Speed vs Tr  (the discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: Tr is HIGHER than expected for its speed (points shift up)  |  "
             "Decoys: Tr tracks speed proportionally (same band as healthy)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("plots/3_radial_wear_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → plots/3_radial_wear_group.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 0 — Overview: normal baseline
# ─────────────────────────────────────────────────────────────────────────────

def plot_normal(results: dict):
    df = results["normal"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("Healthy pump — baseline signals (all NORMAL)",
                 fontsize=12, fontweight="bold")

    pairs = [
        ("shaft_speed_rads", "Shaft speed (rad/s)"),
        ("fluid_temp_K",     "Temperature (K)"),
        ("flow_out_m3s",     "Flow out (m³/s)"),
        ("flow_in_m3s",      "Flow in (m³/s)"),
        ("pump_speed_rads",  "Pump speed (rad/s)"),
    ]

    for idx, (col, label) in enumerate(pairs):
        ax = axes[idx // 3][idx % 3]
        if col in df.columns:
            ax.plot(df["time_h"], df[col], color=COLOURS["normal"], linewidth=1.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    # Last axis: label distribution bar
    ax = axes[1][2]
    counts = df["label"].value_counts()
    ax.bar(counts.index, counts.values, color=["#1D9E75"])
    ax.set_title("Label distribution", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig("plots/0_normal_baseline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → plots/0_normal_baseline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "="*72)
    print(f"  {'Scenario':<35} {'Rows':>6}  {'NORMAL':>8}  {'PRE_FAIL':>9}  {'FAILURE':>7}")
    print("  " + "-"*68)
    for name, df in results.items():
        vc  = df["label"].value_counts()
        n   = vc.get("NORMAL",      0)
        pre = vc.get("PRE_FAILURE", 0)
        f   = vc.get("FAILURE",     0)
        print(f"  {name:<35} {len(df):>6}  {n:>8}  {pre:>9}  {f:>7}")
    print("="*72)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Pump Scenario Sample Generator")
    print("  Horizon: 6 hours, save every 60s, cycling load")
    print("  Wear via x0 initial state (NASA progpy convention)")
    print("="*60 + "\n")

    print("Running simulations...")
    scenarios = build_scenarios()
    results   = run_all(scenarios)

    print("\nGenerating plots...")
    plot_normal(results)
    plot_bearing_group(results)
    plot_impeller_group(results)
    plot_seal_group(results)

    print_summary(results)

    print("\n  Files written:")
    print("    sample_data/*.csv   — one CSV per scenario")
    print("    plots/*.png         — four plot files")
    print("\n  What to look for in the plots:")
    print("    Plot 1 — Bearing:  failure has temp rising, decoys have temp flat")
    print("    Plot 2 — Impeller: failure has ratio > 1, decoys ratio stays ~1")
    print("    Plot 3 — Seal:     failure has growing gap, decoys gap = 0")


if __name__ == "__main__":
    main()