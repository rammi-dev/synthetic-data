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
    for sc in scenarios:
        print(f"  Simulating: {sc.name:<35}", end="", flush=True)
        df = run_progpy(sc.load_fn, sc.wear_x0, sc.faulty)
        df = make_label(df, sc.faulty)
        df["scenario"]    = sc.name
        df["kind"]        = sc.kind
        df["description"] = sc.description

        # Derived column (flow gap — useful for analysis)
        if "flow_in_m3s" in df.columns and "flow_out_m3s" in df.columns:
            df["flow_gap_m3s"] = (df["flow_in_m3s"] - df["flow_out_m3s"]).clip(lower=0)

        # Save CSV
        df.to_csv(f"sample_data/{sc.name}.csv", index=False)
        results[sc.name] = df
        print(f" {len(df)} rows  [{df['label'].value_counts().to_dict()}]")

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
    scenarios = [
        ("pump_bearing_wear",   "Thrust bearing wear (FAILURE)"),
        ("decoy_highload_step", "High-load step (NORMAL decoy)"),
        ("decoy_highload_ramp", "High-load ramp (NORMAL decoy)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(
        "Thrust bearing wear vs high-load decoys\n"
        "Both show Tt rising — but failure rises FASTER than load explains\n"
        "Decoy: Tt rises proportionally to speed increase (no excess heat)",
        fontsize=11, fontweight="bold", y=0.99
    )

    normal_df = results.get("normal")

    for row, (name, title) in enumerate(scenarios):
        df     = results[name]
        colour = COLOURS["failure"] if "wear" in name else COLOURS["decoy"]

        # Left: thrust bearing temperature (both rise — rate is the discriminator)
        ax = axes[row, 0]
        _shade_labels(ax, df)
        _plot_line(ax, df, "thrust_bearing_K", colour, "Tt (this scenario)")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["thrust_bearing_K"],
                    color="#999", linewidth=1.0, linestyle="--", label="Tt (healthy)")
        _finish_ax(ax, "Thrust bearing Tt (K)", title)

        # Right: shaft speed (decoys show speed increasing, failure doesn't)
        ax = axes[row, 1]
        _shade_labels(ax, df)
        _plot_line(ax, df, "shaft_speed_rads", colour, "shaft speed")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["shaft_speed_rads"],
                    color="#999", linewidth=1.0, linestyle="--", label="w (healthy)")
        _finish_ax(ax, "Shaft speed (rad/s)", title)

    axes[0, 0].set_title("thrust_bearing_K (Tt)\n" + axes[0, 0].get_title(),
                          fontsize=9, fontweight="bold", pad=4)
    axes[0, 1].set_title("shaft_speed_rads\n" + axes[0, 1].get_title(),
                          fontsize=9, fontweight="bold", pad=4)

    fig.text(0.5, 0.01,
             "Failure: Tt rises with NO speed increase (excess friction heat)  |  "
             "Decoy: Tt rises WITH speed increase (expected load heat)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    scenarios = [
        ("pump_impeller_wear",       "Impeller wear (FAILURE)"),
        ("decoy_back_pressure_step", "Back pressure step (NORMAL decoy)"),
        ("decoy_back_pressure_ramp", "Back pressure ramp (NORMAL decoy)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(
        "Impeller wear vs back-pressure decoys  (observable signals only)\n"
        "Both show flow dropping — but failure also shows temp rising at same speed\n"
        "Decoy: flow drops because pressure changed, temp stays proportional",
        fontsize=11, fontweight="bold", y=0.99
    )

    normal_df = results.get("normal")

    for row, (name, title) in enumerate(scenarios):
        df     = results[name]
        colour = COLOURS["failure"] if "wear" in name else COLOURS["decoy"]

        # Left: flow_out (both show reduction — this alone can't discriminate)
        ax = axes[row, 0]
        _shade_labels(ax, df)
        _plot_line(ax, df, "flow_out_m3s", colour, "flow out")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["flow_out_m3s"],
                    color="#999", linewidth=1.0, linestyle="--", label="flow (healthy)")
        _finish_ax(ax, "Flow out (m³/s)", title)

        # Right: fluid temp (failure → temp rises from extra friction/recirculation)
        ax = axes[row, 1]
        _shade_labels(ax, df)
        _plot_line(ax, df, "fluid_temp_K", colour, "To (this scenario)")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["fluid_temp_K"],
                    color="#999", linewidth=1.0, linestyle="--", label="To (healthy)")
        _finish_ax(ax, "Oil temperature To (K)", title)

    axes[0, 0].set_title("flow_out_m3s\n" + axes[0,0].get_title(),
                          fontsize=9, fontweight="bold", pad=4)
    axes[0, 1].set_title("fluid_temp_K (To)\n" + axes[0,1].get_title(),
                          fontsize=9, fontweight="bold", pad=4)

    fig.text(0.5, 0.01,
             "Failure: flow drops AND temp diverges from healthy (hidden A degradation)  |  "
             "Decoy: flow drops but temp stays near healthy baseline",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    scenarios = [
        ("pump_radial_wear",             "Radial bearing wear (FAILURE)"),
        ("decoy_radial_highload_step",   "High-load step (NORMAL decoy)"),
        ("decoy_radial_highload_ramp",   "High-load ramp (NORMAL decoy)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(
        "Radial bearing wear vs high-load decoys\n"
        "Both show Tr rising — but failure rises FASTER than load explains\n"
        "Decoy: Tr rises proportionally to speed increase (no excess heat)",
        fontsize=11, fontweight="bold", y=0.99
    )

    normal_df = results.get("normal")

    for row, (name, title) in enumerate(scenarios):
        df     = results[name]
        colour = COLOURS["failure"] if "wear" in name else COLOURS["decoy"]

        # Left: radial bearing temperature
        ax = axes[row, 0]
        _shade_labels(ax, df)
        _plot_line(ax, df, "radial_bearing_K", colour, "Tr (this scenario)")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["radial_bearing_K"],
                    color="#999", linewidth=1.0, linestyle="--", label="Tr (healthy)")
        _finish_ax(ax, "Radial bearing Tr (K)", title)

        # Right: shaft speed
        ax = axes[row, 1]
        _shade_labels(ax, df)
        _plot_line(ax, df, "shaft_speed_rads", colour, "shaft speed")
        if normal_df is not None:
            ax.plot(normal_df["time_h"], normal_df["shaft_speed_rads"],
                    color="#999", linewidth=1.0, linestyle="--", label="w (healthy)")
        _finish_ax(ax, "Shaft speed (rad/s)", title)

    axes[0, 0].set_title("radial_bearing_K (Tr)\n" + axes[0,0].get_title(),
                          fontsize=9, fontweight="bold", pad=4)
    axes[0, 1].set_title("shaft_speed_rads\n" + axes[0,1].get_title(),
                          fontsize=9, fontweight="bold", pad=4)

    fig.text(0.5, 0.01,
             "Failure: Tr rises with NO speed increase (excess friction heat)  |  "
             "Decoy: Tr rises WITH speed increase (expected load heat)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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