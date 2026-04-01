"""
DC Motor (Powertrain) Scenario Sample Generator
================================================
Generates synthetic time-series data for a brushless DC motor powertrain
(ESC + DCMotor + propeller load) using NASA progpy.

Since the Powertrain model has no built-in failure events, degradation is
modelled externally by evolving motor parameters over time:

  1. Winding degradation — R (resistance) increases
  2. Bearing wear         — B (friction/damping) increases
  3. Demagnetization      — K (back-emf constant) decreases

Each failure mode has matching decoy scenarios (load/voltage changes)
that produce similar-looking output shifts without actual degradation.

Setup:
  source .venv/bin/activate
  uv pip install progpy numpy matplotlib pyarrow pandas scipy

Run:
  python dcmotor_generator.py

Output:
  motor_sample_data/  — one CSV per scenario + scenarios.csv
  motor_plots/        — four PNG files
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
if matplotlib.get_backend() == "agg" or not hasattr(matplotlib, "_pylab_helpers"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Output directories ─────────────────────────────────────────────────────────
Path("motor_sample_data").mkdir(exist_ok=True)
Path("motor_plots").mkdir(exist_ok=True)

# ── Simulation config ──────────────────────────────────────────────────────────
SAVE_FREQ     = 60          # seconds between saved rows
HORIZON_S     = 6 * 3600    # max 6-hour window
CHANGE_AT_S   = 2 * 3600    # decoy changes happen at 2-hour mark
PRE_FAIL_S    = 3600        # 1-hour pre-failure label window

# Powertrain operating point
V_NOM         = 23.0        # nominal battery voltage (V)
DUTY_NOM      = 1.0         # full duty cycle
DT_SIM        = 2e-5        # simulation timestep (s) — needed for PWM fidelity
SIM_DURATION  = 2.0         # seconds per steady-state calibration run
SIM_SETTLE    = 1.0         # seconds to discard (transient)

# Propeller load parameters (defaults from Powertrain)
C_Q_COEFF     = 5.42e-7
RHO           = 1.225
PROP_D        = 0.381
C_Q           = C_Q_COEFF * RHO * PROP_D**5   # combined torque coefficient

# DCMotor nominal parameters
R_NOM         = 0.081       # winding resistance (Ohm)
K_NOM         = 0.0265      # back-emf / torque constant (V/(rad/s))
B_NOM         = 0.0         # bearing friction (Nm/(rad/s))
L_NOM         = 83e-6       # inductance (H)
J_NOM         = 26.967e-6   # rotor inertia (kg·m²)

# Degradation end-points (parameter value at failure)
R_FAIL        = 0.30        # ~3.7× nominal — heavy winding degradation
B_FAIL        = 0.0015      # significant friction (nominal is 0)
K_FAIL        = 0.012       # ~45% of nominal — heavy demagnetization

# Failure threshold: speed below this fraction of healthy = FAILURE
SPEED_FAIL_FRAC = 0.40

# Decoy magnitudes
LOAD_MULT_STEP = 2.5        # propeller load multiplier for load-step decoy
V_SAG_FRAC     = 0.75       # voltage drops to 75% for voltage-sag decoy
DUTY_DROP      = 0.60       # duty cycle drops to 60% for duty-change decoy

# Per-signal noise (std dev, applied to the 60s-averaged signals)
PROCESS_NOISE = {
    'v_rot': 0.5,
    'i_rms': 0.01,
    'torque_load': 1e-5,
    'mech_power': 0.01,
    'elec_power': 0.02,
}
MEASUREMENT_NOISE = {
    'v_rot': 0.3,
    'i_rms': 0.005,
    'torque_load': 5e-6,
    'mech_power': 0.005,
    'elec_power': 0.01,
}


# ── Steady-state cache ────────────────────────────────────────────────────────

_SS_CACHE: dict[tuple, dict] = {}


def _run_steady_state(
    voltage: float = V_NOM,
    duty: float = DUTY_NOM,
    R: float = R_NOM,
    K: float = K_NOM,
    B: float = B_NOM,
    c_q_mult: float = 1.0,
) -> dict[str, float]:
    """
    Run a short Powertrain simulation and return steady-state averages.

    Results are cached by parameter tuple for reuse.
    """
    key = (voltage, duty, R, K, B, c_q_mult)
    if key in _SS_CACHE:
        return _SS_CACHE[key]

    import warnings
    warnings.filterwarnings('ignore')

    from progpy.models.dcmotor import DCMotor
    from progpy.models.esc import ESC
    from progpy.models.powertrain import Powertrain

    esc = ESC()
    motor = DCMotor()
    motor.parameters['R'] = R
    motor.parameters['K'] = K
    motor.parameters['B'] = B

    pt = Powertrain(esc, motor,
                    c_q=C_Q_COEFF * c_q_mult,
                    rho=RHO, D=PROP_D)

    def load_fn(t, x=None):
        return pt.InputContainer({'duty': duty, 'v': voltage})

    first_output = pt.output(pt.initialize(load_fn(0), {}))

    r = pt.simulate_to(
        SIM_DURATION, load_fn,
        dt=DT_SIM,
        save_freq=0.05,   # save every 50ms
    )

    # Extract states and outputs from the settled portion
    st_df = r.states.frame
    st_df.index.name = 'time_s'
    st_df = st_df.reset_index()

    settle_mask = st_df['time_s'] >= SIM_SETTLE
    if settle_mask.sum() < 2:
        settle_mask = st_df.index >= len(st_df) // 2

    ss = st_df[settle_mask]

    v_rot = ss['v_rot'].mean()
    i_a = ss['i_a'].values
    i_b = ss['i_b'].values
    i_c = ss['i_c'].values
    i_rms = np.sqrt(np.mean(i_a**2 + i_b**2 + i_c**2))

    torque = pt.parameters['C_q'] * v_rot**2
    mech_power = torque * abs(v_rot)
    elec_power = voltage * i_rms  # approximate

    result = {
        'v_rot': v_rot,
        'i_rms': i_rms,
        'torque_load': torque,
        'mech_power': mech_power,
        'elec_power': elec_power,
    }

    _SS_CACHE[key] = result
    return result


# ── Degradation curves ────────────────────────────────────────────────────────

def _degrade_param(t: float, p_nom: float, p_fail: float,
                   fail_time: float) -> float:
    """Exponential degradation from p_nom at t=0 to p_fail at t=fail_time."""
    if fail_time <= 0:
        return p_fail
    alpha = t / fail_time
    alpha = min(alpha, 1.0)
    # Exponential interpolation for smooth onset
    frac = (np.exp(3 * alpha) - 1) / (np.exp(3) - 1)
    return p_nom + (p_fail - p_nom) * frac


# ── Build steady-state lookup tables ──────────────────────────────────────────

_N_CALIB = 15   # number of calibration points per degradation axis

_GRID_CACHE: tuple | None = None   # (healthy, grids) — built once per process


def _build_calibration_grid():
    """
    Run progpy simulations across degradation parameter ranges.
    Returns (healthy, grids). Cached after first call per process.
    """
    global _GRID_CACHE
    if _GRID_CACHE is not None:
        return _GRID_CACHE

    print("  Building motor steady-state calibration grid...", flush=True)

    # Healthy baseline
    healthy = _run_steady_state()
    print(f"    Healthy baseline: v_rot={healthy['v_rot']:.1f} rad/s, "
          f"i_rms={healthy['i_rms']:.3f} A", flush=True)

    grids = {}

    # --- Winding degradation: R varies ---
    R_vals = np.linspace(R_NOM, R_FAIL, _N_CALIB)
    R_results = []
    for R_val in R_vals:
        R_results.append(_run_steady_state(R=R_val))
    grids['winding'] = {'param_vals': R_vals, 'results': R_results}
    print(f"    Winding grid: {_N_CALIB} points, R={R_NOM:.3f}→{R_FAIL:.3f}", flush=True)

    # --- Bearing wear: B varies ---
    B_vals = np.linspace(B_NOM, B_FAIL, _N_CALIB)
    B_results = []
    for B_val in B_vals:
        B_results.append(_run_steady_state(B=B_val))
    grids['bearing'] = {'param_vals': B_vals, 'results': B_results}
    print(f"    Bearing grid: {_N_CALIB} points, B={B_NOM:.4f}→{B_FAIL:.4f}", flush=True)

    # --- Demagnetization: K varies ---
    K_vals = np.linspace(K_NOM, K_FAIL, _N_CALIB)
    K_results = []
    for K_val in K_vals:
        K_results.append(_run_steady_state(K=K_val))
    grids['demag'] = {'param_vals': K_vals, 'results': K_results}
    print(f"    Demag grid: {_N_CALIB} points, K={K_NOM:.4f}→{K_FAIL:.4f}", flush=True)

    # --- Decoy: load multiplier varies ---
    cq_vals = np.linspace(1.0, LOAD_MULT_STEP, _N_CALIB)
    cq_results = []
    for cq in cq_vals:
        cq_results.append(_run_steady_state(c_q_mult=cq))
    grids['load'] = {'param_vals': cq_vals, 'results': cq_results}
    print(f"    Load grid: {_N_CALIB} points, c_q_mult=1.0→{LOAD_MULT_STEP:.1f}", flush=True)

    # --- Decoy: voltage varies ---
    v_vals = np.linspace(V_NOM, V_NOM * V_SAG_FRAC, _N_CALIB)
    v_results = []
    for v in v_vals:
        v_results.append(_run_steady_state(voltage=v))
    grids['voltage'] = {'param_vals': v_vals, 'results': v_results}
    print(f"    Voltage grid: {_N_CALIB} points, V={V_NOM:.1f}→{V_NOM*V_SAG_FRAC:.1f}", flush=True)

    # --- Decoy: duty cycle varies ---
    d_vals = np.linspace(DUTY_NOM, DUTY_DROP, _N_CALIB)
    d_results = []
    for d in d_vals:
        d_results.append(_run_steady_state(duty=d))
    grids['duty'] = {'param_vals': d_vals, 'results': d_results}
    print(f"    Duty grid: {_N_CALIB} points, duty={DUTY_NOM:.1f}→{DUTY_DROP:.2f}", flush=True)

    _GRID_CACHE = (healthy, grids)
    return healthy, grids


def _interp_grid(grid: dict, param_val: float) -> dict[str, float]:
    """Linearly interpolate steady-state signals at a given parameter value."""
    pvals = grid['param_vals']
    results = grid['results']

    # Clamp to grid range
    p_min, p_max = pvals[0], pvals[-1]
    if p_min > p_max:
        param_val = np.clip(param_val, p_max, p_min)
    else:
        param_val = np.clip(param_val, p_min, p_max)

    out = {}
    for sig in ['v_rot', 'i_rms', 'torque_load', 'mech_power', 'elec_power']:
        sig_vals = np.array([r[sig] for r in results])
        out[sig] = float(np.interp(param_val, pvals if pvals[0] <= pvals[-1] else pvals[::-1],
                                    sig_vals if pvals[0] <= pvals[-1] else sig_vals[::-1]))
    return out


# ── Signal columns ────────────────────────────────────────────────────────────

_SIGNAL_COLS = [
    'rotational_velocity_rads',
    'current_rms_A',
    'torque_load_Nm',
    'mechanical_power_W',
    'electrical_power_W',
    'resistance_ohm',
    'friction_coeff',
    'backemf_constant',
]

_TILE_NOISE_STD = {
    'rotational_velocity_rads': 0.3,
    'current_rms_A':           0.005,
    'torque_load_Nm':          5e-6,
    'mechanical_power_W':      0.005,
    'electrical_power_W':      0.01,
    'resistance_ohm':          0.0,
    'friction_coeff':          0.0,
    'backemf_constant':        0.0,
}


# ── Scenario runner (short 6h windows) ────────────────────────────────────────

def _generate_scenario_series(
    healthy: dict,
    grids: dict,
    degradation_type: str | None,
    decoy_type: str | None,
    faulty: bool,
    save_freq_s: int = SAVE_FREQ,
    horizon_s: int = HORIZON_S,
    change_at_s: int = CHANGE_AT_S,
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate a 6h motor time series.

    For failures: parameter degrades exponentially from t=0 to failure at ~horizon_s.
    For decoys:  input changes at change_at_s (step or ramp).
    For normal:  steady healthy operation.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_rows = horizon_s // save_freq_s + 1
    time_s = np.arange(n_rows) * save_freq_s
    time_h = time_s / 3600.0

    signals = np.zeros((n_rows, len(_SIGNAL_COLS)), dtype=np.float64)
    fail_row = None

    healthy_v_rot = healthy['v_rot']

    for row_i in range(n_rows):
        t = time_s[row_i]

        if degradation_type and faulty:
            # Compute degraded parameter at this time
            fail_time = horizon_s * severity
            if degradation_type == 'winding':
                p_val = _degrade_param(t, R_NOM, R_FAIL, fail_time)
                ss = _interp_grid(grids['winding'], p_val)
                signals[row_i, 5] = p_val          # resistance_ohm
                signals[row_i, 6] = B_NOM           # friction_coeff
                signals[row_i, 7] = K_NOM           # backemf_constant
            elif degradation_type == 'bearing':
                p_val = _degrade_param(t, B_NOM, B_FAIL, fail_time)
                ss = _interp_grid(grids['bearing'], p_val)
                signals[row_i, 5] = R_NOM
                signals[row_i, 6] = p_val
                signals[row_i, 7] = K_NOM
            elif degradation_type == 'demag':
                p_val = _degrade_param(t, K_NOM, K_FAIL, fail_time)
                ss = _interp_grid(grids['demag'], p_val)
                signals[row_i, 5] = R_NOM
                signals[row_i, 6] = B_NOM
                signals[row_i, 7] = p_val
            else:
                ss = healthy
                signals[row_i, 5] = R_NOM
                signals[row_i, 6] = B_NOM
                signals[row_i, 7] = K_NOM

            # Check failure threshold
            if fail_row is None and ss['v_rot'] < healthy_v_rot * SPEED_FAIL_FRAC:
                fail_row = row_i

        elif decoy_type:
            # Input changes at change_at_s
            if decoy_type == 'load_step':
                if t >= change_at_s:
                    ss = _interp_grid(grids['load'], LOAD_MULT_STEP)
                else:
                    ss = healthy.copy()
            elif decoy_type == 'load_ramp':
                if t >= change_at_s:
                    ramp_end = change_at_s + 2 * 3600
                    frac = min(1.0, (t - change_at_s) / (ramp_end - change_at_s))
                    cq_val = 1.0 + frac * (LOAD_MULT_STEP - 1.0)
                    ss = _interp_grid(grids['load'], cq_val)
                else:
                    ss = healthy.copy()
            elif decoy_type == 'voltage_sag_step':
                if t >= change_at_s:
                    ss = _interp_grid(grids['voltage'], V_NOM * V_SAG_FRAC)
                else:
                    ss = healthy.copy()
            elif decoy_type == 'voltage_sag_ramp':
                if t >= change_at_s:
                    ramp_end = change_at_s + 2 * 3600
                    frac = min(1.0, (t - change_at_s) / (ramp_end - change_at_s))
                    v_val = V_NOM - frac * V_NOM * (1 - V_SAG_FRAC)
                    ss = _interp_grid(grids['voltage'], v_val)
                else:
                    ss = healthy.copy()
            elif decoy_type == 'duty_step':
                if t >= change_at_s:
                    ss = _interp_grid(grids['duty'], DUTY_DROP)
                else:
                    ss = healthy.copy()
            elif decoy_type == 'duty_ramp':
                if t >= change_at_s:
                    ramp_end = change_at_s + 2 * 3600
                    frac = min(1.0, (t - change_at_s) / (ramp_end - change_at_s))
                    d_val = DUTY_NOM - frac * (DUTY_NOM - DUTY_DROP)
                    ss = _interp_grid(grids['duty'], d_val)
                else:
                    ss = healthy.copy()
            else:
                ss = healthy.copy()

            signals[row_i, 5] = R_NOM
            signals[row_i, 6] = B_NOM
            signals[row_i, 7] = K_NOM
        else:
            # Healthy
            ss = healthy.copy()
            signals[row_i, 5] = R_NOM
            signals[row_i, 6] = B_NOM
            signals[row_i, 7] = K_NOM

        signals[row_i, 0] = ss['v_rot']
        signals[row_i, 1] = ss['i_rms']
        signals[row_i, 2] = ss['torque_load']
        signals[row_i, 3] = ss['mech_power']
        signals[row_i, 4] = ss['elec_power']

    # Add measurement noise
    for i, col in enumerate(_SIGNAL_COLS):
        std = _TILE_NOISE_STD.get(col, 0)
        if std > 0:
            signals[:, i] += rng.normal(0, std, size=n_rows)

    # Add slow random-walk drift on non-parameter columns
    for i, col in enumerate(_SIGNAL_COLS):
        if col not in ('resistance_ohm', 'friction_coeff', 'backemf_constant'):
            drift = np.cumsum(rng.normal(0, 0.0005, size=n_rows))
            signals[:, i] += drift

    # Build DataFrame
    df = pd.DataFrame(signals, columns=_SIGNAL_COLS)
    df['time_s'] = time_s
    df['time_h'] = time_h

    # Labels
    df['label'] = 'NORMAL'
    df['time_to_failure_s'] = -1.0

    if faulty and fail_row is not None:
        pre_steps = PRE_FAIL_S // save_freq_s
        pre_start = max(0, fail_row - pre_steps)
        df.loc[pre_start:fail_row - 1, 'label'] = 'PRE_FAILURE'
        df.loc[fail_row:, 'label'] = 'FAILURE'
        for i in range(pre_start, len(df)):
            df.loc[i, 'time_to_failure_s'] = max(0, (fail_row - i) * save_freq_s)
    elif faulty and fail_row is None:
        # Degradation didn't reach threshold — mark last row as failure
        fi = n_rows - 1
        pre_steps = PRE_FAIL_S // save_freq_s
        pi = max(0, fi - pre_steps)
        df.loc[pi:fi - 1, 'label'] = 'PRE_FAILURE'
        df.loc[fi:, 'label'] = 'FAILURE'
        for i in range(pi, n_rows):
            df.loc[i, 'time_to_failure_s'] = max(0, (fi - i) * save_freq_s)

    return df


# ── Scenario definitions ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name:              str
    group:             str       # "winding" | "bearing" | "demag"
    kind:              str       # "failure" | "decoy" | "normal"
    degradation_type:  str | None
    decoy_type:        str | None
    faulty:            bool
    description:       str


def build_scenarios() -> list[Scenario]:
    return [
        # ── Normal ────────────────────────────────────────────────────
        Scenario(
            name='normal',
            group='all',
            kind='normal',
            degradation_type=None,
            decoy_type=None,
            faulty=False,
            description='Healthy motor — all signals stable',
        ),

        # ── Winding degradation (R↑) ─────────────────────────────────
        # R increases → speed drops + current rises
        # Decoys: load increase also drops speed + raises current
        Scenario(
            name='motor_winding_degradation',
            group='winding',
            kind='failure',
            degradation_type='winding',
            decoy_type=None,
            faulty=True,
            description='Winding degradation — R rises, speed drops, current rises',
        ),
        Scenario(
            name='decoy_load_step',
            group='winding',
            kind='decoy',
            degradation_type=None,
            decoy_type='load_step',
            faulty=False,
            description='Load step — speed drops from load increase (not winding)',
        ),
        Scenario(
            name='decoy_load_ramp',
            group='winding',
            kind='decoy',
            degradation_type=None,
            decoy_type='load_ramp',
            faulty=False,
            description='Load ramp — speed drops gradually from load (not winding)',
        ),

        # ── Bearing wear (B↑) ────────────────────────────────────────
        # B increases → friction rises → speed drops
        # Decoys: voltage sag also drops speed but B stays constant
        Scenario(
            name='motor_bearing_wear',
            group='bearing',
            kind='failure',
            degradation_type='bearing',
            decoy_type=None,
            faulty=True,
            description='Bearing wear — friction rises, speed drops under same load',
        ),
        Scenario(
            name='decoy_voltage_sag_step',
            group='bearing',
            kind='decoy',
            degradation_type=None,
            decoy_type='voltage_sag_step',
            faulty=False,
            description='Voltage sag step — speed drops from lower voltage (not wear)',
        ),
        Scenario(
            name='decoy_voltage_sag_ramp',
            group='bearing',
            kind='decoy',
            degradation_type=None,
            decoy_type='voltage_sag_ramp',
            faulty=False,
            description='Voltage sag ramp — speed drops gradually (not wear)',
        ),

        # ── Demagnetization (K↓) ─────────────────────────────────────
        # K decreases → back-emf drops → current/speed change
        # Decoys: duty cycle change produces similar speed reduction
        Scenario(
            name='motor_demagnetization',
            group='demag',
            kind='failure',
            degradation_type='demag',
            decoy_type=None,
            faulty=True,
            description='Demagnetization — K drops, speed/current characteristics change',
        ),
        Scenario(
            name='decoy_duty_step',
            group='demag',
            kind='decoy',
            degradation_type=None,
            decoy_type='duty_step',
            faulty=False,
            description='Duty cycle step — speed drops from reduced duty (not demag)',
        ),
        Scenario(
            name='decoy_duty_ramp',
            group='demag',
            kind='decoy',
            degradation_type=None,
            decoy_type='duty_ramp',
            faulty=False,
            description='Duty cycle ramp — speed drops gradually (not demag)',
        ),
    ]


# ── Run all scenarios ─────────────────────────────────────────────────────────

def run_all(scenarios: list[Scenario]) -> dict[str, pd.DataFrame]:
    """Run all motor scenarios and save CSVs."""
    # Build calibration grids first (runs progpy sims)
    healthy, grids = _build_calibration_grid()

    results = {}
    scenario_meta = []
    rng = np.random.default_rng(42)

    for sc_id, sc in enumerate(scenarios):
        print(f"  Generating: {sc.name:<40}", end="", flush=True)

        df = _generate_scenario_series(
            healthy=healthy,
            grids=grids,
            degradation_type=sc.degradation_type,
            decoy_type=sc.decoy_type,
            faulty=sc.faulty,
            rng=np.random.default_rng(sc_id + 100),
        )
        df['scenario_id'] = sc_id

        df.to_csv(f"motor_sample_data/{sc.name}.csv", index=False)

        scenario_meta.append({
            'scenario_id':  sc_id,
            'scenario':     sc.name,
            'group':        sc.group,
            'kind':         sc.kind,
            'faulty':       sc.faulty,
            'description':  sc.description,
        })
        results[sc.name] = df
        vc = df['label'].value_counts().to_dict()
        print(f" {len(df)} rows  [{vc}]")

    meta_df = pd.DataFrame(scenario_meta)
    meta_df.to_csv("motor_sample_data/scenarios.csv", index=False)
    print(f"\n  Scenario lookup table: motor_sample_data/scenarios.csv ({len(meta_df)} rows)")

    return results


# ── Long time-series generator ────────────────────────────────────────────────

_FAILURE_TYPES = ['winding', 'bearing', 'demag']

_DECOY_TYPES = ['load_step', 'load_ramp', 'voltage_sag_step',
                'voltage_sag_ramp', 'duty_step', 'duty_ramp']


def generate_long_series(
    name: str,
    duration_days: float = 365.0,
    save_freq_s: int = SAVE_FREQ,
    failure_type: str | None = None,
    failure_start_day: float | None = None,
    failure_severity: float = 1.0,
    decoy_types: list[str] | None = None,
    decoy_freq_per_day: float = 0.0,
    decoy_duration_h: float = 4.0,
    maintenance_freq_per_month: float = 0.0,
    maintenance_duration_h: float = 4.0,
    ambient_var_K: float = 0.0,
    duty_cycle_var: float = 0.0,
    seed: int = 42,
    noise_scale: float = 1.0,
    v_offset: float = 0.0,
    output_path: str | None = None,
) -> pd.DataFrame | int:
    """
    Generate a long motor time series (days to years).

    If output_path is given, writes parquet directly and returns the row count
    (avoids keeping the full DataFrame in memory — use this in fleet mode).

    Uses the same tiling strategy as the pump generator: build a healthy
    template from a short progpy run, tile it for the full duration, then
    splice in degradation episodes and decoy events.
    """
    rng = np.random.default_rng(seed)
    total_s = duration_days * 86400
    total_rows = int(total_s / save_freq_s) + 1

    print(f"  Generating {name}: {duration_days:.0f} days, {total_rows} rows", flush=True)

    # Get healthy baseline
    healthy = _run_steady_state()

    # Build healthy tiled signal
    signals = np.zeros((total_rows, len(_SIGNAL_COLS)), dtype=np.float64)
    for i, col in enumerate(_SIGNAL_COLS):
        if col == 'rotational_velocity_rads':
            signals[:, i] = healthy['v_rot']
        elif col == 'current_rms_A':
            signals[:, i] = healthy['i_rms']
        elif col == 'torque_load_Nm':
            signals[:, i] = healthy['torque_load']
        elif col == 'mechanical_power_W':
            signals[:, i] = healthy['mech_power']
        elif col == 'electrical_power_W':
            signals[:, i] = healthy['elec_power']
        elif col == 'resistance_ohm':
            signals[:, i] = R_NOM
        elif col == 'friction_coeff':
            signals[:, i] = B_NOM
        elif col == 'backemf_constant':
            signals[:, i] = K_NOM

    # Add measurement noise
    for i, col in enumerate(_SIGNAL_COLS):
        std = _TILE_NOISE_STD.get(col, 0)
        if std > 0:
            signals[:, i] += rng.normal(0, std * noise_scale, size=total_rows)

    # Add slow random-walk drift
    for i, col in enumerate(_SIGNAL_COLS):
        if col not in ('resistance_ohm', 'friction_coeff', 'backemf_constant'):
            drift = np.cumsum(rng.normal(0, 0.0003, size=total_rows))
            signals[:, i] += drift

    # Apply voltage offset
    if v_offset != 0.0:
        v_shifted = _run_steady_state(voltage=V_NOM + v_offset)
        for i, col in enumerate(_SIGNAL_COLS):
            if col == 'rotational_velocity_rads':
                signals[:, i] += (v_shifted['v_rot'] - healthy['v_rot'])
            elif col == 'current_rms_A':
                signals[:, i] += (v_shifted['i_rms'] - healthy['i_rms'])

    # Ambient variation (adds daily/seasonal cycle to electrical power)
    if ambient_var_K > 0:
        time_h = np.arange(total_rows) * save_freq_s / 3600.0
        daily = ambient_var_K * 0.01 * np.sin(2 * np.pi * (time_h - 14) / 24)
        seasonal = ambient_var_K * 0.005 * np.sin(2 * np.pi * time_h / (365 * 24))
        # Ambient temp affects winding resistance slightly
        r_idx = _SIGNAL_COLS.index('resistance_ohm')
        signals[:, r_idx] += daily + seasonal
        # And thus speed/current
        v_rot_idx = _SIGNAL_COLS.index('rotational_velocity_rads')
        signals[:, v_rot_idx] -= (daily + seasonal) * 10  # approximate coupling

    # Duty cycle variation (day/night load pattern)
    if duty_cycle_var > 0:
        time_h = np.arange(total_rows) * save_freq_s / 3600.0
        hour_of_day = time_h % 24
        day_of_week = (time_h / 24).astype(int) % 7

        day_night = np.where(
            (hour_of_day >= 6) & (hour_of_day < 18), 1.0,
            1.0 - 0.4 * duty_cycle_var
        )
        weekend = np.where(day_of_week >= 5, 1.0 - 0.3 * duty_cycle_var, 1.0)
        duty_scale = day_night * weekend

        for i, col in enumerate(_SIGNAL_COLS):
            if col in ('rotational_velocity_rads', 'current_rms_A',
                        'torque_load_Nm', 'mechanical_power_W', 'electrical_power_W'):
                mean_val = signals[:, i].mean()
                signals[:, i] = mean_val + (signals[:, i] - mean_val) * duty_scale

    time_s = np.arange(total_rows) * save_freq_s
    time_h = time_s / 3600.0
    labels = np.full(total_rows, 'NORMAL', dtype=object)
    event_types = np.full(total_rows, 'normal', dtype=object)
    ttf = np.full(total_rows, -1.0)

    used_windows = []

    # ── Maintenance shutdowns ──
    if maintenance_freq_per_month > 0:
        n_maint = rng.poisson(maintenance_freq_per_month * duration_days / 30)
        maint_times = np.sort(rng.uniform(0, total_s, size=n_maint))
        maint_count = 0

        for t_s in maint_times:
            dur_h = rng.uniform(maintenance_duration_h * 0.5,
                                 maintenance_duration_h * 1.5)
            dur_rows = int(dur_h * 3600 / save_freq_s)
            ramp_rows = min(30, dur_rows // 4)
            start_row = int(t_s / save_freq_s)
            end_row = start_row + dur_rows
            if end_row >= total_rows:
                continue

            overlap = any(not (end_row <= ws or start_row >= we)
                          for ws, we in used_windows)
            if overlap:
                continue

            pre_vals = signals[start_row].copy()

            # Ramp down
            for r in range(ramp_rows):
                alpha = 1.0 - 0.5 * (1 - np.cos(np.pi * r / ramp_rows))
                for i, col in enumerate(_SIGNAL_COLS):
                    if col in ('rotational_velocity_rads', 'current_rms_A',
                                'torque_load_Nm', 'mechanical_power_W', 'electrical_power_W'):
                        signals[start_row + r, i] = pre_vals[i] * alpha

            # Flat shutdown
            flat_start = start_row + ramp_rows
            flat_end = end_row - ramp_rows
            for i, col in enumerate(_SIGNAL_COLS):
                n_flat = max(0, flat_end - flat_start)
                if col in ('rotational_velocity_rads', 'current_rms_A',
                            'torque_load_Nm', 'mechanical_power_W', 'electrical_power_W'):
                    signals[flat_start:flat_end, i] = rng.normal(
                        0, _TILE_NOISE_STD.get(col, 0) * 0.3, size=n_flat)

            # Ramp up
            ramp_up_start = max(flat_start, flat_end)
            post_vals = signals[end_row].copy() if end_row < total_rows else pre_vals
            for r in range(min(ramp_rows, total_rows - ramp_up_start)):
                alpha = 0.5 * (1 - np.cos(np.pi * r / ramp_rows))
                row_idx = ramp_up_start + r
                for i, col in enumerate(_SIGNAL_COLS):
                    if col in ('rotational_velocity_rads', 'current_rms_A',
                                'torque_load_Nm', 'mechanical_power_W', 'electrical_power_W'):
                        signals[row_idx, i] = post_vals[i] * alpha

            event_types[start_row:end_row] = 'maintenance'
            used_windows.append((start_row, end_row))
            maint_count += 1

        if maint_count > 0:
            print(f"    Spliced {maint_count} maintenance windows", flush=True)

    # ── Decoy events ──
    if decoy_types and decoy_freq_per_day > 0:
        _, grids = _build_calibration_grid()
        n_decoys = rng.poisson(decoy_freq_per_day * duration_days)
        decoy_times = np.sort(rng.uniform(0, total_s, size=n_decoys))

        for t_s in decoy_times:
            dur_h = rng.uniform(1.0, decoy_duration_h)
            dur_rows = int(dur_h * 3600 / save_freq_s)
            start_row = int(t_s / save_freq_s)
            end_row = start_row + dur_rows
            if end_row >= total_rows:
                continue

            overlap = any(not (end_row <= ws or start_row >= we)
                          for ws, we in used_windows)
            if overlap:
                continue

            dt_name = rng.choice(decoy_types)
            amplitude = rng.uniform(0.4, 1.0)

            # Map decoy type to grid and target value
            if dt_name in ('load_step', 'load_ramp'):
                target_val = 1.0 + amplitude * (LOAD_MULT_STEP - 1.0)
                ss_target = _interp_grid(grids['load'], target_val)
            elif dt_name in ('voltage_sag_step', 'voltage_sag_ramp'):
                target_val = V_NOM - amplitude * V_NOM * (1 - V_SAG_FRAC)
                ss_target = _interp_grid(grids['voltage'], target_val)
            elif dt_name in ('duty_step', 'duty_ramp'):
                target_val = DUTY_NOM - amplitude * (DUTY_NOM - DUTY_DROP)
                ss_target = _interp_grid(grids['duty'], target_val)
            else:
                continue

            # Splice: blend from current signal to decoy steady state
            is_ramp = 'ramp' in dt_name
            blend_rows = min(30, dur_rows // 4) if is_ramp else 5

            for i, col in enumerate(_SIGNAL_COLS):
                if col in ('resistance_ohm', 'friction_coeff', 'backemf_constant'):
                    continue  # parameters don't change for decoys

                sig_key = {'rotational_velocity_rads': 'v_rot',
                           'current_rms_A': 'i_rms',
                           'torque_load_Nm': 'torque_load',
                           'mechanical_power_W': 'mech_power',
                           'electrical_power_W': 'elec_power'}.get(col)
                if sig_key is None:
                    continue

                target = ss_target[sig_key]
                pre_val = signals[start_row, i]
                noise_std = _TILE_NOISE_STD.get(col, 0)

                for r in range(min(dur_rows, total_rows - start_row)):
                    row_idx = start_row + r
                    if is_ramp:
                        frac = min(1.0, r / max(1, dur_rows * 0.6))
                    else:
                        frac = min(1.0, r / max(1, blend_rows))

                    alpha = 0.5 * (1 - np.cos(np.pi * frac))
                    signals[row_idx, i] = (
                        pre_val * (1 - alpha) + target * alpha
                        + rng.normal(0, noise_std)
                    )

            n_actual = min(dur_rows, total_rows - start_row)
            event_types[start_row:start_row + n_actual] = f'decoy_{dt_name}'
            used_windows.append((start_row, start_row + n_actual))

        print(f"    Spliced {len(used_windows)} decoy events", flush=True)

    # ── Failure episode ──
    fail_end_row = None
    if failure_type and failure_type in _FAILURE_TYPES:
        _, grids = _build_calibration_grid()

        if failure_start_day is None:
            failure_start_day = duration_days * rng.uniform(0.7, 0.9)

        sev = failure_severity if failure_severity > 0 else rng.uniform(0.5, 2.0)
        fail_start_s = failure_start_day * 86400
        fail_start_row = int(fail_start_s / save_freq_s)

        # Determine parameter range and grid
        if failure_type == 'winding':
            p_nom, p_fail, grid = R_NOM, R_FAIL, grids['winding']
            p_col_idx = 5
        elif failure_type == 'bearing':
            p_nom, p_fail, grid = B_NOM, B_FAIL, grids['bearing']
            p_col_idx = 6
        elif failure_type == 'demag':
            p_nom, p_fail, grid = K_NOM, K_FAIL, grids['demag']
            p_col_idx = 7
        else:
            p_nom, p_fail, grid, p_col_idx = R_NOM, R_FAIL, grids['winding'], 5

        # Failure progresses over ~6h (scaled by severity)
        fail_duration_s = HORIZON_S * sev
        fail_duration_rows = int(fail_duration_s / save_freq_s)

        print(f"    Simulating {failure_type} failure (day {failure_start_day:.1f}, "
              f"severity {sev:.2f}x)...", flush=True)

        healthy_v_rot = healthy['v_rot']

        for r in range(fail_duration_rows):
            row_idx = fail_start_row + r
            if row_idx >= total_rows:
                break

            t_deg = r * save_freq_s
            p_val = _degrade_param(t_deg, p_nom, p_fail, fail_duration_s)
            ss = _interp_grid(grid, p_val)

            # Blend entry
            if r < 30:
                alpha = 0.5 * (1 - np.cos(np.pi * r / 30))
            else:
                alpha = 1.0

            sig_map = {0: 'v_rot', 1: 'i_rms', 2: 'torque_load',
                       3: 'mech_power', 4: 'elec_power'}
            for i, sig_key in sig_map.items():
                pre = signals[row_idx, i]
                signals[row_idx, i] = pre * (1 - alpha) + ss[sig_key] * alpha
                signals[row_idx, i] += rng.normal(0, _TILE_NOISE_STD.get(_SIGNAL_COLS[i], 0))

            signals[row_idx, p_col_idx] = p_val

            # Check failure
            if fail_end_row is None and ss['v_rot'] < healthy_v_rot * SPEED_FAIL_FRAC:
                fail_end_row = row_idx

        if fail_end_row is None:
            fail_end_row = min(fail_start_row + fail_duration_rows, total_rows - 1)

        pre_fail_rows = PRE_FAIL_S // save_freq_s
        pre_start = max(fail_start_row, fail_end_row - pre_fail_rows - 1)
        labels[pre_start:fail_end_row] = 'PRE_FAILURE'
        labels[fail_end_row] = 'FAILURE'
        event_types[fail_start_row:fail_end_row + 1] = f'failure_{failure_type}'

        for i in range(pre_start, min(fail_end_row + 1, total_rows)):
            ttf[i] = max(0, (fail_end_row - i) * save_freq_s)

        print(f"    Failure at row {fail_end_row} "
              f"(day {fail_end_row * save_freq_s / 86400:.1f})", flush=True)

        # Post-failure downtime (6h)
        DOWNTIME_S = 6 * 3600
        downtime_rows = DOWNTIME_S // save_freq_s

        if fail_end_row + 1 < total_rows:
            down_end = min(fail_end_row + 1 + downtime_rows, total_rows)
            dead_rows = down_end - fail_end_row - 1

            for i, col in enumerate(_SIGNAL_COLS):
                if col in ('rotational_velocity_rads', 'current_rms_A',
                            'torque_load_Nm', 'mechanical_power_W', 'electrical_power_W'):
                    signals[fail_end_row + 1:down_end, i] = rng.normal(
                        0, _TILE_NOISE_STD.get(col, 0) * 0.3, size=dead_rows)

            labels[fail_end_row + 1:down_end] = 'FAILURE'
            event_types[fail_end_row + 1:down_end] = 'post_failure'

            # Resume normal after downtime
            if down_end < total_rows:
                startup_rows = min(240, total_rows - down_end)  # 4h ramp-up
                for i, col in enumerate(_SIGNAL_COLS):
                    if col in ('resistance_ohm', 'friction_coeff', 'backemf_constant'):
                        # Reset to nominal (repaired)
                        signals[down_end:, i] = signals[0, i]
                        continue
                    normal_val = signals[0, i]
                    for r in range(startup_rows):
                        alpha = 0.5 * (1 - np.cos(np.pi * r / startup_rows))
                        signals[down_end + r, i] = alpha * (
                            normal_val + rng.normal(0, _TILE_NOISE_STD.get(col, 0)))

                labels[down_end:] = 'NORMAL'
                event_types[down_end:down_end + startup_rows] = 'startup'

    # ── Assemble DataFrame ──
    df = pd.DataFrame(signals, columns=_SIGNAL_COLS)
    del signals  # free the numpy array — DataFrame owns the data now
    df['time_s'] = time_s
    df['time_h'] = time_h
    df['label'] = labels
    df['time_to_failure_s'] = ttf
    df['event_type'] = event_types

    # Write to parquet immediately if path given (fleet mode — avoids returning
    # the full DataFrame to the caller and keeping two copies in memory).
    if output_path is not None:
        df.to_parquet(output_path, index=False)
        n_rows = len(df)
        del df
        return n_rows   # caller gets row count, not the DataFrame

    return df


def generate_dataset(
    duration_days: float = 365.0,
    save_freq_s: int = SAVE_FREQ,
    decoy_freq_per_day: float = 2.0,
    decoy_types: list[str] | None = None,
    failure_severity: float = 1.0,
    maintenance_freq_per_month: float = 1.0,
    ambient_var_K: float = 0.0,
    duty_cycle_var: float = 0.0,
    seed: int = 42,
    output_dir: str = 'motor_sample_data',
) -> dict[str, pd.DataFrame]:
    """Generate a full dataset: one healthy + one per failure type."""
    if decoy_types is None:
        decoy_types = _DECOY_TYPES

    Path(output_dir).mkdir(exist_ok=True)

    series_specs = [
        ('normal_long',            None),
        ('winding_failure_long',   'winding'),
        ('bearing_failure_long',   'bearing'),
        ('demag_failure_long',     'demag'),
    ]

    results = {}
    meta_rows = []

    for i, (name, fail_type) in enumerate(series_specs):
        df = generate_long_series(
            name=name,
            duration_days=duration_days,
            save_freq_s=save_freq_s,
            failure_type=fail_type,
            failure_severity=failure_severity,
            decoy_types=decoy_types,
            decoy_freq_per_day=decoy_freq_per_day,
            maintenance_freq_per_month=maintenance_freq_per_month,
            ambient_var_K=ambient_var_K,
            duty_cycle_var=duty_cycle_var,
            seed=seed + i,
        )
        df['scenario_id'] = i

        out_path = f"{output_dir}/{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"    → {out_path} ({len(df)} rows)")

        meta_rows.append({
            'scenario_id': i,
            'scenario': name,
            'group': fail_type or 'all',
            'kind': 'failure' if fail_type else 'normal',
            'faulty': fail_type is not None,
            'duration_days': duration_days,
            'decoy_freq_per_day': decoy_freq_per_day,
            'description': f"{'Healthy' if not fail_type else fail_type.title() + ' failure'}"
                           f" — {duration_days:.0f} days",
        })
        results[name] = df

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(f"{output_dir}/scenarios_long.csv", index=False)
    print(f"\n  Lookup table: {output_dir}/scenarios_long.csv ({len(meta_df)} rows)")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

COLOURS = {
    "failure": "#D85A30",
    "decoy":   "#1D9E75",
    "normal":  "#534AB7",
}

LABEL_ALPHA = {"NORMAL": 0.08, "PRE_FAILURE": 0.18, "FAILURE": 0.25}
LABEL_COLOUR = {"NORMAL": "#1D9E75", "PRE_FAILURE": "#BA7517", "FAILURE": "#D85A30"}


def _finish_ax(ax, ylabel, title, show_legend=True):
    ax.set_xlabel("Time (hours)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if show_legend:
        ax.legend(fontsize=7, framealpha=0.7)


def plot_normal(results: dict):
    """Plot 0 — healthy motor baseline."""
    df = results["normal"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("Healthy motor — baseline signals (all NORMAL)",
                 fontsize=12, fontweight="bold")

    pairs = [
        ("rotational_velocity_rads", "Rotational velocity (rad/s)"),
        ("current_rms_A",           "RMS current (A)"),
        ("torque_load_Nm",          "Torque load (Nm)"),
        ("mechanical_power_W",      "Mechanical power (W)"),
        ("electrical_power_W",      "Electrical power (W)"),
    ]

    for idx, (col, label) in enumerate(pairs):
        ax = axes[idx // 3][idx % 3]
        if col in df.columns:
            ax.plot(df["time_h"], df[col], color=COLOURS["normal"], linewidth=1.5)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xlabel("Time (hours)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    ax = axes[1][2]
    counts = df["label"].value_counts()
    ax.bar(counts.index, counts.values, color=["#1D9E75"])
    ax.set_title("Label distribution", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig("motor_plots/0_normal_baseline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → motor_plots/0_normal_baseline.png")


def plot_winding_group(results: dict):
    """Plot 1 — Winding degradation vs load decoys."""
    all_scenarios = [
        ("normal",                       "Healthy",       COLOURS["normal"]),
        ("motor_winding_degradation",    "Winding degr.", COLOURS["failure"]),
        ("decoy_load_step",              "Load step",     COLOURS["decoy"]),
        ("decoy_load_ramp",              "Load ramp",     "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Winding degradation (R↑) vs load-change decoys\n"
        "Both show speed dropping — discriminator is current vs speed relationship",
        fontsize=11, fontweight="bold", y=1.02
    )

    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["rotational_velocity_rads"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Rotational velocity (rad/s)", "Speed over time")

    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["current_rms_A"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "RMS current (A)", "Current over time")

    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["rotational_velocity_rads"], df["current_rms_A"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Rotational velocity (rad/s)", fontsize=8)
    ax.set_ylabel("RMS current (A)", fontsize=8)
    ax.set_title("Speed vs Current (discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: current HIGHER than expected for speed (R↑ causes I²R loss)  |  "
             "Decoys: current tracks speed proportionally",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("motor_plots/1_winding_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → motor_plots/1_winding_group.png")


def plot_bearing_group(results: dict):
    """Plot 2 — Bearing wear vs voltage-sag decoys."""
    all_scenarios = [
        ("normal",                  "Healthy",           COLOURS["normal"]),
        ("motor_bearing_wear",      "Bearing wear",      COLOURS["failure"]),
        ("decoy_voltage_sag_step",  "Voltage sag step",  COLOURS["decoy"]),
        ("decoy_voltage_sag_ramp",  "Voltage sag ramp",  "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Bearing wear (B↑) vs voltage-sag decoys\n"
        "Both show speed dropping — discriminator is mechanical power vs speed",
        fontsize=11, fontweight="bold", y=1.02
    )

    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["rotational_velocity_rads"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Rotational velocity (rad/s)", "Speed over time")

    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["mechanical_power_W"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Mechanical power (W)", "Mechanical power over time")

    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["rotational_velocity_rads"], df["mechanical_power_W"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Rotational velocity (rad/s)", fontsize=8)
    ax.set_ylabel("Mechanical power (W)", fontsize=8)
    ax.set_title("Speed vs Mech Power (discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: power LOWER than expected for speed (friction absorbs energy)  |  "
             "Decoys: power tracks speed proportionally",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("motor_plots/2_bearing_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → motor_plots/2_bearing_group.png")


def plot_demag_group(results: dict):
    """Plot 3 — Demagnetization vs duty-change decoys."""
    all_scenarios = [
        ("normal",                "Healthy",         COLOURS["normal"]),
        ("motor_demagnetization", "Demagnetization", COLOURS["failure"]),
        ("decoy_duty_step",       "Duty step",       COLOURS["decoy"]),
        ("decoy_duty_ramp",       "Duty ramp",       "#2196F3"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Demagnetization (K↓) vs duty-cycle decoys\n"
        "Both show speed changing — discriminator is electrical power vs torque",
        fontsize=11, fontweight="bold", y=1.02
    )

    ax = axes[0]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["rotational_velocity_rads"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Rotational velocity (rad/s)", "Speed over time")

    ax = axes[1]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.plot(df["time_h"], df["electrical_power_W"],
                color=colour, linewidth=1.2, label=label, alpha=0.8)
    _finish_ax(ax, "Electrical power (W)", "Electrical power over time")

    ax = axes[2]
    for name, label, colour in all_scenarios:
        df = results[name]
        ax.scatter(df["torque_load_Nm"], df["electrical_power_W"],
                   color=colour, s=4, alpha=0.5, label=label)
    ax.set_xlabel("Torque load (Nm)", fontsize=8)
    ax.set_ylabel("Electrical power (W)", fontsize=8)
    ax.set_title("Torque vs Elec Power (discriminator)", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    fig.text(0.5, -0.02,
             "Failure: elec power shifts differently (K affects torque-current coupling)  |  "
             "Decoys: power scales with duty (consistent K)",
             ha="center", fontsize=8, color="#555")

    plt.tight_layout()
    fig.savefig("motor_plots/3_demag_group.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → motor_plots/3_demag_group.png")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(results: dict):
    print("\n" + "=" * 76)
    print(f"  {'Scenario':<42} {'Rows':>6}  {'NORMAL':>8}  {'PRE_FAIL':>9}  {'FAILURE':>7}")
    print("  " + "-" * 72)
    for name, df in results.items():
        vc = df["label"].value_counts()
        n   = vc.get("NORMAL",      0)
        pre = vc.get("PRE_FAILURE", 0)
        f   = vc.get("FAILURE",     0)
        print(f"  {name:<42} {len(df):>6}  {n:>8}  {pre:>9}  {f:>7}")
    print("=" * 76)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 64)
    print("  DC Motor (Powertrain) Scenario Sample Generator")
    print("  Horizon: 6 hours, save every 60s")
    print("  Degradation modes: winding (R↑), bearing (B↑), demag (K↓)")
    print("=" * 64 + "\n")

    print("Running simulations...")
    scenarios = build_scenarios()
    results = run_all(scenarios)

    print("\nGenerating plots...")
    plot_normal(results)
    plot_winding_group(results)
    plot_bearing_group(results)
    plot_demag_group(results)

    print_summary(results)

    print("\n  Files written:")
    print("    motor_sample_data/*.csv  — one CSV per scenario")
    print("    motor_plots/*.png        — four plot files")
    print("\n  What to look for in the plots:")
    print("    Plot 1 — Winding:  failure has current rising faster than speed drops")
    print("    Plot 2 — Bearing:  failure has power loss at same speed (friction)")
    print("    Plot 3 — Demag:    failure has different power-torque relationship")


if __name__ == "__main__":
    main()
