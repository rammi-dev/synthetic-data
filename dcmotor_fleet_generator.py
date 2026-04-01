"""
Fleet-scale DC motor data generator.
Creates thousands of unique motor device time series as parquet files.

Usage:
    python dcmotor_fleet_generator.py --num-devices 6000 --duration-days 365 --output-dir /data/motor_fleet
    python dcmotor_fleet_generator.py --num-devices 10 --duration-days 30 --output-dir test_motor_fleet
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from dcmotor_generator import (
    generate_long_series,
    _run_steady_state,
    _build_calibration_grid,
    _SIGNAL_COLS,
    _DECOY_TYPES,
    _FAILURE_TYPES,
    V_NOM,
    SAVE_FREQ,
)

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DeviceConfig:
    device_id: str                  # "motor_000000"
    seed: int
    duration_days: float
    failure_type: str | None        # None, 'winding', 'bearing', 'demag'
    failure_start_day: float | None
    failure_severity: float
    ambient_var_K: float
    duty_cycle_var: float
    decoy_freq_per_day: float
    decoy_types: list[str]
    noise_scale: float
    v_offset: float                 # voltage offset from nominal


@dataclass
class FleetConfig:
    num_devices: int = 6000
    duration_days: float = 365.0
    save_freq_s: int = 60
    max_workers: int = 0            # 0 = cpu_count - 1
    base_seed: int = 12345
    output_dir: str = 'motor_fleet_data'
    # Failure distribution
    healthy_fraction: float = 0.4
    winding_fraction: float = 0.2
    bearing_fraction: float = 0.2
    demag_fraction: float = 0.2


# ---------------------------------------------------------------------------
# Device config sampler
# ---------------------------------------------------------------------------

class DeviceConfigSampler:
    """Generates a list of DeviceConfig from a FleetConfig using seeded RNG."""

    ALL_DECOY_TYPES = [
        'load_step', 'load_ramp',
        'voltage_sag_step', 'voltage_sag_ramp',
        'duty_step', 'duty_ramp',
    ]

    def __init__(self, fleet_cfg: FleetConfig):
        self.cfg = fleet_cfg

    def sample_all(self) -> list[DeviceConfig]:
        rng = np.random.default_rng(self.cfg.base_seed)
        n = self.cfg.num_devices

        # Failure type assignment
        types = (
            [None] * int(n * self.cfg.healthy_fraction)
            + ['winding'] * int(n * self.cfg.winding_fraction)
            + ['bearing'] * int(n * self.cfg.bearing_fraction)
            + ['demag'] * int(n * self.cfg.demag_fraction)
        )
        # Fill remainder with None (healthy) to reach exactly n
        while len(types) < n:
            types.append(None)
        rng.shuffle(types)

        configs = []
        for i in range(n):
            fail_type = types[i]

            # failure_start_day: only for failing devices
            if fail_type is not None:
                fail_start = float(rng.uniform(
                    0.3 * self.cfg.duration_days,
                    0.95 * self.cfg.duration_days,
                ))
            else:
                fail_start = None

            # failure_severity: lognormal clipped
            sev = float(np.clip(rng.lognormal(0, 0.4), 0.3, 3.0))

            # ambient_var_K
            amb = float(rng.uniform(2.0, 12.0))

            # duty_cycle_var
            duty = float(rng.uniform(0.1, 0.8))

            # decoy_freq_per_day: exponential clipped
            decoy_freq = float(np.clip(rng.exponential(scale=2.0), 0, 8))

            # decoy_types: random subset of 1-6
            n_decoys = int(rng.integers(1, len(self.ALL_DECOY_TYPES) + 1))
            decoy_subset = list(rng.choice(
                self.ALL_DECOY_TYPES, size=n_decoys, replace=False,
            ))

            # noise_scale
            ns = float(rng.uniform(0.5, 2.0))

            # v_offset: normal(0, V_NOM * 0.02)
            vo = float(rng.normal(0, V_NOM * 0.02))

            configs.append(DeviceConfig(
                device_id=f"motor_{i:06d}",
                seed=self.cfg.base_seed + i,
                duration_days=self.cfg.duration_days,
                failure_type=fail_type,
                failure_start_day=fail_start,
                failure_severity=sev,
                ambient_var_K=amb,
                duty_cycle_var=duty,
                decoy_freq_per_day=decoy_freq,
                decoy_types=decoy_subset,
                noise_scale=ns,
                v_offset=vo,
            ))

        return configs


# ---------------------------------------------------------------------------
# Worker init and per-device generation
# ---------------------------------------------------------------------------

_WORKER_GRIDS: dict | None = None
_WORKER_HEALTHY: dict | None = None


def _worker_init():
    """Called once per worker process to build shared caches."""
    global _WORKER_GRIDS, _WORKER_HEALTHY

    # Build the calibration grid (populates _SS_CACHE in dcmotor_generator)
    _WORKER_HEALTHY, _WORKER_GRIDS = _build_calibration_grid()


def generate_single_device(cfg: DeviceConfig, output_dir: str) -> dict:
    """Generate data for a single device and write to parquet.

    Uses output_path to write parquet inside generate_long_series,
    avoiding a second copy of the full DataFrame in memory.
    """
    t0 = time.time()
    try:
        devices_dir = Path(output_dir) / 'devices'
        devices_dir.mkdir(parents=True, exist_ok=True)
        out_path = devices_dir / f"{cfg.device_id}.parquet"

        n_rows = generate_long_series(
            name=cfg.device_id,
            duration_days=cfg.duration_days,
            save_freq_s=SAVE_FREQ,
            failure_type=cfg.failure_type,
            failure_start_day=cfg.failure_start_day,
            failure_severity=cfg.failure_severity,
            decoy_types=cfg.decoy_types,
            decoy_freq_per_day=cfg.decoy_freq_per_day,
            ambient_var_K=cfg.ambient_var_K,
            duty_cycle_var=cfg.duty_cycle_var,
            seed=cfg.seed,
            noise_scale=cfg.noise_scale,
            v_offset=cfg.v_offset,
            output_path=str(out_path),
        )

        elapsed = time.time() - t0
        return {
            'device_id': cfg.device_id,
            'num_rows': n_rows,
            'file_size_bytes': out_path.stat().st_size,
            'generation_time_s': round(elapsed, 2),
            'status': 'completed',
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            'device_id': cfg.device_id,
            'num_rows': 0,
            'file_size_bytes': 0,
            'generation_time_s': round(elapsed, 2),
            'status': f'error: {e}',
        }


def _generate_device_wrapper(args):
    """Wrapper for ProcessPoolExecutor (takes a tuple)."""
    cfg, output_dir = args
    return generate_single_device(cfg, output_dir)


# ---------------------------------------------------------------------------
# Fleet orchestrator
# ---------------------------------------------------------------------------

def generate_fleet(fleet_cfg: FleetConfig):
    """Main orchestrator: sample configs, generate in parallel, validate."""
    from tqdm import tqdm

    output_dir = fleet_cfg.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / 'devices').mkdir(exist_ok=True)

    # 1. Sample all DeviceConfigs
    sampler = DeviceConfigSampler(fleet_cfg)
    all_configs = sampler.sample_all()

    # Save pipeline config
    config_path = Path(output_dir) / 'pipeline_config.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(fleet_cfg), f, indent=2)
    print(f"Pipeline config saved to {config_path}")

    # 2. Save device manifest with status='pending'
    manifest_path = Path(output_dir) / 'device_manifest.csv'

    # Check for resumability
    completed_ids = set()
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        completed_ids = set(
            existing.loc[existing['status'] == 'completed', 'device_id']
        )
        print(f"Resuming: {len(completed_ids)} devices already completed")

    # Filter out already-completed devices
    pending_configs = [c for c in all_configs if c.device_id not in completed_ids]
    print(f"Devices to generate: {len(pending_configs)} "
          f"(total: {len(all_configs)}, already done: {len(completed_ids)})")

    if not pending_configs:
        print("All devices already completed.")
        return

    # Write initial manifest if starting fresh
    if not manifest_path.exists():
        manifest_rows = []
        for cfg in all_configs:
            row = asdict(cfg)
            row['status'] = 'pending'
            row['num_rows'] = 0
            row['file_size_bytes'] = 0
            row['generation_time_s'] = 0.0
            # Convert list to string for CSV
            row['decoy_types'] = '|'.join(row['decoy_types'])
            manifest_rows.append(row)
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # 3. Determine workers
    max_workers = fleet_cfg.max_workers
    if max_workers <= 0:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {max_workers} worker processes")

    # 4. Submit to ProcessPoolExecutor
    results = []
    args_list = [(cfg, output_dir) for cfg in pending_configs]

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_worker_init,
    ) as executor:
        futures = {
            executor.submit(_generate_device_wrapper, args): args[0].device_id
            for args in args_list
        }

        with tqdm(total=len(futures), desc="Generating devices", unit="dev") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                pbar.set_postfix(
                    last=result['device_id'],
                    status=result['status'][:10],
                )

    # 5. Update manifest
    manifest_df = pd.read_csv(manifest_path)
    results_df = pd.DataFrame(results)
    for _, row in results_df.iterrows():
        mask = manifest_df['device_id'] == row['device_id']
        manifest_df.loc[mask, 'status'] = row['status']
        manifest_df.loc[mask, 'num_rows'] = row['num_rows']
        manifest_df.loc[mask, 'file_size_bytes'] = row['file_size_bytes']
        manifest_df.loc[mask, 'generation_time_s'] = row['generation_time_s']
    manifest_df.to_csv(manifest_path, index=False)

    # 6. Validation pass
    print("\nValidation pass...")
    devices_dir = Path(output_dir) / 'devices'
    n_ok, n_fail = 0, 0
    for cfg in all_configs:
        pq_path = devices_dir / f"{cfg.device_id}.parquet"
        if pq_path.exists() and pq_path.stat().st_size > 0:
            n_ok += 1
        else:
            n_fail += 1
    print(f"  Valid files: {n_ok}/{len(all_configs)}")
    if n_fail:
        print(f"  Missing/empty files: {n_fail}")

    # 7. Summary
    completed = [r for r in results if r['status'] == 'completed']
    total_bytes = sum(r['file_size_bytes'] for r in completed)
    total_rows = sum(r['num_rows'] for r in completed)
    avg_time = (sum(r['generation_time_s'] for r in completed) / len(completed)
                if completed else 0)

    print(f"\n{'='*60}")
    print(f"Motor fleet generation complete")
    print(f"  Devices generated this run: {len(completed)}")
    print(f"  Total rows:  {total_rows:,}")
    print(f"  Total size:  {total_bytes / 1e9:.2f} GB")
    print(f"  Avg time/device: {avg_time:.1f}s")
    print(f"  Output dir:  {output_dir}")
    print(f"  Manifest:    {manifest_path}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> FleetConfig:
    parser = argparse.ArgumentParser(
        description="Fleet-scale DC motor data generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--num-devices', type=int, default=6000)
    parser.add_argument('--duration-days', type=float, default=365.0)
    parser.add_argument('--save-freq-s', type=int, default=60)
    parser.add_argument('--max-workers', type=int, default=0,
                        help='0 = cpu_count - 1')
    parser.add_argument('--base-seed', type=int, default=12345)
    parser.add_argument('--output-dir', type=str, default='motor_fleet_data')
    parser.add_argument('--healthy-fraction', type=float, default=0.4)
    parser.add_argument('--winding-fraction', type=float, default=0.2)
    parser.add_argument('--bearing-fraction', type=float, default=0.2)
    parser.add_argument('--demag-fraction', type=float, default=0.2)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides CLI args)')

    args = parser.parse_args()

    cfg = FleetConfig(
        num_devices=args.num_devices,
        duration_days=args.duration_days,
        save_freq_s=args.save_freq_s,
        max_workers=args.max_workers,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        healthy_fraction=args.healthy_fraction,
        winding_fraction=args.winding_fraction,
        bearing_fraction=args.bearing_fraction,
        demag_fraction=args.demag_fraction,
    )

    # JSON config overrides
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    print(f"Fleet config: {asdict(cfg)}")
    generate_fleet(cfg)
