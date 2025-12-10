"""
Mass boost sweep utility
------------------------

This helper script demonstrates that the gravity calculator responds to
large-scale Richter-style multipliers and explicit disk mass boosts.
It keeps the observation set fixed and evaluates a grid of
(mass_boost, constant_multiplier) pairs, recording the resulting RMS
against the true Gaia v_phi targets.
"""

import json
import time

import numpy as np
import pandas as pd

from fixed_improved_calculation import (
    FixedGravityCalculator,
    multiplier_constant_scale,
    GPU_AVAILABLE,
)


def select_observations(gaia_df, n_obs=200, seed=42):
    """Return observation indices and v_phi values."""
    if 'v_phi' not in gaia_df.columns:
        raise ValueError("Gaia dataset is missing v_phi column.")
    vphi = gaia_df['v_phi'].values.astype(np.float32)
    valid_mask = np.isfinite(vphi) & (vphi != 0.0)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid v_phi measurements found.")
    if len(valid_idx) < n_obs:
        n_obs = len(valid_idx)
    rng = np.random.default_rng(seed)
    obs_indices = rng.choice(valid_idx, size=n_obs, replace=False)
    return obs_indices, vphi[obs_indices]


def evaluate_pair(calculator, obs_indices, v_observed, period_name, scale):
    """Compute RMS for a given constant multiplier scale."""
    params = [float(scale)]
    t0 = time.time()
    v_model, components = calculator.compute_total_velocity(
        obs_indices, period_name, multiplier_constant_scale, params
    )
    duration = time.time() - t0
    rms = float(np.sqrt(np.mean((v_model - v_observed) ** 2)))
    mean_star = float(np.mean(components['stars']))
    return {
        'scale': float(scale),
        'rms': rms,
        'mean_stellar_speed': mean_star,
        'runtime_sec': duration,
    }


def run_mass_boost_sweep():
    gaia = pd.read_parquet('gravitywavebaseline/gaia_with_periods.parquet')
    obs_indices, v_observed = select_observations(gaia)
    print(f"[OK] Observation subset: {len(obs_indices)} stars")

    calculator = FixedGravityCalculator(
        gaia,
        use_gpu=GPU_AVAILABLE,
        use_bulge=True,
        use_gas=True,
        use_selection_weights=True,
        mass_boost=1.0,
    )

    mass_boost_values = [1.0, 5.0, 25.0, 100.0, 250.0]
    constant_scales = [1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

    results = []

    for mass_boost in mass_boost_values:
        calculator.mass_boost = float(mass_boost)
        print(f"\n=== Mass boost {mass_boost:.1f}x ===")
        for scale in constant_scales:
            entry = evaluate_pair(
                calculator,
                obs_indices,
                v_observed,
                'jeans',
                scale,
            )
            entry['mass_boost'] = float(mass_boost)
            results.append(entry)
            print(
                f"  scale={scale:>6.1f} -> RMS={entry['rms']:6.1f} km/s, "
                f"<v_stars>={entry['mean_stellar_speed']:6.1f} km/s "
                f"({entry['runtime_sec']:.2f}s)"
            )

    output = 'gravitywavebaseline/mass_boost_results.json'
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved sweep results to {output}")


if __name__ == "__main__":
    run_mass_boost_sweep()

