"""
Lambda Multiplier Inversion

Goal:
    For outer-disk stars with observed velocities (from Gaia),
    solve for the gravity multiplier per wavelength bin needed
    to reproduce the observed speeds.

Approach:
    - Sample observation stars in an outer radial band.
    - Sample source stars (disk mass contributors).
    - Compute base Newtonian contribution matrix A (obs x λ bins).
    - Solve least squares (non-negative) for multiplier per bin.
    - Repeat for various λ hypotheses.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls
import numpy.linalg as LA

G_KPC = 4.30091e-6  # (km/s)^2 * kpc / M_sun


def select_observations(gaia, r_min=11.0, r_max=15.0, n_obs=200, seed=42):
    rng = np.random.default_rng(seed)
    mask = (
        (gaia["R"] >= r_min)
        & (gaia["R"] <= r_max)
        & np.isfinite(gaia["v_phi"])
        & (gaia["v_phi"] > 0.0)
    )
    subset = gaia.loc[mask]
    if len(subset) < n_obs:
        raise ValueError(f"Not enough observation stars ({len(subset)}) in radial band.")
    obs = subset.sample(n=n_obs, random_state=seed).copy()
    obs.reset_index(drop=True, inplace=True)
    return obs


def sample_sources(gaia, n_sources=150_000, seed=123):
    rng = np.random.default_rng(seed)
    if len(gaia) <= n_sources:
        idx = np.arange(len(gaia))
    else:
        idx = rng.choice(len(gaia), size=n_sources, replace=False)
    sources = gaia.iloc[idx].copy()
    sources.reset_index(drop=True, inplace=True)
    return sources


def build_bins(lambda_values, n_bins=10):
    # quantile-based bins for balanced counts
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(lambda_values, quantiles)
    # ensure strictly increasing by nudging
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def compute_design_matrix(
    obs, sources, lambda_col, n_bins=10, chunk_size=10_000, mass_scale=1.0
):
    """
    Returns A (n_obs x n_bins) where each entry is the summed contribution
    of sources in that λ bin to the observation's v^2 equation.
    """
    x_obs = obs["x"].values.astype(np.float32)
    y_obs = obs["y"].values.astype(np.float32)
    z_obs = obs["z"].values.astype(np.float32)
    R_obs = obs["R"].values.astype(np.float32)

    x_src = sources["x"].values.astype(np.float32)
    y_src = sources["y"].values.astype(np.float32)
    z_src = sources["z"].values.astype(np.float32)
    M_src = sources["M_star"].values.astype(np.float32) * mass_scale
    lam = sources[lambda_col].values.astype(np.float32)

    edges = build_bins(lam, n_bins=n_bins)
    bin_ids = np.digitize(lam, edges[1:-1], right=False)

    A = np.zeros((len(obs), n_bins), dtype=np.float64)

    for start in range(0, len(sources), chunk_size):
        end = min(start + chunk_size, len(sources))
        xs = x_src[start:end]
        ys = y_src[start:end]
        zs = z_src[start:end]
        ms = M_src[start:end]
        bins_chunk = bin_ids[start:end]

        dx = x_obs[:, None] - xs[None, :]
        dy = y_obs[:, None] - ys[None, :]
        dz = z_obs[:, None] - zs[None, :]
        r = np.sqrt(dx**2 + dy**2 + dz**2 + 0.01**2)

        cos_theta = (dx) / r
        contrib = G_KPC * ms[None, :] / r**2 * cos_theta
        contrib *= R_obs[:, None]

        for b in range(n_bins):
            mask = bins_chunk == b
            if np.any(mask):
                A[:, b] += contrib[:, mask].sum(axis=1)

    return A, edges


def solve_multipliers(A, v_obs, method="nnls", lam=0.0):
    y = v_obs.astype(np.float64) ** 2
    if method == "nnls":
        g, _ = nnls(A, y)
    else:
        ATA = A.T @ A + lam * np.eye(A.shape[1])
        ATy = A.T @ y
        g = LA.solve(ATA, ATy)
        g = np.maximum(g, 0.0)
    v_fit = np.sqrt(np.maximum(A @ g, 0.0))
    residual = v_obs - v_fit
    rms = np.sqrt(np.mean(residual**2))
    return g, v_fit, residual, rms


def run_inversion(
    gaia_path="gravitywavebaseline/gaia_with_periods.parquet",
    output_path="gravitywavebaseline/lambda_inversion_results.json",
    lambda_cols=None,
    methods=None,
    disk_mass=6e10,
):
    if lambda_cols is None:
        lambda_cols = [
            "lambda_jeans",
            "lambda_orbital",
            "lambda_hybrid",
            "lambda_gw",
        ]
    if methods is None:
        methods = [
            ("nnls", 0.0),
            ("ridge1e-1", 1e-1),
            ("ridge1", 1.0),
            ("ridge10", 10.0),
        ]

    gaia = pd.read_parquet(gaia_path)
    obs = select_observations(gaia)
    sources = sample_sources(gaia)
    total_source_mass = sources["M_star"].sum()
    if total_source_mass <= 0:
        raise ValueError("Sampled sources have zero total mass.")
    mass_scale = disk_mass / total_source_mass
    print(
        f"[INFO] Source sample mass={total_source_mass:.2e} Msun; "
        f"scaling to disk_mass={disk_mass:.2e} Msun (scale={mass_scale:.2e})"
    )

    results = []
    for col in lambda_cols:
        if col not in gaia.columns:
            print(f"[WARN] Column {col} missing; skipping.")
            continue
        print(f"\n=== Inverting multipliers for {col} ===")
        A, edges = compute_design_matrix(
            obs,
            sources,
            col,
            n_bins=10,
            mass_scale=mass_scale,
        )
        for method_name, lam in methods:
            g, v_fit, residual, rms = solve_multipliers(
                A, obs["v_phi"].values, method="nnls" if lam == 0 else "ridge", lam=lam
            )
            result = {
                "lambda_column": col,
                "method": method_name,
                "lambda": lam,
                "bin_edges": edges.tolist(),
                "multipliers": g.tolist(),
                "rms": float(rms),
                "n_obs": int(len(obs)),
                "n_sources": int(len(sources)),
            }
            print(f"  [{method_name}] RMS: {rms:.2f} km/s; multipliers: {g}")
            results.append(result)

    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Saved results to {output_path}")


if __name__ == "__main__":
    run_inversion()

