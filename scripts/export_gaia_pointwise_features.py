#!/usr/bin/env python3
"""Export Gaia star-by-star residuals + local phase-space structure features.

Goal
----
Bulges are pressure-supported and have different orbit families than disks.
If the missing 'coherence root cause' depends on *how stellar velocities line up
in 3D* (laminar rotation vs isotropic/chaotic motion), we need features beyond
just a rotation curve.

This script:
1) loads a Gaia 6D catalogue (e.g., the Eilers+ APOGEE×Gaia disk sample)
2) computes baseline baryonic circular speed V_bar(R) from the same MW model
   used in the regression suite (McMillan-like components)
3) optionally computes your Σ-Gravity prediction V_pred(R)
4) computes neighborhood-based local structure features using KNN:
   - local number density proxy n_star (from distance to k-th neighbor)
   - mean interstellar separation d_star ~ n^{-1/3}
   - local velocity dispersion tensor + isotropy/anisotropy proxies
   - local velocity-gradient tensor ∇v via least squares
     -> vorticity |∇×v|, expansion ∇·v, shear ||S||

These are the exact kinds of 'xyz alignment / spacing' quantities that could
explain why bulges behave differently.

Usage
-----
  python scripts/export_gaia_pointwise_features.py \
      data/gaia/eilers_apogee_6d_disk.csv \
      --out scripts/regression_results/gaia_pointwise_features.csv \
      --k 64 --compute-sigma-gravity

Notes
-----
- The Eilers disk sample is not a bulge sample; but this pipeline is reusable
  for any Gaia bulge selection you later ingest.
- Computing per-star ∇v is O(N*k) and is feasible for N~3e4.

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors


# -----------------------------
# Constants / defaults
# -----------------------------
H0_SI = 2.27e-18
c = 2.998e8
G_DAGGER = c * H0_SI / (4.0 * math.sqrt(math.pi))

# MW baryonic model parameters (match test_gaia() in run_regression_experimental.py)
MW_SCALE = 1.16
M_DISK = 4.6e10 * MW_SCALE**2
M_BULGE = 1.0e10 * MW_SCALE**2
M_GAS = 1.0e10 * MW_SCALE**2
G_KPC = 4.302e-6  # (km/s)^2 kpc / Msun


def compute_mw_vbar(R_kpc: np.ndarray) -> np.ndarray:
    """McMillan-like analytic baryonic circular-speed curve."""
    R = np.asarray(R_kpc)
    v2_disk = G_KPC * M_DISK * R**2 / (R**2 + 3.3**2) ** 1.5
    v2_bulge = G_KPC * M_BULGE * R / (R + 0.5) ** 2
    v2_gas = G_KPC * M_GAS * R**2 / (R**2 + 7.0**2) ** 1.5
    return np.sqrt(np.maximum(v2_disk + v2_bulge + v2_gas, 0.0))


def cylindrical_to_cartesian(R: np.ndarray, phi_rad: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = R * np.cos(phi_rad)
    y = R * np.sin(phi_rad)
    return np.column_stack([x, y, z])


def estimate_grad_v(coords: np.ndarray, vels: np.ndarray, nn_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate ∇v at each point via least squares from neighbors.

    coords: (N,3)
    vels:   (N,3)
    nn_idx: (N,k) integer indices of neighbors (excluding self)

    Returns:
        omega2: (N,) vorticity magnitude squared
        shear2: (N,) shear Frobenius norm squared
        theta:  (N,) expansion scalar (divergence)
    """
    N, _ = coords.shape
    k = nn_idx.shape[1]

    omega2 = np.zeros(N)
    shear2 = np.zeros(N)
    theta = np.zeros(N)

    I = np.eye(3)

    for i in range(N):
        idx = nn_idx[i]
        # Relative positions and velocities
        dx = coords[idx] - coords[i]   # (k,3)
        dv = vels[idx] - vels[i]       # (k,3)

        # If neighbors are degenerate (rare), skip.
        if not np.isfinite(dx).all() or not np.isfinite(dv).all():
            omega2[i] = np.nan
            shear2[i] = np.nan
            theta[i] = np.nan
            continue

        # Solve dv ≈ J · dx, where J = ∇v (3x3).
        # For each velocity component separately: dv_comp = dx @ grad_comp
        # grad_comp is 3-vector of partial derivatives.
        J = np.zeros((3, 3))
        # Use rcond=None for stable lstsq
        for comp in range(3):
            grad_comp, *_ = np.linalg.lstsq(dx, dv[:, comp], rcond=None)
            J[comp, :] = grad_comp

        # Decompose J into rotation (antisymmetric), shear (symmetric traceless), expansion.
        theta_i = float(np.trace(J))
        S = 0.5 * (J + J.T)
        A = 0.5 * (J - J.T)
        shear_tensor = S - (theta_i / 3.0) * I

        # Vorticity vector from antisymmetric part: ω = (∇×v)
        omega_vec = np.array([
            J[2, 1] - J[1, 2],
            J[0, 2] - J[2, 0],
            J[1, 0] - J[0, 1],
        ])

        omega2[i] = float(np.dot(omega_vec, omega_vec))
        shear2[i] = float(np.sum(shear_tensor**2))
        theta[i] = theta_i

    return omega2, shear2, theta


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Gaia residuals + local flow structure features")
    ap.add_argument("gaia_csv", type=str, help="Input Gaia 6D CSV")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    ap.add_argument("--k", type=int, default=64, help="# neighbors for local estimates")
    ap.add_argument("--compute-sigma-gravity", action="store_true", help="Also compute Σ-Gravity V_pred using run_regression_experimental.predict_velocity")
    ap.add_argument("--regression-script", type=str, default="scripts/run_regression_experimental.py", help="Path to regression script to import predict_velocity")
    ap.add_argument("--max-stars", type=int, default=0, help="Optional cap for quick tests (0 = all)")
    args = ap.parse_args()

    df = pd.read_csv(args.gaia_csv)

    # Required columns (we keep this flexible)
    if "R_gal" not in df.columns:
        raise SystemExit("Input CSV must contain column 'R_gal' (kpc).")
    if "v_R" not in df.columns or "v_phi" not in df.columns:
        raise SystemExit("Input CSV must contain columns 'v_R' and 'v_phi' (km/s).")

    # Conventions: in the repo loader, v_phi_obs = -v_phi (file has counter-rot positive)
    v_phi_obs = -df["v_phi"].to_numpy(dtype=float)
    v_R = df["v_R"].to_numpy(dtype=float)

    # z coordinate if available
    if "z_gal" in df.columns:
        z = df["z_gal"].to_numpy(dtype=float)
    elif "z" in df.columns:
        z = df["z"].to_numpy(dtype=float)
    else:
        z = np.zeros(len(df), dtype=float)

    # phi coordinate if available
    if "phi_gal" in df.columns:
        phi = df["phi_gal"].to_numpy(dtype=float)
    elif "phi" in df.columns:
        phi = df["phi"].to_numpy(dtype=float)
    else:
        # Without phi we can't do full 3D cartesian; set phi=0 so x=R, y=0
        phi = np.zeros(len(df), dtype=float)

    # Limit for quick experiments
    if args.max_stars and args.max_stars > 0:
        df = df.iloc[: args.max_stars].copy()
        v_phi_obs = v_phi_obs[: args.max_stars]
        v_R = v_R[: args.max_stars]
        z = z[: args.max_stars]
        phi = phi[: args.max_stars]

    R = df["R_gal"].to_numpy(dtype=float)

    # Build observed velocity vector (in a cylindrical basis proxy)
    # We keep (v_R, v_phi, v_z) even if v_z missing.
    if "v_z" in df.columns:
        v_z = df["v_z"].to_numpy(dtype=float)
    else:
        v_z = np.zeros(len(df), dtype=float)

    # Coordinates in kpc (cartesian)
    coords = cylindrical_to_cartesian(R, phi, z)
    vels = np.column_stack([v_R, v_phi_obs, v_z])

    # KNN in position space
    k = int(args.k)
    if k < 8:
        raise SystemExit("Use k>=8 for stable gradients")

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree")
    nn.fit(coords)
    dists, idx = nn.kneighbors(coords, return_distance=True)

    # Drop self neighbor (index 0)
    dists = dists[:, 1:]
    idx = idx[:, 1:]

    d_k = dists[:, -1]
    # number density proxy (kpc^-3): n = k / (4/3 π d_k^3)
    n_star = k / (4.0 / 3.0 * math.pi * np.clip(d_k, 1e-6, None) ** 3)
    d_star = n_star ** (-1.0 / 3.0)  # mean separation proxy

    # Local velocity dispersion (use neighbors)
    v_mean = np.zeros_like(vels)
    sigma_1d = np.zeros(len(df))
    sigma_R = np.zeros(len(df))
    sigma_phi = np.zeros(len(df))
    sigma_z = np.zeros(len(df))

    for i in range(len(df)):
        vv = vels[idx[i]]
        mu = vv.mean(axis=0)
        v_mean[i] = mu
        cov = np.cov(vv.T)
        # robust fallback if cov is scalar
        if np.ndim(cov) == 0:
            cov = np.eye(3) * float(cov)
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
        sigma_R[i], sigma_phi[i], sigma_z[i] = sig
        sigma_1d[i] = float(np.sqrt(np.mean(sig**2)))

    # Velocity-gradient tensor invariants
    omega2, shear2, theta = estimate_grad_v(coords, vels, idx)

    # Build baryonic curve + optional Σ-Gravity prediction
    V_bar = compute_mw_vbar(R)
    V_pred = np.full_like(V_bar, np.nan)

    if args.compute_sigma_gravity:
        # Import predict_velocity from regression script
        import importlib.util
        reg_path = Path(args.regression_script)
        if not reg_path.exists():
            raise SystemExit(f"Could not find regression script: {reg_path}")
        spec = importlib.util.spec_from_file_location("sigma_regression", reg_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)

        # MW params consistent with test_gaia
        R_d_mw = 2.6
        # NOTE: using same f_bulge/h_disk placeholders as test_gaia; update if you refine.
        V_pred = mod.predict_velocity(R, V_bar, R_d_mw, h_disk=0.3, f_bulge=0.1, sigma_profile_kms=None)

    # Residual in v_phi space (ignoring asymmetric drift here; for pattern mining this is OK)
    resid = v_phi_obs - V_pred if np.isfinite(V_pred).any() else np.nan * np.ones_like(v_phi_obs)

    out = pd.DataFrame({
        "R_kpc": R,
        "phi_rad": phi,
        "z_kpc": z,
        "v_R_kms": v_R,
        "v_phi_obs_kms": v_phi_obs,
        "v_z_kms": v_z,
        "V_bar_kms": V_bar,
        "V_pred_kms": V_pred,
        "resid_kms": resid,
        # neighborhood geometry
        "d_k_kpc": d_k,
        "n_star_kpc3": n_star,
        "d_star_kpc": d_star,
        # neighborhood kinematics
        "sigma_1d_kms": sigma_1d,
        "sigma_R_kms": sigma_R,
        "sigma_phi_kms": sigma_phi,
        "sigma_z_kms": sigma_z,
        # flow invariants
        "omega2": omega2,
        "shear2": shear2,
        "theta": theta,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path} (N={len(out)})")


if __name__ == "__main__":
    main()


