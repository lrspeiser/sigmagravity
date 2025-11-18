#!/usr/bin/env python
"""
pure_lambda_gr_test.py

Test a *pure* lambda_gw-based modification of GR:

    g_eff(R) = g_GR(R) * f(lambda_gw)

with

    f(lambda) = 1 + A * (lambda0 / lambda)^alpha  (A>0, alpha>0)

Goal:
  - Calibrate (A, lambda0, alpha) on the MW outer disk
    using the Gaia GR baseline (no capacity, no shell budgets).
  - Then show, on toy disks, that short-λ dwarfs get a larger
    boost than the MW, especially once we gate amplitude by σ_v.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

G_KPC = 4.30091e-6  # (km/s)^2 * kpc / Msun
KPC_TO_M = 3.085677581491367e19
KM_TO_M = 1000.0


def multiplier_shortlambda_boost(lam, params, xp=np):
    """
    SHORT wavelength -> STRONG boost.

    f(lambda) = 1 + A * (lambda0 / lambda)^alpha

    - Dwarfs (lambda ~ 0.5 kpc): f >> 1 (strong enhancement)
    - MW outer disk (lambda ~ 10–100 kpc): f ~ 1–2 (mild enhancement)
    """
    A, lambda_0, alpha = params
    lam_safe = xp.maximum(lam, 1.0e-3 * lambda_0)
    return 1.0 + A * (lambda_0 / lam_safe) ** alpha


def load_gaia_baseline(path):
    df = pd.read_parquet(path)
    required = ["R", "v_phi", "v_phi_GR", "lambda_gw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def select_observations(df, r_min, r_max, n_obs, seed=42):
    mask = np.isfinite(df["v_phi"]) & np.isfinite(df["v_phi_GR"])
    mask &= (df["R"] >= r_min) & (df["R"] <= r_max)
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        raise ValueError(f"No Gaia stars in R=[{r_min},{r_max}] kpc with valid velocities.")

    n = min(n_obs, len(sub))
    rng = np.random.default_rng(seed)
    idx = rng.choice(sub.index.values, size=n, replace=False)
    return sub.loc[idx].reset_index(drop=True)


def calibrate_mw_pure_lambda(
    gaia_path="gravitywavebaseline/gaia_with_gr_baseline.parquet",
    r_min=12.0,
    r_max=16.0,
    n_obs=1000,
    max_mult=6.0,
    output_path="gravitywavebaseline/mw_pure_lambda_fit.json",
):
    print("\n" + "=" * 80)
    print("MW OUTER DISK: PURE λ-GW × GR FIT")
    print("=" * 80)

    df = load_gaia_baseline(gaia_path)
    obs = select_observations(df, r_min, r_max, n_obs)
    R = obs["R"].values
    v_obs = obs["v_phi"].values
    v_GR = obs["v_phi_GR"].values
    lam = obs["lambda_gw"].values

    print(f"\n[INFO] Using {len(obs)} stars in R=[{r_min:.1f},{r_max:.1f}] kpc")
    print(f"       gaia file: {gaia_path}")

    def objective(params):
        A, lambda_0, alpha = params
        if A <= 0 or lambda_0 <= 0 or alpha <= 0:
            return 1e30
        mult = multiplier_shortlambda_boost(lam, params, np)
        mult = np.clip(mult, 1.0, max_mult)
        v_model = v_GR * np.sqrt(mult)
        resid = v_model - v_obs
        return float(np.mean(resid**2))

    bounds = [
        (0.1, 10.0),
        (0.5, 30.0),
        (0.2, 3.0),
    ]

    result = differential_evolution(objective, bounds, maxiter=60, polish=True, disp=True)
    A_best, lambda0_best, alpha_best = result.x
    mult_best = multiplier_shortlambda_boost(lam, result.x, np)
    mult_best = np.clip(mult_best, 1.0, max_mult)

    v_model = v_GR * np.sqrt(mult_best)
    residuals = v_model - v_obs
    rms = float(np.sqrt(np.mean(residuals**2)))
    rms_gr = float(np.sqrt(np.mean((v_GR - v_obs) ** 2)))

    print("\n[RESULT] Best-fit λ-law on MW outer disk (pure multiplicative GR):")
    print(f"  A       = {A_best:.3f}")
    print(f"  lambda0 = {lambda0_best:.3f} kpc")
    print(f"  alpha   = {alpha_best:.3f}")
    print(f"  RMS (GR only)      = {rms_gr:.2f} km/s")
    print(f"  RMS (with λ boost) = {rms:.2f} km/s")
    print(f"  Improvement        = {rms_gr - rms:.2f} km/s")

    print("\n[STATS] MW outer disk multipliers:")
    print(f"  f(lambda)_min    = {mult_best.min():.2f}")
    print(f"  f(lambda)_median = {np.median(mult_best):.2f}")
    print(f"  f(lambda)_max    = {mult_best.max():.2f}")

    out = {
        "A": float(A_best),
        "lambda0": float(lambda0_best),
        "alpha": float(alpha_best),
        "rms_gr": rms_gr,
        "rms_lambda": rms,
        "r_min": r_min,
        "r_max": r_max,
        "n_obs": int(len(obs)),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print(f"\n[OK] Saved MW λ-law fit to {output_path}")

    return out, obs


def exponential_disk_mass_enclosed(R, M_disk, R_d):
    x = np.maximum(R / max(R_d, 1e-3), 0.0)
    return M_disk * (1.0 - np.exp(-x) * (1.0 + x))


def exponential_disk_velocity(R, M_disk, R_d):
    mass_enclosed = exponential_disk_mass_enclosed(R, M_disk, R_d)
    denom = np.maximum(R, 1e-3)
    v_sq = G_KPC * mass_enclosed / denom
    return np.sqrt(np.maximum(v_sq, 0.0))


def simulate_one_toy_disk(
    name,
    M_disk,
    R_d,
    params,
    radii=None,
    output_dir="gravitywavebaseline/toy_models_sigma",
):
    if radii is None:
        radii = np.linspace(0.2, 12.0, 200)

    v_gr = exponential_disk_velocity(radii, M_disk, R_d)
    lam = 2.0 * np.pi * np.maximum(radii, 1e-3)
    multiplier = np.asarray(multiplier_shortlambda_boost(lam, params, np), dtype=np.float64)
    multiplier = np.clip(multiplier, 1.0, None)

    v_eff = v_gr * np.sqrt(multiplier)
    ratio = v_eff / np.maximum(v_gr, 1e-3)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    profile_df = pd.DataFrame(
        {
            "R_kpc": radii,
            "v_GR": v_gr,
            "lambda_gw": lam,
            "multiplier": multiplier,
            "v_eff": v_eff,
            "velocity_ratio": ratio,
        }
    )
    profile_path = output_path / f"toy_profile_{name}.csv"
    profile_df.to_csv(profile_path, index=False)

    summary = {
        "name": name,
        "M_disk": float(M_disk),
        "R_d": float(R_d),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "multiplier_min": float(multiplier.min()),
        "multiplier_median": float(np.median(multiplier)),
        "multiplier_max": float(multiplier.max()),
        "velocity_ratio_median": float(np.median(ratio)),
        "velocity_ratio_max": float(ratio.max()),
        "profile_path": str(profile_path),
    }
    return summary


def apply_sigma_gating_to_toys(
    mw_fit,
    beta=0.4,
    sigma_ref=30.0,
    output_dir="gravitywavebaseline/toy_models_sigma",
):
    A_MW = mw_fit["A"]
    lambda0 = mw_fit["lambda0"]
    alpha = mw_fit["alpha"]

    toy_sigmas = {
        "dwarf": 10.0,
        "lmc_like": 20.0,
        "mw_disk": 30.0,
    }

    disk_specs = {
        "dwarf": {"M_disk": 1.0e9, "R_d": 1.0},
        "lmc_like": {"M_disk": 3.0e9, "R_d": 1.5},
        "mw_disk": {"M_disk": 4.0e10, "R_d": 3.0},
    }

    summaries = []
    for name, props in disk_specs.items():
        sigma_v = toy_sigmas[name]
        ratio = sigma_ref / max(sigma_v, 1e-6)
        A_gal = A_MW * min(1.0, ratio**beta)
        params_gal = np.array([A_gal, lambda0, alpha], dtype=np.float64)
        print(f"\n[TOY] {name}: sigma_v={sigma_v:.1f} km/s, A_gal={A_gal:.3f} (beta={beta:.2f})")
        summary = simulate_one_toy_disk(
            name=name,
            M_disk=props["M_disk"],
            R_d=props["R_d"],
            params=params_gal,
            output_dir=output_dir,
        )
        print(
            f"      median f(lambda)={summary['multiplier_median']:.2f}, "
            f"median v_ratio={summary['velocity_ratio_median']:.2f}"
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    summary_path = Path(output_dir) / "toy_disk_sigma_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved toy σ-gated summary to {summary_path}")
    return summary_df, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Pure λ-GW × GR test (MW calibration + toy dwarf/MW disks)"
    )
    parser.add_argument(
        "--gaia-path",
        type=str,
        default="gravitywavebaseline/gaia_with_gr_baseline.parquet",
        help="Parquet file with MW GR baseline and lambda_gw.",
    )
    parser.add_argument("--r-min", type=float, default=12.0, help="Minimum radius for MW fit (kpc).")
    parser.add_argument("--r-max", type=float, default=16.0, help="Maximum radius for MW fit (kpc).")
    parser.add_argument(
        "--n-obs", type=int, default=1000, help="Number of Gaia stars to use in MW outer-ring fit."
    )
    parser.add_argument(
        "--max-mult",
        type=float,
        default=6.0,
        help="Hard cap on f(lambda) during MW fit (avoids insane boosts).",
    )
    parser.add_argument(
        "--mw-fit-out",
        type=str,
        default="gravitywavebaseline/mw_pure_lambda_fit.json",
        help="Where to save best-fit MW lambda parameters.",
    )
    parser.add_argument(
        "--sigma-beta",
        type=float,
        default=0.4,
        help="Exponent β for σ-v gating: A_gal = A_MW (σ_ref/σ_v)^β.",
    )
    parser.add_argument(
        "--sigma-ref",
        type=float,
        default=30.0,
        help="Reference dispersion σ_ref (km/s) for MW.",
    )
    parser.add_argument(
        "--toy-output-dir",
        type=str,
        default="gravitywavebaseline/toy_models_sigma",
        help="Directory for toy disk CSVs.",
    )

    args = parser.parse_args()

    mw_fit, _ = calibrate_mw_pure_lambda(
        gaia_path=args.gaia_path,
        r_min=args.r_min,
        r_max=args.r_max,
        n_obs=args.n_obs,
        max_mult=args.max_mult,
        output_path=args.mw_fit_out,
    )

    apply_sigma_gating_to_toys(
        mw_fit,
        beta=args.sigma_beta,
        sigma_ref=args.sigma_ref,
        output_dir=args.toy_output_dir,
    )


if __name__ == "__main__":
    main()



