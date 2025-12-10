"""
Fit a simple capacity-growth law (e.g., razor-thin disk that thickens with radius)
so that the required lambda_gw enhancement per shell matches the Gaia rotation curve.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
import test_lambda_enhancement as tle


def compute_required_extra(v_obs, v_gr):
    """Velocity contribution (km/s) needed from lambda_gw to match observations."""
    delta_sq = np.maximum(v_obs**2 - v_gr**2, 0.0)
    return np.sqrt(delta_sq)


def build_capacity(profile, model, alpha, gamma, geometry):
    centers = profile["centers"]
    base = np.zeros_like(centers, dtype=np.float64)
    geometry = geometry.lower()
    model = model.lower()

    if model == "surface_density":
        if geometry == "sphere":
            area = profile["sphere_area"]
            base = np.divide(
                profile["mass"],
                area,
                out=np.zeros_like(centers),
                where=area > 0,
            )
        else:
            base = profile["surface_density"]
    elif model == "velocity_dispersion":
        sigma = profile["sigma_v"] + 1e-6
        base = np.divide(
            profile["mass"],
            sigma,
            out=np.zeros_like(centers),
            where=sigma > 0,
        )
    elif model == "flatness":
        base = profile["mass"] * profile["flatness"]
    elif model == "wavelength":
        base = profile["lambda_median"]
    else:
        raise ValueError(f"Unknown capacity model '{model}'")

    r_norm = np.maximum(centers, 1e-3)
    r_ref = np.median(r_norm[base > 0]) if np.any(base > 0) else 1.0
    growth = np.power(r_norm / r_ref, gamma)
    capacity = alpha * base * growth
    return np.maximum(capacity, 0.0)


def fit_capacity(calculator, R_obs, v_required, model, geometry, alpha_bounds, gamma_bounds):
    profile = calculator.shell_profiles
    shell_edges = profile["edges"]

    def objective(params):
        alpha, gamma = params
        capacity = build_capacity(profile, model, alpha, gamma, geometry)
        provided = tle.apply_capacity_to_enhancement(R_obs, v_required, shell_edges, capacity)
        deficit = v_required - provided
        return np.sum(deficit**2)

    bounds = [alpha_bounds, gamma_bounds]
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=30,
        popsize=8,
        seed=42,
        polish=True,
    )

    best_capacity = build_capacity(profile, model, result.x[0], result.x[1], geometry)
    provided = tle.apply_capacity_to_enhancement(R_obs, v_required, shell_edges, best_capacity)
    residuals = v_required - provided
    rms_shortfall = np.sqrt(np.mean(residuals**2))

    return result, best_capacity, provided, rms_shortfall


def main():
    parser = argparse.ArgumentParser(description="Fit capacity-growth law for lambda_gw enhancement.")
    parser.add_argument("--data-path", default="gravitywavebaseline/gaia_with_gr_baseline.parquet")
    parser.add_argument("--r-min", type=float, default=12.0)
    parser.add_argument("--r-max", type=float, default=16.0)
    parser.add_argument("--n-obs", type=int, default=1000)
    parser.add_argument("--n-sample-stars", type=int, default=50000)
    parser.add_argument("--stellar-scale", type=float, default=5.0)
    parser.add_argument("--disk-mass", type=float, default=4.0e10)
    parser.add_argument("--capacity-model", type=str, default="surface_density",
                        choices=["surface_density", "velocity_dispersion", "flatness", "wavelength"])
    parser.add_argument("--capacity-geometry", type=str, default="disk",
                        choices=["disk", "sphere"])
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=20.0)
    parser.add_argument("--gamma-min", type=float, default=-3.0)
    parser.add_argument("--gamma-max", type=float, default=3.0)
    parser.add_argument("--output", default="gravitywavebaseline/capacity_profile_fit.json")
    args = parser.parse_args()

    print("=" * 80)
    print("CAPACITY PROFILE FIT")
    print("=" * 80)

    gaia = pd.read_parquet(args.data_path)
    mask = (
        (gaia["R"] >= args.r_min)
        & (gaia["R"] <= args.r_max)
        & np.isfinite(gaia["v_phi"])
        & (gaia["v_phi"] > 0)
        & np.isfinite(gaia["v_phi_GR"])
    )
    candidates = gaia.loc[mask]
    if len(candidates) == 0:
        raise ValueError("No valid stars in requested radial band.")

    if len(candidates) > args.n_obs:
        obs = candidates.sample(n=args.n_obs, random_state=42)
    else:
        obs = candidates
    print(f"  Observations used: {len(obs)}")

    R_obs = obs["R"].values
    v_obs = obs["v_phi"].values
    v_gr = obs["v_phi_GR"].values
    v_required = compute_required_extra(v_obs, v_gr)

    calculator = tle.LambdaEnhancementCalculator(
        gaia,
        use_gpu=tle.GPU_AVAILABLE,
        n_sample_stars=args.n_sample_stars,
        stellar_mass_scale=args.stellar_scale,
        disk_mass=args.disk_mass,
    )

    result, capacity, provided, rms_shortfall = fit_capacity(
        calculator,
        R_obs,
        v_required,
        model=args.capacity_model,
        geometry=args.capacity_geometry,
        alpha_bounds=(args.alpha_min, args.alpha_max),
        gamma_bounds=(args.gamma_min, args.gamma_max),
    )

    print("\nBest-fit capacity parameters:")
    print(f"  alpha = {result.x[0]:.4f}")
    print(f"  gamma = {result.x[1]:.4f}")
    print(f"  RMS shortfall = {rms_shortfall:.2f} km/s")

    v_model = np.sqrt(v_gr**2 + provided**2)
    rms_velocity = np.sqrt(np.mean((v_model - v_obs) ** 2))
    print(f"  RMS velocity error after capacity fit = {rms_velocity:.2f} km/s")

    profile = calculator.shell_profiles
    shell_report = []
    for i in range(len(profile["centers"])):
        shell_report.append(
            {
                "R_center": float(profile["centers"][i]),
                "surface_density": float(profile["surface_density"][i]),
                "capacity": float(capacity[i]),
            }
        )

    output = {
        "config": {
            "r_min": args.r_min,
            "r_max": args.r_max,
            "n_obs": int(len(obs)),
            "stellar_scale": args.stellar_scale,
            "disk_mass": args.disk_mass,
            "capacity_model": args.capacity_model,
            "capacity_geometry": args.capacity_geometry,
            "alpha_bounds": [args.alpha_min, args.alpha_max],
            "gamma_bounds": [args.gamma_min, args.gamma_max],
        },
        "fit": {
            "alpha": result.x.tolist()[0],
            "gamma": result.x.tolist()[1],
            "rms_shortfall": float(rms_shortfall),
            "rms_velocity": float(rms_velocity),
        },
        "shells": shell_report,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\n[OK] Saved: {args.output}")


if __name__ == "__main__":
    main()

