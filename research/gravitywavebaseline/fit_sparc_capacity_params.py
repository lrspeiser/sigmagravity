"""
Fit (alpha, gamma) per SPARC galaxy and correlate with global properties.
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

from sparc_capacity_test import (
    read_sparc_rotmod,
    compute_shell_edges,
    build_capacity,
)
from test_lambda_enhancement import apply_capacity_to_enhancement


def fit_capacity_for_galaxy(
    R, V_obs, V_gr, capacity_model, alpha_bounds, gamma_bounds
):
    V_req = np.sqrt(np.maximum(V_obs**2 - V_gr**2, 0.0))
    edges = compute_shell_edges(R)

    def objective(params):
        alpha, gamma = params
        capacity = build_capacity(capacity_model, R, alpha, gamma)
        provided = apply_capacity_to_enhancement(R, V_req, edges, capacity)
        residual = V_req - provided
        return np.sum(residual**2)

    result = differential_evolution(
        objective,
        bounds=[alpha_bounds, gamma_bounds],
        maxiter=25,
        popsize=6,
        seed=42,
        polish=True,
    )
    alpha, gamma = result.x
    capacity = build_capacity(capacity_model, R, alpha, gamma)
    provided = apply_capacity_to_enhancement(R, V_req, edges, capacity)
    V_model = np.sqrt(np.maximum(V_gr**2 + provided**2, 0.0))
    rms_vel = np.sqrt(np.mean((V_model - V_obs) ** 2))
    return alpha, gamma, rms_vel, capacity


def main():
    parser = argparse.ArgumentParser(description="Fit capacity parameters per SPARC galaxy.")
    parser.add_argument("--galaxy-list", default="data/sparc/sparc_combined.csv")
    parser.add_argument("--rotmod-dir", default="data/Rotmod_LTG")
    parser.add_argument("--capacity-model", default="surface_density")
    parser.add_argument("--alpha-min", type=float, default=0.01)
    parser.add_argument("--alpha-max", type=float, default=20.0)
    parser.add_argument("--gamma-min", type=float, default=-3.0)
    parser.add_argument("--gamma-max", type=float, default=3.0)
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--output", default="gravitywavebaseline/sparc_capacity_fits.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.galaxy_list)
    galaxies = df["galaxy_name"].str.strip().tolist()
    if args.max_galaxies:
        galaxies = galaxies[: args.max_galaxies]

    rows = []
    for name in galaxies:
        rotmod_path = Path(args.rotmod_dir) / f"{name}_rotmod.dat"
        if not rotmod_path.exists():
            print(f"[WARN] Missing rotmod file for {name}")
            continue

        rot = read_sparc_rotmod(rotmod_path)
        if len(rot) < 3:
            continue
        R = rot["R"].values
        V_obs = rot["Vobs"].values
        V_gr = np.sqrt(
            np.maximum(
                rot["Vgas"].values**2
                + rot["Vdisk"].values**2
                + rot["Vbulge"].values**2,
                0.0,
            )
        )
        capacity_model = rot["SBdisk"].values + rot["SBbulge"].values

        try:
            alpha, gamma, rms, capacity = fit_capacity_for_galaxy(
                R,
                V_obs,
                V_gr,
                capacity_model,
                (args.alpha_min, args.alpha_max),
                (args.gamma_min, args.gamma_max),
            )
            rows.append(
                {
                    "galaxy": name,
                    "alpha": alpha,
                    "gamma": gamma,
                    "rms_velocity": rms,
                    "n_points": len(R),
                    "M_baryon": float(df.loc[df["galaxy_name"] == name, "M_baryon"].values[0]),
                    "R_disk": float(df.loc[df["galaxy_name"] == name, "R_disk"].values[0]),
                    "sigma_velocity": float(df.loc[df["galaxy_name"] == name, "sigma_velocity"].values[0]),
                }
            )
            print(f"{name}: alpha={alpha:.2f}, gamma={gamma:.2f}, RMS={rms:.1f}")
        except Exception as exc:
            print(f"[WARN] Fit failed for {name}: {exc}")

    out_df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\n[OK] Saved capacity fits: {args.output}")


if __name__ == "__main__":
    main()


