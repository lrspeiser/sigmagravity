"""
Sweep unified kernel parameters to find optimal performance.

Explores gamma_sigma, F_max, extra_amp, and sigma_gate_ref to match empirical Σ-Gravity.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Dict, Any

import numpy as np
import pandas as pd

# Import modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from unified_kernel import UnifiedKernelParams, compute_unified_kernel
from f_missing_mass_model import FMissingParams
from sparc_utils import load_rotmod, load_sparc_summary

G_MSUN_KPC_KM2_S2 = 4.302e-6


def evaluate_galaxy(
    galaxy_row: pd.Series,
    uk_params: UnifiedKernelParams,
    rotmod_dir: Path,
) -> Dict[str, Any] | None:
    """
    Compute GR and unified-kernel RMS for a single SPARC galaxy.

    Assumes:
      * rotmod files live in rotmod_dir / f"{galaxy_name}_rotmod.dat"
      * galaxy_row contains: galaxy_name, sigma_velocity, R_disk, morph_class
    """
    name = str(galaxy_row["galaxy_name"])
    sigma_v = float(galaxy_row.get("sigma_velocity", galaxy_row.get("sigma_v", 20.0)))
    R_d = float(galaxy_row.get("R_disk", galaxy_row.get("R_d", 5.0)))
    morph = galaxy_row.get("morph_class", galaxy_row.get("morphology", None))

    rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
    if not rotmod_path.exists():
        return None

    try:
        df = load_rotmod(rotmod_path)  # must return columns R_kpc, V_obs, V_gr
    except Exception:
        return None

    R = df["R_kpc"].to_numpy(float)
    V_obs = df["V_obs"].to_numpy(float)
    V_gr = df["V_gr"].to_numpy(float)

    if len(R) < 4:
        return None

    # Compute g_bar and density
    g_bar_kms2 = (V_gr**2) / (R * 1e3)
    rho_bar_msun_pc3 = g_bar_kms2 / (G_MSUN_KPC_KM2_S2 * R * 1e3) * 1e-9

    # GR RMS
    mask = (V_gr > 1e-3) & (V_obs > 0) & np.isfinite(V_gr) & np.isfinite(V_obs)
    if np.sum(mask) < 4:
        return None

    rms_gr = float(np.sqrt(np.mean((V_obs[mask] - V_gr[mask]) ** 2)))

    # Unified-kernel RMS
    try:
        galaxy_props = {
            "sigma_v": sigma_v,
            "R_disk": R_d,
            "M_baryon": float(galaxy_row.get("M_baryon", 1e10)),
        }
        if morph is not None:
            galaxy_props["morphology"] = morph

        K_total, info = compute_unified_kernel(
            R_kpc=R,
            g_bar_kms2=g_bar_kms2,
            sigma_v_kms=sigma_v,
            rho_bar_msun_pc3=rho_bar_msun_pc3,
            galaxy_props=galaxy_props,
            params=uk_params,
        )

        g_eff = g_bar_kms2 * (1.0 + K_total)
        V_model = np.sqrt(R * 1e3 * g_eff)

        mask_model = mask & np.isfinite(V_model)
        if np.sum(mask_model) < 4:
            return None

        rms_model = float(np.sqrt(np.mean((V_obs[mask_model] - V_model[mask_model]) ** 2)))
    except Exception:
        return None

    return {
        "galaxy_name": name,
        "sigma_v": sigma_v,
        "R_disk": R_d,
        "morph_class": morph,
        "rms_gr": rms_gr,
        "rms_model": rms_model,
        "delta_rms": rms_model - rms_gr,
        "F_missing": info.get("F_missing", 1.0),
        "K_total_mean": info.get("K_total_mean", 0.0),
    }


def sweep_parameter_grid(
    summary_df: pd.DataFrame,
    rotmod_dir: Path,
    base_params: UnifiedKernelParams,
    gamma_sigma_grid: Iterable[float],
    F_max_grid: Iterable[float],
    extra_amp_grid: Iterable[float],
    sigma_gate_ref_grid: Iterable[float],
) -> pd.DataFrame:
    """
    Sweep over {gamma_sigma, F_max, extra_amp, sigma_gate_ref} and evaluate all galaxies.
    """
    rows: list[Dict[str, Any]] = []

    total_combinations = (
        len(list(gamma_sigma_grid))
        * len(list(F_max_grid))
        * len(list(extra_amp_grid))
        * len(list(sigma_gate_ref_grid))
    )
    print(f"Total parameter combinations: {total_combinations}")
    print(f"Total galaxies: {len(summary_df)}")
    print(f"Estimated evaluations: {total_combinations * len(summary_df)}")
    print()

    combo_count = 0
    for gamma_sigma in gamma_sigma_grid:
        for F_max in F_max_grid:
            for extra_amp in extra_amp_grid:
                for sigma_gate_ref in sigma_gate_ref_grid:
                    combo_count += 1
                    print(f"[{combo_count}/{total_combinations}] Testing: "
                          f"gamma_sigma={gamma_sigma:.2f}, F_max={F_max:.1f}, "
                          f"extra_amp={extra_amp:.3f}, sigma_gate_ref={sigma_gate_ref:.1f}")

                    # Construct a new parameter set from the base
                    f_params = replace(
                        base_params.f_missing,
                        F_max=F_max,
                        gamma_sigma=gamma_sigma,
                        sigma_gate_ref=sigma_gate_ref,
                    )
                    uk_params = replace(
                        base_params,
                        extra_amp=extra_amp,
                        f_missing=f_params,
                    )

                    # Evaluate all galaxies
                    per_gal = []
                    for _, gal_row in summary_df.iterrows():
                        res = evaluate_galaxy(gal_row, uk_params, rotmod_dir)
                        if res is not None:
                            per_gal.append(res)

                    if not per_gal:
                        print(f"  No valid galaxies")
                        continue

                    gal_df = pd.DataFrame(per_gal)
                    improved = gal_df["delta_rms"] < 0.0
                    frac_improved = float(improved.mean())
                    mean_delta = float(gal_df["delta_rms"].mean())
                    median_delta = float(gal_df["delta_rms"].median())
                    std_delta = float(gal_df["delta_rms"].std())

                    # Also compute performance by sigma_v bins
                    gal_df["sigma_bin"] = pd.cut(gal_df["sigma_v"], bins=[0, 15, 25, 50, 200], labels=["low", "med", "high", "very_high"])
                    sigma_stats = {}
                    for bin_name in ["low", "med", "high", "very_high"]:
                        bin_data = gal_df[gal_df["sigma_bin"] == bin_name]
                        if len(bin_data) > 0:
                            sigma_stats[f"mean_delta_{bin_name}"] = float(bin_data["delta_rms"].mean())
                            sigma_stats[f"n_{bin_name}"] = int(len(bin_data))

                    rows.append({
                        "gamma_sigma": gamma_sigma,
                        "F_max": F_max,
                        "extra_amp": extra_amp,
                        "sigma_gate_ref": sigma_gate_ref,
                        "frac_improved": frac_improved,
                        "mean_delta_rms": mean_delta,
                        "median_delta_rms": median_delta,
                        "std_delta_rms": std_delta,
                        "n_galaxies": len(gal_df),
                        **sigma_stats,
                    })

                    print(f"  -> improved={frac_improved:.3f}, "
                          f"mean_delta={mean_delta:.2f}, median_delta={median_delta:.2f}, "
                          f"n={len(gal_df)}")
                    print()

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep unified-kernel parameters over the SPARC sample."
    )
    parser.add_argument(
        "--sparc-summary",
        default="data/sparc/sparc_combined.csv",
        help="Path to SPARC summary CSV (must contain galaxy_name, sigma_velocity, R_disk).",
    )
    parser.add_argument(
        "--rotmod-dir",
        default="data/Rotmod_LTG",
        help="Directory containing *_rotmod.dat files.",
    )
    parser.add_argument(
        "--base-params-json",
        default=None,
        help="JSON file with a baseline UnifiedKernelParams. If not provided, uses defaults.",
    )
    parser.add_argument(
        "--out-csv",
        default="time-coherence/unified_kernel_sweep_results.csv",
        help="Output CSV for sweep summary.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with fewer parameter combinations.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    summary_path = project_root / args.sparc_summary
    rotmod_dir = project_root / args.rotmod_dir

    summary_df = load_sparc_summary(summary_path)
    print(f"Loaded {len(summary_df)} galaxies from {summary_path}")

    if args.base_params_json:
        base_params = UnifiedKernelParams.from_json(project_root / args.base_params_json)
    else:
        base_params = UnifiedKernelParams()

    # Parameter grids
    if args.quick:
        print("Running QUICK test with reduced parameter space")
        gamma_sigma_grid = [0.0, 1.0, 2.0]
        F_max_grid = [3.0, 5.0]
        extra_amp_grid = [0.15, 0.25]
        sigma_gate_ref_grid = [20.0, 25.0]
    else:
        gamma_sigma_grid = [0.0, 0.5, 1.0, 1.5, 2.0]
        F_max_grid = [2.0, 3.0, 4.0, 5.0]
        extra_amp_grid = [0.10, 0.15, 0.20, 0.25]
        sigma_gate_ref_grid = [20.0, 25.0, 30.0]

    sweep_df = sweep_parameter_grid(
        summary_df=summary_df,
        rotmod_dir=rotmod_dir,
        base_params=base_params,
        gamma_sigma_grid=gamma_sigma_grid,
        F_max_grid=F_max_grid,
        extra_amp_grid=extra_amp_grid,
        sigma_gate_ref_grid=sigma_gate_ref_grid,
    )

    out_path = project_root / args.out_csv
    sweep_df.to_csv(out_path, index=False)
    print(f"\nSaved sweep summary to {out_path}")

    # Find and report best parameters
    if len(sweep_df) > 0:
        print("\n" + "=" * 80)
        print("BEST PARAMETERS")
        print("=" * 80)

        # Best by fraction improved
        best_frac = sweep_df.loc[sweep_df["frac_improved"].idxmax()]
        print(f"\nBest by fraction improved:")
        print(f"  gamma_sigma: {best_frac['gamma_sigma']:.2f}")
        print(f"  F_max: {best_frac['F_max']:.1f}")
        print(f"  extra_amp: {best_frac['extra_amp']:.3f}")
        print(f"  sigma_gate_ref: {best_frac['sigma_gate_ref']:.1f}")
        print(f"  Fraction improved: {best_frac['frac_improved']:.3f}")
        print(f"  Mean delta_RMS: {best_frac['mean_delta_rms']:.2f} km/s")
        print(f"  Median delta_RMS: {best_frac['median_delta_rms']:.2f} km/s")

        # Best by mean delta_RMS (most negative = best)
        best_rms = sweep_df.loc[sweep_df["mean_delta_rms"].idxmin()]
        print(f"\nBest by mean delta_RMS:")
        print(f"  gamma_sigma: {best_rms['gamma_sigma']:.2f}")
        print(f"  F_max: {best_rms['F_max']:.1f}")
        print(f"  extra_amp: {best_rms['extra_amp']:.3f}")
        print(f"  sigma_gate_ref: {best_rms['sigma_gate_ref']:.1f}")
        print(f"  Mean delta_RMS: {best_rms['mean_delta_rms']:.2f} km/s")
        print(f"  Fraction improved: {best_rms['frac_improved']:.3f}")

        # Best by median delta_RMS
        best_median = sweep_df.loc[sweep_df["median_delta_rms"].idxmin()]
        print(f"\nBest by median delta_RMS:")
        print(f"  gamma_sigma: {best_median['gamma_sigma']:.2f}")
        print(f"  F_max: {best_median['F_max']:.1f}")
        print(f"  extra_amp: {best_median['extra_amp']:.3f}")
        print(f"  sigma_gate_ref: {best_median['sigma_gate_ref']:.1f}")
        print(f"  Median delta_RMS: {best_median['median_delta_rms']:.2f} km/s")
        print(f"  Fraction improved: {best_median['frac_improved']:.3f}")


if __name__ == "__main__":
    main()
