"""
Test mass-coherence model against F_missing data.

Fits parameters (K_max, psi0, gamma) to match observed F_missing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from f_missing_mass_model import predict_F_missing_mass_model
from mass_coherence_model import dimensionless_potential_depth


def load_data():
    """Load SPARC data with F_missing."""
    roughness_df = pd.read_csv("time-coherence/sparc_roughness_amplitude.csv")
    summary_df = pd.read_csv("data/sparc/sparc_combined.csv")

    # Merge
    galaxy_col_rough = "galaxy"
    galaxy_col_summary = None
    for col in ["galaxy_name", "Galaxy", "name", "gal"]:
        if col in summary_df.columns:
            galaxy_col_summary = col
            break

    if galaxy_col_summary is None:
        raise ValueError("Could not find galaxy name column")

    merged = roughness_df.merge(
        summary_df,
        left_on=galaxy_col_rough,
        right_on=galaxy_col_summary,
        how="inner",
    )

    return merged


def objective_mass_model(params, df_valid):
    """
    Objective function for mass-coherence model fit.

    Parameters
    ----------
    params : tuple
        (K_max, psi0, gamma, R_eff_factor)
    df_valid : pd.DataFrame
        DataFrame with valid data

    Returns
    -------
    float
        RMS error between predicted and observed F_missing
    """
    K_max, psi0, gamma, R_eff_factor = params

    # Get data
    F_true = df_valid["F_missing"].values
    M_b = df_valid["M_baryon"].values
    R_d = df_valid["R_disk"].values
    ell0 = df_valid["ell0_kpc"].values

    # Predict
    try:
        F_pred = predict_F_missing_mass_model(
            M_baryon_msun=M_b,
            R_d_kpc=R_d,
            ell0_kpc=ell0,
            R_eff_factor=R_eff_factor,
            K_max=K_max,
            psi0=psi0,
            gamma=gamma,
        )

        # Compute RMS
        valid = np.isfinite(F_pred) & np.isfinite(F_true) & (F_pred > 0)
        if np.sum(valid) < 10:
            return 1e10

        resid = F_pred[valid] - F_true[valid]
        rms = float(np.sqrt(np.mean(resid**2)))
        return rms
    except Exception:
        return 1e10


def fit_mass_coherence_model(df):
    """Fit mass-coherence model parameters to F_missing data."""
    # Filter valid data
    valid = (
        df["F_missing"].notna()
        & (df["F_missing"] > 0)
        & df["M_baryon"].notna()
        & (df["M_baryon"] > 0)
        & df["R_disk"].notna()
        & (df["R_disk"] > 0)
        & df["ell0_kpc"].notna()
        & (df["ell0_kpc"] > 0)
    )

    df_valid = df[valid].copy()

    if len(df_valid) < 10:
        print("Error: Insufficient valid data")
        return None

    print(f"Fitting on {len(df_valid)} galaxies")

    # Bounds: (K_max, psi0, gamma, R_eff_factor)
    bounds = [
        (0.1, 20.0),  # K_max
        (1e-10, 1e-4),  # psi0
        (0.1, 3.0),  # gamma
        (1.0, 5.0),  # R_eff_factor
    ]

    # Objective wrapper
    def objective(params):
        return objective_mass_model(params, df_valid)

    # Fit
    print("Running differential evolution...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=100,
        popsize=10,
        seed=42,
        polish=True,
    )

    if not result.success:
        print(f"Warning: Fit did not converge: {result.message}")
        return None

    K_max, psi0, gamma, R_eff_factor = result.x

    # Evaluate fit
    F_true = df_valid["F_missing"].values
    M_b = df_valid["M_baryon"].values
    R_d = df_valid["R_disk"].values
    ell0 = df_valid["ell0_kpc"].values

    F_pred = predict_F_missing_mass_model(
        M_baryon_msun=M_b,
        R_d_kpc=R_d,
        ell0_kpc=ell0,
        R_eff_factor=R_eff_factor,
        K_max=K_max,
        psi0=psi0,
        gamma=gamma,
    )

    valid_pred = np.isfinite(F_pred) & np.isfinite(F_true) & (F_pred > 0)
    F_true_good = F_true[valid_pred]
    F_pred_good = F_pred[valid_pred]

    rms = float(np.sqrt(np.mean((F_pred_good - F_true_good) ** 2)))
    corr = (
        float(np.corrcoef(F_true_good, F_pred_good)[0, 1])
        if np.std(F_true_good) > 1e-6 and np.std(F_pred_good) > 1e-6
        else 0.0
    )

    return {
        "params": {
            "K_max": float(K_max),
            "psi0": float(psi0),
            "gamma": float(gamma),
            "R_eff_factor": float(R_eff_factor),
        },
        "rms": float(rms),
        "correlation": float(corr),
        "n_galaxies": int(len(df_valid)),
        "n_valid": int(np.sum(valid_pred)),
    }


def main():
    print("=" * 80)
    print("TESTING MASS-COHERENCE MODEL ON F_MISSING")
    print("=" * 80)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} galaxies")

    # Check required columns and find alternatives
    # M_baryon might be in summary CSV
    if "M_baryon" not in df.columns:
        # Try to compute from other columns or use placeholder
        if "M_star" in df.columns and "M_gas" in df.columns:
            df["M_baryon"] = df["M_star"] + df["M_gas"]
        elif "M_star" in df.columns:
            df["M_baryon"] = df["M_star"] * 1.5  # Rough estimate: gas ~50% of stars
        else:
            # Use rough estimate from V_flat: M ~ V^2 R / G
            if "V_flat" in df.columns or "v_flat" in df.columns:
                v_col = "V_flat" if "V_flat" in df.columns else "v_flat"
                R_col = "R_disk" if "R_disk" in df.columns else "R_d"
                G = 4.30091e-6  # kpc km^2 / s^2 / Msun
                df["M_baryon"] = (df[v_col] ** 2) * df[R_col] / G
            else:
                print("Error: Cannot find or estimate M_baryon")
                print(f"Available columns: {list(df.columns)}")
                return
    
    # ell0_kpc might need to be computed or use default
    if "ell0_kpc" not in df.columns:
        # Use default from roughness amplitude test (5.0 kpc)
        df["ell0_kpc"] = 5.0
        print("Warning: Using default ell0_kpc = 5.0 kpc")
    
    # Check required columns
    required_cols = ["F_missing", "M_baryon", "R_disk", "ell0_kpc"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {list(df.columns)}")
        return

    # Fit model
    result = fit_mass_coherence_model(df)

    if result is None:
        print("\nFit failed")
        return

    # Save results
    outpath = Path("time-coherence/mass_coherence_fit.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 80)
    print("FIT RESULTS")
    print("=" * 80)
    print(f"\nParameters:")
    for key, value in result["params"].items():
        print(f"  {key}: {value:.6e}" if value < 0.01 else f"  {key}: {value:.3f}")

    print(f"\nPerformance:")
    print(f"  RMS: {result['rms']:.3f}")
    print(f"  Correlation: {result['correlation']:.3f}")
    print(f"  N galaxies: {result['n_galaxies']}")
    print(f"  N valid predictions: {result['n_valid']}")

    print(f"\nResults saved to {outpath}")

    # Compare to functional form fit
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nMass-coherence model:")
    print(f"  RMS: {result['rms']:.3f}")
    print(f"  Correlation: {result['correlation']:.3f}")

    # Load functional form fit for comparison
    func_fit_path = Path("time-coherence/F_missing_functional_fit.json")
    if func_fit_path.exists():
        with open(func_fit_path, "r") as f:
            func_fit = json.load(f)
        best_func = min(func_fit.items(), key=lambda x: x[1]["rms"])
        print(f"\nFunctional form ({best_func[0]}):")
        print(f"  RMS: {best_func[1]['rms']:.3f}")
        print(f"  Correlation: {best_func[1]['corr']:.3f}")

        if result["rms"] < best_func[1]["rms"]:
            print("\n*** Mass-coherence model is BETTER ***")
        else:
            print("\n*** Functional form is better ***")


if __name__ == "__main__":
    main()

