"""
Test velocity-based coherence model against F_missing data.

Alternative: F_missing from circular velocity only (no mass estimates needed).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from f_missing_mass_model import predict_F_missing_velocity_model


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


def objective_velocity_model(params, df_valid):
    """Objective function for velocity-based model fit."""
    F_max, psi0, delta, v_ref = params

    # Get data
    F_true = df_valid["F_missing"].values
    v_flat = df_valid["v_flat"].values

    # Predict
    try:
        F_pred = predict_F_missing_velocity_model(
            v_flat_kms=v_flat,
            F_max=F_max,
            psi0=psi0,
            delta=delta,
            v_ref_kms=v_ref,
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


def fit_velocity_coherence_model(df):
    """Fit velocity-based coherence model parameters to F_missing data."""
    # Filter valid data
    valid = (
        df["F_missing"].notna()
        & (df["F_missing"] > 0)
        & df["v_flat"].notna()
        & (df["v_flat"] > 0)
    )

    df_valid = df[valid].copy()

    if len(df_valid) < 10:
        print("Error: Insufficient valid data")
        return None

    print(f"Fitting on {len(df_valid)} galaxies")

    # Bounds: (F_max, psi0, delta, v_ref)
    bounds = [
        (1.0, 50.0),  # F_max
        (1e-3, 1.0),  # psi0
        (0.1, 3.0),  # delta
        (50.0, 500.0),  # v_ref
    ]

    # Objective wrapper
    def objective(params):
        return objective_velocity_model(params, df_valid)

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

    F_max, psi0, delta, v_ref = result.x

    # Evaluate fit
    F_true = df_valid["F_missing"].values
    v_flat = df_valid["v_flat"].values

    F_pred = predict_F_missing_velocity_model(
        v_flat_kms=v_flat,
        F_max=F_max,
        psi0=psi0,
        delta=delta,
        v_ref_kms=v_ref,
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
            "F_max": float(F_max),
            "psi0": float(psi0),
            "delta": float(delta),
            "v_ref": float(v_ref),
        },
        "rms": float(rms),
        "correlation": float(corr),
        "n_galaxies": int(len(df_valid)),
        "n_valid": int(np.sum(valid_pred)),
    }


def main():
    print("=" * 80)
    print("TESTING VELOCITY-BASED COHERENCE MODEL ON F_MISSING")
    print("=" * 80)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} galaxies")

    # Check required columns
    if "v_flat" not in df.columns:
        print("Error: v_flat column not found")
        print(f"Available columns: {list(df.columns)}")
        return

    # Fit model
    result = fit_velocity_coherence_model(df)

    if result is None:
        print("\nFit failed")
        return

    # Save results
    outpath = Path("time-coherence/velocity_coherence_fit.json")
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

    # Compare to other models
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print("\nVelocity-coherence model:")
    print(f"  RMS: {result['rms']:.3f}")
    print(f"  Correlation: {result['correlation']:.3f}")

    # Load mass-coherence fit for comparison
    mass_fit_path = Path("time-coherence/mass_coherence_fit.json")
    if mass_fit_path.exists():
        with open(mass_fit_path, "r") as f:
            mass_fit = json.load(f)
        print(f"\nMass-coherence model:")
        print(f"  RMS: {mass_fit['rms']:.3f}")
        print(f"  Correlation: {mass_fit['correlation']:.3f}")

    # Load functional form fit for comparison
    func_fit_path = Path("time-coherence/F_missing_functional_fit.json")
    if func_fit_path.exists():
        with open(func_fit_path, "r") as f:
            func_fit = json.load(f)
        best_func = min(func_fit.items(), key=lambda x: x[1]["rms"])
        print(f"\nFunctional form ({best_func[0]}):")
        print(f"  RMS: {best_func[1]['rms']:.3f}")
        print(f"  Correlation: {best_func[1]['corr']:.3f}")


if __name__ == "__main__":
    main()

