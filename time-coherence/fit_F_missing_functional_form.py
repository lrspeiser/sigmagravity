"""
Fit functional form for F_missing(σ_v, R_d).

Based on correlations:
- F_missing ∝ 1/σ_v^α (strongest correlation)
- F_missing ∝ 1/R_d^β (moderate correlation)

Try: F_missing = A × (σ_ref / σ_v)^α × (R_ref / R_d)^β
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def model_sigma_only(sigma_v, A, alpha, sigma_ref=18.0):
    """Simple model: F_missing = A × (σ_ref / σ_v)^α"""
    return A * (sigma_ref / np.clip(sigma_v, 1e-3, None)) ** alpha


def model_sigma_Rd(sigma_v, R_d, A, alpha, beta, sigma_ref=18.0, R_ref=20.0):
    """Full model: F_missing = A × (σ_ref / σ_v)^α × (R_ref / R_d)^β"""
    return (
        A
        * (sigma_ref / np.clip(sigma_v, 1e-3, None)) ** alpha
        * (R_ref / np.clip(R_d, 1e-3, None)) ** beta
    )


def fit_and_report(name, model_func, x_data, y_data, p0, bounds=None):
    """Fit model and report statistics."""
    mask = np.isfinite(y_data) & np.all(np.isfinite(x_data), axis=0)
    
    if isinstance(x_data, tuple):
        x_fit = tuple(x[mask] for x in x_data)
    else:
        x_fit = x_data[mask]
    y_fit = y_data[mask]
    
    if len(y_fit) < 10:
        print(f"\n[{name}]")
        print(f"  Insufficient data points")
        return None, np.inf, 0.0
    
    try:
        if bounds is None:
            popt, pcov = curve_fit(model_func, x_fit, y_fit, p0=p0, maxfev=10000)
        else:
            popt, pcov = curve_fit(
                model_func, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000
            )
    except Exception as e:
        print(f"\n[{name}]")
        print(f"  Fit failed: {e}")
        return None, np.inf, 0.0
    
    y_pred = model_func(x_fit, *popt)
    
    resid = y_fit - y_pred
    rms = float(np.sqrt(np.mean(resid**2)))
    
    if np.std(y_fit) > 1e-6 and np.std(y_pred) > 1e-6:
        corr = float(np.corrcoef(y_fit, y_pred)[0, 1])
    else:
        corr = 0.0
    
    print(f"\n[{name}]")
    print(f"  N points: {len(y_fit)}")
    print(f"  Params: {popt}")
    print(f"  RMS(F_missing - F_pred): {rms:.3f}")
    print(f"  corr(F_missing, F_pred): {corr:.3f}")
    print(f"  Mean F_missing: {np.mean(y_fit):.3f}, Mean F_pred: {np.mean(y_pred):.3f}")
    
    return popt, rms, corr


def main():
    print("=" * 80)
    print("FITTING F_MISSING FUNCTIONAL FORM")
    print("=" * 80)
    
    # Load data
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
        print("Error: Could not find galaxy name column")
        return
    
    merged = roughness_df.merge(
        summary_df,
        left_on=galaxy_col_rough,
        right_on=galaxy_col_summary,
        how="inner",
    )
    
    # Extract variables
    F_missing = merged["F_missing"].values
    sigma_v = merged["sigma_velocity"].values
    R_d = merged["R_disk"].values
    
    # Filter valid
    valid = (
        np.isfinite(F_missing)
        & np.isfinite(sigma_v)
        & np.isfinite(R_d)
        & (sigma_v > 0)
        & (R_d > 0)
        & (F_missing > 0)
    )
    
    F_valid = F_missing[valid]
    sigma_valid = sigma_v[valid]
    R_d_valid = R_d[valid]
    
    print(f"\nData: {len(F_valid)} galaxies with valid F_missing")
    print(f"  Mean F_missing: {np.mean(F_valid):.3f}")
    print(f"  Mean sigma_v: {np.mean(sigma_valid):.3f} km/s")
    print(f"  Mean R_d: {np.mean(R_d_valid):.3f} kpc")
    
    # Reference values (medians)
    sigma_ref = float(np.median(sigma_valid))
    R_ref = float(np.median(R_d_valid))
    
    print(f"\nReference values:")
    print(f"  sigma_ref: {sigma_ref:.2f} km/s")
    print(f"  R_ref: {R_ref:.2f} kpc")
    
    # Fit models
    print("\n" + "-" * 80)
    print("MODEL 1: F_missing = A * (sigma_ref / sigma_v)^alpha")
    print("-" * 80)
    
    p_sigma, rms_sigma, corr_sigma = fit_and_report(
        "sigma_v only",
        lambda s, A, alpha: model_sigma_only(s, A, alpha, sigma_ref=sigma_ref),
        sigma_valid,
        F_valid,
        p0=[10.0, 1.0],
        bounds=([0.1, 0.1], [100, 5]),
    )
    
    print("\n" + "-" * 80)
    print("MODEL 2: F_missing = A * (sigma_ref / sigma_v)^alpha * (R_ref / R_d)^beta")
    print("-" * 80)
    
    # Fix lambda to properly unpack tuple
    def model_wrapper(x_tuple, A, alpha, beta):
        s, r = x_tuple
        return model_sigma_Rd(s, r, A, alpha, beta, sigma_ref=sigma_ref, R_ref=R_ref)
    
    p_full, rms_full, corr_full = fit_and_report(
        "sigma_v + R_d",
        model_wrapper,
        (sigma_valid, R_d_valid),
        F_valid,
        p0=[10.0, 1.0, 0.5],
        bounds=([0.1, 0.1, 0.1], [100, 5, 5]),
    )
    
    # Save results
    results = {}
    if p_sigma is not None:
        results["sigma_only"] = {
            "params": p_sigma.tolist(),
            "rms": float(rms_sigma),
            "corr": float(corr_sigma),
            "sigma_ref": float(sigma_ref),
        }
    if p_full is not None:
        results["sigma_Rd"] = {
            "params": p_full.tolist(),
            "rms": float(rms_full),
            "corr": float(corr_full),
            "sigma_ref": float(sigma_ref),
            "R_ref": float(R_ref),
        }
    
    outpath = Path("time-coherence/F_missing_functional_fit.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]["rms"])
        print(f"\nBest model: {best_model[0]}")
        print(f"  RMS: {best_model[1]['rms']:.3f}")
        print(f"  Correlation: {best_model[1]['corr']:.3f}")
        print(f"  Params: {best_model[1]['params']}")
        
        if best_model[0] == "sigma_only":
            A, alpha = best_model[1]["params"]
            print(f"\nFunctional form:")
            print(f"  F_missing = {A:.2f} * ({sigma_ref:.1f} / sigma_v)^{alpha:.2f}")
        elif best_model[0] == "sigma_Rd":
            A, alpha, beta = best_model[1]["params"]
            print(f"\nFunctional form:")
            print(f"  F_missing = {A:.2f} * ({sigma_ref:.1f} / sigma_v)^{alpha:.2f} * ({R_ref:.1f} / R_d)^{beta:.2f}")
    
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

