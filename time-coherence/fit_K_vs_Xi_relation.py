"""
Fit a universal K(Ξ) relation across MW, SPARC, and clusters.

Tests if "extra time in the field" translates to enhancement via a universal function.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from glob import glob
from scipy.optimize import curve_fit

MW_PROFILE = Path("time-coherence/mw_exposure_profile.json")
SPARC_SUMMARY = Path("time-coherence/sparc_exposure_summary.csv")
CLUSTER_PATTERN = "time-coherence/cluster_time_delay_*.json"


def load_mw_points():
    """Load MW exposure profile data."""
    if not MW_PROFILE.exists():
        print(f"Warning: {MW_PROFILE} not found")
        return np.array([]), np.array([])
    
    with open(MW_PROFILE, "r") as f:
        data = json.load(f)
    
    Xi = np.array(data.get("Xi", []), dtype=float)
    K = np.array(data.get("K", []), dtype=float)
    
    return Xi, K


def load_sparc_points():
    """Load SPARC exposure summary data."""
    if not SPARC_SUMMARY.exists():
        print(f"Warning: {SPARC_SUMMARY} not found")
        return np.array([]), np.array([])
    
    df = pd.read_csv(SPARC_SUMMARY)
    
    Xi = np.array(df["Xi_mean"], dtype=float)
    K = np.array(df["K_mean"], dtype=float)
    
    return Xi, K


def load_cluster_points():
    """Load cluster time delay data."""
    Xi_list, K_list = [], []
    
    cluster_files = glob(str(CLUSTER_PATTERN))
    if not cluster_files:
        print(f"Warning: No cluster files found matching {CLUSTER_PATTERN}")
        return np.array([]), np.array([])
    
    for path in cluster_files:
        try:
            with open(path, "r") as f:
                d = json.load(f)
            # Get Xi_E and K_E at Einstein radius
            if "Xi_E" in d and "K_E" in d:
                Xi_list.append(d["Xi_E"])
                K_list.append(d["K_E"])
        except Exception as e:
            print(f"Warning: Error loading {path}: {e}")
            continue
    
    return np.array(Xi_list, dtype=float), np.array(K_list, dtype=float)


def model_linear(Xi, a):
    """Linear model: K = a * Xi"""
    return a * np.clip(Xi, 0, None)


def model_extratime(Xi, a, b):
    """
    Extra-time model: K = (a * Xi) / (1 - b * Xi)
    
    Represents: more boost as you spend larger fraction of orbit coherent.
    """
    Xi_clipped = np.clip(Xi, 0, None)
    denominator = np.clip(1.0 - b * Xi_clipped, 1e-6, None)
    return (a * Xi_clipped) / denominator


def model_power(Xi, a, n):
    """Power law: K = a * Xi^n"""
    return a * (np.clip(Xi, 0, None) ** n)


def fit_and_report(name, Xi, K, model, p0, bounds=None):
    """Fit model and report statistics."""
    mask = np.isfinite(Xi) & np.isfinite(K) & (Xi >= 0) & (K >= 0)
    
    if np.sum(mask) < 3:
        print(f"\n[{name}]")
        print(f"  Insufficient data points")
        return None, np.inf, 0.0
    
    Xi_fit = Xi[mask]
    K_fit = K[mask]
    
    try:
        if bounds is None:
            popt, pcov = curve_fit(model, Xi_fit, K_fit, p0=p0, maxfev=10000)
        else:
            popt, pcov = curve_fit(model, Xi_fit, K_fit, p0=p0, bounds=bounds, maxfev=10000)
    except Exception as e:
        print(f"\n[{name}]")
        print(f"  Fit failed: {e}")
        return None, np.inf, 0.0
    
    K_pred = model(Xi_fit, *popt)
    
    resid = K_fit - K_pred
    rms = float(np.sqrt(np.mean(resid**2)))
    
    if np.std(K_fit) > 1e-6 and np.std(K_pred) > 1e-6:
        corr = float(np.corrcoef(K_fit, K_pred)[0, 1])
    else:
        corr = 0.0
    
    print(f"\n[{name}]")
    print(f"  N points: {len(Xi_fit)}")
    print(f"  Params: {popt}")
    print(f"  RMS(K - K_pred): {rms:.4f}")
    print(f"  corr(K, K_pred): {corr:.3f}")
    print(f"  Mean K: {np.mean(K_fit):.3f}, Mean K_pred: {np.mean(K_pred):.3f}")
    
    return popt, rms, corr


def main():
    print("=" * 80)
    print("FITTING UNIVERSAL K(Xi) RELATION")
    print("=" * 80)
    
    # Load data
    Xi_mw, K_mw = load_mw_points()
    Xi_s, K_s = load_sparc_points()
    Xi_c, K_c = load_cluster_points()
    
    print(f"\nData loaded:")
    print(f"  MW: {len(Xi_mw)} points")
    print(f"  SPARC: {len(Xi_s)} points")
    print(f"  Clusters: {len(Xi_c)} points")
    
    if len(Xi_mw) == 0 and len(Xi_s) == 0:
        print("\nError: No MW or SPARC data available")
        return
    
    # Stack MW+SPARC for fitting; reserve clusters for validation
    Xi_fit = np.concatenate([Xi_mw, Xi_s])
    K_fit = np.concatenate([K_mw, K_s])
    
    print(f"\nFitting on MW+SPARC ({len(Xi_fit)} points)")
    print(f"Validating on clusters ({len(Xi_c)} points)")
    
    # Linear hypothesis
    print("\n" + "-" * 80)
    print("HYPOTHESIS A: Linear K = a * Xi")
    print("-" * 80)
    p_lin, rms_lin, corr_lin = fit_and_report(
        "Linear (MW+SPARC)", Xi_fit, K_fit, model_linear, p0=[5.0]
    )
    
    if len(Xi_c) > 0 and p_lin is not None:
        fit_and_report(
            "Linear (clusters, out-of-sample)",
            Xi_c, K_c,
            lambda x, a: model_linear(x, a),
            p0=[p_lin[0]]
        )
    
    # Extra-time hypothesis
    print("\n" + "-" * 80)
    print("HYPOTHESIS B: Extra-time K = (a * Xi) / (1 - b * Xi)")
    print("-" * 80)
    p_xt, rms_xt, corr_xt = fit_and_report(
        "Extra-time (MW+SPARC)",
        Xi_fit, K_fit,
        model_extratime,
        p0=[5.0, 1.0],
        bounds=([0, 0], [100, 10])
    )
    
    if len(Xi_c) > 0 and p_xt is not None:
        fit_and_report(
            "Extra-time (clusters, out-of-sample)",
            Xi_c, K_c,
            lambda x, a, b: model_extratime(x, a, b),
            p0=p_xt,
            bounds=([0, 0], [100, 10])
        )
    
    # Power law hypothesis
    print("\n" + "-" * 80)
    print("HYPOTHESIS C: Power law K = a * Xi^n")
    print("-" * 80)
    p_pow, rms_pow, corr_pow = fit_and_report(
        "Power law (MW+SPARC)",
        Xi_fit, K_fit,
        model_power,
        p0=[5.0, 1.0],
        bounds=([0, 0.1], [100, 5])
    )
    
    if len(Xi_c) > 0 and p_pow is not None:
        fit_and_report(
            "Power law (clusters, out-of-sample)",
            Xi_c, K_c,
            lambda x, a, n: model_power(x, a, n),
            p0=p_pow,
            bounds=([0, 0.1], [100, 5])
        )
    
    # Save best fit
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    results = {}
    if p_lin is not None:
        results["linear"] = {
            "params": p_lin.tolist(),
            "rms": float(rms_lin),
            "corr": float(corr_lin),
        }
    if p_xt is not None:
        results["extra_time"] = {
            "params": p_xt.tolist(),
            "rms": float(rms_xt),
            "corr": float(corr_xt),
        }
    if p_pow is not None:
        results["power_law"] = {
            "params": p_pow.tolist(),
            "rms": float(rms_pow),
            "corr": float(corr_pow),
        }
    
    outpath = Path("time-coherence/K_vs_Xi_fit.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {outpath}")
    
    # Determine best model
    if results:
        best_model = min(results.items(), key=lambda x: x[1]["rms"])
        print(f"\nBest model: {best_model[0]}")
        print(f"  RMS: {best_model[1]['rms']:.4f}")
        print(f"  Correlation: {best_model[1]['corr']:.3f}")
        print(f"  Params: {best_model[1]['params']}")


if __name__ == "__main__":
    main()

