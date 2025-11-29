#!/usr/bin/env python3
"""
Compute bootstrap confidence intervals for Simpson's paradox partial correlation.
"""

import numpy as np
from scipy import stats
from pathlib import Path
import json

# Load SPARC data
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "Rotmod_LTG"

def load_sparc_data():
    """Load all SPARC rotation curve data."""
    K_values = []
    sigma_v_values = []
    R_values = []
    
    g_dagger = 1.2e-10  # m/s^2
    A0 = 0.591
    p = 0.757
    ell0 = 5.0  # kpc
    
    for f in DATA_DIR.glob("*_rotmod.dat"):
        try:
            data = np.loadtxt(f, comments='#')
            if len(data.shape) != 2 or data.shape[1] < 4:
                continue
            
            R = data[:, 0]  # kpc
            V_obs = data[:, 1]  # km/s
            V_bar = data[:, 3]  # km/s (baryonic)
            
            # Compute g_bar from V_bar
            g_bar = (V_bar * 1e3)**2 / (R * 3.086e19)  # m/s^2
            
            # Valid points
            mask = (R > 0.5) & (V_bar > 10) & (g_bar > 0)
            if np.sum(mask) < 5:
                continue
            
            R = R[mask]
            V_obs = V_obs[mask]
            V_bar = V_bar[mask]
            g_bar = g_bar[mask]
            
            # Compute K from the formula
            K = A0 * (g_dagger / g_bar)**p * (ell0 / (ell0 + R))
            
            # Estimate sigma_v from scatter (proxy)
            sigma_v = np.abs(V_obs - V_bar)
            
            K_values.extend(K)
            sigma_v_values.extend(sigma_v)
            R_values.extend(R)
            
        except Exception as e:
            continue
    
    return np.array(K_values), np.array(sigma_v_values), np.array(R_values)


def partial_correlation(x, y, z):
    """Compute partial correlation of x and y controlling for z."""
    slope_xz = np.polyfit(z, x, 1)[0]
    x_resid = x - slope_xz * z
    
    slope_yz = np.polyfit(z, y, 1)[0]
    y_resid = y - slope_yz * z
    
    return stats.spearmanr(x_resid, y_resid)


def bootstrap_partial_correlation(K, sigma_v, R, n_bootstrap=10000):
    """Bootstrap confidence interval for partial correlation."""
    n = len(K)
    partial_corrs = []
    
    np.random.seed(42)
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        K_boot = K[idx]
        sigma_boot = sigma_v[idx]
        R_boot = R[idx]
        
        # Residualize
        slope_KR = np.polyfit(R_boot, K_boot, 1)[0]
        slope_sR = np.polyfit(R_boot, sigma_boot, 1)[0]
        K_resid = K_boot - slope_KR * R_boot
        s_resid = sigma_boot - slope_sR * R_boot
        
        r, _ = stats.spearmanr(K_resid, s_resid)
        if not np.isnan(r):
            partial_corrs.append(r)
    
    return np.array(partial_corrs)


def main():
    print("Loading SPARC data...")
    K, sigma_v, R = load_sparc_data()
    print(f"Loaded {len(K)} data points")
    
    # Raw correlation
    r_raw, p_raw = stats.spearmanr(K, sigma_v)
    print(f"\nRaw correlation K vs σ_v: r = {r_raw:.3f}, p = {p_raw:.2e}")
    
    # Partial correlation
    r_partial, p_partial = partial_correlation(K, sigma_v, R)
    print(f"Partial correlation K vs σ_v | R: r = {r_partial:.3f}, p = {p_partial:.2e}")
    
    # Bootstrap
    print("\nRunning bootstrap (10,000 iterations)...")
    partial_corrs = bootstrap_partial_correlation(K, sigma_v, R, n_bootstrap=10000)
    
    ci_low = np.percentile(partial_corrs, 2.5)
    ci_high = np.percentile(partial_corrs, 97.5)
    mean_partial = np.mean(partial_corrs)
    
    print(f"\nBootstrap results:")
    print(f"  Mean partial correlation: {mean_partial:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Check Simpson's paradox
    is_simpsons = (np.sign(r_raw) != np.sign(r_partial)) and p_raw < 0.05 and p_partial < 0.05
    print(f"\nSimpson's paradox detected: {is_simpsons}")
    print(f"  Raw: {'+' if r_raw > 0 else '-'}, Partial: {'+' if r_partial > 0 else '-'}")
    
    # Stratified analysis
    print("\nStratified analysis (5 R bins):")
    n_bins = 5
    R_bins = np.percentile(R, np.linspace(0, 100, n_bins + 1))
    n_negative = 0
    
    for i in range(n_bins):
        mask = (R >= R_bins[i]) & (R < R_bins[i+1])
        if np.sum(mask) > 10:
            r_bin, p_bin = stats.spearmanr(K[mask], sigma_v[mask])
            if r_bin < 0:
                n_negative += 1
            print(f"  R ∈ [{R_bins[i]:.1f}, {R_bins[i+1]:.1f}): r = {r_bin:.3f}, n = {np.sum(mask)}")
    
    print(f"\n{n_negative}/5 bins show negative correlation (within-stratum)")
    
    # Save results
    results = {
        'raw_correlation': float(r_raw),
        'raw_p_value': float(p_raw),
        'partial_correlation': float(r_partial),
        'partial_p_value': float(p_partial),
        'bootstrap_mean': float(mean_partial),
        'bootstrap_ci_low': float(ci_low),
        'bootstrap_ci_high': float(ci_high),
        'is_simpsons_paradox': bool(is_simpsons),
        'n_points': len(K)
    }
    
    out_path = Path(__file__).parent / "simpsons_paradox_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
