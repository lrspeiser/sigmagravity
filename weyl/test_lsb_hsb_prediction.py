#!/usr/bin/env python3
"""
Test the Weyl LSB/HSB Prediction on SPARC Data
===============================================

The Weyl derivation makes a specific, falsifiable prediction:
LSB galaxies should show larger effective coherence lengths than HSB galaxies
at fixed R_d (by factor ~1.5×).

This script:
1. Loads SPARC rotation curves and galaxy properties
2. Splits galaxies by surface brightness
3. Fits ℓ₀ independently for each subsample
4. Compares ℓ₀/R_d ratios

If confirmed → strong evidence for Weyl connection
If refuted → constrains or rules out this derivation

Author: Leonard Speiser
Date: December 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize_scalar, minimize
from dataclasses import dataclass
import warnings

# Physical constants
C = 2.998e8           # m/s
G = 6.674e-11         # m³/kg/s²
H0_SI = 2.27e-18      # s⁻¹ (70 km/s/Mpc)
KPC_TO_M = 3.086e19   # m
M_SUN = 1.989e30      # kg

# Derived constants
cH0 = C * H0_SI
g_dagger = cH0 / 6  # Critical acceleration

@dataclass
class GalaxyData:
    """Container for galaxy rotation curve data"""
    name: str
    R_kpc: np.ndarray      # Radii [kpc]
    V_obs: np.ndarray      # Observed velocity [km/s]
    V_bar: np.ndarray      # Baryonic velocity (V_GR) [km/s]
    R_d_kpc: float         # Disk scale length [kpc]
    M_stellar: float       # Stellar mass [M_sun]
    surface_brightness: float  # Central surface brightness proxy


def load_sparc_metadata(repo_root: Path) -> pd.DataFrame:
    """Load SPARC galaxy metadata"""
    combined_path = repo_root / "data" / "sparc" / "sparc_combined.csv"
    true_rdisk_path = repo_root / "data" / "sparc" / "sparc_true_rdisk.csv"
    
    df_combined = pd.read_csv(combined_path)
    df_rdisk = pd.read_csv(true_rdisk_path)
    
    # Merge on galaxy name
    df_rdisk = df_rdisk.rename(columns={"Name": "galaxy_name", "Rdisk": "R_disk_true"})
    df = df_combined.merge(df_rdisk[["galaxy_name", "R_disk_true"]], on="galaxy_name", how="left")
    
    return df


def load_rotation_curve(galaxy: str, sparc_dir: Path) -> Optional[Dict]:
    """Load rotation curve data from JSON"""
    json_path = sparc_dir / f"{galaxy}_capacity_test.json"
    if not json_path.exists():
        return None
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    return data


def compute_surface_brightness(M_stellar: float, R_d_kpc: float) -> float:
    """
    Compute central surface brightness proxy
    
    Σ_0 = M_stellar / (2π R_d²)
    
    Returns log10(Σ_0) in M_sun/kpc²
    """
    R_d_kpc = max(R_d_kpc, 0.1)  # Avoid division issues
    Sigma_0 = M_stellar / (2 * np.pi * R_d_kpc**2)
    return np.log10(max(Sigma_0, 1e-6))


def sigma_gravity_enhancement(r: np.ndarray, g_bar: np.ndarray, 
                              ell_0: float, A: float = np.sqrt(3),
                              n_coh: float = 0.5) -> np.ndarray:
    """
    Compute Σ-Gravity enhancement factor
    
    Σ = 1 + A · W(r) · h(g)
    
    where:
        h(g) = √(g†/g) · g†/(g†+g)
        W(r) = 1 - (ℓ₀/(ℓ₀+r))^n_coh
    """
    g_safe = np.maximum(g_bar, 1e-20)
    r_safe = np.maximum(r, 1e-10)
    
    # h(g) - acceleration function
    sqrt_term = np.sqrt(g_dagger / g_safe)
    gate_term = g_dagger / (g_dagger + g_safe)
    h = sqrt_term * gate_term
    
    # W(r) - coherence window
    W = 1.0 - (ell_0 / (ell_0 + r_safe)) ** n_coh
    
    return 1.0 + A * W * h


def fit_ell0_for_galaxy(galaxy_data: GalaxyData, 
                        A: float = np.sqrt(3),
                        n_coh: float = 0.5) -> Tuple[float, float, float]:
    """
    Fit coherence length ℓ₀ to minimize velocity RMS error
    
    Returns: (best_ell0_kpc, rms_velocity, chi_squared)
    """
    R = galaxy_data.R_kpc * KPC_TO_M  # Convert to meters
    V_obs = galaxy_data.V_obs * 1000   # Convert to m/s
    V_bar = galaxy_data.V_bar * 1000   # Convert to m/s
    
    # Compute baryonic acceleration
    R_safe = np.maximum(R, 1e-10)
    g_bar = V_bar**2 / R_safe
    
    def objective(ell_0_kpc: float) -> float:
        ell_0 = ell_0_kpc * KPC_TO_M
        Sigma = sigma_gravity_enhancement(R, g_bar, ell_0, A, n_coh)
        V_pred = V_bar * np.sqrt(Sigma)
        return np.sqrt(np.mean((V_pred - V_obs)**2))
    
    # Search over reasonable range of ell_0
    result = minimize_scalar(objective, bounds=(0.1, 50.0), method='bounded')
    
    best_ell0 = result.x
    rms = result.fun / 1000  # Convert back to km/s
    
    # Compute chi-squared (assuming 5% velocity errors)
    ell_0 = best_ell0 * KPC_TO_M
    Sigma = sigma_gravity_enhancement(R, g_bar, ell_0, A, n_coh)
    V_pred = V_bar * np.sqrt(Sigma)
    errors = 0.05 * V_obs  # 5% relative error
    chi2 = np.sum(((V_pred - V_obs) / np.maximum(errors, 1))**2) / max(len(V_obs) - 1, 1)
    
    return best_ell0, rms, chi2


def run_lsb_hsb_test(repo_root: Path) -> Dict:
    """
    Main test: Compare ℓ₀/R_d between LSB and HSB galaxies
    """
    print("=" * 70)
    print("WEYL LSB/HSB PREDICTION TEST")
    print("=" * 70)
    print("\nPrediction: LSB galaxies should show ℓ₀/R_d ~ 1.5× larger than HSB")
    print("            due to √(g†/g_char) factor in Weyl derivation\n")
    
    # Load metadata
    df = load_sparc_metadata(repo_root)
    sparc_dir = repo_root / "gravitywavebaseline" / "sparc_results"
    
    # Process each galaxy
    results = []
    
    print("Loading and fitting galaxies...")
    print("-" * 70)
    
    for _, row in df.iterrows():
        galaxy = row["galaxy_name"]
        
        # Load rotation curve
        rc_data = load_rotation_curve(galaxy, sparc_dir)
        if rc_data is None:
            continue
        
        # Extract data
        data_points = rc_data.get("data", [])
        if len(data_points) < 5:
            continue
        
        R_kpc = np.array([p["R"] for p in data_points])
        V_obs = np.array([p["V_obs"] for p in data_points])
        V_bar = np.array([p.get("V_GR", p.get("V_bar", 0)) for p in data_points])
        
        # Filter valid points
        valid = (R_kpc > 0) & (V_bar > 1) & (V_obs > 1)
        if np.sum(valid) < 5:
            continue
        
        R_kpc = R_kpc[valid]
        V_obs = V_obs[valid]
        V_bar = V_bar[valid]
        
        # Get disk scale length (prefer true value)
        R_d = row.get("R_disk_true", row.get("R_disk", 2.0))
        if pd.isna(R_d) or R_d <= 0:
            R_d = row.get("R_disk", 2.0)
        if pd.isna(R_d) or R_d <= 0:
            continue
        
        # Get stellar mass
        M_stellar = row.get("M_stellar", 1e9)
        if pd.isna(M_stellar) or M_stellar <= 0:
            continue
        
        # Compute surface brightness
        sb = compute_surface_brightness(M_stellar, R_d)
        
        # Create galaxy data object
        gal_data = GalaxyData(
            name=galaxy,
            R_kpc=R_kpc,
            V_obs=V_obs,
            V_bar=V_bar,
            R_d_kpc=R_d,
            M_stellar=M_stellar,
            surface_brightness=sb
        )
        
        # Fit ℓ₀
        try:
            ell0, rms, chi2 = fit_ell0_for_galaxy(gal_data)
        except Exception as e:
            print(f"  {galaxy}: fit failed - {e}")
            continue
        
        # Compute characteristic acceleration
        R_char = np.median(R_kpc) * KPC_TO_M
        V_char = np.median(V_bar) * 1000
        g_char = V_char**2 / R_char
        
        # Weyl prediction
        g_ratio = np.sqrt(g_dagger / g_char)
        ell0_weyl_pred = (2/3) * R_d * min(g_ratio, 1.5)
        
        results.append({
            "galaxy": galaxy,
            "R_d_kpc": R_d,
            "M_stellar": M_stellar,
            "surface_brightness": sb,
            "g_char": g_char,
            "ell0_fit": ell0,
            "ell0_over_Rd": ell0 / R_d,
            "ell0_weyl_pred": ell0_weyl_pred,
            "g_ratio": g_ratio,
            "rms_velocity": rms,
            "chi2": chi2,
            "n_points": len(R_kpc)
        })
        
        print(f"  {galaxy:12s}: R_d={R_d:5.2f} kpc, log(Σ)={sb:5.2f}, "
              f"ℓ₀={ell0:5.2f} kpc, ℓ₀/R_d={ell0/R_d:.3f}, RMS={rms:.1f} km/s")
    
    print("-" * 70)
    print(f"Successfully processed {len(results)} galaxies")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Split into LSB and HSB by median surface brightness
    median_sb = df_results["surface_brightness"].median()
    
    lsb_mask = df_results["surface_brightness"] < median_sb
    hsb_mask = df_results["surface_brightness"] >= median_sb
    
    df_lsb = df_results[lsb_mask]
    df_hsb = df_results[hsb_mask]
    
    print(f"\n{'=' * 70}")
    print("RESULTS: LSB vs HSB Comparison")
    print("=" * 70)
    
    print(f"\nSurface brightness split at log(Σ) = {median_sb:.2f} M_sun/kpc²")
    print(f"  LSB sample: {len(df_lsb)} galaxies (lower surface brightness)")
    print(f"  HSB sample: {len(df_hsb)} galaxies (higher surface brightness)")
    
    # Key statistics
    lsb_ell0_Rd = df_lsb["ell0_over_Rd"].values
    hsb_ell0_Rd = df_hsb["ell0_over_Rd"].values
    
    lsb_mean = np.mean(lsb_ell0_Rd)
    lsb_std = np.std(lsb_ell0_Rd) / np.sqrt(len(lsb_ell0_Rd))
    hsb_mean = np.mean(hsb_ell0_Rd)
    hsb_std = np.std(hsb_ell0_Rd) / np.sqrt(len(hsb_ell0_Rd))
    
    ratio = lsb_mean / hsb_mean if hsb_mean > 0 else float('inf')
    
    print(f"\n  ℓ₀/R_d Statistics:")
    print(f"    LSB galaxies: {lsb_mean:.3f} ± {lsb_std:.3f}")
    print(f"    HSB galaxies: {hsb_mean:.3f} ± {hsb_std:.3f}")
    print(f"    Ratio (LSB/HSB): {ratio:.3f}")
    
    # Compare to Weyl prediction
    lsb_g_ratio = np.mean(df_lsb["g_ratio"].values)
    hsb_g_ratio = np.mean(df_hsb["g_ratio"].values)
    predicted_ratio = lsb_g_ratio / hsb_g_ratio
    
    print(f"\n  Weyl Prediction Check:")
    print(f"    LSB mean √(g†/g_char): {lsb_g_ratio:.3f}")
    print(f"    HSB mean √(g†/g_char): {hsb_g_ratio:.3f}")
    print(f"    Predicted LSB/HSB ratio: {predicted_ratio:.3f}")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(lsb_ell0_Rd, hsb_ell0_Rd)
    
    print(f"\n  Statistical Test (two-sample t-test):")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    
    # Verdict
    print(f"\n{'=' * 70}")
    print("VERDICT")
    print("=" * 70)
    
    # Check if ratio is in predicted direction and significant
    weyl_confirmed = (ratio > 1.0) and (p_value < 0.1)
    
    if ratio > 1.2 and p_value < 0.05:
        verdict = "✓ STRONGLY SUPPORTS Weyl prediction"
        print(f"\n  {verdict}")
        print(f"  LSB galaxies show significantly larger ℓ₀/R_d ({ratio:.2f}×)")
        print(f"  This is consistent with the Weyl derivation")
    elif ratio > 1.0 and p_value < 0.1:
        verdict = "✓ WEAKLY SUPPORTS Weyl prediction"
        print(f"\n  {verdict}")
        print(f"  LSB galaxies show larger ℓ₀/R_d ({ratio:.2f}×) but marginally significant")
    elif abs(ratio - 1.0) < 0.2:
        verdict = "? INCONCLUSIVE - no significant difference"
        print(f"\n  {verdict}")
        print(f"  LSB and HSB show similar ℓ₀/R_d ratios ({ratio:.2f}×)")
        print(f"  May indicate the Weyl acceleration dependence is weaker than predicted")
    else:
        verdict = "✗ DOES NOT SUPPORT Weyl prediction"
        print(f"\n  {verdict}")
        print(f"  Observed ratio ({ratio:.2f}×) contradicts prediction")
    
    # Additional analysis: correlation with g_char
    print(f"\n{'=' * 70}")
    print("SUPPLEMENTARY ANALYSIS")
    print("=" * 70)
    
    # Correlation between ℓ₀/R_d and g_char
    corr_g = np.corrcoef(df_results["g_char"], df_results["ell0_over_Rd"])[0, 1]
    corr_sb = np.corrcoef(df_results["surface_brightness"], df_results["ell0_over_Rd"])[0, 1]
    
    print(f"\n  Correlation Analysis:")
    print(f"    ℓ₀/R_d vs g_char: r = {corr_g:.3f}")
    print(f"    ℓ₀/R_d vs log(Σ): r = {corr_sb:.3f}")
    
    if corr_g < -0.2:
        print(f"    → Negative correlation with g_char supports Weyl mechanism")
    elif corr_g > 0.2:
        print(f"    → Positive correlation contradicts Weyl mechanism")
    else:
        print(f"    → Weak correlation, inconclusive for Weyl mechanism")
    
    # Compare fit quality
    lsb_rms = np.mean(df_lsb["rms_velocity"])
    hsb_rms = np.mean(df_hsb["rms_velocity"])
    
    print(f"\n  Fit Quality:")
    print(f"    LSB mean RMS: {lsb_rms:.1f} km/s")
    print(f"    HSB mean RMS: {hsb_rms:.1f} km/s")
    
    # Save results
    output = {
        "summary": {
            "n_galaxies": len(results),
            "n_lsb": len(df_lsb),
            "n_hsb": len(df_hsb),
            "median_surface_brightness": median_sb,
            "lsb_ell0_over_Rd": {"mean": lsb_mean, "std_err": lsb_std},
            "hsb_ell0_over_Rd": {"mean": hsb_mean, "std_err": hsb_std},
            "ratio_lsb_hsb": ratio,
            "predicted_ratio": predicted_ratio,
            "t_statistic": t_stat,
            "p_value": p_value,
            "correlation_g_char": corr_g,
            "correlation_surface_brightness": corr_sb,
            "verdict": verdict
        },
        "galaxies": results
    }
    
    output_path = repo_root / "weyl" / "lsb_hsb_test_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    
    print(f"\n  Results saved to: {output_path}")
    
    return output


def main():
    """Main entry point"""
    repo_root = Path(__file__).resolve().parent.parent
    
    print("\n" + "=" * 70)
    print("TESTING WEYL LSB/HSB PREDICTION ON SPARC DATA")
    print("=" * 70)
    
    results = run_lsb_hsb_test(repo_root)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
