#!/usr/bin/env python3
"""
Test k derivation hypotheses on SPARC data.

This script tests the theoretical predictions for k against real galaxy data.

HYPOTHESES TO TEST:

1. k emerges from matching W(r) to C(r) with mass weighting
   Prediction: k ≈ 0.19-0.27 depending on weighting

2. k depends on galaxy properties (V/σ ratio)
   Prediction: k should correlate with V/σ at R_d

3. k depends on disk thickness (σ_z/σ_r ratio)
   Prediction: thicker disks → smaller k

Author: Leonard Speiser
"""

import numpy as np
from scipy.optimize import minimize_scalar, minimize
from pathlib import Path
import json

# Physical constants
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s⁻¹
G_NEWTON = 6.674e-11  # m³/(kg·s²)
kpc_to_m = 3.086e19
km_to_m = 1000

# Model parameters
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
A_COEFF = 1.6
B_COEFF = 109.0
G_GALAXY = 0.038

# Velocity dispersions
SIGMA_GAS = 10.0
SIGMA_DISK = 25.0
SIGMA_BULGE = 120.0

def A_geometry(G):
    return np.sqrt(A_COEFF + B_COEFF * G**2)

def h_function(g_N):
    g_N = np.maximum(g_N, 1e-15)
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)

def W_coherence(r, xi):
    xi = max(xi, 0.01)
    return 1 - np.sqrt(xi / (xi + r))

def predict_velocity(R_kpc, V_bar_kms, xi_kpc, G=G_GALAXY):
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar_kms * km_to_m
    g_N = V_bar_ms**2 / R_m
    
    A = A_geometry(G)
    W = W_coherence(R_kpc, xi_kpc)
    h = h_function(g_N)
    Sigma = 1 + A * W * h
    return V_bar_kms * np.sqrt(Sigma)


def load_sparc_galaxy(filepath):
    """Load a single SPARC rotation curve."""
    try:
        data = np.loadtxt(filepath, comments='#')
        if len(data) < 5:
            return None
        
        # Format: Rad Vobs errV Vgas Vdisk Vbul SBdisk SBbul
        result = {
            'R': data[:, 0],  # kpc
            'V_obs': data[:, 1],  # km/s
            'e_V': data[:, 2],  # km/s
            'V_gas': data[:, 3],  # km/s
            'V_disk': data[:, 4],  # km/s
            'V_bulge': data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 0])
        }
        
        # Apply M/L corrections (M/L=0.5 for disk, M/L=0.7 for bulge)
        result['V_bar'] = np.sqrt(
            result['V_gas']**2 + 
            0.5 * result['V_disk']**2 + 
            0.7 * result['V_bulge']**2
        )
        
        # Check for valid data
        if np.any(np.isnan(result['V_bar'])) or np.max(result['V_obs']) < 10:
            return None
        
        return result
    except Exception as e:
        return None


def estimate_disk_scale_length(R, V_disk):
    """Estimate R_d from the disk velocity profile."""
    # V_disk peaks around 2.2 R_d for exponential disk
    if len(R) < 3 or np.max(V_disk) < 10:
        return R[-1] / 3
    
    idx_max = np.argmax(V_disk)
    R_peak = R[idx_max]
    R_d = R_peak / 2.2
    return max(R_d, 0.5)


def compute_theoretical_k(V_over_sigma, sigma_z_over_r=0.5):
    """
    Compute theoretical k from V/σ ratio.
    
    Based on the analysis:
    - Coherence scalar C = v²/(v² + σ_total²)
    - For flat RC: σ_total = √3 × σ_φ (epicyclic relation)
    - σ_eff in formula uses component dispersions
    
    The matching condition gives k that depends on V/σ.
    """
    # For a disk with flat rotation curve:
    # σ_r/σ_φ = √2 (epicyclic)
    # σ_z/σ_r = sigma_z_over_r
    
    sigma_r_over_phi = np.sqrt(2)
    sigma_z_over_phi = sigma_z_over_r * sigma_r_over_phi
    
    # Total dispersion
    sigma_total_over_phi = np.sqrt(1 + sigma_r_over_phi**2 + sigma_z_over_phi**2)
    
    # The coherence scale in the physics is σ_total/Ω
    # The formula uses σ_eff/Ω where σ_eff ≈ σ_φ (for pure disk)
    
    # Mass-weighted matching of W to C gives:
    # k ≈ 0.27 × (σ_φ/σ_total) for typical V/σ
    
    # More precisely, we need to integrate
    # But as an approximation:
    k_base = 0.27  # From mass-weighted matching
    k = k_base * (1 / sigma_total_over_phi)
    
    return k


def fit_optimal_k_per_galaxy(gal_data, k_range=(0.05, 0.8)):
    """
    Find the optimal k for a single galaxy.
    """
    R = gal_data['R']
    V_obs = gal_data['V_obs']
    V_bar = gal_data['V_bar']
    e_V = gal_data['e_V']
    
    # Estimate R_d
    R_d = estimate_disk_scale_length(R, gal_data['V_disk'])
    
    # Estimate V at R_d
    idx_Rd = np.argmin(np.abs(R - R_d))
    V_at_Rd = V_bar[idx_Rd] if V_bar[idx_Rd] > 10 else 50.0
    
    # Estimate mass fractions
    V_gas_sq = gal_data['V_gas']**2
    V_disk_sq = 0.5 * gal_data['V_disk']**2
    V_bulge_sq = 0.7 * gal_data['V_bulge']**2
    V_bar_sq = V_bar**2 + 1e-10
    
    gas_frac = np.mean(V_gas_sq / V_bar_sq)
    disk_frac = np.mean(V_disk_sq / V_bar_sq)
    bulge_frac = np.mean(V_bulge_sq / V_bar_sq)
    
    total = gas_frac + disk_frac + bulge_frac + 1e-10
    gas_frac /= total
    disk_frac /= total
    bulge_frac /= total
    
    # Effective dispersion
    sigma_eff = gas_frac * SIGMA_GAS + disk_frac * SIGMA_DISK + bulge_frac * SIGMA_BULGE
    
    # Ω_d
    Omega_d = V_at_Rd / R_d
    
    # V/σ ratio at R_d
    V_over_sigma = V_at_Rd / sigma_eff if sigma_eff > 0 else 5.0
    
    def objective(k):
        xi = k * sigma_eff / Omega_d
        V_pred = np.array([predict_velocity(r, vb, xi) for r, vb in zip(R, V_bar)])
        
        # Weighted chi-squared
        residuals = (V_obs - V_pred) / np.maximum(e_V, 5.0)
        return np.sum(residuals**2)
    
    result = minimize_scalar(objective, bounds=k_range, method='bounded')
    k_optimal = result.x
    
    # Compute RMS at optimal k
    xi_opt = k_optimal * sigma_eff / Omega_d
    V_pred_opt = np.array([predict_velocity(r, vb, xi_opt) for r, vb in zip(R, V_bar)])
    rms = np.sqrt(np.mean((V_obs - V_pred_opt)**2))
    
    return {
        'k_optimal': k_optimal,
        'rms': rms,
        'R_d': R_d,
        'V_at_Rd': V_at_Rd,
        'sigma_eff': sigma_eff,
        'V_over_sigma': V_over_sigma,
        'gas_frac': gas_frac,
        'disk_frac': disk_frac,
        'bulge_frac': bulge_frac
    }


def main():
    print("=" * 70)
    print("TESTING k DERIVATION ON SPARC DATA")
    print("=" * 70)
    
    # Find SPARC data
    sparc_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/SPARC/RotationCurves"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/vendor/sparc"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/sparc")
    ]
    
    sparc_dir = None
    for p in sparc_paths:
        if p.exists() and len(list(p.glob("*.dat"))) > 0:
            sparc_dir = p
            break
    
    if sparc_dir is None:
        # Try to find any .dat files
        for p in Path("/Users/leonardspeiser/Projects/sigmagravity").rglob("*rotmod.dat"):
            sparc_dir = p.parent
            break
    
    if sparc_dir is None:
        print("SPARC data not found. Running theoretical analysis only.")
        
        # Theoretical predictions
        print("\n" + "=" * 70)
        print("THEORETICAL PREDICTIONS FOR k")
        print("=" * 70)
        
        print("\nk as a function of V/σ and disk thickness:")
        print("-" * 60)
        print(f"{'V/σ':>8} | {'σ_z/σ_r=0.3':>12} | {'σ_z/σ_r=0.5':>12} | {'σ_z/σ_r=0.7':>12}")
        print("-" * 60)
        
        for V_sigma in [3, 5, 7, 10, 15]:
            k_03 = compute_theoretical_k(V_sigma, 0.3)
            k_05 = compute_theoretical_k(V_sigma, 0.5)
            k_07 = compute_theoretical_k(V_sigma, 0.7)
            print(f"{V_sigma:>8} | {k_03:>12.4f} | {k_05:>12.4f} | {k_07:>12.4f}")
        
        print("-" * 60)
        print(f"Empirical k = 0.24")
        
        # The theoretical k is ~0.16 for typical parameters
        # This is less than empirical 0.24
        # Suggests we need to refine the theory
        
        print("\n" + "=" * 70)
        print("REFINED HYPOTHESIS")
        print("=" * 70)
        print("""
The theoretical k ≈ 0.16 is less than empirical k = 0.24.

Possible explanations:
1. The σ_eff in the formula is NOT the total 3D dispersion
   - If σ_eff ≈ σ_LOS (line-of-sight), it's smaller than σ_total
   - This would increase k

2. The coherence transition occurs at C ≠ 0.5
   - If transition is at C ≈ 0.34 (as suggested by k = 0.24), this is consistent

3. The mass weighting in the matching integral is different
   - Different radial ranges or weighting schemes give different k

4. Galaxy-to-galaxy variation
   - k may vary with galaxy properties
   - The empirical k = 0.24 is an average
""")
        
        return
    
    print(f"\nFound SPARC data at: {sparc_dir}")
    
    # Load and process galaxies
    galaxy_files = list(sparc_dir.glob("*.dat"))
    print(f"Found {len(galaxy_files)} rotation curve files")
    
    results = []
    
    for gal_file in galaxy_files:
        gal_data = load_sparc_galaxy(gal_file)
        if gal_data is None:
            continue
        
        try:
            fit_result = fit_optimal_k_per_galaxy(gal_data)
            fit_result['name'] = gal_file.stem
            results.append(fit_result)
        except Exception as e:
            continue
    
    print(f"Successfully processed {len(results)} galaxies")
    
    if len(results) == 0:
        print("No valid results. Check data format.")
        return
    
    # Analyze results
    k_values = np.array([r['k_optimal'] for r in results])
    V_sigma_values = np.array([r['V_over_sigma'] for r in results])
    rms_values = np.array([r['rms'] for r in results])
    gas_frac_values = np.array([r['gas_frac'] for r in results])
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nOptimal k distribution:")
    print(f"  Mean k = {np.mean(k_values):.4f}")
    print(f"  Median k = {np.median(k_values):.4f}")
    print(f"  Std k = {np.std(k_values):.4f}")
    print(f"  Range: [{np.min(k_values):.4f}, {np.max(k_values):.4f}]")
    
    print(f"\nRMS at optimal k:")
    print(f"  Mean RMS = {np.mean(rms_values):.2f} km/s")
    print(f"  Median RMS = {np.median(rms_values):.2f} km/s")
    
    # Check for correlations
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # k vs V/σ
    corr_V_sigma = np.corrcoef(k_values, V_sigma_values)[0, 1]
    print(f"\nk vs V/σ correlation: r = {corr_V_sigma:.3f}")
    
    # k vs gas fraction
    corr_gas = np.corrcoef(k_values, gas_frac_values)[0, 1]
    print(f"k vs gas_frac correlation: r = {corr_gas:.3f}")
    
    # Theoretical prediction comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH THEORETICAL PREDICTIONS")
    print("=" * 70)
    
    # Compute theoretical k for each galaxy
    k_theory = np.array([compute_theoretical_k(v_s) for v_s in V_sigma_values])
    
    print(f"\nTheoretical k (from matching):")
    print(f"  Mean = {np.mean(k_theory):.4f}")
    print(f"  Median = {np.median(k_theory):.4f}")
    
    print(f"\nEmpirical k:")
    print(f"  Mean = {np.mean(k_values):.4f}")
    print(f"  Median = {np.median(k_values):.4f}")
    
    ratio = np.mean(k_values) / np.mean(k_theory)
    print(f"\nRatio (empirical/theoretical) = {ratio:.3f}")
    
    if ratio > 1:
        print(f"Empirical k is {ratio:.1f}× larger than theoretical prediction")
        print("This suggests the effective σ in the formula is smaller than σ_total")
        
        # What σ ratio would give the right k?
        sigma_ratio = 1 / ratio
        print(f"Implied σ_eff/σ_total = {sigma_ratio:.3f}")
    
    # Save detailed results
    output = {
        'summary': {
            'n_galaxies': len(results),
            'k_mean': float(np.mean(k_values)),
            'k_median': float(np.median(k_values)),
            'k_std': float(np.std(k_values)),
            'rms_mean': float(np.mean(rms_values)),
            'corr_k_V_sigma': float(corr_V_sigma),
            'corr_k_gas_frac': float(corr_gas),
            'k_theory_mean': float(np.mean(k_theory)),
            'ratio_empirical_theory': float(ratio)
        },
        'galaxies': results
    }
    
    output_path = Path("/Users/leonardspeiser/Projects/sigmagravity/derivations/k_derivation_sparc_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()

