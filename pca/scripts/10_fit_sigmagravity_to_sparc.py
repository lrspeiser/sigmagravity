#!/usr/bin/env python3
"""
Fit Sigma-Gravity model to SPARC galaxies for PCA comparison.

Uses simplified fitting approach with fixed hyperparameters from calibration:
- A = 0.6 (galaxy amplitude)
- ℓ₀ = 5 kpc (coherence scale)
- p = 2.0 (coherence shape)
- n_coh = 1.5 (coherence exponent)

For each galaxy, we compute the model prediction and residuals.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sigma-Gravity kernel (Burr-XII coherence function)
def coherence_function(R, l0=5.0, p=2.0, n_coh=1.5):
    """
    Burr-XII coherence function: C(R) = 1 - [1 + (R/ℓ₀)^p]^{-n_coh}
    
    Parameters:
    -----------
    R : float or array
        Radius in kpc
    l0 : float
        Coherence scale in kpc (default 5.0)
    p : float
        Shape parameter (default 2.0)
    n_coh : float
        Coherence exponent (default 1.5)
    """
    x = (R / l0)**p
    return 1.0 - (1.0 + x)**(-n_coh)

def sigma_gravity_boost(R, A=0.6, l0=5.0, p=2.0, n_coh=1.5):
    """
    Sigma-Gravity boost factor: K(R) = A · C(R/ℓ₀)
    
    The modified acceleration is: g_eff = g_bar * (1 + K)
    """
    C = coherence_function(R, l0, p, n_coh)
    return A * C

def compute_ring_acceleration(R_eval, R_rings, Sigma_rings, dR):
    """
    Compute gravitational acceleration from ring mass distribution.
    
    Uses elliptic integral approximation for speed.
    
    Parameters:
    -----------
    R_eval : array
        Radii where to evaluate acceleration (kpc)
    R_rings : array  
        Ring radii (kpc)
    Sigma_rings : array
        Surface density of each ring (M_sun/kpc^2)
    dR : float
        Ring width (kpc)
    """
    G = 4.30091e-6  # kpc (km/s)^2 / M_sun
    
    g_acc = np.zeros_like(R_eval)
    
    for i, R in enumerate(R_eval):
        if R < 0.01:  # Too close to center
            continue
            
        for j, R_ring in enumerate(R_rings):
            M_ring = 2 * np.pi * R_ring * Sigma_rings[j] * dR
            
            # Distance
            if R_ring < 0.01:
                r_dist = R
            else:
                # Simplified: treat as point mass at R_ring for R > R_ring
                # or as enclosed mass for R < R_ring
                if R > R_ring:
                    r_dist = np.sqrt((R - R_ring)**2 + 0.01**2)  # small softening
                else:
                    r_dist = R_ring
            
            # Acceleration contribution
            g_acc[i] += G * M_ring / r_dist**2 if r_dist > 0 else 0
    
    return g_acc

def fit_sparc_galaxy(curve_data, meta_row, A=0.6, l0=5.0, p=2.0, n_coh=1.5):
    """
    Fit Sigma-Gravity to a single SPARC galaxy.
    
    Parameters:
    -----------
    curve_data : DataFrame
        Columns: R_kpc, V_obs, eV_obs, V_disk, V_gas, V_bul
    meta_row : Series
        Metadata: Rd, Vf, Mbar, etc.
    
    Returns:
    --------
    dict with fit results
    """
    # Extract data
    R = curve_data['R_kpc'].values
    V_obs = curve_data['V_obs'].values
    eV_obs = curve_data['eV_obs'].values
    
    # Get baryonic components
    V_disk = curve_data.get('V_disk', np.zeros_like(R)).values
    V_gas = curve_data.get('V_gas', np.zeros_like(R)).values
    V_bul = curve_data.get('V_bul', np.zeros_like(R)).values if 'V_bul' in curve_data else np.zeros_like(R)
    
    # Baryonic circular velocity
    V_bar = np.sqrt(V_disk**2 + V_gas**2 + V_bul**2)
    
    # Baryonic acceleration (g = V^2/R)
    g_bar = V_bar**2 / np.maximum(R, 0.1) / 3.086e16  # kpc/Myr^2
    
    # Sigma-Gravity boost
    K = sigma_gravity_boost(R, A, l0, p, n_coh)
    
    # Modified acceleration
    g_model = g_bar * (1 + K)
    
    # Convert back to velocity
    V_model = np.sqrt(g_model * R * 3.086e16)  # km/s (approximate)
    
    # Compute residuals
    residuals = V_obs - V_model
    weighted_residuals = residuals / np.maximum(eV_obs, 1.0)
    
    # Metrics
    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum(weighted_residuals**2)
    chi2_red = chi2 / max(len(R) - 4, 1)  # 4 fixed parameters
    
    # Fractional error
    ape = np.mean(np.abs(residuals / V_obs)) * 100  # percent
    
    return {
        'rms': rms,
        'chi2': chi2,
        'chi2_red': chi2_red,
        'ape': ape,
        'n_points': len(R),
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'max_residual': np.max(np.abs(residuals))
    }

def main():
    """
    Fit Sigma-Gravity to all SPARC galaxies in the PCA dataset.
    """
    print("=" * 70)
    print("SIGMA-GRAVITY FITTING FOR PCA COMPARISON")
    print("=" * 70)
    
    # Paths
    repo_root = Path(__file__).parent.parent.parent
    curves_dir = repo_root / 'pca' / 'data' / 'raw' / 'sparc_curves'
    meta_file = repo_root / 'pca' / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'
    output_dir = repo_root / 'pca' / 'outputs' / 'sigmagravity_fits'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv(meta_file)
    print(f"\nLoaded metadata for {len(meta)} galaxies")
    
    # Fixed hyperparameters (from calibration)
    A = 0.6      # Galaxy amplitude
    l0 = 5.0     # Coherence scale (kpc)
    p = 2.0      # Shape parameter
    n_coh = 1.5  # Coherence exponent
    
    print(f"\nFixed hyperparameters:")
    print(f"  A = {A}")
    print(f"  l0 = {l0} kpc")
    print(f"  p = {p}")
    print(f"  n_coh = {n_coh}")
    
    # Fit each galaxy
    results = []
    print(f"\nFitting Sigma-Gravity to SPARC galaxies...")
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Fitting"):
        name = row['name']
        curve_file = curves_dir / f"{name}.csv"
        
        if not curve_file.exists():
            continue
        
        try:
            # Load curve
            curve = pd.read_csv(curve_file)
            
            # Fit
            fit_result = fit_sparc_galaxy(curve, row, A, l0, p, n_coh)
            
            # Store results
            results.append({
                'name': name,
                'Rd': row.get('Rd', np.nan),
                'Vf': row.get('Vf', np.nan),
                'Mbar': row.get('Mbar', np.nan),
                'Sigma0': row.get('Sigma0', np.nan),
                'HSB_LSB': row.get('HSB_LSB', 'Unknown'),
                'residual_rms': fit_result['rms'],
                'chi2': fit_result['chi2'],
                'chi2_red': fit_result['chi2_red'],
                'ape': fit_result['ape'],
                'n_points': fit_result['n_points'],
                'mean_residual': fit_result['mean_residual'],
                'std_residual': fit_result['std_residual'],
                'max_residual': fit_result['max_residual'],
                'A': A,
                'l0': l0,
                'p': p,
                'n_coh': n_coh
            })
            
        except Exception as e:
            print(f"\nWarning: Failed to fit {name}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = output_dir / 'sparc_sigmagravity_fits.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{' ' * 70}")
    print("=" * 70)
    print("FITTING COMPLETE")
    print("=" * 70)
    print(f"\nFitted {len(results)} galaxies successfully")
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    print(f"\nFit Quality Summary:")
    print(f"  Mean RMS residual: {results_df['residual_rms'].mean():.2f} km/s")
    print(f"  Median RMS residual: {results_df['residual_rms'].median():.2f} km/s")
    print(f"  Mean chi2_red: {results_df['chi2_red'].mean():.2f}")
    print(f"  Median chi2_red: {results_df['chi2_red'].median():.2f}")
    print(f"  Mean APE: {results_df['ape'].mean():.1f}%")
    
    print(f"\nReady for PCA comparison!")
    print(f"Run: python pca/scripts/08_compare_models.py")

if __name__ == '__main__':
    main()

