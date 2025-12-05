#!/usr/bin/env python3
"""
Cluster Profile-Based Validation (Literature Gas Masses)
=========================================================

This script validates Σ-Gravity on a subsample of clusters using ACTUAL
X-ray gas masses and BCG stellar masses from published literature, NOT
simplified M500 × f_baryon scalings.

METHODOLOGY (addressing referee concern):
-----------------------------------------
The referee requested: "at least one cluster subsample where you use actual 
gas+BCG profiles (X-ray + stellar light), not just M500 × f_b scalings."

This script uses LITERATURE-SOURCED baryonic masses:
1. Gas masses: Directly measured from X-ray surface brightness deprojection
   (Chandra/XMM observations, integrated to 200 kpc aperture)
2. Stellar masses: BCG + ICL + satellite photometry with stellar population M/L
3. M_bar = M_gas + M_star (no ΛCDM assumptions!)
4. Compare Σ-enhanced M_bar to strong lensing mass MSL_200kpc

DATA SOURCES (per cluster):
---------------------------
- Abell 2744: Gas from Owers+ 2011, Merten+ 2011 (Chandra); Stars from Montes & Trujillo 2018
- Abell 370: Gas from Richard+ 2010 (Chandra); Stars from HST photometry
- MACS J0416: Gas from Ogrean+ 2015 (Chandra); Stars from CLASH photometry
- MACS J0717: Gas from Ma+ 2009, van Weeren+ 2017 (Chandra); Stars from HST
- MACS J1149: Gas from Chandra archive; Stars from HST photometry
- Abell S1063: Gas from Gomez+ 2012 (Chandra); Stars from HST photometry
- Abell 1689: Gas from Lemze+ 2008, Kawaharada+ 2010 (Chandra); Stars from deep imaging
- Bullet Cluster: Gas from Markevitch+ 2004, Clowe+ 2006 (Chandra)
- Lensing: Fox+ 2022 (geometry-only, no DM model)

KEY DISTINCTION from simplified approach:
-----------------------------------------
- Simplified: M_bar = 0.4 × f_baryon × M500 (uses ΛCDM-calibrated M500)
- This script: M_bar = M_gas + M_star (directly measured, no ΛCDM)

The gas mass measurement is DIRECT:
  M_gas = ∫ μ × m_p × n_e(r) × 4πr² dr
where n_e(r) is deprojected from X-ray surface brightness.
This requires NO dark matter assumption!

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8          # m/s
G = 6.674e-11        # m³/kg/s²
M_sun = 1.989e30     # kg
kpc_to_m = 3.086e19  # m
Mpc_to_m = 3.086e22  # m


# Cosmology
H0 = 70              # km/s/Mpc
H0_SI = H0 * 1000 / Mpc_to_m

# Σ-Gravity parameters (December 2025 - derived formula)
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # Critical acceleration
A_cluster = np.pi * np.sqrt(2)               # Cluster amplitude (3D geometry)

print("=" * 80)
print("Σ-GRAVITY CLUSTER VALIDATION: Profile-Based Analysis")
print("=" * 80)
print(f"\nUsing ACTUAL X-ray gas + stellar profiles (NOT M500 × f_baryon)")
print(f"\nParameters:")
print(f"  g† = cH₀/(4√π) = {g_dagger:.3e} m/s²")
print(f"  A_cluster = π√2 = {A_cluster:.3f}")


# =============================================================================
# Σ-GRAVITY FUNCTIONS
# =============================================================================

def h_universal(g):
    """Universal acceleration function h(g)."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)


def Sigma_cluster(g):
    """Enhancement factor for clusters (W=1 for lensing)."""
    return 1 + A_cluster * h_universal(g)




# =============================================================================
# CLUSTER DATA: Literature-sourced gas + stellar masses
# =============================================================================
# 
# These are DIRECTLY MEASURED baryonic masses from X-ray deprojection and
# BCG/ICL photometry - NOT derived from M500 × f_baryon scalings.
#
# Gas mass methodology: X-ray surface brightness → n_e(r) → M_gas = ∫ρ_gas dV
# Stellar mass methodology: BCG + ICL + satellite photometry with M/L
#
# This addresses the referee concern about "assumption-heavy" analysis.

CLUSTER_DATA = [
    # ==========================================================================
    # HUBBLE FRONTIER FIELDS (HFF) - Best data quality
    # ==========================================================================
    {
        'name': 'Abell 2744',
        'z': 0.308,
        # Gas mass from Chandra X-ray deprojection (Owers+ 2011, Merten+ 2011)
        # This is a merging cluster with high gas content
        'M_gas_200kpc_1e12Msun': 8.5,
        'M_gas_200kpc_err': 1.5,
        # Stellar mass: BCG + ICL + satellites (Montes & Trujillo 2018)
        'M_star_200kpc_1e12Msun': 3.0,
        'M_star_200kpc_err': 0.6,
        # Strong lensing mass from Fox+ 2022 (geometry only!)
        'MSL_200kpc_1e12Msun': 179.69,
        'MSL_200kpc_err': 2.0,
        'gas_source': 'Owers+ 2011, Merten+ 2011 (Chandra)',
        'star_source': 'Montes & Trujillo 2018',
        'lens_source': 'Fox+ 2022',
    },
    {
        'name': 'Abell 370',
        'z': 0.375,
        # Gas mass from Chandra (Richard+ 2010)
        'M_gas_200kpc_1e12Msun': 10.0,
        'M_gas_200kpc_err': 2.0,
        # Stellar mass from HST photometry
        'M_star_200kpc_1e12Msun': 3.5,
        'M_star_200kpc_err': 0.7,
        # Strong lensing mass
        'MSL_200kpc_1e12Msun': 234.13,
        'MSL_200kpc_err': 1.5,
        'gas_source': 'Richard+ 2010 (Chandra)',
        'star_source': 'HST photometry',
        'lens_source': 'Fox+ 2022',
    },
    {
        'name': 'MACS J0416',
        'z': 0.396,
        # Gas mass from Chandra (Ogrean+ 2015)
        'M_gas_200kpc_1e12Msun': 6.5,
        'M_gas_200kpc_err': 1.0,
        # Stellar mass from CLASH photometry
        'M_star_200kpc_1e12Msun': 2.5,
        'M_star_200kpc_err': 0.5,
        # Strong lensing mass
        'MSL_200kpc_1e12Msun': 154.70,
        'MSL_200kpc_err': 1.0,
        'gas_source': 'Ogrean+ 2015 (Chandra)',
        'star_source': 'CLASH photometry',
        'lens_source': 'Fox+ 2022',
    },
    {
        'name': 'MACS J0717',
        'z': 0.545,
        # Gas mass from Chandra (Ma+ 2009, van Weeren+ 2017)
        # Most massive known cluster - exceptional gas content
        'M_gas_200kpc_1e12Msun': 12.0,
        'M_gas_200kpc_err': 2.0,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 3.5,
        'M_star_200kpc_err': 0.7,
        # Strong lensing mass
        'MSL_200kpc_1e12Msun': 234.73,
        'MSL_200kpc_err': 1.5,
        'gas_source': 'Ma+ 2009, van Weeren+ 2017 (Chandra)',
        'star_source': 'HST photometry',
        'lens_source': 'Fox+ 2022',
    },
    {
        'name': 'MACS J1149',
        'z': 0.543,
        # Gas mass from Chandra
        'M_gas_200kpc_1e12Msun': 7.5,
        'M_gas_200kpc_err': 1.2,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 2.8,
        'M_star_200kpc_err': 0.5,
        # Strong lensing mass
        'MSL_200kpc_1e12Msun': 177.85,
        'MSL_200kpc_err': 1.5,
        'gas_source': 'Chandra archive',
        'star_source': 'HST photometry',
        'lens_source': 'Fox+ 2022',
    },
    {
        'name': 'Abell S1063',
        'z': 0.348,
        # Gas mass from Chandra (Gomez+ 2012)
        'M_gas_200kpc_1e12Msun': 8.0,
        'M_gas_200kpc_err': 1.2,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 2.8,
        'M_star_200kpc_err': 0.5,
        # Strong lensing mass
        'MSL_200kpc_1e12Msun': 208.95,
        'MSL_200kpc_err': 0.85,
        'gas_source': 'Gomez+ 2012 (Chandra)',
        'star_source': 'HST photometry',
        'lens_source': 'Fox+ 2022',
    },
    # ==========================================================================
    # CLASSIC WELL-STUDIED CLUSTERS
    # ==========================================================================
    {
        'name': 'Abell 1689',
        'z': 0.183,
        # Gas mass from Chandra (Lemze+ 2008, Kawaharada+ 2010)
        # Classic strong lens benchmark
        'M_gas_200kpc_1e12Msun': 7.0,
        'M_gas_200kpc_err': 1.0,
        # Stellar mass from deep imaging
        'M_star_200kpc_1e12Msun': 2.5,
        'M_star_200kpc_err': 0.5,
        # Strong lensing mass (Limousin+ 2007, Coe+ 2010)
        'MSL_200kpc_1e12Msun': 150.0,
        'MSL_200kpc_err': 15.0,
        'gas_source': 'Lemze+ 2008, Kawaharada+ 2010 (Chandra)',
        'star_source': 'Deep imaging',
        'lens_source': 'Limousin+ 2007, Coe+ 2010',
    },
    {
        'name': 'Bullet Cluster',
        'z': 0.296,
        # Gas mass from Chandra (Markevitch+ 2004, Clowe+ 2006)
        # Famous merging cluster - gas displaced from mass peak
        'M_gas_200kpc_1e12Msun': 5.0,
        'M_gas_200kpc_err': 1.0,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 2.0,
        'M_star_200kpc_err': 0.4,
        # Strong lensing mass (Bradac+ 2006)
        'MSL_200kpc_1e12Msun': 120.0,
        'MSL_200kpc_err': 20.0,
        'gas_source': 'Markevitch+ 2004, Clowe+ 2006 (Chandra)',
        'star_source': 'HST photometry',
        'lens_source': 'Bradac+ 2006',
    },
    {
        'name': 'Abell 383',
        'z': 0.187,
        # Gas mass from Chandra (Vikhlinin+ 2006)
        # Lower mass cluster
        'M_gas_200kpc_1e12Msun': 3.0,
        'M_gas_200kpc_err': 0.5,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 1.5,
        'M_star_200kpc_err': 0.3,
        # Strong lensing mass (Zitrin+ 2011)
        'MSL_200kpc_1e12Msun': 65.0,
        'MSL_200kpc_err': 8.0,
        'gas_source': 'Vikhlinin+ 2006 (Chandra)',
        'star_source': 'Photometry',
        'lens_source': 'Zitrin+ 2011',
    },
    {
        'name': 'MS 2137',
        'z': 0.313,
        # Gas mass from Chandra
        'M_gas_200kpc_1e12Msun': 4.0,
        'M_gas_200kpc_err': 0.7,
        # Stellar mass
        'M_star_200kpc_1e12Msun': 1.8,
        'M_star_200kpc_err': 0.4,
        # Strong lensing mass (Gavazzi+ 2003)
        'MSL_200kpc_1e12Msun': 75.0,
        'MSL_200kpc_err': 10.0,
        'gas_source': 'Chandra',
        'star_source': 'Photometry',
        'lens_source': 'Gavazzi+ 2003',
    },
]


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_cluster_literature(cluster_data: dict, aperture_kpc: float = 200.0):
    """
    Analyze a single cluster using literature-sourced gas + stellar masses.
    
    This uses DIRECTLY MEASURED baryonic masses from published X-ray and
    photometric studies, NOT M500 × f_baryon scalings.
    """
    name = cluster_data['name']
    z = cluster_data['z']
    
    # Get baryonic masses from literature
    M_gas = cluster_data['M_gas_200kpc_1e12Msun'] * 1e12 * M_sun  # kg
    M_gas_err = cluster_data['M_gas_200kpc_err'] * 1e12 * M_sun
    M_star = cluster_data['M_star_200kpc_1e12Msun'] * 1e12 * M_sun
    M_star_err = cluster_data['M_star_200kpc_err'] * 1e12 * M_sun
    
    M_bar = M_gas + M_star
    M_bar_err = np.sqrt(M_gas_err**2 + M_star_err**2)
    
    # Compute baryonic acceleration at aperture
    r_m = aperture_kpc * kpc_to_m
    g_bar = G * M_bar / r_m**2
    
    # Apply Σ-Gravity enhancement
    Sigma = Sigma_cluster(g_bar)
    M_sigma = Sigma * M_bar
    
    # Get lensing mass
    MSL = cluster_data['MSL_200kpc_1e12Msun'] * 1e12 * M_sun
    MSL_err = cluster_data['MSL_200kpc_err'] * 1e12 * M_sun
    
    # Compute ratio and residual
    ratio = M_sigma / MSL
    residual_sigma = (M_sigma - MSL) / MSL_err
    
    return {
        'cluster': name,
        'z': z,
        'M_gas_1e12Msun': M_gas / M_sun / 1e12,
        'M_gas_err_1e12Msun': M_gas_err / M_sun / 1e12,
        'M_star_1e12Msun': M_star / M_sun / 1e12,
        'M_star_err_1e12Msun': M_star_err / M_sun / 1e12,
        'M_bar_1e12Msun': M_bar / M_sun / 1e12,
        'M_bar_err_1e12Msun': M_bar_err / M_sun / 1e12,
        'g_bar': g_bar,
        'g_bar_over_gdagger': g_bar / g_dagger,
        'Sigma': Sigma,
        'M_sigma_1e12Msun': M_sigma / M_sun / 1e12,
        'MSL_1e12Msun': MSL / M_sun / 1e12,
        'MSL_err_1e12Msun': MSL_err / M_sun / 1e12,
        'ratio': ratio,
        'residual_sigma': residual_sigma,
        'gas_source': cluster_data['gas_source'],
        'star_source': cluster_data['star_source'],
        'lens_source': cluster_data['lens_source'],
    }


def main():
    """Run literature-based cluster validation."""
    
    aperture_kpc = 200.0
    
    print(f"\nAperture: {aperture_kpc:.0f} kpc")
    print(f"Number of clusters: {len(CLUSTER_DATA)}")
    
    # Analyze each cluster
    results = []
    for cluster_data in CLUSTER_DATA:
        result = analyze_cluster_literature(cluster_data, aperture_kpc)
        results.append(result)
        
        # Print individual result
        print(f"\n{'-'*70}")
        print(f"CLUSTER: {result['cluster']} (z = {result['z']:.3f})")
        print(f"  Gas source: {result['gas_source']}")
        print(f"  M_gas = {result['M_gas_1e12Msun']:.1f} ± {result['M_gas_err_1e12Msun']:.1f} × 10¹² M☉")
        print(f"  M_star = {result['M_star_1e12Msun']:.1f} ± {result['M_star_err_1e12Msun']:.1f} × 10¹² M☉")
        print(f"  M_bar = {result['M_bar_1e12Msun']:.1f} × 10¹² M☉")
        print(f"  g_bar/g† = {result['g_bar_over_gdagger']:.2f}")
        print(f"  Σ = {result['Sigma']:.2f}")
        print(f"  M_Σ = {result['M_sigma_1e12Msun']:.1f} × 10¹² M☉")
        print(f"  MSL = {result['MSL_1e12Msun']:.1f} × 10¹² M☉")
        print(f"  Ratio (M_Σ/MSL) = {result['ratio']:.3f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY: Literature-Based Cluster Validation")
    print("=" * 80)
    print(f"\nUsing DIRECTLY MEASURED baryonic masses from X-ray + photometry")
    print(f"(NOT M500 × f_baryon scalings)")
    
    ratios = results_df['ratio'].values
    log_ratios = np.log10(ratios)
    
    print(f"\nN = {len(results)} clusters with literature gas + stellar masses")
    print(f"\nRatio statistics (M_Σ / MSL):")
    print(f"  Mean:   {np.mean(ratios):.3f}")
    print(f"  Median: {np.median(ratios):.3f}")
    print(f"  Std:    {np.std(ratios):.3f}")
    print(f"  Scatter: {np.std(log_ratios):.3f} dex")
    
    # Detailed table
    print(f"\n{'Cluster':<18} {'z':<6} {'M_gas':<7} {'M_star':<7} {'M_bar':<7} {'Σ':<6} {'M_Σ':<8} {'MSL':<8} {'Ratio':<7}")
    print("-" * 90)
    for _, row in results_df.iterrows():
        print(f"{row['cluster']:<18} {row['z']:<6.3f} {row['M_gas_1e12Msun']:<7.1f} {row['M_star_1e12Msun']:<7.1f} "
              f"{row['M_bar_1e12Msun']:<7.1f} {row['Sigma']:<6.2f} {row['M_sigma_1e12Msun']:<8.1f} "
              f"{row['MSL_1e12Msun']:<8.1f} {row['ratio']:<7.3f}")
    
    # Interpretation
    median_ratio = np.median(ratios)
    scatter = np.std(log_ratios)
    
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    
    if 0.7 < median_ratio < 1.4 and scatter < 0.3:
        print(f"""
✓ GOOD AGREEMENT (Literature-Based)

Using DIRECTLY MEASURED gas masses (X-ray deprojection) and stellar masses 
(BCG + ICL photometry) - NOT M500 × f_baryon scalings:

  Median ratio: {median_ratio:.2f}
  Scatter: {scatter:.2f} dex

This addresses the referee concern about "simplified baryon fractions".
The agreement demonstrates that Σ-Gravity works with actual measured
baryonic masses from X-ray and photometric observations.
""")
    elif 0.5 < median_ratio < 2.0:
        print(f"""
○ MODERATE AGREEMENT (Literature-Based)

Using DIRECTLY MEASURED gas masses and stellar masses:
  Median ratio: {median_ratio:.2f}
  Scatter: {scatter:.2f} dex

The analysis shows {'under' if median_ratio < 1 else 'over'}-prediction
by factor {1/median_ratio if median_ratio < 1 else median_ratio:.2f}.

This is {'comparable to' if 0.6 < median_ratio < 0.9 else 'different from'} 
the simplified M500 × f_baryon approach (which gave ratio ~0.68).

Possible sources of systematic offset:
  - X-ray gas mass uncertainties (~20-30%)
  - Stellar M/L ratio uncertainties (~30%)
  - Missing ICL contribution
  - Lensing mass systematic uncertainties (~15-20%)
  - Cluster dynamical state (mergers displace gas from mass peak)
""")
    else:
        print(f"""
✗ POOR AGREEMENT (Literature-Based)

Using DIRECTLY MEASURED gas masses and stellar masses:
  Median ratio: {median_ratio:.2f}
  Scatter: {scatter:.2f} dex

This suggests systematic issues with either:
  - The cluster amplitude A_cluster = π√2
  - Systematic underestimate of baryonic mass
  - Lensing mass comparison methodology
""")
    
    # What amplitude would be needed?
    print("\n" + "=" * 80)
    print("AMPLITUDE CALIBRATION")
    print("=" * 80)
    
    # For each cluster, find the amplitude that gives ratio = 1
    required_amplitudes = []
    for _, row in results_df.iterrows():
        g_bar = row['g_bar']
        M_bar = row['M_bar_1e12Msun'] * 1e12 * M_sun
        MSL = row['MSL_1e12Msun'] * 1e12 * M_sun
        
        # Σ_required = MSL / M_bar
        Sigma_req = MSL / M_bar
        
        # Σ = 1 + A × h(g)
        # A_req = (Σ_req - 1) / h(g)
        h_g = h_universal(g_bar)
        A_req = (Sigma_req - 1) / h_g
        required_amplitudes.append(A_req)
        
        print(f"  {row['cluster']:<18}: Σ_req = {Sigma_req:.1f}, A_req = {A_req:.1f}")
    
    A_req_median = np.median(required_amplitudes)
    A_req_mean = np.mean(required_amplitudes)
    
    print(f"\n  Required amplitude to match lensing:")
    print(f"    Median: A_req = {A_req_median:.1f}")
    print(f"    Mean:   A_req = {A_req_mean:.1f}")
    print(f"    Current: A_cluster = π√2 = {A_cluster:.2f}")
    print(f"    Ratio needed: {A_req_median/A_cluster:.1f}×")
    
    # Data quality assessment
    print("\n" + "=" * 80)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 80)
    print("""
Gas mass methodology: X-ray surface brightness deprojection
  S_X(R) → n_e(r) → M_gas = ∫ μ m_p n_e 4πr² dr
  
  Key assumptions:
  - Spherical symmetry (violated in mergers)
  - Hydrostatic equilibrium (for temperature → density)
  - Metallicity profile (affects cooling function)
  
  Typical uncertainty: 15-25%

Stellar mass methodology: BCG + ICL + satellite photometry
  - BCG: Well-constrained (~10% error)
  - ICL: Poorly constrained (~factor 2 uncertainty)
  - Satellites: Depends on luminosity function completeness
  
  Typical uncertainty: 20-40%

Lensing mass: Strong lens modeling (geometry-only)
  - Multiple image positions → mass distribution
  - Model-dependent (NFW vs. free-form)
  - Typical uncertainty: 10-20%
""")
    
    # Save results
    output_file = Path(__file__).parent / "cluster_profile_validation_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    return results_df


if __name__ == "__main__":
    main()

