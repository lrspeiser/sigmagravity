#!/usr/bin/env python3
"""
Pantheon+ Coherence Cosmology Test
===================================

This script tests coherence cosmology against the REAL Pantheon+ Type Ia
supernova dataset (1701 supernovae, 0.001 < z < 2.3).

We compare:
- Standard ΛCDM (expanding universe)
- Coherence cosmology (static universe with coherence-induced redshift)

Author: Sigma Gravity Research
Date: December 2025
"""

import numpy as np
from scipy import integrate
from scipy.optimize import minimize, differential_evolution
import os

# Physical constants
c = 2.998e5  # km/s (for convenience with H0 units)

print("=" * 100)
print("PANTHEON+ COHERENCE COSMOLOGY TEST")
print("=" * 100)

# =============================================================================
# LOAD PANTHEON+ DATA
# =============================================================================

print("\n" + "=" * 80)
print("LOADING PANTHEON+ DATA")
print("=" * 80)

data_path = "/Users/leonardspeiser/Projects/sigmagravity/data/pantheon/Pantheon+SH0ES.dat"

# Read the data
with open(data_path, 'r') as f:
    header = f.readline().strip().split()
    
# Find column indices
col_indices = {name: i for i, name in enumerate(header)}

# Load data
data = np.genfromtxt(data_path, skip_header=1, dtype=None, encoding='utf-8')

# Extract relevant columns
# zHD = Hubble diagram redshift (corrected for peculiar velocities)
# m_b_corr = corrected apparent magnitude
# m_b_corr_err_DIAG = diagonal error (we'll use this for simplicity)

z_all = np.array([float(row[col_indices['zHD']]) for row in data])
m_b_all = np.array([float(row[col_indices['m_b_corr']]) for row in data])
m_b_err_all = np.array([float(row[col_indices['m_b_corr_err_DIAG']]) for row in data])

# Filter out problematic data
mask = (z_all > 0.01) & (z_all < 2.5) & (m_b_err_all > 0) & (m_b_err_all < 2)
z = z_all[mask]
m_b = m_b_all[mask]
m_b_err = m_b_err_all[mask]

print(f"Total supernovae in catalog: {len(z_all)}")
print(f"After quality cuts: {len(z)}")
print(f"Redshift range: {z.min():.4f} to {z.max():.4f}")
print(f"Median error: {np.median(m_b_err):.3f} mag")

# =============================================================================
# DISTANCE FORMULAS
# =============================================================================

def luminosity_distance_LCDM(z, H0, Om):
    """
    Luminosity distance in flat ΛCDM cosmology.
    Returns distance in Mpc.
    """
    OL = 1.0 - Om  # Flat universe
    
    def E(zp):
        return np.sqrt(Om * (1 + zp)**3 + OL)
    
    z = np.atleast_1d(z)
    d_L = np.zeros_like(z, dtype=float)
    
    for i, zi in enumerate(z):
        if zi <= 0:
            d_L[i] = 1e-10
        else:
            integral, _ = integrate.quad(lambda zp: 1.0/E(zp), 0, zi, limit=100)
            d_L[i] = (1 + zi) * c / H0 * integral
    
    return d_L


def luminosity_distance_coherence(z, H0, alpha):
    """
    Luminosity distance in coherence cosmology.
    
    The coherence-induced redshift has a non-linear correction:
        z = (H₀/c) × d × (1 + α × (H₀/c) × d)
    
    Solving for d and computing d_L = d × (1+z)
    """
    z = np.atleast_1d(z)
    
    if abs(alpha) < 1e-10:
        # Linear case
        d = c / H0 * z
    else:
        # Quadratic case: solve z = x(1 + αx) where x = H₀d/c
        # αx² + x - z = 0
        # x = (-1 + √(1 + 4αz)) / (2α)
        discriminant = 1 + 4 * alpha * z
        discriminant = np.maximum(discriminant, 0)  # Avoid negative sqrt
        x = (-1 + np.sqrt(discriminant)) / (2 * alpha)
        d = c / H0 * x
    
    d_L = d * (1 + z)
    return np.maximum(d_L, 1e-10)


def distance_modulus(d_L):
    """Convert luminosity distance (Mpc) to distance modulus."""
    return 5 * np.log10(np.maximum(d_L, 1e-10)) + 25

# =============================================================================
# CHI-SQUARED FITTING
# =============================================================================

def chi2_LCDM(params, z, m_b, m_b_err):
    """Chi-squared for ΛCDM model."""
    H0, Om, M = params
    if H0 < 50 or H0 > 100 or Om < 0.01 or Om > 0.99:
        return 1e20
    
    d_L = luminosity_distance_LCDM(z, H0, Om)
    mu_pred = distance_modulus(d_L)
    m_pred = mu_pred + M  # M is the absolute magnitude
    
    chi2 = np.sum(((m_b - m_pred) / m_b_err)**2)
    return chi2


def chi2_coherence(params, z, m_b, m_b_err):
    """Chi-squared for coherence model."""
    H0, alpha, M = params
    if H0 < 50 or H0 > 100 or alpha < -0.5 or alpha > 5:
        return 1e20
    
    d_L = luminosity_distance_coherence(z, H0, alpha)
    mu_pred = distance_modulus(d_L)
    m_pred = mu_pred + M
    
    chi2 = np.sum(((m_b - m_pred) / m_b_err)**2)
    return chi2

# =============================================================================
# FIT BOTH MODELS
# =============================================================================

print("\n" + "=" * 80)
print("FITTING MODELS TO PANTHEON+ DATA")
print("=" * 80)

# Estimate M from low-z data
low_z_mask = z < 0.1
if np.sum(low_z_mask) > 10:
    d_L_low = c / 70 * z[low_z_mask]  # Approximate
    mu_low = distance_modulus(d_L_low)
    M_init = np.median(m_b[low_z_mask] - mu_low)
else:
    M_init = -19.3

print(f"Initial absolute magnitude estimate: M = {M_init:.2f}")

# Fit ΛCDM
print("\nFitting ΛCDM model...")
result_LCDM = minimize(
    chi2_LCDM, 
    x0=[70.0, 0.3, M_init],
    args=(z, m_b, m_b_err),
    method='Nelder-Mead',
    options={'maxiter': 5000}
)
H0_LCDM, Om_LCDM, M_LCDM = result_LCDM.x
chi2_LCDM_val = result_LCDM.fun

print(f"  H₀ = {H0_LCDM:.2f} km/s/Mpc")
print(f"  Ω_m = {Om_LCDM:.3f}")
print(f"  M = {M_LCDM:.3f}")
print(f"  χ² = {chi2_LCDM_val:.1f}")
print(f"  χ²/N = {chi2_LCDM_val/len(z):.3f}")

# Fit coherence model
print("\nFitting coherence model...")
result_coh = minimize(
    chi2_coherence,
    x0=[70.0, 0.5, M_init],
    args=(z, m_b, m_b_err),
    method='Nelder-Mead',
    options={'maxiter': 5000}
)
H0_coh, alpha_coh, M_coh = result_coh.x
chi2_coh_val = result_coh.fun

print(f"  H₀ = {H0_coh:.2f} km/s/Mpc")
print(f"  α = {alpha_coh:.4f}")
print(f"  M = {M_coh:.3f}")
print(f"  χ² = {chi2_coh_val:.1f}")
print(f"  χ²/N = {chi2_coh_val/len(z):.3f}")

# =============================================================================
# MODEL COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

# Both models have 3 parameters, so we can compare chi2 directly
delta_chi2 = chi2_coh_val - chi2_LCDM_val

print(f"""
                        ΛCDM            Coherence
                        ----            ---------
H₀ (km/s/Mpc)          {H0_LCDM:6.2f}          {H0_coh:6.2f}
Ω_m                    {Om_LCDM:6.3f}          (n/a)
Ω_Λ                    {1-Om_LCDM:6.3f}          (n/a)
α (coherence)          (n/a)           {alpha_coh:6.4f}
M (abs mag)            {M_LCDM:6.3f}          {M_coh:6.3f}
χ²                     {chi2_LCDM_val:8.1f}        {chi2_coh_val:8.1f}
χ²/N                   {chi2_LCDM_val/len(z):8.3f}        {chi2_coh_val/len(z):8.3f}

Δχ² = χ²(coherence) - χ²(ΛCDM) = {delta_chi2:.1f}
""")

if delta_chi2 < 0:
    print("→ Coherence model fits BETTER than ΛCDM!")
elif delta_chi2 < 10:
    print("→ Models are statistically comparable (Δχ² < 10)")
else:
    print(f"→ ΛCDM fits better by Δχ² = {delta_chi2:.1f}")

# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("RESIDUAL ANALYSIS")
print("=" * 80)

# Compute residuals for both models
d_L_LCDM = luminosity_distance_LCDM(z, H0_LCDM, Om_LCDM)
mu_LCDM = distance_modulus(d_L_LCDM)
m_pred_LCDM = mu_LCDM + M_LCDM
resid_LCDM = m_b - m_pred_LCDM

d_L_coh = luminosity_distance_coherence(z, H0_coh, alpha_coh)
mu_coh = distance_modulus(d_L_coh)
m_pred_coh = mu_coh + M_coh
resid_coh = m_b - m_pred_coh

# Binned residuals
z_bins = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.5]
print("\nBinned residuals (observed - predicted):")
print("-" * 70)
print(f"{'z range':>15} | {'N':>5} | {'ΛCDM resid':>12} | {'Coh resid':>12}")
print("-" * 70)

for i in range(len(z_bins) - 1):
    mask = (z >= z_bins[i]) & (z < z_bins[i+1])
    n = np.sum(mask)
    if n > 0:
        mean_LCDM = np.mean(resid_LCDM[mask])
        mean_coh = np.mean(resid_coh[mask])
        std_LCDM = np.std(resid_LCDM[mask]) / np.sqrt(n)
        std_coh = np.std(resid_coh[mask]) / np.sqrt(n)
        print(f"{z_bins[i]:.2f} - {z_bins[i+1]:.2f}    | {n:5d} | {mean_LCDM:+.4f}±{std_LCDM:.4f} | {mean_coh:+.4f}±{std_coh:.4f}")

# =============================================================================
# PHYSICAL INTERPRETATION
# =============================================================================

print("\n" + "=" * 80)
print("PHYSICAL INTERPRETATION OF COHERENCE PARAMETER")
print("=" * 80)

print(f"""
The best-fit coherence parameter is α = {alpha_coh:.4f}

MEANING:
The coherence-induced redshift is:
    z = (H₀/c) × d × (1 + α × (H₀/c) × d)

At the Hubble radius (d = c/H₀):
    z = 1 × (1 + α) = {1 + alpha_coh:.3f}

For α > 0:
    - Coherence effect INCREASES with distance
    - Distant regions contribute MORE to redshift per unit distance
    - This mimics "cosmic acceleration" without dark energy

CONNECTION TO Σ-GRAVITY:
    The coherence parameter α relates to the coherence potential:
    
    Ψ_coh(d) = (H₀/2c) × d × (1 + α × (H₀/c) × d)
    
    At large scales, the coherence field becomes stronger,
    just as we expect from cumulative matter correlations.
""")

# =============================================================================
# HIGH-Z PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("HIGH-Z PREDICTIONS: WHERE MODELS DIVERGE")
print("=" * 80)

z_future = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 15.0])

d_L_LCDM_future = luminosity_distance_LCDM(z_future, H0_LCDM, Om_LCDM)
mu_LCDM_future = distance_modulus(d_L_LCDM_future)

d_L_coh_future = luminosity_distance_coherence(z_future, H0_coh, alpha_coh)
mu_coh_future = distance_modulus(d_L_coh_future)

print("\nDistance modulus predictions at high z:")
print("-" * 60)
print(f"{'z':>6} | {'ΛCDM μ':>10} | {'Coherence μ':>12} | {'Δμ':>8}")
print("-" * 60)

for i, zf in enumerate(z_future):
    delta = mu_coh_future[i] - mu_LCDM_future[i]
    print(f"{zf:6.1f} | {mu_LCDM_future[i]:10.2f} | {mu_coh_future[i]:12.2f} | {delta:+8.2f}")

print("""
JWST OPPORTUNITY:
At z > 5, the models diverge by > 1 magnitude.
JWST can observe galaxies at z = 10-15.
If we find standardizable candles (e.g., from lensing time delays),
we can definitively distinguish the models.
""")

# =============================================================================
# UNIQUE COHERENCE PREDICTIONS
# =============================================================================

print("\n" + "=" * 80)
print("UNIQUE COHERENCE PREDICTIONS (TESTABLE)")
print("=" * 80)

print("""
1. ENVIRONMENT-DEPENDENT REDSHIFT
   Coherence predicts that lines of sight through overdense regions
   (galaxy clusters, cosmic web filaments) should show slightly
   MORE redshift than lines through voids.
   
   Expected effect: Δz/z ~ δρ/ρ × (small factor)
   
   TEST: Compare supernova redshifts with local galaxy density maps.

2. TIME DILATION VARIATIONS
   If coherence causes time dilation, it should vary with environment.
   Supernovae in clusters should show MORE time dilation.
   
   TEST: Correlate light curve stretch with local density.

3. GRAVITATIONAL WAVE PROPAGATION
   GW should also be affected by coherence.
   The GW "luminosity distance" should match EM luminosity distance.
   
   TEST: Compare GW170817-like events at higher z.

4. NO ANGULAR SIZE MINIMUM (CHALLENGE)
   In coherence cosmology, angular sizes should decrease monotonically.
   In ΛCDM, there's a minimum at z ~ 1.5.
   
   This is a STRONG test that could falsify coherence cosmology.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
PANTHEON+ RESULTS ({len(z)} supernovae):

                        ΛCDM            Coherence
                        ----            ---------
Best-fit χ²/N          {chi2_LCDM_val/len(z):.3f}           {chi2_coh_val/len(z):.3f}
Parameters             H₀, Ω_m, M      H₀, α, M
                       (3 params)      (3 params)

CONCLUSION:
""")

if abs(delta_chi2) < 10:
    print("""Both models fit the Pantheon+ data comparably well.
The coherence model with α ~ {:.3f} can replace dark energy (Ω_Λ ~ 0.7).

This means:
- The universe might NOT be expanding
- "Dark energy" might be the coherence field
- The g† = cH₀/(4√π) connection is explained naturally

Further tests needed:
- Angular diameter distance (BAO, galaxy sizes)
- CMB power spectrum
- Environment-dependent effects
""".format(alpha_coh))
elif delta_chi2 > 0:
    print(f"""ΛCDM fits better by Δχ² = {delta_chi2:.1f}.
However, coherence cosmology is not ruled out - it may need refinement.

Possible improvements:
- More complex coherence model (varying α with z)
- Include systematic uncertainties
- Account for peculiar velocities differently
""")
else:
    print(f"""Coherence fits BETTER by Δχ² = {-delta_chi2:.1f}!
This is surprising and warrants further investigation.
""")

print("=" * 100)
print("END OF PANTHEON+ ANALYSIS")
print("=" * 100)

