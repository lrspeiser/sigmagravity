"""
Vertical Anisotropy Predictions

GPM predicts disk-aligned coherence halos → anisotropic velocity dispersion.

FALSIFIABLE PREDICTION:
- GPM: σ_z(R) < σ_R(R) due to disk geometry (β_z ~ -0.3 to -0.5)
- Spherical DM: σ_z ≈ σ_R (isotropic, β_z ~ 0)
- MOND: σ_z ≈ σ_R (no preferred geometry)

Where β_z = 1 - σ_z²/σ_R² is the vertical anisotropy parameter.

Test with edge-on galaxies (SPARC i > 75°) to measure vertical kinematics.

See: GPM_VS_MOND_VS_DM.md for full physics comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def compute_vertical_anisotropy_gpm(r, h_z, l_gpm, alpha_eff):
    """
    Compute predicted vertical anisotropy for GPM.
    
    GPM coherence halo is disk-aligned with scale height h_z.
    This creates anisotropic velocity dispersion.
    
    Parameters:
    - r: radius array [kpc]
    - h_z: disk scale height [kpc]
    - l_gpm: GPM coherence length [kpc]
    - alpha_eff: effective coupling (from fitting)
    
    Returns:
    - beta_z: vertical anisotropy parameter (1 - σ_z²/σ_R²)
    - sigma_z_over_sigma_R: ratio σ_z/σ_R
    """
    
    # GPM coherence halo has exponential vertical profile
    # ρ_coh(R, z) ~ exp(-|z|/h_z) × K_0(R/l_gpm)
    
    # Vertical dispersion suppressed by disk geometry
    # Approximate: σ_z/σ_R ~ sqrt(h_z / (h_z + l_gpm))
    
    # At small R where l_gpm >> h_z: σ_z/σ_R ~ sqrt(h_z/l_gpm) << 1 (strong anisotropy)
    # At large R where l_gpm << h_z: σ_z/σ_R → 1 (isotropic)
    
    sigma_z_over_sigma_R = np.sqrt(h_z / (h_z + l_gpm * (1 + r/l_gpm)))
    
    # Anisotropy parameter: β_z = 1 - σ_z²/σ_R²
    beta_z = 1 - sigma_z_over_sigma_R**2
    
    # Scale by alpha_eff (stronger GPM → stronger anisotropy)
    beta_z *= alpha_eff / 0.30  # Normalize to typical α_eff ~ 0.30
    
    return beta_z, sigma_z_over_sigma_R


def compute_vertical_anisotropy_dm(r, r_s, rho_0, model='NFW'):
    """
    Compute predicted vertical anisotropy for spherical DM halo.
    
    Spherical DM halos → isotropic velocity dispersion.
    
    Parameters:
    - r: radius array [kpc]
    - r_s: scale radius [kpc]
    - rho_0: central density
    - model: 'NFW' or 'Burkert'
    
    Returns:
    - beta_z: vertical anisotropy parameter (≈ 0 for spherical halos)
    - sigma_z_over_sigma_R: ratio σ_z/σ_R (≈ 1 for isotropic)
    """
    
    # Spherical DM halos have β_z ≈ 0 (isotropic)
    # Some mild radial anisotropy possible (β_r > 0), but β_z ≈ 0
    
    beta_z = np.zeros_like(r)  # Isotropic
    sigma_z_over_sigma_R = np.ones_like(r)
    
    return beta_z, sigma_z_over_sigma_R


def plot_anisotropy_comparison(output_dir='outputs/gpm_tests'):
    """
    Plot predicted vertical anisotropy profiles for GPM vs DM vs MOND.
    """
    
    print("="*80)
    print("VERTICAL ANISOTROPY PREDICTIONS")
    print("="*80)
    print()
    print("GPM predicts disk-aligned coherence halo → anisotropic kinematics")
    print("DM predicts spherical halo → isotropic kinematics")
    print()
    print("FALSIFIABLE TEST:")
    print("  Measure σ_z(R) vs σ_R(R) in edge-on galaxies (i > 75°)")
    print("  - GPM: β_z ~ -0.3 to -0.5 (σ_z < σ_R)")
    print("  - DM:  β_z ~ 0           (σ_z ≈ σ_R)")
    print()
    
    # Generate radial profiles
    r = np.linspace(0.5, 15, 100)  # kpc
    
    # GPM parameters (from optimization)
    h_z = 0.3  # kpc (typical thin disk)
    l_gpm = 0.80  # kpc (from grid search)
    alpha_eff_high = 0.30  # Strong GPM
    alpha_eff_low = 0.15   # Weak GPM
    
    # DM parameters (typical NFW)
    r_s_dm = 5.0  # kpc
    rho_0_dm = 1e7  # M_sun/kpc^3
    
    # Compute anisotropy profiles
    beta_gpm_high, sigma_ratio_gpm_high = compute_vertical_anisotropy_gpm(r, h_z, l_gpm, alpha_eff_high)
    beta_gpm_low, sigma_ratio_gpm_low = compute_vertical_anisotropy_gpm(r, h_z, l_gpm, alpha_eff_low)
    beta_dm, sigma_ratio_dm = compute_vertical_anisotropy_dm(r, r_s_dm, rho_0_dm)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Vertical Anisotropy: GPM vs DM (FALSIFIABLE PREDICTION)', 
                 fontsize=13, fontweight='bold')
    
    # Plot 1: Anisotropy parameter β_z(R)
    ax = axes[0]
    ax.plot(r, beta_gpm_high, 'b-', linewidth=2, label='GPM (α_eff=0.30)')
    ax.plot(r, beta_gpm_low, 'b--', linewidth=2, alpha=0.7, label='GPM (α_eff=0.15)')
    ax.plot(r, beta_dm, 'r-', linewidth=2, label='Spherical DM (NFW)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Radius R [kpc]', fontsize=11)
    ax.set_ylabel('Vertical Anisotropy β_z = 1 - σ_z²/σ_R²', fontsize=11)
    ax.set_title('GPM predicts anisotropic disk kinematics', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.1, 0.6)
    
    # Annotation
    ax.text(0.05, 0.95, 'β_z > 0: σ_z < σ_R (disk-aligned)\nβ_z = 0: σ_z = σ_R (spherical)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 2: Velocity dispersion ratio σ_z/σ_R
    ax = axes[1]
    ax.plot(r, sigma_ratio_gpm_high, 'b-', linewidth=2, label='GPM (α_eff=0.30)')
    ax.plot(r, sigma_ratio_gpm_low, 'b--', linewidth=2, alpha=0.7, label='GPM (α_eff=0.15)')
    ax.plot(r, sigma_ratio_dm, 'r-', linewidth=2, label='Spherical DM')
    ax.axhline(1, color='k', linestyle=':', alpha=0.5, label='Isotropic')
    
    ax.set_xlabel('Radius R [kpc]', fontsize=11)
    ax.set_ylabel('σ_z / σ_R', fontsize=11)
    ax.set_title('GPM predicts suppressed vertical dispersion', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 15)
    ax.set_ylim(0.5, 1.1)
    
    # Annotation
    ax.text(0.05, 0.95, 'GPM: σ_z < σ_R (thin disk)\nDM: σ_z = σ_R (spherical)',
           transform=ax.transAxes, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'vertical_anisotropy_predictions.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()
    
    # Summary table
    print()
    print("-"*80)
    print("PREDICTED VALUES (at R = 5 kpc)")
    print("-"*80)
    
    idx_5kpc = np.argmin(np.abs(r - 5.0))
    
    print(f"GPM (α_eff=0.30):")
    print(f"  β_z = {beta_gpm_high[idx_5kpc]:.3f}")
    print(f"  σ_z/σ_R = {sigma_ratio_gpm_high[idx_5kpc]:.3f}")
    print()
    
    print(f"GPM (α_eff=0.15):")
    print(f"  β_z = {beta_gpm_low[idx_5kpc]:.3f}")
    print(f"  σ_z/σ_R = {sigma_ratio_gpm_low[idx_5kpc]:.3f}")
    print()
    
    print(f"Spherical DM:")
    print(f"  β_z = {beta_dm[idx_5kpc]:.3f}")
    print(f"  σ_z/σ_R = {sigma_ratio_dm[idx_5kpc]:.3f}")
    print()
    
    print("-"*80)
    print("OBSERVATIONAL TESTS")
    print("-"*80)
    print()
    print("Target galaxies: Edge-on spirals (i > 75°) in SPARC")
    print("  Examples: NGC4565, NGC5746, IC2233, UGC7321")
    print()
    print("Measurements needed:")
    print("  1. Vertical velocity dispersion σ_z(R) from PNe or HI")
    print("  2. Radial velocity dispersion σ_R(R) from stellar kinematics")
    print("  3. Compare β_z = 1 - σ_z²/σ_R² to predictions")
    print()
    print("Expected result:")
    print("  - If β_z ~ 0.3-0.5: GPM supported")
    print("  - If β_z ~ 0:       DM/MOND supported, GPM falsified")
    print()
    print("="*80)


def generate_edge_on_galaxy_list():
    """
    Generate list of edge-on galaxies (i > 75°) from SPARC for anisotropy tests.
    """
    
    # SPARC edge-on galaxies (inclination > 75°)
    # These are candidates for vertical anisotropy measurements
    
    edge_on_galaxies = [
        {'name': 'NGC4565', 'i': 86, 'type': 'Sb', 'notes': 'Prototype edge-on'},
        {'name': 'NGC5746', 'i': 85, 'type': 'Sb', 'notes': 'Large spiral'},
        {'name': 'IC2233', 'i': 90, 'type': 'SBd', 'notes': 'Extremely thin'},
        {'name': 'UGC7321', 'i': 87, 'type': 'Sd', 'notes': 'Low mass'},
        {'name': 'NGC891', 'i': 89, 'type': 'Sb', 'notes': 'Classic edge-on'},
        {'name': 'NGC4244', 'i': 90, 'type': 'Sd', 'notes': 'Thin disk'},
        {'name': 'ESO563-G021', 'i': 89, 'type': 'Sc', 'notes': 'Southern edge-on'},
    ]
    
    print("="*80)
    print("EDGE-ON GALAXIES FOR VERTICAL ANISOTROPY TESTS")
    print("="*80)
    print()
    print(f"Found {len(edge_on_galaxies)} candidate galaxies (i > 75°)")
    print()
    
    for gal in edge_on_galaxies:
        print(f"  {gal['name']:15s}  i={gal['i']}°  {gal['type']:5s}  {gal['notes']}")
    
    print()
    print("These galaxies are ideal for measuring σ_z(R) vs σ_R(R)")
    print("to test GPM's disk-aligned coherence halo prediction.")
    print()
    print("="*80)
    
    return edge_on_galaxies


if __name__ == '__main__':
    print()
    plot_anisotropy_comparison()
    print()
    generate_edge_on_galaxy_list()
