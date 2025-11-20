#!/usr/bin/env python3
"""
Debug DDO154 velocity mismatch between batch and single tests.

Batch test: alpha=0.181, chi2_gpm=1128, improvement +89.6%
Single test: alpha=0.225, chi2_gpm=125949, improvement -93%

Both use similar environment (Q~1-1.5, sigma_v~2 km/s) but vastly different results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_integration.load_real_data import RealDataLoader
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve


def exponential_disk_density(r, Sigma0, R_d, h_z):
    """3D density for exponential disk."""
    r_scalar = np.atleast_1d(r)
    rho = Sigma0 / (2.0 * h_z) * np.exp(-r_scalar / R_d)
    if np.isscalar(r):
        return float(rho[0])
    return rho


def create_baryon_density(gal, galaxy_name):
    """Create baryon density matching batch test."""
    from data_integration.load_real_data import RealDataLoader
    
    r = gal['r']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    
    # Load SBdisk
    loader = RealDataLoader()
    rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = [l for l in lines if not l.startswith('#')]
    SBdisk = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7:
            SBdisk.append(float(parts[6]))
    
    SBdisk = np.array(SBdisk)
    
    # Estimate R_disk from exponential fit
    mask = SBdisk > 0
    if np.sum(mask) >= 3:
        from scipy.optimize import curve_fit
        
        def exp_profile(r_fit, SB0, R_d):
            return SB0 * np.exp(-r_fit / R_d)
        
        try:
            popt, _ = curve_fit(exp_profile, r[mask], SBdisk[mask], 
                               p0=[SBdisk[mask][0], 2.0],
                               bounds=([0, 0.1], [1e10, 20.0]))
            SB0, R_disk = popt
        except:
            SB0 = np.max(SBdisk)
            idx_1e = np.argmin(np.abs(SBdisk - SB0/np.e))
            R_disk = r[idx_1e] if idx_1e > 0 else 2.0
    else:
        idx_peak = np.argmax(v_disk)
        R_disk = r[idx_peak] / 2.2
        SB0 = np.max(SBdisk) if len(SBdisk) > 0 else 1.0
    
    # Convert to mass
    M_L = 0.5
    Sigma0 = SB0 * M_L * 1e6
    M_disk = 2.0 * np.pi * Sigma0 * R_disk**2
    
    h_z = 0.3
    def rho_disk(r_eval):
        return exponential_disk_density(r_eval, Sigma0, R_disk, h_z)
    
    # Gas
    G_kpc = 4.302e-3
    M_gas_enc = r * v_gas**2 / G_kpc
    M_gas_total = M_gas_enc[-1] if len(M_gas_enc) > 0 else 0
    
    Sigma0_gas = M_gas_total / (2.0 * np.pi * R_disk**2)
    
    def rho_gas(r_eval):
        return exponential_disk_density(r_eval, Sigma0_gas, R_disk, h_z)
    
    def rho_b(r_eval):
        rho_d = rho_disk(r_eval)
        rho_g = rho_gas(r_eval)
        result = rho_d + rho_g
        if np.isscalar(r_eval):
            return float(result) if not np.isscalar(result) else result
        return result
    
    return rho_b, M_disk + M_gas_total, R_disk, SBdisk


def main():
    print("="*80)
    print("DEBUG: DDO154 Batch vs Single Test Mismatch")
    print("="*80)
    
    # Load data
    print("\n1. Loading SPARC data...")
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy('DDO154')
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    
    print(f"   {len(r_data)} data points, v_obs range: {v_obs.min():.1f}-{v_obs.max():.1f} km/s")
    
    # Create baryon density
    print("\n2. Creating baryon density...")
    rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, 'DDO154')
    print(f"   M_total = {M_total:.2e} Msun")
    print(f"   R_disk = {R_disk:.3f} kpc")
    
    # Batch test parameters (from CSV)
    print("\n3. Batch test parameters:")
    Q_batch = 1.5
    sigma_v_batch = 1.76  # From CSV
    alpha_batch = 0.181   # From CSV
    ell_batch = 0.94      # From CSV
    print(f"   Q = {Q_batch}, sigma_v = {sigma_v_batch:.2f} km/s")
    print(f"   alpha_eff = {alpha_batch:.3f}, ell = {ell_batch:.2f} kpc")
    
    # Single test parameters (computed)
    print("\n4. Single test parameters:")
    v_mean = np.mean(v_obs[v_obs > 0])
    sigma_v_single = 0.06 * v_mean  # Dwarf scaling
    Q_single = 1.0  # From environment estimator
    print(f"   Q = {Q_single}, sigma_v = {sigma_v_single:.2f} km/s")
    
    # Create GPM with batch parameters
    print("\n5. Creating GPM with batch-like parameters...")
    gpm_batch = GravitationalPolarizationMemory(
        alpha0=0.3, ell0_kpc=2.0, Qstar=2.0, sigmastar=25.0,
        nQ=2.0, nsig=2.0, p=0.5, Mstar_Msun=2e8, nM=1.5
    )
    
    # Try to reproduce batch alpha
    alpha_test, ell_test = gpm_batch.environment_factors(
        Q=Q_batch, sigma_v=sigma_v_batch, R_disk=R_disk, M_total=M_total
    )
    print(f"   Computed: alpha = {alpha_test:.3f}, ell = {ell_test:.2f} kpc")
    print(f"   Batch CSV: alpha = {alpha_batch:.3f}, ell = {ell_batch:.2f} kpc")
    print(f"   Match: {'YES' if abs(alpha_test - alpha_batch) < 0.01 else 'NO'}")
    
    # Create rho_coh with both sets of parameters
    print("\n6. Computing coherence densities...")
    
    # Use batch environment
    rho_coh_func_batch, _ = gpm_batch.make_rho_coh(
        rho_b, Q=Q_batch, sigma_v=sigma_v_batch, R_disk=R_disk, M_total=M_total
    )
    
    # Evaluate at data radii
    r_eval = np.linspace(r_data.min(), r_data.max(), 100)
    rho_b_vals = np.array([rho_b(ri) for ri in r_eval])
    rho_coh_vals_batch = rho_coh_func_batch(r_eval)
    
    print(f"   rho_b range: {rho_b_vals.min():.2e} - {rho_b_vals.max():.2e} Msun/kpc^3")
    print(f"   rho_coh range: {rho_coh_vals_batch.min():.2e} - {rho_coh_vals_batch.max():.2e} Msun/kpc^3")
    print(f"   rho_coh/rho_b ratio: {(rho_coh_vals_batch.mean() / rho_b_vals.mean()):.2f}")
    
    # Compute rotation curves
    print("\n7. Computing rotation curves...")
    
    galaxy_batch = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_batch.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy_batch.set_coherence_halo_gpm(rho_coh_func_batch, {'alpha': alpha_test, 'ell_kpc': ell_test})
    
    v_model_batch = galaxy_batch.circular_velocity(r_eval)
    
    print(f"   v_model range: {v_model_batch.min():.1f} - {v_model_batch.max():.1f} km/s")
    print(f"   v_obs range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
    
    # Interpolate to data points
    from scipy.interpolate import PchipInterpolator
    v_model_at_data = PchipInterpolator(r_eval, v_model_batch)(r_data)
    
    # Compute chi2
    chi2 = np.sum((v_obs - v_model_at_data)**2 / v_err**2)
    chi2_red = chi2 / len(r_data)
    
    print(f"   chi2_red = {chi2_red:.1f}")
    print(f"   Batch CSV chi2_red = 1128.3 / 12 = 94.0")
    
    # Plot comparison
    print("\n8. Creating diagnostic plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('DDO154: Batch vs Single Test Debug', fontsize=14, fontweight='bold')
    
    # Top left: Rotation curves
    ax = axes[0, 0]
    ax.errorbar(r_data, v_obs, yerr=v_err, fmt='ko', label='Observed', alpha=0.6, capsize=3)
    ax.plot(r_eval, v_model_batch, 'r-', label=f'GPM (alpha={alpha_test:.3f})', linewidth=2)
    ax.axhline(v_obs.mean(), color='gray', linestyle='--', alpha=0.5, label='Mean v_obs')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Velocity (km/s)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title(f'Rotation Curve (chi2_red={chi2_red:.1f})')
    
    # Top right: Density profiles
    ax = axes[0, 1]
    ax.semilogy(r_eval, rho_b_vals, 'b-', label='rho_baryons', linewidth=2)
    ax.semilogy(r_eval, rho_coh_vals_batch, 'r-', label='rho_coherence', linewidth=2)
    ax.semilogy(r_eval, rho_b_vals + rho_coh_vals_batch, 'k--', label='rho_total', linewidth=1.5)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Density (Msun/kpc^3)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Density Profiles')
    
    # Bottom left: Enclosed mass
    ax = axes[1, 0]
    
    # Compute enclosed masses
    M_bar_enc = []
    M_coh_enc = []
    for ri in r_eval:
        # Baryon mass
        r_int = np.linspace(0, ri, 100)
        rho_b_int = np.array([rho_b(r) for r in r_int])
        M_bar_enc.append(np.trapz(4*np.pi*r_int**2 * rho_b_int, r_int))
        
        # Coherence mass
        rho_coh_int = rho_coh_func_batch(r_int)
        M_coh_enc.append(np.trapz(4*np.pi*r_int**2 * rho_coh_int, r_int))
    
    M_bar_enc = np.array(M_bar_enc)
    M_coh_enc = np.array(M_coh_enc)
    
    ax.loglog(r_eval, M_bar_enc, 'b-', label='M_baryon(<r)', linewidth=2)
    ax.loglog(r_eval, M_coh_enc, 'r-', label='M_coherence(<r)', linewidth=2)
    ax.loglog(r_eval, M_bar_enc + M_coh_enc, 'k--', label='M_total(<r)', linewidth=1.5)
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Enclosed Mass (Msun)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Enclosed Mass')
    
    # Bottom right: Parameter comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison_text = f"""
BATCH TEST (from CSV):
  Q = {Q_batch:.2f}
  σ_v = {sigma_v_batch:.2f} km/s
  α_eff = {alpha_batch:.3f}
  ℓ = {ell_batch:.2f} kpc
  χ²_red = 94.0
  Improvement: +89.6%

SINGLE TEST (computed):
  Q = {Q_single:.2f}
  σ_v = {sigma_v_single:.2f} km/s
  α_eff = {alpha_test:.3f}
  ℓ = {ell_test:.2f} kpc
  χ²_red = {chi2_red:.1f}
  Improvement: {((chi2_red - 5431)/5431*100):.1f}%

KEY FINDINGS:
  M_coh/M_bar ~ {M_coh_enc[-1]/M_bar_enc[-1]:.2f}
  v_model max = {v_model_batch.max():.1f} km/s
  v_obs max = {v_obs.max():.1f} km/s
  
ISSUE:
  Model velocities too low!
  Need ~3-4× more mass or 
  different ℓ/α combination
"""
    
    ax.text(0.1, 0.5, comparison_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gpm_tests')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'DDO154_debug_mismatch.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    plt.close()
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    print(f"Model produces v ~ {v_model_batch.max():.1f} km/s vs observed {v_obs.max():.1f} km/s")
    print(f"Coherence mass is {M_coh_enc[-1]/M_bar_enc[-1]:.1%} of baryon mass")
    print(f"Need ~{(v_obs.max()/v_model_batch.max())**2:.1f}× more total mass for correct velocities")
    print("\nPossible causes:")
    print("1. Baryon mass underestimated (M_total too low)")
    print("2. Alpha or ell incorrect despite matching batch values")
    print("3. Rotation curve integration issue")
    print("="*80)


if __name__ == '__main__':
    main()
