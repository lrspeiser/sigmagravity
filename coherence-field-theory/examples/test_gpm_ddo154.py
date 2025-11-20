"""
Test GPM (Gravitational Polarization with Memory) on Real SPARC Data

This tests the first-principles GPM microphysics on DDO154.

Comparison:
- Baryons only (v_bar from SPARC)
- Baryons + GPM coherence halo
- Observed v_obs from SPARC
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from galaxies.coherence_microphysics import GravitationalPolarizationMemory, exponential_disk_density
from galaxies.rotation_curves import GalaxyRotationCurve


def load_sparc_data(galaxy_name='DDO154'):
    """Load SPARC data for a galaxy."""
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    return gal


def create_baryon_density(gal, galaxy_name):
    """
    Create baryon density function from SPARC velocity components.
    
    CRITICAL FIX: Uses v_disk and v_gas directly (already have correct M/L),
    not SBdisk × M/L which severely underestimates masses (~30×).
    
    SPARC velocity components encode correct M/L ratios:
    - v_disk: includes fitted M/L for stellar disk ([3.6] band)
    - v_gas: from HI 21cm observations (atomic) + CO (molecular)
    
    Returns:
    --------
    rho_b : callable
        Baryon density ρ_b(r) in M☉/kpc³
    M_total : float
        Total baryon mass
    R_disk : float
        Disk scale length
    SBdisk : array
        Surface brightness profile (needed for Q estimation)
    """
    from data_integration.load_real_data import RealDataLoader
    from data_integration.load_sparc_masses import load_sparc_masses
    from scipy.interpolate import PchipInterpolator
    
    # === USE SPARC MASTER TABLE FOR MASSES (CRITICAL FIX) ===
    # v_disk/v_gas are rotation curve fits that don't extend far enough
    # Master table has integrated masses: M_stellar from L[3.6], M_HI from 21cm
    sparc_masses = load_sparc_masses(galaxy_name)
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk_sparc = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    r = gal['r']
    
    # === CONSTRUCT DENSITY PROFILES FROM SPARC MASSES ===
    # Use exponential disk profiles with total masses from master table
    # For M_total = 2π Σ₀ R_d², we have Σ₀ = M/(2π R_d²)
    
    # Use R_disk from SPARC (stellar disk scale length)
    # Use R_HI for gas (typically more extended)
    R_disk = R_disk_sparc
    R_gas = max(R_HI, 1.5 * R_disk)  # Gas at least 1.5× more extended
    
    # Central surface densities from total masses
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)  # M☉/kpc²
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)  # M☉/kpc²
    
    # Scale height (thin disk approximation)
    h_z = 0.3  # kpc
    
    def rho_b(r_eval):
        """Total baryon volume density: stellar + gas exponential disks."""
        r_safe = np.maximum(np.atleast_1d(r_eval), 1e-6)
        scalar_input = np.isscalar(r_eval)
        
        # Exponential profiles: Σ(r) = Σ₀ exp(-r/R)
        Sigma_stellar = Sigma0_stellar * np.exp(-r_safe / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-r_safe / R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        
        # Volume density: ρ = Σ / (2 h_z)
        rho = Sigma_total / (2.0 * h_z)
        
        return float(rho[0]) if scalar_input else rho
    
    # Load SBdisk for Q estimation (needed by EnvironmentEstimator)
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
            SBdisk.append(float(parts[6]))  # L☉/pc²
    SBdisk = np.array(SBdisk)
    
    return rho_b, M_total, R_disk, SBdisk


def estimate_environment_proper(gal, SBdisk, R_disk, M_total, galaxy_name='DDO154'):
    """
    Estimate Q and σ_v from actual SPARC data.
    
    Uses EnvironmentEstimator to compute proper Toomre Q and
    velocity dispersion from observables.
    """
    from galaxies.environment_estimator import EnvironmentEstimator
    
    estimator = EnvironmentEstimator()
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    
    Q, sigma_v = estimator.estimate_from_sparc(
        gal, SBdisk, R_disk, M_L=0.5, morphology=morphology
    )
    
    print(f"   Morphology: {morphology}")
    print(f"   Q method: Toomre Q = κσ_R/(3.36 G Σ) from SBdisk")
    print(f"   σ_v method: scaling relation σ_v/v_c ~ {0.06 if morphology=='dwarf' else 0.17}")
    
    return Q, sigma_v, morphology


def test_gpm_on_ddo154():
    """Main test: GPM on DDO154."""
    print("="*80)
    print("Testing GPM on DDO154")
    print("="*80)
    
    # 1. Load SPARC data
    print("\n1. Loading SPARC data...")
    gal = load_sparc_data('DDO154')
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    
    print(f"   Loaded {len(r_data)} data points")
    print(f"   Radius range: {r_data.min():.2f} - {r_data.max():.2f} kpc")
    print(f"   Velocity range: {v_obs.min():.1f} - {v_obs.max():.1f} km/s")
    
    # 2. Create baryon density
    print("\n2. Creating baryon density...")
    rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, 'DDO154')
    
    print(f"   Total baryon mass: {M_total:.2e} Msun")
    print(f"   Disk scale length: {R_disk:.2f} kpc")
    
    # 3. Estimate environment from data
    print("\n3. Estimating environment from SPARC data...")
    Q, sigma_v, morphology = estimate_environment_proper(gal, SBdisk, R_disk, M_total, 'DDO154')
    
    print(f"   Q (Toomre): {Q:.2f}")
    print(f"   sigma_v: {sigma_v:.1f} km/s")
    
    # 4. Create GPM model
    print("\n4. Creating GPM coherence halo...")
    # Use validated parameters from batch test (80% success rate)
    gpm = GravitationalPolarizationMemory(
        alpha0=0.3,         # Tuned from 0.9
        ell0_kpc=2.0,
        Qstar=2.0,
        sigmastar=25.0,
        nQ=2.0,
        nsig=2.0,
        p=0.5,              # ell ~ R_disk^0.5 scaling
        Mstar_Msun=2e8,     # Mass-dependent gating
        nM=1.5
    )
    
    rho_coh_func, gpm_params = gpm.make_rho_coh(
        rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, M_total=M_total
    )
    
    print(f"   Effective α: {gpm_params['alpha']:.3f} ({gpm_params['gate_strength']:.1%} gate)")
    print(f"   Coherence length ℓ: {gpm_params['ell_kpc']:.2f} kpc (ℓ/R_d = {gpm_params['coherence_scale']:.2f})")
    
    # 5. Compute rotation curves
    print("\n5. Computing rotation curves...")
    
    # Create GalaxyRotationCurve object
    galaxy = GalaxyRotationCurve(G=4.30091e-6)
    galaxy.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy.set_coherence_halo_gpm(rho_coh_func, gpm_params)
    
    # Compute velocities
    r_model = np.linspace(r_data.min(), r_data.max(), 100)
    v_model = galaxy.circular_velocity(r_model)
    
    # Baryons only
    v_bar = np.sqrt(v_disk**2 + v_gas**2)
    
    # Also compute baryons-only model
    galaxy_baryon = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_baryon.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    v_model_baryon = galaxy_baryon.circular_velocity(r_model)
    
    print(f"   Model velocity range: {v_model.min():.1f} - {v_model.max():.1f} km/s")
    
    # 6. Compute χ²
    print("\n6. Computing fit quality...")
    
    # Interpolate model to data points using PCHIP (shape-preserving, no overshoots)
    from scipy.interpolate import PchipInterpolator
    v_model_at_data = PchipInterpolator(r_model, v_model)(r_data)
    
    # χ² for baryons only
    chi2_baryon = np.sum((v_bar - v_obs)**2 / (v_err**2 + 1e-10))
    dof = len(r_data)
    chi2_red_baryon = chi2_baryon / dof
    
    # χ² for baryons + GPM
    chi2_gpm = np.sum((v_model_at_data - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_red_gpm = chi2_gpm / dof
    
    print(f"   Baryons only:")
    print(f"      χ² = {chi2_baryon:.1f}, χ²_red = {chi2_red_baryon:.2f}")
    print(f"   Baryons + GPM:")
    print(f"      χ² = {chi2_gpm:.1f}, χ²_red = {chi2_red_gpm:.2f}")
    
    improvement = (chi2_baryon - chi2_gpm) / chi2_baryon * 100
    print(f"   Improvement: {improvement:.1f}%")
    
    # 7. Plot
    print("\n7. Creating plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('DDO154: GPM Test', fontsize=14, fontweight='bold')
    
    # Left: Rotation curve
    ax = axes[0]
    ax.errorbar(r_data, v_obs, yerr=v_err, fmt='ko', 
               label='Observed', alpha=0.6, capsize=3, markersize=5)
    ax.plot(r_data, v_bar, 'b--', label='Baryons only (SPARC)', linewidth=2, alpha=0.7)
    ax.plot(r_model, v_model_baryon, 'b:', label='Baryons model', linewidth=2, alpha=0.5)
    ax.plot(r_model, v_model, 'r-', label=f'Baryons + GPM (χ²_red={chi2_red_gpm:.2f})', linewidth=2.5)
    ax.set_xlabel('Radius (kpc)', fontsize=11)
    ax.set_ylabel('Velocity (km/s)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title('Rotation Curve')
    
    # Right: Density profiles
    ax = axes[1]
    r_dens = np.linspace(0.1, r_data.max(), 100)
    rho_b_vals = np.array([rho_b(ri) for ri in r_dens])
    rho_coh_vals = rho_coh_func(r_dens)
    
    ax.semilogy(r_dens, rho_b_vals, 'b-', label='ρ_baryons', linewidth=2)
    ax.semilogy(r_dens, rho_coh_vals, 'r-', label='ρ_coherence (GPM)', linewidth=2)
    ax.semilogy(r_dens, rho_b_vals + rho_coh_vals, 'k--', label='ρ_total', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Radius (kpc)', fontsize=11)
    ax.set_ylabel('Density (M☉/kpc³)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f'Density Profiles (α={gpm_params["alpha"]:.3f}, ℓ={gpm_params["ell_kpc"]:.2f} kpc)')
    
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gpm_tests')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'DDO154_gpm_test.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_file}")
    
    plt.close()
    
    # 8. Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Galaxy: DDO154 (dwarf)")
    print(f"Environment: Q={Q:.2f}, σ_v={sigma_v:.1f} km/s")
    print(f"GPM parameters: α={gpm_params['alpha']:.3f}, ℓ={gpm_params['ell_kpc']:.2f} kpc")
    print(f"\nFit quality:")
    print(f"  Baryons only:  χ²_red = {chi2_red_baryon:.2f}")
    print(f"  Baryons + GPM: χ²_red = {chi2_red_gpm:.2f}")
    print(f"  Improvement:   {improvement:.1f}%")
    
    if chi2_red_gpm < chi2_red_baryon:
        print("\n✓ GPM IMPROVES FIT!")
    else:
        print("\n⚠ GPM does not improve fit (needs parameter tuning)")
    
    print("="*80)
    
    return {
        'galaxy': 'DDO154',
        'chi2_baryon': chi2_baryon,
        'chi2_gpm': chi2_gpm,
        'chi2_red_baryon': chi2_red_baryon,
        'chi2_red_gpm': chi2_red_gpm,
        'improvement_percent': improvement,
        'gpm_params': gpm_params
    }


if __name__ == '__main__':
    results = test_gpm_on_ddo154()
