"""
Test Bulge Coupling on Massive Spirals

Tests NGC2841 and NGC0801 with bulge component included in coherence source.
These galaxies previously failed due to mass gating suppressing alpha_eff to ~0.

WITH BULGE: Coherence source = disk + bulge surface densities
Expects improvement as coherence halo now tracks complete baryon geometry.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.coherence_microphysics_axisym import AxiSymmetricYukawaConvolver
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator
from scipy.interpolate import PchipInterpolator


def create_baryon_components(gal, galaxy_name):
    """
    Create disk and bulge surface density functions from SPARC data.
    
    Returns disk, bulge, and total baryon densities.
    """
    sparc_masses = load_sparc_masses(galaxy_name)
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    # Disk components
    R_gas = max(R_HI, 1.5 * R_disk)
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)
    h_z = 0.3  # kpc
    
    # Check if SPARC has bulge velocity component
    v_bulge = gal.get('v_bulge', None)
    has_bulge = v_bulge is not None and np.any(v_bulge > 0)
    
    if has_bulge:
        # Extract bulge mass from v_bulge
        # v_bulge² = G M_bulge(<r) / r
        # Assume Hernquist profile for bulge: M(<r) = M_bulge × r² / (r + a)²
        # For simplicity, estimate M_bulge from peak v_bulge
        r_data = gal['r']
        v_bulge_max = np.max(v_bulge)
        r_at_max = r_data[np.argmax(v_bulge)]
        
        # Rough estimate: M_bulge ~ v_max² × r_max / G
        G_kpc = 4.30091e-6  # kpc (km/s)² / M_sun
        M_bulge_est = v_bulge_max**2 * r_at_max / G_kpc
        
        # Hernquist scale radius: a ~ 0.1 × R_disk (typical for spirals)
        a_bulge = 0.1 * R_disk
        
        print(f"  Bulge detected: M_bulge ~ {M_bulge_est:.2e} M☉, a = {a_bulge:.2f} kpc")
    else:
        # No bulge data, estimate from morphology
        # Massive spirals typically have M_bulge ~ 0.2-0.3 × M_stellar
        M_bulge_est = 0.25 * M_stellar
        a_bulge = 0.15 * R_disk
        
        print(f"  No v_bulge data, estimating: M_bulge ~ {M_bulge_est:.2e} M☉, a = {a_bulge:.2f} kpc")
    
    # Surface density functions
    def Sigma_disk(R):
        """Exponential disk (stellar + gas)."""
        Sigma_stellar = Sigma0_stellar * np.exp(-R / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-R / R_gas)
        return Sigma_stellar + Sigma_gas
    
    def Sigma_bulge(R):
        """Hernquist bulge surface density projection."""
        # Σ(R) for Hernquist: Σ(R) = M/(2π a²) × X(R) where X is complicated
        # Approximation: Σ(R) ≈ (M_bulge/(2π a²)) × (1 + R/a)^(-3)
        R_safe = np.maximum(R, 1e-6)
        return M_bulge_est / (2.0 * np.pi * a_bulge**2) * (1.0 + R_safe / a_bulge)**(-3)
    
    def rho_b_total(R):
        """Total baryon volume density at midplane."""
        return (Sigma_disk(R) + Sigma_bulge(R)) / (2.0 * h_z)
    
    # Load SBdisk for Q estimation
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
    
    return {
        'Sigma_disk': Sigma_disk,
        'Sigma_bulge': Sigma_bulge,
        'rho_total': rho_b_total,
        'M_total': M_total + M_bulge_est,
        'M_bulge': M_bulge_est,
        'R_disk': R_disk,
        'SBdisk': SBdisk,
        'h_z': h_z
    }


def test_galaxy_with_bulge(galaxy_name, alpha0=0.30, ell0=0.80, 
                           Mstar=2e10, nM=2.5, eta=1.0):
    """Test one galaxy with disk + bulge coherence source."""
    
    print(f"\n{'='*80}")
    print(f"Testing {galaxy_name} with Bulge Coupling")
    print('='*80)
    
    # Load data
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    print(f"\nData: {len(r_data)} points, R = {r_data.min():.2f}-{r_data.max():.2f} kpc")
    print(f"      v_obs = {v_obs.min():.1f}-{v_obs.max():.1f} km/s")
    
    # Create baryon components
    print("\nBaryon Components:")
    components = create_baryon_components(gal, galaxy_name)
    Sigma_disk = components['Sigma_disk']
    Sigma_bulge = components['Sigma_bulge']
    M_total = components['M_total']
    M_bulge = components['M_bulge']
    R_disk = components['R_disk']
    SBdisk = components['SBdisk']
    h_z = components['h_z']
    
    print(f"  M_total = {M_total:.2e} M☉ (disk + gas + bulge)")
    print(f"  M_bulge = {M_bulge:.2e} M☉ ({M_bulge/M_total*100:.1f}% of total)")
    print(f"  R_disk = {R_disk:.2f} kpc")
    
    # Environment estimation
    estimator = EnvironmentEstimator()
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    Q, sigma_v = estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L=0.5, morphology=morphology)
    
    print(f"\nEnvironment:")
    print(f"  Morphology: {morphology}")
    print(f"  Q = {Q:.2f}, σ_v = {sigma_v:.1f} km/s")
    
    # GPM parameters
    gpm = GravitationalPolarizationMemory(
        alpha0=alpha0, ell0_kpc=ell0,
        Qstar=2.0, sigmastar=25.0,
        nQ=2.0, nsig=2.0, p=0.5,
        Mstar_Msun=Mstar, nM=nM
    )
    
    # Compute effective parameters via GPM gates
    # Q gating
    f_Q = 1.0 / (1.0 + (Q / 2.0)**2.0)
    # Velocity dispersion gating
    f_sigma = 1.0 / (1.0 + (sigma_v / 25.0)**2.0)
    # Mass gating
    f_M = 1.0 / (1.0 + (M_total / Mstar)**nM)
    
    alpha_eff = alpha0 * f_Q * f_sigma * f_M
    ell_eff = ell0 * (R_disk**0.5)
    
    print(f"\nGPM Parameters:")
    print(f"  α₀ = {alpha0}, ℓ₀ = {ell0} kpc")
    print(f"  α_eff = {alpha_eff:.4f} (gating: {alpha_eff/alpha0*100:.1f}%)")
    print(f"  ℓ_eff = {ell_eff:.2f} kpc")
    
    # ===== TEST 1: Disk only (old method) =====
    print("\n" + "-"*80)
    print("TEST 1: Disk Only (without bulge)")
    print("-"*80)
    
    convolver_disk = AxiSymmetricYukawaConvolver(h_z=h_z)
    rho_coh_disk = convolver_disk.convolve_surface_density(
        Sigma_disk, alpha_eff, ell_eff, r_data,
        R_max=r_data.max() * 2, use_3d=False, apply_thickness_correction=True
    )
    
    # Rotation curve with disk-only coherence  
    # Use v_bar (baryons) + coherence contribution
    # For simplicity: v_total² = v_bar² + v_coh²
    # where v_coh comes from rho_coh via circular velocity formula
    
    # Coherence contribution to velocity
    # v_coh² = G M_coh(<r) / r where M_coh = 4π ∫ rho_coh r² dr
    # Approximate: v_coh ~ sqrt(G × rho_coh × ℓ² × R)
    G_kpc = 4.30091e-6
    v_coh_disk = np.sqrt(G_kpc * rho_coh_disk * ell_eff**2 * r_data)
    
    v_model_disk_only = np.sqrt(v_bar**2 + v_coh_disk**2)
    
    chi2_disk_only = np.sum((v_model_disk_only - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_red_disk_only = chi2_disk_only / len(r_data)
    
    # ===== TEST 2: Disk + Bulge (new method) =====
    print("\n" + "-"*80)
    print("TEST 2: Disk + Bulge (with bulge in coherence source)")
    print("-"*80)
    
    convolver_bulge = AxiSymmetricYukawaConvolver(h_z=h_z)
    rho_coh_bulge = convolver_bulge.convolve_disk_plus_bulge(
        Sigma_disk, Sigma_bulge, alpha_eff, ell_eff, r_data,
        R_max=r_data.max() * 2, apply_thickness_correction=True
    )
    
    # Apply temporal memory smoothing
    # Need v_circ for memory timescale
    v_circ_approx = v_bar  # Use baryon velocity as approximation
    rho_coh_bulge_smoothed = convolver_bulge.apply_temporal_memory(
        rho_coh_bulge, r_data, v_circ_approx, eta=eta
    )
    
    # Rotation curve with disk+bulge coherence
    # Coherence contribution with bulge included
    v_coh_bulge = np.sqrt(G_kpc * rho_coh_bulge_smoothed * ell_eff**2 * r_data)
    
    v_model_bulge = np.sqrt(v_bar**2 + v_coh_bulge**2)
    
    chi2_bulge = np.sum((v_model_bulge - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_red_bulge = chi2_bulge / len(r_data)
    
    # ===== COMPARISON =====
    chi2_baryon = np.sum((v_bar - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_red_baryon = chi2_baryon / len(r_data)
    
    improvement_disk_only = (chi2_baryon - chi2_disk_only) / chi2_baryon * 100
    improvement_bulge = (chi2_baryon - chi2_bulge) / chi2_baryon * 100
    improvement_vs_disk = (chi2_disk_only - chi2_bulge) / chi2_disk_only * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Baryons only:        χ²_red = {chi2_red_baryon:>8.2f}")
    print(f"GPM (disk only):     χ²_red = {chi2_red_disk_only:>8.2f}  ({improvement_disk_only:+6.1f}%)")
    print(f"GPM (disk+bulge):    χ²_red = {chi2_red_bulge:>8.2f}  ({improvement_bulge:+6.1f}%)")
    print(f"\nBulge effect:        Δχ² = {improvement_vs_disk:+.1f}% vs disk-only")
    
    if improvement_bulge > improvement_disk_only:
        print("\n✓ BULGE IMPROVES FIT!")
    elif improvement_bulge > 0:
        print("\n⚠ Bulge helps but still needs tuning")
    else:
        print("\n✗ Still failing (mass gating too strong)")
    
    print("="*80)
    
    return {
        'galaxy': galaxy_name,
        'M_total': M_total,
        'M_bulge': M_bulge,
        'alpha_eff': alpha_eff,
        'chi2_red_baryon': chi2_red_baryon,
        'chi2_red_disk_only': chi2_red_disk_only,
        'chi2_red_bulge': chi2_red_bulge,
        'improvement_disk_only': improvement_disk_only,
        'improvement_bulge': improvement_bulge,
        'improvement_vs_disk': improvement_vs_disk
    }


def main():
    """Test NGC2841 and NGC0801 with bulge coupling."""
    
    print("="*80)
    print("BULGE COUPLING TEST: Massive Spiral Failures")
    print("="*80)
    print()
    print("Testing NGC2841 and NGC0801 with bulge mass in coherence source.")
    print("Previous failures: α_eff ≈ 0 due to mass gating.")
    print("Expected: Bulge mass increases coherence response in inner regions.")
    print()
    
    # Optimal parameters from grid search
    alpha0 = 0.30
    ell0 = 0.80
    Mstar = 2e10
    nM = 2.5
    eta = 1.0  # Memory timescale factor
    
    galaxies = ['NGC2841', 'NGC0801']
    results = []
    
    for galaxy_name in galaxies:
        try:
            result = test_galaxy_with_bulge(
                galaxy_name, alpha0=alpha0, ell0=ell0,
                Mstar=Mstar, nM=nM, eta=eta
            )
            results.append(result)
        except Exception as e:
            print(f"\nERROR testing {galaxy_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if len(results) > 0:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print()
        print(f"{'Galaxy':<12} {'M_bulge/M_tot':<15} {'α_eff':<10} {'Disk-only':<12} {'Disk+Bulge':<12} {'Δ vs Disk'}")
        print("-"*80)
        
        for r in results:
            bulge_frac = r['M_bulge'] / r['M_total'] * 100
            print(f"{r['galaxy']:<12} {bulge_frac:>6.1f}%        {r['alpha_eff']:>8.4f}  "
                  f"{r['improvement_disk_only']:>+8.1f}%    {r['improvement_bulge']:>+8.1f}%    "
                  f"{r['improvement_vs_disk']:>+8.1f}%")
        
        print()
        print("="*80)


if __name__ == '__main__':
    main()
