"""
Test Axisymmetric vs Spherical Kernels on Spiral Galaxies

This tests the impact of disk geometry on spiral galaxies where we expect
the largest improvements from axisymmetric convolution.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics import GravitationalPolarizationMemory
from galaxies.rotation_curves import GalaxyRotationCurve
from galaxies.environment_estimator import EnvironmentEstimator
from scipy.interpolate import PchipInterpolator


def create_baryon_density(gal, galaxy_name):
    """Create baryon density from SPARC data."""
    sparc_masses = load_sparc_masses(galaxy_name)
    M_stellar = sparc_masses['M_stellar']
    M_HI = sparc_masses['M_HI']
    M_total = sparc_masses['M_total']
    R_disk = sparc_masses['R_disk']
    R_HI = sparc_masses['R_HI']
    
    R_gas = max(R_HI, 1.5 * R_disk)
    Sigma0_stellar = M_stellar / (2.0 * np.pi * R_disk**2)
    Sigma0_gas = M_HI / (2.0 * np.pi * R_gas**2)
    h_z = 0.3  # kpc
    
    def rho_b(r_eval):
        r_safe = np.maximum(np.atleast_1d(r_eval), 1e-6)
        scalar_input = np.isscalar(r_eval)
        
        Sigma_stellar = Sigma0_stellar * np.exp(-r_safe / R_disk)
        Sigma_gas = Sigma0_gas * np.exp(-r_safe / R_gas)
        Sigma_total = Sigma_stellar + Sigma_gas
        rho = Sigma_total / (2.0 * h_z)
        
        return float(rho[0]) if scalar_input else rho
    
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
    
    return rho_b, M_total, R_disk, SBdisk


def test_galaxy(galaxy_name, alpha0=0.25, ell0=1.0, Qstar=2.0, sigmastar=25.0,
                nQ=2.0, nsig=2.0, p=0.5, Mstar=1e10, nM=2.5):
    """Test one galaxy with both spherical and axisymmetric kernels."""
    
    # Load data
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    
    r_data = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bar = np.sqrt(v_disk**2 + v_gas**2)
    
    # Create baryon density
    rho_b, M_total, R_disk, SBdisk = create_baryon_density(gal, galaxy_name)
    
    # Environment estimation
    estimator = EnvironmentEstimator()
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    Q, sigma_v = estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L=0.5, morphology=morphology)
    
    # GPM model
    gpm = GravitationalPolarizationMemory(
        alpha0=alpha0, ell0_kpc=ell0,
        Qstar=Qstar, sigmastar=sigmastar,
        nQ=nQ, nsig=nsig, p=p,
        Mstar_Msun=Mstar, nM=nM
    )
    
    h_z = 0.3  # kpc
    
    # Spherical kernel
    rho_coh_sph, gpm_params_sph = gpm.make_rho_coh(
        rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, M_total=M_total,
        use_axisymmetric=False
    )
    
    # Axisymmetric kernel
    rho_coh_axi, gpm_params_axi = gpm.make_rho_coh(
        rho_b, Q=Q, sigma_v=sigma_v, R_disk=R_disk, M_total=M_total,
        use_axisymmetric=True, h_z=h_z
    )
    
    # Compute rotation curves
    r_model = np.linspace(r_data.min(), r_data.max(), 100)
    
    # Spherical
    galaxy_sph = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_sph.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy_sph.set_coherence_halo_gpm(rho_coh_sph, gpm_params_sph)
    v_model_sph = galaxy_sph.circular_velocity(r_model)
    
    # Axisymmetric
    galaxy_axi = GalaxyRotationCurve(G=4.30091e-6)
    galaxy_axi.set_baryon_profile(M_disk=M_total, R_disk=R_disk)
    galaxy_axi.set_coherence_halo_gpm(rho_coh_axi, gpm_params_axi)
    v_model_axi = galaxy_axi.circular_velocity(r_model)
    
    # Interpolate to data points
    v_model_sph_at_data = PchipInterpolator(r_model, v_model_sph)(r_data)
    v_model_axi_at_data = PchipInterpolator(r_model, v_model_axi)(r_data)
    
    # Compute χ²
    chi2_baryon = np.sum((v_bar - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_gpm_sph = np.sum((v_model_sph_at_data - v_obs)**2 / (v_err**2 + 1e-10))
    chi2_gpm_axi = np.sum((v_model_axi_at_data - v_obs)**2 / (v_err**2 + 1e-10))
    
    dof = len(r_data)
    chi2_red_baryon = chi2_baryon / dof
    chi2_red_sph = chi2_gpm_sph / dof
    chi2_red_axi = chi2_gpm_axi / dof
    
    improvement_sph = (chi2_baryon - chi2_gpm_sph) / chi2_baryon * 100
    improvement_axi = (chi2_baryon - chi2_gpm_axi) / chi2_baryon * 100
    improvement_vs_sph = (chi2_gpm_sph - chi2_gpm_axi) / chi2_gpm_sph * 100
    
    return {
        'galaxy': galaxy_name,
        'morphology': morphology,
        'M_total': M_total,
        'R_disk': R_disk,
        'Q': Q,
        'sigma_v': sigma_v,
        'chi2_red_baryon': chi2_red_baryon,
        'chi2_red_sph': chi2_red_sph,
        'chi2_red_axi': chi2_red_axi,
        'improvement_sph': improvement_sph,
        'improvement_axi': improvement_axi,
        'improvement_vs_sph': improvement_vs_sph,
        'alpha': gpm_params_axi['alpha'],
        'ell_kpc': gpm_params_axi['ell_kpc']
    }


def main():
    """Test multiple galaxies spanning dwarf to massive spirals."""
    
    print("="*80)
    print("AXISYMMETRIC VS SPHERICAL KERNEL COMPARISON")
    print("="*80)
    print()
    
    # Test galaxies: dwarfs and spirals
    galaxies = [
        'DDO154',     # Dwarf
        'NGC2403',    # Normal spiral
        'NGC6503',    # Normal spiral
        'NGC2841',    # Massive spiral
        'NGC3198',    # Normal spiral
        'UGC06614',   # Normal spiral
    ]
    
    # Use optimal parameters from grid search
    alpha0 = 0.25
    ell0 = 1.0
    Mstar = 1e10
    nM = 2.5
    
    results = []
    
    for galaxy_name in galaxies:
        print(f"\nTesting {galaxy_name}...")
        print("-" * 40)
        
        try:
            result = test_galaxy(
                galaxy_name,
                alpha0=alpha0, ell0=ell0,
                Qstar=2.0, sigmastar=25.0,
                nQ=2.0, nsig=2.0, p=0.5,
                Mstar=Mstar, nM=nM
            )
            results.append(result)
            
            print(f"  Morphology: {result['morphology']}")
            print(f"  M_total: {result['M_total']:.2e} Msun")
            print(f"  R_disk: {result['R_disk']:.2f} kpc")
            print(f"  α_eff: {result['alpha']:.3f}, ℓ: {result['ell_kpc']:.2f} kpc")
            print(f"  Baryons only: χ²_red = {result['chi2_red_baryon']:.2f}")
            print(f"  GPM spherical: χ²_red = {result['chi2_red_sph']:.2f} ({result['improvement_sph']:+.1f}%)")
            print(f"  GPM axisymmetric: χ²_red = {result['chi2_red_axi']:.2f} ({result['improvement_axi']:+.1f}%)")
            print(f"  Axisym vs spherical: {result['improvement_vs_sph']:+.1f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'Galaxy':<12} {'Morph':<8} {'M [M☉]':<12} {'χ²_sph':<8} {'χ²_axi':<8} {'Δχ²[%]':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['galaxy']:<12} {r['morphology']:<8} {r['M_total']:.2e}  "
              f"{r['chi2_red_sph']:>6.2f}  {r['chi2_red_axi']:>6.2f}  {r['improvement_vs_sph']:>+6.1f}")
    
    # Statistics
    improvements = [r['improvement_vs_sph'] for r in results]
    mean_improvement = np.mean(improvements)
    median_improvement = np.median(improvements)
    
    print()
    print(f"Mean axisymmetric improvement: {mean_improvement:+.1f}%")
    print(f"Median axisymmetric improvement: {median_improvement:+.1f}%")
    
    # Morphology breakdown
    dwarf_improvements = [r['improvement_vs_sph'] for r in results if r['morphology'] == 'dwarf']
    spiral_improvements = [r['improvement_vs_sph'] for r in results if r['morphology'] in ['normal_spiral', 'massive_spiral']]
    
    if dwarf_improvements:
        print(f"Dwarfs: {np.mean(dwarf_improvements):+.1f}% (n={len(dwarf_improvements)})")
    if spiral_improvements:
        print(f"Spirals: {np.mean(spiral_improvements):+.1f}% (n={len(spiral_improvements)})")
    
    print()
    print("="*80)


if __name__ == '__main__':
    main()
