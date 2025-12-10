"""
Test the resonant halo solver on real SPARC galaxies.

This script:
1. Loads SPARC galaxy rotation curves
2. Computes gain function g(r) from observed profiles
3. Solves for coherence field φ(r)
4. Compares rotation curve fits to baselines
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from galaxies.resonant_halo_solver import (
    ResonantParams, ResonantHaloSolver, 
    gain_function, toomre_Q
)


def estimate_dispersion(v_obs, morphology_guess='dwarf'):
    """
    Estimate velocity dispersion from morphology.
    
    Parameters:
    -----------
    v_obs : array
        Observed velocities (km/s)
    morphology_guess : str
        'dwarf', 'spiral', 'lsb', 'hsb'
        
    Returns:
    --------
    sigma_v : float
        Velocity dispersion (km/s)
    """
    v_max = np.max(v_obs)
    
    if morphology_guess == 'dwarf':
        # Dwarfs are cold: σ ~ 10-20 km/s
        return 15.0
    elif morphology_guess == 'lsb':
        # LSBs are cold: σ ~ 15-25 km/s
        return 20.0
    elif morphology_guess == 'spiral':
        # Spirals: σ ~ v_max/10
        return max(v_max / 10, 20.0)
    elif morphology_guess == 'hsb':
        # HSBs are hot: σ ~ 30-50 km/s
        return 40.0
    else:
        # Default
        return 25.0


def load_SBdisk_from_file(galaxy_name):
    """
    Load SBdisk (surface brightness) directly from SPARC file.
    
    Returns:
    --------
    SBdisk : array
        Surface brightness in L☉/pc² (SPARC [3.6] band)
    """
    from data_integration.load_real_data import RealDataLoader
    loader = RealDataLoader()
    rotmod_dir = os.path.join(loader.base_data_dir, 'Rotmod_LTG')
    filepath = os.path.join(rotmod_dir, f'{galaxy_name}_rotmod.dat')
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse data (skip comments)
    data_lines = [l for l in lines if not l.startswith('#')]
    
    # Expected columns: Rad  Vobs  errV  Vgas  Vdisk  Vbul  SBdisk  SBbul
    SBdisk = []
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 7:
            SBdisk.append(float(parts[6]))  # SBdisk is column 7 (0-indexed: 6)
    
    return np.array(SBdisk)


def compute_surface_density(galaxy_name, v_gas, r):
    """
    Compute proper baryonic surface density from SBdisk + gas.
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name (to load SBdisk)
    v_gas : array
        Gas contribution to velocity (km/s)
    r : array
        Radius (kpc)
        
    Returns:
    --------
    Sigma_b : array
        Surface density (M☉/kpc²)
    mask : array (bool)
        True where Σ_b is valid (SBdisk > 0)
    """
    # Get SBdisk from file
    SBdisk = load_SBdisk_from_file(galaxy_name)
    
    # Convert to mass surface density
    M_L = 0.5  # M/L ratio for [3.6] band
    Sigma_disk = SBdisk * M_L  # M☉/pc²
    Sigma_disk_kpc2 = Sigma_disk * 1e6  # M☉/kpc²
    
    # Gas surface density from v_gas
    G_kpc = 4.302e-3  # kpc (M☉)⁻¹ (km/s)²
    M_gas_enc = r * v_gas**2 / G_kpc
    dM_gas = np.gradient(M_gas_enc, r)
    Sigma_gas_kpc2 = dM_gas / (2 * np.pi * np.maximum(r, 0.1))
    Sigma_gas_kpc2 = np.maximum(Sigma_gas_kpc2, 0)
    
    # Total
    Sigma_b = Sigma_disk_kpc2 + Sigma_gas_kpc2
    
    # Mask: only valid where SBdisk > 0
    mask = SBdisk > 0
    
    return Sigma_b, mask


def auto_lambda_phi(r, Sigma_b, sigma_v, Omega, dlnOm_dlnr, base_params):
    """Choose λ_φ by aligning m=1 resonance with max S_Q (coldest radius)."""
    Q = toomre_Q(r, Sigma_b, sigma_v * np.ones_like(r), Omega, dlnOm_dlnr)
    S_Q = 0.5 * (1.0 + np.tanh((base_params.Q_c - Q) / base_params.Delta_Q))
    idx = int(np.nanargmax(S_Q))
    r_target = float(np.clip(r[idx], 0.3, 8.0))  # stay in data range
    lambda_phi = float(np.clip(2.0 * np.pi * r_target, 2.5, 12.0))
    return lambda_phi, r_target


def fit_galaxy(galaxy_name, params, morphology='dwarf', plot=True):
    """
    Fit resonant halo model to a SPARC galaxy.
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    params : ResonantParams
        Model parameters
    morphology : str
        Morphology guess for dispersion
    plot : bool
        Whether to plot results
        
    Returns:
    --------
    results : dict
        Fit results including χ²
    """
    print("\n" + "="*80)
    print(f"FITTING: {galaxy_name}")
    print("="*80)
    
    # Load data
    loader = RealDataLoader()
    try:
        gal = loader.load_rotmod_galaxy(galaxy_name)
    except Exception as e:
        print(f"ERROR: Could not load galaxy: {e}")
        return None
    
    # Extract data
    r = gal['r']
    v_obs = gal['v_obs']
    v_err = gal['v_err']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal['v_bulge']
    
    # Compute baryonic velocity
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # Estimate dispersion
    sigma_v = estimate_dispersion(v_obs, morphology)
    print(f"\nAssumed σ_v = {sigma_v:.1f} km/s ({morphology})")
    
    # Compute surface density
    Sigma_b, valid_mask = compute_surface_density(galaxy_name, v_gas, r)
    print(f"Surface density range: {Sigma_b.min():.1e} - {Sigma_b.max():.1e} M☉/kpc²")
    print(f"Valid points (SBdisk > 0): {np.sum(valid_mask)} / {len(r)}")
    
    # Filter to valid points only
    if np.sum(valid_mask) < 3:
        print(f"ERROR: Too few valid points ({np.sum(valid_mask)})")
        return None
    
    r = r[valid_mask]
    v_obs = v_obs[valid_mask]
    v_err = v_err[valid_mask]
    v_disk = v_disk[valid_mask]
    v_gas = v_gas[valid_mask]
    v_bulge = v_bulge[valid_mask]
    Sigma_b = Sigma_b[valid_mask]
    
    # Recompute baryonic velocity after filtering
    v_bar = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
    
    # Compute rotation curve quantities
    Omega = v_obs / r  # Angular frequency (km/s/kpc)
    
    # Compute dlnΩ/dlnr
    ln_Omega = np.log(Omega + 1e-10)
    ln_r = np.log(r)
    dlnOm_dlnr = np.gradient(ln_Omega, ln_r)
    
    # Auto-tune λ_φ to align resonance with coldest radius (m=1)
    lambda_phi_auto, r_target = auto_lambda_phi(r, Sigma_b, sigma_v, Omega, dlnOm_dlnr, params)
    params_local = ResonantParams(
        m0=params.m0,
        R_coh=params.R_coh,
        alpha=params.alpha,
        lambda_phi=lambda_phi_auto,
        Q_c=max(params.Q_c, 1.5),
        Delta_Q=max(params.Delta_Q, 0.6),  # broaden coldness gate
        sigma_c=params.sigma_c,
        sigma_m=max(params.sigma_m, 0.5),  # broaden resonance
        m_max=params.m_max,
        beta=params.beta,
        lambda_4=params.lambda_4,
    )

    print(f"\nComputing gain function (auto λ_φ = {params_local.lambda_phi:.2f} kpc at r*={r_target:.2f} kpc)...")
    g = gain_function(r, Sigma_b, sigma_v * np.ones_like(r), Omega, dlnOm_dlnr, params_local)
    
    # Compute Toomre Q
    Q = toomre_Q(r, Sigma_b, sigma_v * np.ones_like(r), Omega, dlnOm_dlnr)
    
    print(f"Gain range: {g.min():.4f} - {g.max():.4f} kpc⁻²")
    print(f"Toomre Q range: {Q.min():.2f} - {Q.max():.2f}")
    print(f"Tachyonic zone: {np.sum(g > params.m0**2)} / {len(r)} points")
    
    # Convert to volume density (assume scale height)
    h_z = 0.3  # kpc (typical)
    rho_b = Sigma_b / (2 * h_z)
    
    # Solve field equation
    print("\nSolving field equation...")
    solver = ResonantHaloSolver(params_local)
    phi, diagnostics = solver.solve_phi(r, rho_b, g)
    
    if not diagnostics['success']:
        print(f"WARNING: Solver did not converge!")
        print(f"  Iterations: {diagnostics['niter']}")
        print(f"  Residual: {diagnostics['residual']:.2e}")
        return None
    
    print(f"✓ Converged in {diagnostics['niter']} iterations")
    print(f"  Residual: {diagnostics['residual']:.2e}")
    print(f"  Field amplitude: max(|φ|) = {np.max(np.abs(phi)):.2f}")
    
    # Compute field energy density
    energy = solver.field_energy_density(r, phi, g)
    rho_phi = energy['total']
    
    print(f"  Energy density range: {rho_phi.min():.2e} - {rho_phi.max():.2e} M☉/kpc³")
    
    # Convert to circular velocity contribution
    # M_phi(<r) = 4π ∫ rho_phi(r') r'² dr'
    dr = np.gradient(r)
    integrand = rho_phi * r**2 * dr
    M_phi = 4 * np.pi * np.cumsum(integrand)
    
    # Velocity from field (km/s)
    G_kpc = 4.302e-3  # kpc (M☉)^-1 (km/s)²
    v_phi = np.sqrt(G_kpc * M_phi / (r + 1e-10))
    
    print(f"  Field velocity range: {v_phi.min():.1f} - {v_phi.max():.1f} km/s")
    
    # Total effective velocity
    v_eff = np.sqrt(v_bar**2 + v_phi**2)
    
    # Compute χ²
    chi2 = np.sum((v_eff - v_obs)**2 / (v_err**2 + 1e-10))
    dof = len(r)
    chi2_reduced = chi2 / dof
    
    print(f"\nFIT QUALITY:")
    print(f"  χ² = {chi2:.1f}")
    print(f"  DOF = {dof}")
    print(f"  χ²_red = {chi2_reduced:.2f}")
    
    # Plot results
    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{galaxy_name} - Resonant Halo Model', fontsize=14, fontweight='bold')
        
        # 1. Rotation curve
        ax = axes[0, 0]
        ax.errorbar(r, v_obs, yerr=v_err, fmt='ko', label='Observed', alpha=0.5)
        ax.plot(r, v_bar, 'b--', label='Baryons only', linewidth=2)
        ax.plot(r, v_eff, 'r-', label='Baryons + Field', linewidth=2)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Rotation Curve')
        
        # 2. Field profile
        ax = axes[0, 1]
        ax.plot(r, phi, 'purple', linewidth=2)
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Field φ')
        ax.grid(True, alpha=0.3)
        ax.set_title('Coherence Field')
        
        # 3. Gain function
        ax = axes[0, 2]
        ax.plot(r, g, 'orange', linewidth=2, label='g(r)')
        ax.axhline(params.m0**2, color='r', linestyle='--', label='m₀²', alpha=0.5)
        ax.fill_between(r, 0, g, where=(g > params.m0**2), 
                        alpha=0.2, color='pink', label='Tachyonic')
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Gain g(r) (kpc⁻²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Amplification Gain')
        
        # 4. Energy density
        ax = axes[1, 0]
        ax.semilogy(r, rho_phi, 'purple', linewidth=2)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Energy Density ρ_φ (M☉/kpc³)')
        ax.grid(True, alpha=0.3)
        ax.set_title('Field Energy Density')
        
        # 5. Velocity contributions
        ax = axes[1, 1]
        ax.plot(r, v_bar, 'b--', label='Baryons', linewidth=2)
        ax.plot(r, v_phi, 'purple', label='Field', linewidth=2)
        ax.plot(r, v_eff, 'r-', label='Total', linewidth=2)
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Velocity (km/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Velocity Decomposition')
        
        # 6. Toomre Q
        ax = axes[1, 2]
        ax.plot(r, Q, 'green', linewidth=2)
        ax.axhline(params.Q_c, color='r', linestyle='--', label=f'Q_c = {params.Q_c}', alpha=0.5)
        ax.fill_between(r, 0, Q, where=(Q < params.Q_c), 
                        alpha=0.2, color='lightgreen', label='Unstable')
        ax.set_xlabel('Radius (kpc)')
        ax.set_ylabel('Toomre Q')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Disk Stability')
        
        plt.tight_layout()
        
        # Save figure
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'sparc_resonant_fits')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{galaxy_name}_fit.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot: {output_file}")
        
        plt.close(fig)
    
    # Return results
    results = {
        'galaxy': galaxy_name,
        'chi2': chi2,
        'chi2_reduced': chi2_reduced,
        'dof': dof,
        'r': r,
        'v_obs': v_obs,
        'v_err': v_err,
        'v_bar': v_bar,
        'v_phi': v_phi,
        'v_eff': v_eff,
        'phi': phi,
        'g': g,
        'Q': Q,
        'rho_phi': rho_phi,
        'diagnostics': diagnostics
    }
    
    return results


def test_multiple_galaxies(galaxy_names, params):
    """
    Test resonant model on multiple galaxies.
    
    Parameters:
    -----------
    galaxy_names : list
        List of galaxy names
    params : ResonantParams
        Model parameters
        
    Returns:
    --------
    all_results : list
        List of fit results
    """
    all_results = []
    
    for gal_name in galaxy_names:
        results = fit_galaxy(gal_name, params, morphology='dwarf', plot=True)
        if results is not None:
            all_results.append(results)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF FITS")
    print("="*80)
    print(f"{'Galaxy':<15} {'χ²':<10} {'χ²_red':<10} {'DOF':<6}")
    print("-"*80)
    
    total_chi2 = 0
    total_dof = 0
    
    for res in all_results:
        print(f"{res['galaxy']:<15} {res['chi2']:<10.1f} {res['chi2_reduced']:<10.2f} {res['dof']:<6}")
        total_chi2 += res['chi2']
        total_dof += res['dof']
    
    print("-"*80)
    print(f"{'TOTAL':<15} {total_chi2:<10.1f} {total_chi2/total_dof:<10.2f} {total_dof:<6}")
    print("="*80)
    
    return all_results


def main():
    """Main test script."""
    print("="*80)
    print("RESONANT HALO SOLVER - SPARC GALAXY TEST")
    print("="*80)
    
    # List available galaxies
    loader = RealDataLoader()
    galaxies = loader.list_available_galaxies()
    
    if not galaxies:
        print("\nERROR: No galaxies found!")
        return
    
    # Define model parameters - BOOSTED for real galaxies
    params = ResonantParams(
        m0=0.01,          # kpc⁻¹ (REDUCED for wider tachyonic zones)
        R_coh=2.0,        # kpc (REDUCED for stronger g0)
        alpha=10.0,       # dimensionless (BOOSTED 10x for stronger gain)
        lambda_phi=8.0,   # kpc (initial; will be auto-tuned per galaxy)
        Q_c=2.0,          # allow even stable disks
        sigma_c=30.0,     # km/s (dispersion cutoff)
        sigma_m=0.6,      # resonance width (broad)
        m_max=2,          # max resonance order
        beta=2.0,         # coupling strength (BOOSTED)
        lambda_4=0.5      # quartic coupling
    )
    
    print("\nModel Parameters:")
    print(f"  m₀ = {params.m0} kpc⁻¹")
    print(f"  R_coh = {params.R_coh} kpc")
    print(f"  λ_φ = {params.lambda_phi} kpc")
    print(f"  Q_c = {params.Q_c}")
    print(f"  σ_c = {params.sigma_c} km/s")
    
    # Test on specific galaxies
    # Based on data verification, NGC2403 and NGC6503 have many Q < 1.5 points
    test_galaxies = ['NGC2403', 'NGC6503']
    
    # Verify they exist
    test_galaxies = [g for g in test_galaxies if g in galaxies]
    
    # If none found, use first few
    if not test_galaxies:
        test_galaxies = galaxies[:3]
    
    print(f"\nTesting on {len(test_galaxies)} galaxies:")
    for gal in test_galaxies:
        print(f"  - {gal}")
    
    # Run fits
    results = test_multiple_galaxies(test_galaxies, params)
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    
    return results


if __name__ == '__main__':
    results = main()
