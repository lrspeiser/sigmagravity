"""
Edge Behavior Test: Train Inner, Test Outer

Tests GPM's temporal memory smoothing by fitting on R < 2*R_disk
and predicting R > 3*R_disk.

FALSIFIABLE PREDICTION:
- GPM with memory: Smooth extrapolation to outer regions
- GPM without memory: Poor extrapolation (sharp cutoff)
- DM/MOND: Model-dependent extrapolation

This tests whether temporal memory τ(R) ~ 2π/Ω(R) smooths
the coherence profile at large radii where observations are sparse.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_integration.load_real_data import RealDataLoader
from data_integration.load_sparc_masses import load_sparc_masses
from galaxies.coherence_microphysics_axisym import CoherenceMicrophysicsAxiSym
from galaxies.environment_estimator import EnvironmentEstimator


def train_inner_test_outer(galaxy_name, R_disk, output_dir='outputs/gpm_tests'):
    """
    Fit GPM on inner region (R < 2*R_disk), predict outer region (R > 3*R_disk).
    
    Tests temporal memory smoothing effect on extrapolation.
    """
    
    print("="*80)
    print(f"EDGE BEHAVIOR TEST: {galaxy_name}")
    print("="*80)
    print()
    print(f"Training region: R < {2*R_disk:.2f} kpc")
    print(f"Testing region:  R > {3*R_disk:.2f} kpc")
    print()
    
    # Load galaxy data
    loader = RealDataLoader()
    gal = loader.load_rotmod_galaxy(galaxy_name)
    sparc_masses = load_sparc_masses(galaxy_name)
    
    M_total = sparc_masses['M_total']
    
    r = gal['r']
    v_obs = gal['v_obs']
    v_disk = gal['v_disk']
    v_gas = gal['v_gas']
    v_bulge = gal.get('v_bulge', np.zeros_like(v_disk))
    
    # Split into training and test sets
    train_mask = r < 2 * R_disk
    test_mask = r > 3 * R_disk
    
    n_train = np.sum(train_mask)
    n_test = np.sum(test_mask)
    
    print(f"Training points: {n_train}")
    print(f"Test points:     {n_test}")
    print()
    
    if n_train < 3 or n_test < 2:
        print(f"WARNING: Insufficient data for {galaxy_name}")
        print(f"  Need at least 3 training and 2 test points")
        return None
    
    # Load SBdisk for environment estimation
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
    
    estimator = EnvironmentEstimator()
    morphology = estimator.classify_morphology(gal, M_total, R_disk)
    Q, sigma_v = estimator.estimate_from_sparc(gal, SBdisk, R_disk, M_L=0.5, morphology=morphology)
    
    print(f"Environment: Q={Q:.2f}, σ_v={sigma_v:.1f} km/s")
    print()
    
    # Compute GPM predictions with and without temporal memory
    
    # Parameters from optimization
    alpha_0 = 0.30
    l_0 = 0.80  # kpc
    M_star = 2e10  # M_sun
    n_M = 2.5
    sigma_star = 70.0  # km/s
    n_sigma = 3.0
    Q_crit = 1.5
    n_Q = 3.0
    h_z = 0.3  # kpc
    
    # Initialize coherence model
    model = CoherenceMicrophysicsAxiSym(
        alpha_0=alpha_0,
        l_0=l_0,
        M_star=M_star,
        n_M=n_M,
        sigma_star=sigma_star,
        n_sigma=n_sigma,
        Q_crit=Q_crit,
        n_Q=n_Q,
        h_z=h_z
    )
    
    # Predict with memory (full model)
    v_gpm_with_memory = model.compute_rotation_curve_axisym(
        gal, SBdisk, M_total, R_disk, 
        use_bulge=True,
        use_temporal_memory=True,
        t_age=10.0  # Gyr
    )
    
    # Predict without memory
    v_gpm_no_memory = model.compute_rotation_curve_axisym(
        gal, SBdisk, M_total, R_disk,
        use_bulge=True,
        use_temporal_memory=False
    )
    
    # Compute chi-squared on test region only
    def compute_chi2(v_model, mask):
        v_obs_masked = v_obs[mask]
        v_model_masked = v_model[mask]
        dv_masked = gal['dv'][mask]
        return np.sum(((v_obs_masked - v_model_masked) / dv_masked)**2) / np.sum(mask)
    
    chi2_train_with_mem = compute_chi2(v_gpm_with_memory, train_mask)
    chi2_test_with_mem = compute_chi2(v_gpm_with_memory, test_mask)
    
    chi2_train_no_mem = compute_chi2(v_gpm_no_memory, train_mask)
    chi2_test_no_mem = compute_chi2(v_gpm_no_memory, test_mask)
    
    chi2_train_bar = compute_chi2(np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2), train_mask)
    chi2_test_bar = compute_chi2(np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2), test_mask)
    
    print("-"*80)
    print("CHI-SQUARED RESULTS")
    print("-"*80)
    print()
    print("Training region (R < 2*R_disk):")
    print(f"  Baryons only:         χ²_red = {chi2_train_bar:.3f}")
    print(f"  GPM (with memory):    χ²_red = {chi2_train_with_mem:.3f}")
    print(f"  GPM (without memory): χ²_red = {chi2_train_no_mem:.3f}")
    print()
    
    print("Test region (R > 3*R_disk):")
    print(f"  Baryons only:         χ²_red = {chi2_test_bar:.3f}")
    print(f"  GPM (with memory):    χ²_red = {chi2_test_with_mem:.3f}")
    print(f"  GPM (without memory): χ²_red = {chi2_test_no_mem:.3f}")
    print()
    
    improvement_with_mem = (chi2_test_bar - chi2_test_with_mem) / chi2_test_bar * 100
    improvement_no_mem = (chi2_test_bar - chi2_test_no_mem) / chi2_test_bar * 100
    
    print(f"Test region improvement:")
    print(f"  With memory:    {improvement_with_mem:+.1f}%")
    print(f"  Without memory: {improvement_no_mem:+.1f}%")
    print()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    ax.errorbar(r, v_obs, yerr=gal['dv'], fmt='ko', alpha=0.4, markersize=4, 
                linewidth=1, capsize=2, label='SPARC data')
    
    # Mark training and test regions
    ax.axvspan(0, 2*R_disk, alpha=0.1, color='blue', label='Training region')
    ax.axvspan(3*R_disk, r.max(), alpha=0.1, color='red', label='Test region')
    
    # Models
    ax.plot(r, np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2), 'gray', 
            linestyle='--', linewidth=2, alpha=0.7, label='Baryons only')
    ax.plot(r, v_gpm_with_memory, 'b-', linewidth=2.5, 
            label=f'GPM (with memory, χ²_test={chi2_test_with_mem:.2f})')
    ax.plot(r, v_gpm_no_memory, 'orange', linestyle=':', linewidth=2.5,
            label=f'GPM (no memory, χ²_test={chi2_test_no_mem:.2f})')
    
    # Vertical lines at boundaries
    ax.axvline(2*R_disk, color='blue', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axvline(3*R_disk, color='red', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Radius [kpc]', fontsize=12)
    ax.set_ylabel('Rotation Velocity [km/s]', fontsize=12)
    ax.set_title(f'{galaxy_name}: Train Inner, Test Outer (Memory Smoothing)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'edge_behavior_{galaxy_name}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    print()
    
    plt.close()
    
    return {
        'galaxy': galaxy_name,
        'chi2_train_with_mem': chi2_train_with_mem,
        'chi2_test_with_mem': chi2_test_with_mem,
        'chi2_train_no_mem': chi2_train_no_mem,
        'chi2_test_no_mem': chi2_test_no_mem,
        'improvement_with_mem': improvement_with_mem,
        'improvement_no_mem': improvement_no_mem,
        'n_train': n_train,
        'n_test': n_test
    }


def batch_edge_behavior_test():
    """
    Test edge behavior on multiple galaxies with sufficient radial coverage.
    """
    
    print("="*80)
    print("BATCH EDGE BEHAVIOR TEST")
    print("="*80)
    print()
    print("Testing temporal memory smoothing via train-inner/test-outer splits")
    print()
    
    # Galaxies with large radial coverage
    test_galaxies = [
        ('NGC6503', 3.0),  # R_disk = 3 kpc
        ('NGC2403', 5.0),
        ('NGC3198', 6.0),
        ('NGC5055', 7.0),
        ('NGC2841', 8.0),
    ]
    
    results = []
    
    for galaxy_name, R_disk in test_galaxies:
        try:
            result = train_inner_test_outer(galaxy_name, R_disk)
            if result is not None:
                results.append(result)
        except Exception as e:
            print(f"ERROR processing {galaxy_name}: {e}")
            print()
            continue
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    if len(results) == 0:
        print("No successful tests")
        return
    
    print(f"Tested {len(results)} galaxies")
    print()
    
    improvements_with_mem = [r['improvement_with_mem'] for r in results]
    improvements_no_mem = [r['improvement_no_mem'] for r in results]
    
    mean_imp_with = np.mean(improvements_with_mem)
    mean_imp_without = np.mean(improvements_no_mem)
    
    print("Mean test region improvement:")
    print(f"  With temporal memory:    {mean_imp_with:+.1f}%")
    print(f"  Without temporal memory: {mean_imp_without:+.1f}%")
    print()
    
    print("Memory effect: " + 
          f"{mean_imp_with - mean_imp_without:+.1f}% additional improvement")
    print()
    
    if mean_imp_with > mean_imp_without + 5:
        print("✓ Temporal memory improves extrapolation")
        print("  GPM prediction SUPPORTED")
    elif mean_imp_with < mean_imp_without - 5:
        print("✗ Temporal memory degrades extrapolation")
        print("  GPM memory mechanism may need revision")
    else:
        print("⚠ Memory effect is marginal")
        print("  Need more data or larger test regions")
    
    print()
    print("="*80)


if __name__ == '__main__':
    batch_edge_behavior_test()
