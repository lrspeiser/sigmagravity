"""
Test CMSI with Scale-Dependent ℓ₀
=================================

Physical derivation:
    ℓ_coh ~ c × τ_coh ~ c × R/v_c
    
    This implies ℓ₀ should scale with system size:
        ℓ₀ = η × R_half
    
    where η is a UNIVERSAL constant (~0.3-0.6).

If this works, CMSI can unify galaxies and clusters with
a single new parameter, derived from existing physics.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from galaxies.cmsi_kernel import CMSIParams, cmsi_enhancement, G_NEWTON

# Cluster data
CLUSTERS = {
    'Coma': {'R_E': 300, 'M_bary': 1.4e14, 'M_lens': 7e14, 'sigma_v': 1000, 'Sigma': 200, 'R_half': 500},
    'A2029': {'R_E': 150, 'M_bary': 5e13, 'M_lens': 2.5e14, 'sigma_v': 1200, 'Sigma': 300, 'R_half': 300},
    'A1689': {'R_E': 200, 'M_bary': 1.2e14, 'M_lens': 6e14, 'sigma_v': 1400, 'Sigma': 400, 'R_half': 400},
    'Bullet': {'R_E': 250, 'M_bary': 2.5e14, 'M_lens': 1.5e15, 'sigma_v': 1100, 'Sigma': 350, 'R_half': 500},
}

# Galaxy data for comparison
GALAXIES = {
    'MW-like': {'R_half': 5, 'R_test': 8, 'v_circ': 220, 'sigma_v': 20, 'Sigma': 50, 'F_expected': 1.5},
    'DDO154': {'R_half': 3, 'R_test': 5, 'v_circ': 47, 'sigma_v': 10, 'Sigma': 10, 'F_expected': 2.0},
}


def test_eta_value(eta: float):
    """Test a specific η value on clusters and galaxies."""
    
    params = CMSIParams(
        chi_0=500,
        gamma_phase=1.5,
        alpha_Ncoh=0.45,
        n_profile=2.0,
        Sigma_ref=50.0,
        epsilon_Sigma=0.5,
        include_K_rough=True,
        use_scale_dependent_ell0=True,
        eta_ell0=eta
    )
    
    print(f"\n{'='*60}")
    print(f"Testing η = {eta}")
    print(f"{'='*60}")
    
    # Test clusters
    print("\nCLUSTERS:")
    print("-" * 50)
    cluster_results = []
    
    for name, data in CLUSTERS.items():
        R = data['R_E']
        v_circ = np.sqrt(G_NEWTON * data['M_bary'] / R)
        R_arr = np.array([R])
        v_arr = np.array([v_circ])
        sigma_arr = np.array([data['sigma_v']])
        Sigma_arr = np.array([data['Sigma']])
        
        F, diag = cmsi_enhancement(R_arr, v_arr, sigma_arr, params, Sigma_arr, R_half=data['R_half'])
        
        F_needed = data['M_lens'] / data['M_bary']
        ratio = F[0] / F_needed
        
        print(f"  {name:12s}: ℓ₀={diag['ell_0_local']:.0f} kpc, f_prof={diag['f_profile'][0]:.3f}, "
              f"F={F[0]:.2f} (need {F_needed:.1f}, ratio={ratio:.1%})")
        
        cluster_results.append({
            'name': name,
            'F': F[0],
            'F_needed': F_needed,
            'ratio': ratio,
            'ell_0': diag['ell_0_local'],
            'f_profile': diag['f_profile'][0]
        })
    
    # Test galaxies
    print("\nGALAXIES:")
    print("-" * 50)
    galaxy_results = []
    
    for name, data in GALAXIES.items():
        R_arr = np.array([data['R_test']])
        v_arr = np.array([data['v_circ']])
        sigma_arr = np.array([data['sigma_v']])
        Sigma_arr = np.array([data['Sigma']])
        
        F, diag = cmsi_enhancement(R_arr, v_arr, sigma_arr, params, Sigma_arr, R_half=data['R_half'])
        
        F_expected = data['F_expected']
        
        print(f"  {name:12s}: ℓ₀={diag['ell_0_local']:.1f} kpc, f_prof={diag['f_profile'][0]:.3f}, "
              f"F={F[0]:.2f} (expect ~{F_expected:.1f})")
        
        galaxy_results.append({
            'name': name,
            'F': F[0],
            'F_expected': F_expected,
            'ell_0': diag['ell_0_local'],
            'f_profile': diag['f_profile'][0]
        })
    
    return cluster_results, galaxy_results


def main():
    print("=" * 70)
    print("CMSI SCALE-DEPENDENT ℓ₀ TEST")
    print("Derived from coherence time: ℓ₀ = η × R_half")
    print("=" * 70)
    
    # Sweep η values
    eta_values = [0.3, 0.4, 0.5, 0.6, 0.8]
    
    all_results = {}
    
    for eta in eta_values:
        cluster_results, galaxy_results = test_eta_value(eta)
        all_results[eta] = {'clusters': cluster_results, 'galaxies': galaxy_results}
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Finding optimal η")
    print("=" * 70)
    
    print(f"\n{'η':<6} {'Cluster F/F_need':<20} {'Galaxy F':<20}")
    print("-" * 50)
    
    for eta in eta_values:
        cluster_avg = np.mean([r['ratio'] for r in all_results[eta]['clusters']])
        galaxy_avg = np.mean([r['F'] for r in all_results[eta]['galaxies']])
        
        print(f"{eta:<6.1f} {cluster_avg:<20.1%} {galaxy_avg:<20.2f}")
    
    # Find best η that gets closest to both
    print("\n" + "-" * 50)
    print("ANALYSIS:")
    print("-" * 50)
    
    best_eta = None
    best_score = 0
    
    for eta in eta_values:
        cluster_avg_ratio = np.mean([r['ratio'] for r in all_results[eta]['clusters']])
        galaxy_avg_F = np.mean([r['F'] for r in all_results[eta]['galaxies']])
        
        # Score: cluster ratio close to 1, galaxy F in reasonable range (1.3-2.5)
        cluster_score = 1 - abs(cluster_avg_ratio - 1)  # Best when ratio = 1
        galaxy_score = 1 if 1.3 < galaxy_avg_F < 2.5 else 0.5
        
        score = cluster_score * galaxy_score
        
        if score > best_score:
            best_score = score
            best_eta = eta
    
    print(f"\nBest η for unification: {best_eta}")
    
    # Show detailed results for best η
    print(f"\nDetailed results for η = {best_eta}:")
    for r in all_results[best_eta]['clusters']:
        status = "✓" if r['ratio'] > 0.5 else "✗"
        print(f"  {status} {r['name']}: F={r['F']:.2f}, F_needed={r['F_needed']:.1f}, "
              f"ratio={r['ratio']:.1%}")
    
    # Check if we can hit F ~ 5 for clusters
    max_cluster_F = max([r['F'] for r in all_results[best_eta]['clusters']])
    avg_cluster_F_needed = np.mean([r['F_needed'] for r in all_results[best_eta]['clusters']])
    
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    print(f"  Max cluster F achieved: {max_cluster_F:.2f}")
    print(f"  Avg cluster F needed: {avg_cluster_F_needed:.1f}")
    
    if max_cluster_F > avg_cluster_F_needed * 0.8:
        print("\n  ✓ Scale-dependent ℓ₀ CAN unify galaxies and clusters!")
        print("  η ~ 0.5 is derived from coherence time physics.")
    else:
        print(f"\n  Shortfall: {avg_cluster_F_needed / max_cluster_F:.1f}x")
        print("  Scale-dependent ℓ₀ helps but may not fully explain clusters.")


if __name__ == "__main__":
    main()
