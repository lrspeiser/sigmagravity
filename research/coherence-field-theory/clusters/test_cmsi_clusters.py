"""
CMSI Cluster Lensing Test
=========================

Critical test: Can CMSI produce the required enhancement at cluster scales?

Clusters have:
- High Σ (ICM + galaxies) → source factor OK
- High σ_v (~1000 km/s) → low N_coh → suppressed enhancement
- Need F ~ 5-10 to match lensing mass without dark matter

Test clusters:
- Coma (Abell 1656): nearby, well-studied
- A2029: massive, relaxed
- A1689: strong lensing, well-constrained Einstein radius

Key question: Does CMSI's natural suppression at high-σ kill it for clusters,
or is there enough enhancement from the huge baryonic mass?
"""

import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from galaxies.cmsi_kernel import (
    CMSIParams,
    cmsi_enhancement,
    compute_phase_coherence,
    compute_radial_coherence_profile,
    G_NEWTON,
    C_LIGHT
)


# =============================================================================
# Cluster Data (from observations)
# =============================================================================

@dataclass
class ClusterData:
    """Observed cluster properties."""
    name: str
    z: float                    # Redshift
    R_E_kpc: float             # Einstein radius [kpc]
    M_lens_Msun: float         # Lensing mass within R_E [M_sun]
    M_bary_Msun: float         # Baryonic mass within R_E [M_sun] (gas + galaxies)
    sigma_v_kms: float         # Velocity dispersion [km/s]
    R_core_kpc: float          # Core radius [kpc]
    Sigma_gas_Msunpc2: float   # Gas surface density at R_E [M_sun/pc²]


# Real cluster data (approximate values from literature)
CLUSTERS = {
    'Coma': ClusterData(
        name='Coma (A1656)',
        z=0.023,
        R_E_kpc=300,            # No strong lensing, use ~300 kpc for dynamics
        M_lens_Msun=7e14,       # Total mass within 300 kpc
        M_bary_Msun=1.4e14,     # ~20% baryonic
        sigma_v_kms=1000,       # Velocity dispersion
        R_core_kpc=200,
        Sigma_gas_Msunpc2=200   # ICM surface density
    ),
    'A2029': ClusterData(
        name='A2029',
        z=0.077,
        R_E_kpc=150,            # Strong lensing Einstein radius
        M_lens_Msun=2.5e14,     # Lensing mass within R_E
        M_bary_Msun=5e13,       # ~20% baryonic
        sigma_v_kms=1200,
        R_core_kpc=100,
        Sigma_gas_Msunpc2=300
    ),
    'A1689': ClusterData(
        name='A1689',
        z=0.183,
        R_E_kpc=200,            # Famous strong lensing arc at ~200 kpc
        M_lens_Msun=6e14,       # Well-constrained lensing mass
        M_bary_Msun=1.2e14,     # ~20% baryonic
        sigma_v_kms=1400,
        R_core_kpc=150,
        Sigma_gas_Msunpc2=400
    ),
    'Bullet': ClusterData(
        name='Bullet Cluster',
        z=0.296,
        R_E_kpc=250,
        M_lens_Msun=1.5e15,     # Total lensing mass
        M_bary_Msun=2.5e14,     # Separated from lensing centroid!
        sigma_v_kms=1100,       # Pre-merger dispersion
        R_core_kpc=200,
        Sigma_gas_Msunpc2=350
    )
}


# =============================================================================
# CMSI Cluster Calculation
# =============================================================================

def compute_cluster_v_circ(M_bary: float, R_kpc: float) -> float:
    """
    Estimate circular velocity from enclosed baryonic mass.
    
    v_c = sqrt(G M / R)
    """
    # G in kpc (km/s)^2 / M_sun
    return np.sqrt(G_NEWTON * M_bary / R_kpc)


def compute_cmsi_cluster_enhancement(
    cluster: ClusterData,
    params: CMSIParams
) -> Dict:
    """
    Compute CMSI enhancement factor for a cluster.
    
    Key physics:
    - High σ_v → low N_coh → suppressed phase coherence
    - High Σ → good source factor
    - Question: Does the suppression kill the enhancement?
    """
    R = cluster.R_E_kpc
    
    # Circular velocity from baryonic mass
    v_circ = compute_cluster_v_circ(cluster.M_bary_Msun, R)
    
    # Velocity dispersion
    sigma_v = cluster.sigma_v_kms
    
    # Surface density
    Sigma = cluster.Sigma_gas_Msunpc2
    
    # Arrays for CMSI kernel
    R_arr = np.array([R])
    v_arr = np.array([v_circ])
    sigma_arr = np.array([sigma_v])
    Sigma_arr = np.array([Sigma])
    
    # Compute CMSI enhancement
    F_CMSI, diag = cmsi_enhancement(R_arr, v_arr, sigma_arr, params, Sigma_arr)
    
    # What enhancement would we NEED to match lensing mass?
    F_needed = cluster.M_lens_Msun / cluster.M_bary_Msun
    
    # Predicted lensing mass with CMSI
    M_pred = cluster.M_bary_Msun * F_CMSI[0]
    
    return {
        'cluster': cluster.name,
        'R_E': R,
        'v_circ': v_circ,
        'sigma_v': sigma_v,
        'Sigma': Sigma,
        'N_coh': diag['N_coh'][0],
        'source_factor': diag['source_factor'][0],
        'v_over_c_squared': diag['v_over_c_squared'][0],
        'f_profile': diag['f_profile'][0],
        'F_CMSI': F_CMSI[0],
        'F_needed': F_needed,
        'M_bary': cluster.M_bary_Msun,
        'M_lens_obs': cluster.M_lens_Msun,
        'M_lens_pred': M_pred,
        'ratio_pred_obs': M_pred / cluster.M_lens_Msun,
        'success': F_CMSI[0] >= F_needed * 0.5  # Within 50% counts as partial success
    }


def analyze_why_suppressed(result: Dict, params: CMSIParams):
    """
    Analyze why CMSI might be suppressed for clusters.
    """
    print(f"\n  PHYSICS BREAKDOWN:")
    print(f"  ------------------")
    
    # Phase coherence
    N_coh = result['N_coh']
    v_over_sigma = result['v_circ'] / result['sigma_v']
    print(f"  Phase coherence:")
    print(f"    v_c/σ_v = {v_over_sigma:.2f}")
    print(f"    N_coh = (v/σ)^{params.gamma_phase} = {N_coh:.2f}")
    print(f"    N_coh^α = {N_coh**params.alpha_Ncoh:.3f}")
    
    # Relativistic factor
    v_c2 = result['v_over_c_squared']
    print(f"  Relativistic factor:")
    print(f"    (v/c)² = {v_c2:.2e}")
    print(f"    χ_0 × (v/c)² = {params.chi_0 * v_c2:.2f}")
    
    # Source factor
    sf = result['source_factor']
    print(f"  Source factor:")
    print(f"    (Σ/Σ_ref)^ε = {sf:.2f}")
    
    # Combined amplitude
    amplitude = params.chi_0 * v_c2 * sf * (N_coh ** params.alpha_Ncoh) * result['f_profile']
    print(f"  Combined amplitude:")
    print(f"    χ_0 × (v/c)² × source × N_coh^α × f(R) = {amplitude:.3f}")
    print(f"    F_CMSI = 1 + amplitude = {1 + amplitude:.3f}")


# =============================================================================
# Main Test
# =============================================================================

def main():
    print("=" * 70)
    print("CMSI CLUSTER LENSING TEST")
    print("Critical test: Can CMSI work at cluster scales?")
    print("=" * 70)
    
    # Use best SPARC parameters
    params = CMSIParams(
        chi_0=500,
        gamma_phase=1.5,
        alpha_Ncoh=0.45,
        ell_0_kpc=3.0,
        n_profile=2.0,
        Sigma_ref=50.0,
        epsilon_Sigma=0.5,
        include_K_rough=True
    )
    
    print(f"\nCMSI Parameters (from SPARC optimization):")
    print(f"  χ_0 = {params.chi_0}")
    print(f"  γ_phase = {params.gamma_phase}")
    print(f"  α_Ncoh = {params.alpha_Ncoh}")
    print(f"  ℓ_0 = {params.ell_0_kpc} kpc")
    
    print("\n" + "=" * 70)
    print("CLUSTER RESULTS")
    print("=" * 70)
    
    results = []
    
    for name, cluster in CLUSTERS.items():
        print(f"\n{'─' * 60}")
        print(f"CLUSTER: {cluster.name}")
        print(f"{'─' * 60}")
        print(f"  z = {cluster.z}")
        print(f"  R_E = {cluster.R_E_kpc} kpc")
        print(f"  M_bary = {cluster.M_bary_Msun:.2e} M☉")
        print(f"  M_lens = {cluster.M_lens_Msun:.2e} M☉ (observed)")
        print(f"  σ_v = {cluster.sigma_v_kms} km/s")
        print(f"  Σ_gas = {cluster.Sigma_gas_Msunpc2} M☉/pc²")
        
        result = compute_cmsi_cluster_enhancement(cluster, params)
        results.append(result)
        
        print(f"\n  CMSI PREDICTION:")
        print(f"    v_circ (bary) = {result['v_circ']:.1f} km/s")
        print(f"    F_CMSI = {result['F_CMSI']:.3f}")
        print(f"    F_needed = {result['F_needed']:.1f}")
        print(f"    M_pred = {result['M_lens_pred']:.2e} M☉")
        print(f"    M_pred / M_obs = {result['ratio_pred_obs']:.2%}")
        
        status = "✓ POSSIBLE" if result['success'] else "✗ FAILS"
        print(f"\n  STATUS: {status}")
        
        analyze_why_suppressed(result, params)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Cluster':<20} {'F_CMSI':<10} {'F_needed':<10} {'M_pred/M_obs':<15} {'Status'}")
    print("-" * 65)
    for r in results:
        status = "✓" if r['success'] else "✗"
        print(f"{r['cluster']:<20} {r['F_CMSI']:<10.3f} {r['F_needed']:<10.1f} {r['ratio_pred_obs']:<15.1%} {status}")
    
    # Critical analysis
    print("\n" + "=" * 70)
    print("CRITICAL ANALYSIS")
    print("=" * 70)
    
    avg_F = np.mean([r['F_CMSI'] for r in results])
    avg_needed = np.mean([r['F_needed'] for r in results])
    
    print(f"\n  Mean F_CMSI achieved: {avg_F:.3f}")
    print(f"  Mean F_needed: {avg_needed:.1f}")
    print(f"  Shortfall: {avg_needed / avg_F:.1f}x")
    
    print("\n  KEY INSIGHT:")
    print("  The high velocity dispersion (~1000 km/s) gives v_c/σ_v ~ 1")
    print("  This means N_coh ~ 1 (no coherent enhancement)")
    print("  CMSI is naturally suppressed at cluster scales!")
    
    if avg_F < 2:
        print("\n  VERDICT: CMSI cannot explain cluster lensing masses.")
        print("  The σ-gating that helps galaxies kills clusters.")
        print("  Either:")
        print("    1. Dark matter exists at cluster scales")
        print("    2. A different mechanism operates in clusters")
        print("    3. The phase coherence model needs modification")
    else:
        print("\n  VERDICT: CMSI provides partial enhancement.")
        print("  May need supplemental physics or parameter tuning.")
    
    # Can we tune χ_0 to fix clusters?
    print("\n" + "-" * 50)
    print("WHAT χ_0 WOULD BE NEEDED?")
    print("-" * 50)
    
    # Find χ_0 needed for Coma
    coma = results[0]
    needed_amplitude = coma['F_needed'] - 1
    current_amplitude = coma['F_CMSI'] - 1
    chi_0_needed = params.chi_0 * (needed_amplitude / current_amplitude) if current_amplitude > 0 else float('inf')
    
    print(f"  For Coma (F_needed = {coma['F_needed']:.1f}):")
    print(f"    Current χ_0 = {params.chi_0} → F = {coma['F_CMSI']:.3f}")
    print(f"    Would need χ_0 ~ {chi_0_needed:.0f} to match")
    print(f"    That's {chi_0_needed/params.chi_0:.0f}x higher!")
    print(f"\n  BUT: This would over-boost galaxies by the same factor.")
    print("  CMSI cannot simultaneously fit galaxies and clusters.")
    
    print("\n" + "=" * 70)
    print("Test complete.")


if __name__ == "__main__":
    main()
