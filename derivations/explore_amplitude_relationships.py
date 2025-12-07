#!/usr/bin/env python3
"""
Explore Theoretical Relationships Between A_galaxy and A_cluster

This script tests multiple hypotheses for deriving A_cluster from A_galaxy
(or vice versa) based on physical principles:

1. Volume filling factor
2. Mode counting (solid angle geometry)
3. Jeans length scaling
4. Coherence saturation factor
5. Density profile integration
6. Combined geometric + coherence models

For each hypothesis, we:
- Derive a predicted A_cluster from A_galaxy (or a ratio)
- Test against SPARC galaxies and Fox+ 2022 clusters
- Report which relationships best match observations

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import json

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # 1/s
kpc_to_m = 3.086e19
G_const = 6.674e-11
M_sun = 1.989e30
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))

# M/L ratios (Lelli+ 2016)
ML_DISK = 0.5
ML_BULGE = 0.7

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)

def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """Coherence window W(r) = 1 - (ξ/(ξ+r))^0.5"""
    xi = max(xi, 0.01)
    return 1 - np.power(xi / (xi + r), 0.5)

def predict_galaxy_velocity(R_kpc: np.ndarray, V_bar: np.ndarray, R_d: float, A: float) -> np.ndarray:
    """Predict rotation velocity for a galaxy."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    h = h_function(g_bar)
    xi = (2/3) * R_d
    W = W_coherence(R_kpc, xi)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)

def predict_cluster_mass(M_bar: float, r_kpc: float, A: float) -> float:
    """Predict lensing mass for a cluster."""
    r_m = r_kpc * kpc_to_m
    g_bar = G_const * M_bar * M_sun / r_m**2
    
    h = h_function(np.array([g_bar]))[0]
    # For clusters, W ≈ 1 at lensing radii
    Sigma = 1 + A * h
    
    return M_bar * Sigma

# =============================================================================
# DATA LOADERS
# =============================================================================

def load_sparc(data_dir: Path) -> List[Dict]:
    """Load SPARC galaxy data."""
    sparc_dir = data_dir / "Rotmod_LTG"
    if not sparc_dir.exists():
        return []
    
    galaxies = []
    for gf in sorted(sparc_dir.glob("*_rotmod.dat")):
        data = []
        with open(gf) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        data.append({
                            'R': float(parts[0]),
                            'V_obs': float(parts[1]),
                            'V_gas': float(parts[3]),
                            'V_disk': float(parts[4]),
                            'V_bulge': float(parts[5]) if len(parts) > 5 else 0.0
                        })
                    except ValueError:
                        continue
        
        if len(data) < 5:
            continue
        
        df = pd.DataFrame(data)
        df['V_disk_scaled'] = df['V_disk'] * np.sqrt(ML_DISK)
        df['V_bulge_scaled'] = df['V_bulge'] * np.sqrt(ML_BULGE)
        V_bar_sq = (np.sign(df['V_gas']) * df['V_gas']**2 + 
                    df['V_disk_scaled']**2 + df['V_bulge_scaled']**2)
        df['V_bar'] = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)
        
        valid = (df['V_bar'] > 0) & (df['R'] > 0) & (df['V_obs'] > 0)
        df = df[valid]
        
        if len(df) >= 5:
            idx = len(df) // 3
            R_d = df['R'].iloc[idx] if idx > 0 else df['R'].iloc[-1] / 2
            
            galaxies.append({
                'name': gf.stem.replace('_rotmod', ''),
                'R': df['R'].values,
                'V_obs': df['V_obs'].values,
                'V_bar': df['V_bar'].values,
                'R_d': R_d
            })
    
    return galaxies

def load_clusters(data_dir: Path) -> List[Dict]:
    """Load Fox+ 2022 cluster data."""
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    if not cluster_file.exists():
        return []
    
    df = pd.read_csv(cluster_file)
    df_valid = df[
        df['M500_1e14Msun'].notna() & 
        df['MSL_200kpc_1e12Msun'].notna() &
        (df['spec_z_constraint'] == 'yes')
    ].copy()
    df_valid = df_valid[df_valid['M500_1e14Msun'] > 2.0].copy()
    
    clusters = []
    f_baryon = 0.15
    
    for idx, row in df_valid.iterrows():
        M500 = row['M500_1e14Msun'] * 1e14
        M_bar_200 = 0.4 * f_baryon * M500
        M_lens_200 = row['MSL_200kpc_1e12Msun'] * 1e12
        
        clusters.append({
            'name': row['cluster'],
            'M_bar': M_bar_200,
            'M_lens': M_lens_200,
            'r_kpc': 200,
        })
    
    return clusters

# =============================================================================
# AMPLITUDE RELATIONSHIP HYPOTHESES
# =============================================================================

@dataclass
class AmplitudeHypothesis:
    name: str
    description: str
    compute_A_cluster: Callable[[float], float]  # Given A_galaxy, return A_cluster
    theoretical_basis: str

def get_hypotheses() -> List[AmplitudeHypothesis]:
    """Define all amplitude relationship hypotheses to test."""
    
    hypotheses = []
    
    # 1. Pure geometric mode counting (2D disk vs 3D sphere)
    hypotheses.append(AmplitudeHypothesis(
        name="mode_counting_pure",
        description="A_cl = A_gal × √2 (3D vs 2D modes)",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(2),
        theoretical_basis="Solid angle: 4π (sphere) vs 2π (disk) → √2 ratio"
    ))
    
    # 2. Mode counting with π factors
    hypotheses.append(AmplitudeHypothesis(
        name="mode_counting_pi",
        description="A_cl = A_gal × π/√3 (sphere/disk geometry)",
        compute_A_cluster=lambda A_gal: A_gal * np.pi / np.sqrt(3),
        theoretical_basis="π√2 (sphere) / √3 (disk) from coherence integrals"
    ))
    
    # 3. Coherence saturation factor
    # Galaxies: ⟨W⟩ ≈ 0.55, Clusters: W ≈ 1.0
    hypotheses.append(AmplitudeHypothesis(
        name="coherence_saturation",
        description="A_cl = A_gal × (1.0/0.55) (W saturation)",
        compute_A_cluster=lambda A_gal: A_gal * (1.0 / 0.55),
        theoretical_basis="Clusters have W=1, galaxies have ⟨W⟩≈0.55"
    ))
    
    # 4. Combined: mode counting × coherence saturation
    hypotheses.append(AmplitudeHypothesis(
        name="modes_plus_coherence",
        description="A_cl = A_gal × √2 × (1/0.55)",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(2) * (1.0 / 0.55),
        theoretical_basis="Geometric modes × coherence saturation"
    ))
    
    # 5. Volume filling factor
    # Disk: ~2.5% of sphere, Cluster ICM: ~10-15%
    hypotheses.append(AmplitudeHypothesis(
        name="volume_filling",
        description="A_cl = A_gal × √(0.10/0.025)",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(0.10 / 0.025),
        theoretical_basis="Volume filling: ICM (~10%) vs disk (~2.5%)"
    ))
    
    # 6. Jeans length scaling
    # λ_J ∝ σ/√(Gρ), ratio ~ 15-20 for clusters vs galaxies
    hypotheses.append(AmplitudeHypothesis(
        name="jeans_scaling",
        description="A_cl = A_gal × (λ_J_cl/λ_J_gal)^0.5",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(15),
        theoretical_basis="Jeans length ratio ~15 → √15 amplitude scaling"
    ))
    
    # 7. Density contrast scaling
    # ρ_disk/ρ_mean ~ 10^6, ρ_ICM/ρ_mean ~ 10^3 → ratio ~ 10^3
    hypotheses.append(AmplitudeHypothesis(
        name="density_contrast",
        description="A_cl = A_gal × (ρ_disk/ρ_ICM)^0.25",
        compute_A_cluster=lambda A_gal: A_gal * (1000)**0.25,
        theoretical_basis="Density contrast: disk ~10^6, ICM ~10^3 over mean"
    ))
    
    # 8. Surface density scaling
    # Σ_disk ~ 100 M☉/pc², Σ_cluster ~ 10 M☉/pc² at 200 kpc
    hypotheses.append(AmplitudeHypothesis(
        name="surface_density",
        description="A_cl = A_gal × (Σ_disk/Σ_cluster)^0.5",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(100 / 10),
        theoretical_basis="Surface density: disk ~100, cluster ~10 M☉/pc²"
    ))
    
    # 9. Crossing time / dynamical time
    # t_cross ~ R/σ: galaxy ~0.1 Gyr, cluster ~1 Gyr
    hypotheses.append(AmplitudeHypothesis(
        name="dynamical_time",
        description="A_cl = A_gal × (t_cl/t_gal)^0.5",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(10),
        theoretical_basis="Dynamical time ratio: cluster/galaxy ~ 10"
    ))
    
    # 10. Empirical power law fit
    # A_cluster = A_galaxy^α where α is fitted
    for alpha in [1.5, 2.0, 2.5, 3.0]:
        hypotheses.append(AmplitudeHypothesis(
            name=f"power_law_{alpha}",
            description=f"A_cl = A_gal^{alpha}",
            compute_A_cluster=lambda A_gal, a=alpha: A_gal**a,
            theoretical_basis=f"Empirical power law with exponent {alpha}"
        ))
    
    # 11. Linear offset
    # A_cluster = A_galaxy + offset
    for offset in [4.0, 5.0, 6.0, 7.0]:
        hypotheses.append(AmplitudeHypothesis(
            name=f"linear_offset_{offset}",
            description=f"A_cl = A_gal + {offset}",
            compute_A_cluster=lambda A_gal, o=offset: A_gal + o,
            theoretical_basis=f"Linear offset of {offset}"
        ))
    
    # 12. Multiplicative factor (scan range)
    for factor in [2.0, 3.0, 4.0, 4.6, 5.0, 6.0]:
        hypotheses.append(AmplitudeHypothesis(
            name=f"multiply_{factor}",
            description=f"A_cl = A_gal × {factor}",
            compute_A_cluster=lambda A_gal, f=factor: A_gal * f,
            theoretical_basis=f"Simple multiplicative factor {factor}"
        ))
    
    # 13. Geometric mean of mode counting and coherence
    hypotheses.append(AmplitudeHypothesis(
        name="geometric_mean",
        description="A_cl = A_gal × √(√2 × 1/0.55)",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(np.sqrt(2) * (1/0.55)),
        theoretical_basis="Geometric mean of mode counting and coherence factors"
    ))
    
    # 14. Dimensionality scaling (2D → 3D)
    hypotheses.append(AmplitudeHypothesis(
        name="dimensionality",
        description="A_cl = A_gal × (3/2)^1.5",
        compute_A_cluster=lambda A_gal: A_gal * (3/2)**1.5,
        theoretical_basis="Dimensionality: (D_cluster/D_disk)^1.5"
    ))
    
    # 15. Virial ratio scaling
    # 2K/|W| ~ 1 for both, but different K and W
    hypotheses.append(AmplitudeHypothesis(
        name="virial_scaling",
        description="A_cl = A_gal × (σ_cl/v_rot)^0.5",
        compute_A_cluster=lambda A_gal: A_gal * np.sqrt(1000/200),
        theoretical_basis="Virial: σ_cluster ~1000 km/s, v_rot ~200 km/s"
    ))
    
    return hypotheses

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_hypothesis(
    hypothesis: AmplitudeHypothesis,
    galaxies: List[Dict],
    clusters: List[Dict],
    A_galaxy: float = np.sqrt(3)
) -> Dict:
    """Evaluate a hypothesis against real data."""
    
    # Compute predicted A_cluster
    A_cluster_pred = hypothesis.compute_A_cluster(A_galaxy)
    
    # Evaluate on galaxies (using A_galaxy)
    galaxy_rms_list = []
    for gal in galaxies:
        V_pred = predict_galaxy_velocity(gal['R'], gal['V_bar'], gal['R_d'], A_galaxy)
        rms = np.sqrt(((gal['V_obs'] - V_pred)**2).mean())
        galaxy_rms_list.append(rms)
    
    galaxy_mean_rms = np.mean(galaxy_rms_list)
    
    # Evaluate on clusters (using predicted A_cluster)
    cluster_ratios = []
    for cl in clusters:
        M_pred = predict_cluster_mass(cl['M_bar'], cl['r_kpc'], A_cluster_pred)
        ratio = M_pred / cl['M_lens']
        if np.isfinite(ratio) and ratio > 0:
            cluster_ratios.append(ratio)
    
    if len(cluster_ratios) > 0:
        cluster_median_ratio = np.median(cluster_ratios)
        cluster_scatter = np.std(np.log10(cluster_ratios))
    else:
        cluster_median_ratio = np.nan
        cluster_scatter = np.nan
    
    # Score: how close is cluster ratio to 1.0?
    cluster_score = abs(np.log10(cluster_median_ratio)) if cluster_median_ratio > 0 else 999
    
    return {
        'hypothesis': hypothesis.name,
        'description': hypothesis.description,
        'A_galaxy': A_galaxy,
        'A_cluster_pred': A_cluster_pred,
        'ratio_A': A_cluster_pred / A_galaxy,
        'galaxy_rms': galaxy_mean_rms,
        'cluster_median_ratio': cluster_median_ratio,
        'cluster_scatter': cluster_scatter,
        'cluster_score': cluster_score,  # Lower is better (0 = perfect)
        'theoretical_basis': hypothesis.theoretical_basis
    }

def scan_A_galaxy(
    hypothesis: AmplitudeHypothesis,
    galaxies: List[Dict],
    clusters: List[Dict],
    A_galaxy_range: np.ndarray
) -> List[Dict]:
    """Scan over A_galaxy values to find optimal."""
    results = []
    for A_gal in A_galaxy_range:
        result = evaluate_hypothesis(hypothesis, galaxies, clusters, A_gal)
        results.append(result)
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("EXPLORING AMPLITUDE RELATIONSHIPS: A_galaxy ↔ A_cluster")
    print("=" * 80)
    
    data_dir = Path(__file__).parent.parent / "data"
    
    # Load data
    print("\nLoading data...")
    galaxies = load_sparc(data_dir)
    clusters = load_clusters(data_dir)
    print(f"  SPARC galaxies: {len(galaxies)}")
    print(f"  Clusters: {len(clusters)}")
    
    # Get hypotheses
    hypotheses = get_hypotheses()
    print(f"\nTesting {len(hypotheses)} hypotheses...")
    
    # Evaluate each hypothesis with A_galaxy = √3
    A_galaxy_default = np.sqrt(3)
    results = []
    
    for hyp in hypotheses:
        result = evaluate_hypothesis(hyp, galaxies, clusters, A_galaxy_default)
        results.append(result)
    
    # Sort by cluster score (how close to ratio = 1.0)
    results.sort(key=lambda x: x['cluster_score'])
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (sorted by cluster fit quality)")
    print("=" * 80)
    print(f"\n{'Hypothesis':<25} {'A_cl/A_gal':>10} {'A_cluster':>10} {'Cl Ratio':>10} {'Cl Score':>10}")
    print("-" * 70)
    
    for r in results[:20]:  # Top 20
        print(f"{r['hypothesis']:<25} {r['ratio_A']:>10.2f} {r['A_cluster_pred']:>10.2f} "
              f"{r['cluster_median_ratio']:>10.3f} {r['cluster_score']:>10.3f}")
    
    # Find the best hypothesis
    best = results[0]
    print(f"\n{'=' * 80}")
    print("BEST HYPOTHESIS")
    print(f"{'=' * 80}")
    print(f"  Name: {best['hypothesis']}")
    print(f"  Description: {best['description']}")
    print(f"  Theoretical basis: {best['theoretical_basis']}")
    print(f"  A_galaxy: {best['A_galaxy']:.3f}")
    print(f"  A_cluster (predicted): {best['A_cluster_pred']:.3f}")
    print(f"  Ratio A_cluster/A_galaxy: {best['ratio_A']:.3f}")
    print(f"  Galaxy RMS: {best['galaxy_rms']:.2f} km/s")
    print(f"  Cluster median ratio: {best['cluster_median_ratio']:.3f}")
    print(f"  Cluster scatter: {best['cluster_scatter']:.3f} dex")
    
    # Now scan A_galaxy to find optimal for best hypothesis
    print(f"\n{'=' * 80}")
    print("SCANNING A_galaxy FOR BEST HYPOTHESIS")
    print(f"{'=' * 80}")
    
    A_galaxy_range = np.linspace(1.0, 3.0, 41)
    
    # Find best theoretical hypothesis (excluding pure multiplicative)
    theoretical_hypotheses = [h for h in hypotheses 
                            if not h.name.startswith('multiply_') 
                            and not h.name.startswith('linear_offset_')
                            and not h.name.startswith('power_law_')]
    
    best_theoretical = None
    best_score = 999
    
    for hyp in theoretical_hypotheses:
        scan_results = scan_A_galaxy(hyp, galaxies, clusters, A_galaxy_range)
        for r in scan_results:
            # Combined score: galaxy RMS + 10 × cluster deviation
            combined_score = r['galaxy_rms'] + 10 * r['cluster_score']
            if combined_score < best_score:
                best_score = combined_score
                best_theoretical = r
                best_theoretical['hypothesis_obj'] = hyp
    
    if best_theoretical:
        print(f"\nBest theoretical relationship:")
        print(f"  Hypothesis: {best_theoretical['hypothesis']}")
        print(f"  Description: {best_theoretical['description']}")
        print(f"  Optimal A_galaxy: {best_theoretical['A_galaxy']:.3f}")
        print(f"  Predicted A_cluster: {best_theoretical['A_cluster_pred']:.3f}")
        print(f"  Galaxy RMS: {best_theoretical['galaxy_rms']:.2f} km/s")
        print(f"  Cluster ratio: {best_theoretical['cluster_median_ratio']:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / "amplitude_exploration"
    output_dir.mkdir(exist_ok=True)
    
    # Convert to serializable format
    results_json = []
    for r in results:
        r_copy = {k: v for k, v in r.items()}
        for k, v in r_copy.items():
            if isinstance(v, np.floating):
                r_copy[k] = float(v)
        results_json.append(r_copy)
    
    with open(output_dir / "amplitude_relationships.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'amplitude_relationships.json'}")
    
    # Summary table for README
    print(f"\n{'=' * 80}")
    print("SUMMARY: THEORETICAL RELATIONSHIPS")
    print(f"{'=' * 80}")
    print("\n| Hypothesis | Physical Basis | A_cl/A_gal | Cluster Ratio |")
    print("|------------|----------------|------------|---------------|")
    
    for r in results:
        if not r['hypothesis'].startswith('multiply_') and not r['hypothesis'].startswith('linear_offset_') and not r['hypothesis'].startswith('power_law_'):
            print(f"| {r['hypothesis']:<20} | {r['theoretical_basis'][:30]:<30} | {r['ratio_A']:.2f} | {r['cluster_median_ratio']:.3f} |")

if __name__ == "__main__":
    main()

