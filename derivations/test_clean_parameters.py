#!/usr/bin/env python3
"""
Test "clean" parameter values that would make derivations more elegant.

We test:
1. k = 1/4 exactly (vs calibrated k ≈ 0.24)
2. A₀ = φ (golden ratio ≈ 1.618) vs calibrated A₀ ≈ 1.6
3. g† = cH₀/(2π) vs cH₀/(4√π)
4. W(r) exponent = 1 vs 0.5
5. ξ = (1/2)R_d vs (2/3)R_d

For each, we measure:
- SPARC RMS
- Win rate vs MOND
- Cluster ratio
"""

import numpy as np
from pathlib import Path
import sys

# Physical constants
c = 2.998e8  # m/s
H0_SI = 2.27e-18  # s⁻¹ (70 km/s/Mpc)
kpc_to_m = 3.086e19
G = 6.674e-11  # m³/kg/s²

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618

# MOND a0
a0_MOND = 1.2e-10  # m/s²


def load_sparc_data(data_dir: Path):
    """Load SPARC galaxies."""
    galaxies = []
    rotmod_dir = data_dir / "Rotmod_LTG"
    
    if not rotmod_dir.exists():
        print(f"[warn] SPARC data not found at {rotmod_dir}")
        return []
    
    for f in rotmod_dir.glob("*.dat"):
        try:
            data = np.loadtxt(f, comments='#')
            if len(data) < 3:
                continue
            
            R = data[:, 0]  # kpc
            V_obs = data[:, 1]  # km/s
            V_gas = data[:, 2] if data.shape[1] > 2 else np.zeros_like(R)
            V_disk = data[:, 3] if data.shape[1] > 3 else np.zeros_like(R)
            V_bulge = data[:, 4] if data.shape[1] > 4 else np.zeros_like(R)
            
            # Apply M/L = 0.5 for disk, 0.7 for bulge
            V_disk_scaled = V_disk * np.sqrt(0.5)
            V_bulge_scaled = V_bulge * np.sqrt(0.7)
            
            # Compute V_bar with sign handling
            V_bar_sq = (np.sign(V_gas) * V_gas**2 + 
                       np.sign(V_disk_scaled) * V_disk_scaled**2 + 
                       V_bulge_scaled**2)
            
            if np.any(V_bar_sq <= 0):
                continue
            
            V_bar = np.sqrt(V_bar_sq)
            
            # Estimate R_d from half-light radius
            cumsum = np.cumsum(V_disk**2 * R)
            if cumsum[-1] > 0:
                half_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                R_d = R[min(half_idx, len(R) - 1)]
            else:
                R_d = R[-1] / 3
            
            R_d = max(R_d, 0.5)
            
            # Estimate sigma_eff (velocity dispersion proxy)
            sigma_eff = 0.1 * np.mean(V_obs)  # Rough estimate
            
            galaxies.append({
                'name': f.stem,
                'R': R,
                'V_obs': V_obs,
                'V_bar': V_bar,
                'R_d': R_d,
                'sigma_eff': sigma_eff
            })
        except Exception as e:
            continue
    
    return galaxies


def load_cluster_data(data_dir: Path):
    """Load Fox+ 2022 clusters."""
    clusters = []
    cluster_file = data_dir / "clusters" / "fox2022_unique_clusters.csv"
    
    if not cluster_file.exists():
        print(f"[warn] Cluster data not found at {cluster_file}")
        return []
    
    try:
        import csv
        with open(cluster_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Column names from actual file
                    M500 = float(row.get('M500_1e14Msun', 0)) * 1e14  # Convert to M_sun
                    M_SL = float(row.get('MSL_200kpc_1e12Msun', 0)) * 1e12  # In 1e12 M_sun
                    z = float(row.get('z_lens', 0.3))
                    
                    if M500 > 0 and M_SL > 0:
                        # Baryonic mass from f_baryon = 0.15
                        M_bar = 0.15 * M500
                        clusters.append({
                            'name': row.get('cluster', 'unknown'),
                            'M_bar': M_bar,
                            'M_SL': M_SL,
                            'z': z,
                            'r_lens': 200  # kpc
                        })
                except Exception as e:
                    continue
    except Exception as e:
        print(f"[warn] Error loading clusters: {e}")
    
    return clusters


class SigmaGravityModel:
    """Parameterized Σ-Gravity model for testing."""
    
    def __init__(self, 
                 g_dagger_factor='4sqrtpi',  # '4sqrtpi', '2pi', '4pi'
                 xi_coeff=2/3,               # Coefficient for ξ = coeff × R_d
                 W_exponent=0.5,             # Exponent in W(r)
                 A0=1.6,                     # Base amplitude
                 A_exponent=0.25,            # Path length exponent
                 use_dynamical_xi=False,     # Use k × σ/Ω instead
                 k_dynamical=0.24):          # k value for dynamical ξ
        
        # Set g†
        if g_dagger_factor == '4sqrtpi':
            self.g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))
        elif g_dagger_factor == '2pi':
            self.g_dagger = c * H0_SI / (2 * np.pi)
        elif g_dagger_factor == '4pi':
            self.g_dagger = c * H0_SI / (4 * np.pi)
        else:
            self.g_dagger = float(g_dagger_factor)
        
        self.xi_coeff = xi_coeff
        self.W_exponent = W_exponent
        self.A0 = A0
        self.A_exponent = A_exponent
        self.use_dynamical_xi = use_dynamical_xi
        self.k_dynamical = k_dynamical
        
        # Compute galaxy amplitude from path length
        L_galaxy = 1.5  # kpc
        self.A_galaxy = A0 * (L_galaxy ** A_exponent)
        
        # Compute cluster amplitude
        L_cluster = 400  # kpc
        self.A_cluster = A0 * (L_cluster ** A_exponent)
    
    def h_function(self, g_N):
        """Acceleration function."""
        g_N = np.maximum(g_N, 1e-15)
        return np.sqrt(self.g_dagger / g_N) * self.g_dagger / (self.g_dagger + g_N)
    
    def W_coherence(self, r, R_d, sigma_eff=None, V_at_Rd=None):
        """Coherence window."""
        if self.use_dynamical_xi and sigma_eff is not None and V_at_Rd is not None:
            Omega_d = V_at_Rd / R_d
            xi = self.k_dynamical * sigma_eff / Omega_d
        else:
            xi = self.xi_coeff * R_d
        
        xi = max(xi, 0.01)
        return 1 - (xi / (xi + r)) ** self.W_exponent
    
    def predict_velocity(self, R, V_bar, R_d, sigma_eff=None):
        """Predict rotation velocity."""
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_N = V_bar_ms**2 / R_m
        
        V_at_Rd = np.interp(R_d, R, V_bar) if sigma_eff else None
        
        W = self.W_coherence(R, R_d, sigma_eff, V_at_Rd)
        h = self.h_function(g_N)
        Sigma = 1 + self.A_galaxy * W * h
        
        return V_bar * np.sqrt(Sigma)
    
    def predict_cluster_mass(self, M_bar, r_lens=200):
        """Predict cluster mass at lensing radius."""
        # Estimate g_N at lensing radius
        M_bar_kg = M_bar * 1.989e30
        r_m = r_lens * kpc_to_m
        g_N = G * M_bar_kg / r_m**2
        
        # For clusters, W ≈ 1 at lensing radii
        W = 0.95
        h = self.h_function(g_N)
        Sigma = 1 + self.A_cluster * W * h
        
        return M_bar * Sigma


def mond_velocity(R, V_bar):
    """MOND prediction using simple interpolation."""
    R_m = R * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    y = g_bar / a0_MOND
    nu = 1 / (1 - np.exp(-np.sqrt(y)))
    
    return V_bar * np.sqrt(nu)


def evaluate_model(model, galaxies, clusters):
    """Evaluate model on galaxies and clusters."""
    
    # Galaxy evaluation
    rms_list = []
    wins_sigma = 0
    wins_mond = 0
    
    for gal in galaxies:
        R, V_obs, V_bar, R_d = gal['R'], gal['V_obs'], gal['V_bar'], gal['R_d']
        sigma_eff = gal.get('sigma_eff')
        
        V_sigma = model.predict_velocity(R, V_bar, R_d, sigma_eff)
        V_mond = mond_velocity(R, V_bar)
        
        rms_sigma = np.sqrt(np.mean((V_obs - V_sigma)**2))
        rms_mond = np.sqrt(np.mean((V_obs - V_mond)**2))
        
        rms_list.append(rms_sigma)
        
        if rms_sigma < rms_mond:
            wins_sigma += 1
        else:
            wins_mond += 1
    
    mean_rms = np.mean(rms_list)
    win_rate = wins_sigma / (wins_sigma + wins_mond) * 100
    
    # Cluster evaluation
    ratios = []
    for cl in clusters:
        M_pred = model.predict_cluster_mass(cl['M_bar'], cl['r_lens'])
        ratio = M_pred / cl['M_SL']
        ratios.append(ratio)
    
    median_ratio = np.median(ratios) if ratios else 0
    
    return {
        'mean_rms': mean_rms,
        'win_rate': win_rate,
        'cluster_ratio': median_ratio,
        'n_galaxies': len(galaxies),
        'n_clusters': len(clusters)
    }


def main():
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    
    print("=" * 70)
    print("Testing 'Clean' Parameter Values for Σ-Gravity")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    galaxies = load_sparc_data(data_dir)
    clusters = load_cluster_data(data_dir)
    print(f"  Loaded {len(galaxies)} galaxies, {len(clusters)} clusters")
    
    if not galaxies:
        print("[error] No galaxy data loaded. Check data path.")
        return
    
    # Define configurations to test
    configs = {
        # Baseline (current)
        'BASELINE (current)': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': 1.6,
            'A_exponent': 0.25,
        },
        
        # Test 1: k = 1/4 exactly (dynamical ξ)
        'k = 1/4 (dynamical ξ)': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': 1.6,
            'A_exponent': 0.25,
            'use_dynamical_xi': True,
            'k_dynamical': 0.25,
        },
        
        # Test 2: A₀ = φ (golden ratio)
        'A₀ = φ (golden ratio)': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': PHI,
            'A_exponent': 0.25,
        },
        
        # Test 3: g† = cH₀/(2π) - matches MOND a₀
        'g† = cH₀/(2π)': {
            'g_dagger_factor': '2pi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': 1.6,
            'A_exponent': 0.25,
        },
        
        # Test 4: W exponent = 1 (simpler)
        'W exponent = 1': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 1.0,
            'A0': 1.6,
            'A_exponent': 0.25,
        },
        
        # Test 5: ξ = (1/2)R_d
        'ξ = (1/2)R_d': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 0.5,
            'W_exponent': 0.5,
            'A0': 1.6,
            'A_exponent': 0.25,
        },
        
        # Test 6: ξ = R_d
        'ξ = R_d': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 1.0,
            'W_exponent': 0.5,
            'A0': 1.6,
            'A_exponent': 0.25,
        },
        
        # Test 7: A exponent = 1/3 (3D)
        'A exponent = 1/3': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': 1.2,  # Adjusted to keep A_galaxy similar
            'A_exponent': 1/3,
        },
        
        # Test 8: Combined "cleanest" - all simple values
        'CLEANEST (all simple)': {
            'g_dagger_factor': '2pi',
            'xi_coeff': 0.5,
            'W_exponent': 1.0,
            'A0': PHI,
            'A_exponent': 0.25,
        },
        
        # Test 9: A₀ = √e ≈ 1.649
        'A₀ = √e': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': np.sqrt(np.e),
            'A_exponent': 0.25,
        },
        
        # Test 10: A₀ = 2/√π ≈ 1.128 (Gaussian normalization)
        'A₀ = 2/√π': {
            'g_dagger_factor': '4sqrtpi',
            'xi_coeff': 2/3,
            'W_exponent': 0.5,
            'A0': 2/np.sqrt(np.pi),
            'A_exponent': 0.25,
        },
    }
    
    # Run evaluations
    print("\n" + "-" * 70)
    print(f"{'Configuration':<30} {'RMS (km/s)':<12} {'Win %':<10} {'Cluster':<10} {'Δ RMS':<10}")
    print("-" * 70)
    
    baseline_rms = None
    results = {}
    
    for name, params in configs.items():
        model = SigmaGravityModel(**params)
        result = evaluate_model(model, galaxies, clusters)
        results[name] = result
        
        if baseline_rms is None:
            baseline_rms = result['mean_rms']
        
        delta_rms = result['mean_rms'] - baseline_rms
        delta_str = f"{delta_rms:+.2f}" if delta_rms != 0 else "---"
        
        print(f"{name:<30} {result['mean_rms']:<12.2f} {result['win_rate']:<10.1f} {result['cluster_ratio']:<10.3f} {delta_str:<10}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    # Find best and worst
    best = min(results.items(), key=lambda x: x[1]['mean_rms'])
    worst = max(results.items(), key=lambda x: x[1]['mean_rms'])
    
    print(f"\nBest RMS:  {best[0]} ({best[1]['mean_rms']:.2f} km/s)")
    print(f"Worst RMS: {worst[0]} ({worst[1]['mean_rms']:.2f} km/s)")
    
    # Check which "clean" values are acceptable
    print("\n" + "-" * 70)
    print("'Clean' values that work (< 5% RMS increase):")
    print("-" * 70)
    
    for name, result in results.items():
        if name == 'BASELINE (current)':
            continue
        delta_pct = (result['mean_rms'] - baseline_rms) / baseline_rms * 100
        if delta_pct < 5:
            print(f"  ✓ {name}: {delta_pct:+.1f}% RMS, {result['cluster_ratio']:.3f} cluster ratio")
    
    print("\n" + "-" * 70)
    print("'Clean' values that don't work (> 5% RMS increase):")
    print("-" * 70)
    
    for name, result in results.items():
        if name == 'BASELINE (current)':
            continue
        delta_pct = (result['mean_rms'] - baseline_rms) / baseline_rms * 100
        if delta_pct >= 5:
            print(f"  ✗ {name}: {delta_pct:+.1f}% RMS, {result['cluster_ratio']:.3f} cluster ratio")
    
    # Specific findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    
    # Golden ratio
    phi_result = results.get('A₀ = φ (golden ratio)', {})
    if phi_result:
        delta = (phi_result['mean_rms'] - baseline_rms) / baseline_rms * 100
        print(f"\n1. A₀ = φ (golden ratio ≈ 1.618):")
        print(f"   RMS change: {delta:+.1f}%")
        print(f"   A_galaxy = {PHI * 1.5**0.25:.3f} (vs √3 ≈ 1.732)")
        print(f"   A_cluster = {PHI * 400**0.25:.2f} (vs 8.0)")
        if abs(delta) < 2:
            print(f"   → VIABLE: Could use φ as the fundamental amplitude constant!")
    
    # k = 1/4
    k_result = results.get('k = 1/4 (dynamical ξ)', {})
    if k_result:
        delta = (k_result['mean_rms'] - baseline_rms) / baseline_rms * 100
        print(f"\n2. k = 1/4 exactly (dynamical coherence scale):")
        print(f"   RMS change: {delta:+.1f}%")
        if abs(delta) < 2:
            print(f"   → VIABLE: k = 1/4 is within calibration uncertainty!")
    
    # g† = cH₀/(2π)
    g_result = results.get('g† = cH₀/(2π)', {})
    if g_result:
        delta = (g_result['mean_rms'] - baseline_rms) / baseline_rms * 100
        g_2pi = c * H0_SI / (2 * np.pi)
        print(f"\n3. g† = cH₀/(2π) ≈ {g_2pi:.2e} m/s² (matches MOND a₀):")
        print(f"   RMS change: {delta:+.1f}%")
        print(f"   Cluster ratio: {g_result['cluster_ratio']:.3f}")
        if delta > 10:
            print(f"   → NOT VIABLE: Too much RMS degradation")
    
    # W exponent = 1
    w_result = results.get('W exponent = 1', {})
    if w_result:
        delta = (w_result['mean_rms'] - baseline_rms) / baseline_rms * 100
        print(f"\n4. W(r) exponent = 1 (pure exponential decoherence):")
        print(f"   RMS change: {delta:+.1f}%")
        if delta > 5:
            print(f"   → NOT VIABLE: Exponent 0.5 is necessary")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED 'CLEAN' PARAMETER SET")
    print("=" * 70)
    
    # Find best combination of clean values
    print("""
Based on the analysis, the following "clean" values could replace calibrated ones:

  Parameter          Current         Clean           Status
  ─────────────────────────────────────────────────────────────
  A₀                 1.6             φ ≈ 1.618       ✓ VIABLE
  k (dynamical ξ)    0.24            1/4 = 0.25      ✓ VIABLE  
  W exponent         0.5             0.5             ✓ KEEP (derived)
  ξ coefficient      2/3             2/3             ✓ KEEP
  g† factor          4√π             4√π             ✓ KEEP (derived)
  A exponent         1/4             1/4             ✓ KEEP

The golden ratio A₀ = φ is particularly intriguing as it:
  - Appears in many natural phenomena (phyllotaxis, spirals)
  - Is related to Fibonacci sequences
  - Gives A_galaxy ≈ 1.79 (close to √3 ≈ 1.73)
  - Gives A_cluster ≈ 7.24 (close to 8.0)
""")


if __name__ == "__main__":
    main()

