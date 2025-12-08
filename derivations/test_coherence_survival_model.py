#!/usr/bin/env python3
"""
Coherence Survival Model: First-Principles Test
================================================

This implements the survival/first-passage framework for gravitational coherence:

CORE IDEA:
- Coherence must propagate a minimum distance λ_coh (the Jeans length) without disruption
- Disruption is a Poisson process with mean free path λ_D(x)
- Disruption RESETS the counter (not just attenuates)
- This creates a sharp threshold: either coherence survives or it doesn't

FIELD EQUATION:
    ∇²Φ(x) = 4πG ρ(x) [ 1 + A · exp( -λ_J(x) / λ_D(x) ) ]

Where:
    λ_J = σ_v / √(4πGρ)         # Jeans length (the "finish line")
    λ_D = λ_D† / (g/g† + σ/σ† + Γ_dist)   # Decoherence mean free path

The survival probability is:
    P_survive = exp( -λ_J / λ_D )

KEY PREDICTION:
    Two galaxies with identical acceleration profiles but different radial 
    coherence paths (one smooth, one with a disruptive bar/ring/warp) should 
    show different outer enhancements.

Author: Sigma Gravity Team
Date: December 2025
"""

import numpy as np
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

c = 2.998e8              # Speed of light [m/s]
H0_SI = 2.27e-18         # Hubble constant [1/s] (70 km/s/Mpc)
cH0 = c * H0_SI          # c × H₀ [m/s²]
kpc_to_m = 3.086e19      # meters per kpc
G_SI = 6.674e-11         # Gravitational constant [m³/kg/s²]
M_sun = 1.989e30         # Solar mass [kg]

# Critical scales (from cosmology)
g_dagger = cH0 / (4 * math.sqrt(math.pi))  # ≈ 9.6×10⁻¹¹ m/s²
sigma_dagger = 100e3     # Critical velocity dispersion [m/s] ≈ 100 km/s
lambda_D_dagger = 10 * kpc_to_m  # Critical decoherence scale [m] ≈ 10 kpc

# MOND for comparison
a0_mond = 1.2e-10

print("=" * 80)
print("COHERENCE SURVIVAL MODEL: FIRST-PRINCIPLES TEST")
print("=" * 80)
print(f"\nCritical scales:")
print(f"  g†     = {g_dagger:.4e} m/s² (acceleration)")
print(f"  σ†     = {sigma_dagger/1000:.0f} km/s (velocity dispersion)")
print(f"  λ_D†   = {lambda_D_dagger/kpc_to_m:.1f} kpc (decoherence scale)")


# =============================================================================
# JEANS LENGTH CALCULATION
# =============================================================================

def jeans_length(sigma_v: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Compute the Jeans length: λ_J = σ_v / √(4πGρ)
    
    This is the "finish line" - coherence must propagate this far to activate.
    
    Parameters:
        sigma_v: velocity dispersion [m/s]
        rho: mass density [kg/m³]
    
    Returns:
        λ_J: Jeans length [m]
    """
    rho_safe = np.maximum(rho, 1e-30)
    return sigma_v / np.sqrt(4 * np.pi * G_SI * rho_safe)


def estimate_local_density(M_enc: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Estimate local density from enclosed mass profile.
    
    Uses ρ ≈ M_enc / (4/3 π R³) as rough estimate.
    For more accuracy, would need dM/dR.
    """
    R_safe = np.maximum(R, 1e-10)
    return 3 * M_enc / (4 * np.pi * R_safe**3)


def estimate_sigma_v_from_rotation(V_rot: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Estimate velocity dispersion from rotation curve.
    
    For a disk galaxy:
    - σ_z ≈ V_rot / (√2 × R/h_z) where h_z is scale height
    - Typical: σ ≈ V_rot × 0.1-0.3 for thin disks
    
    We use σ ≈ V_rot × 0.2 as default estimate.
    """
    return V_rot * 0.2


# =============================================================================
# DECOHERENCE RATE CALCULATIONS
# =============================================================================

@dataclass
class DecoherenceRates:
    """Container for different disruption rate contributions."""
    gamma_g: np.ndarray      # Acceleration-induced rate
    gamma_sigma: np.ndarray  # Velocity dispersion rate
    gamma_grad: np.ndarray   # Density gradient rate
    gamma_tidal: np.ndarray  # Tidal field rate
    total: np.ndarray        # Total rate (1/λ_D)


def compute_decoherence_rates(
    g: np.ndarray,           # Local acceleration [m/s²]
    sigma_v: np.ndarray,     # Velocity dispersion [m/s]
    rho: np.ndarray,         # Density [kg/m³]
    grad_rho: np.ndarray,    # |∇ρ| [kg/m⁴]
    tidal: np.ndarray,       # |∇²Φ_ext| [s⁻²]
    g_dag: float = g_dagger,
    sigma_dag: float = sigma_dagger,
    lambda_dag: float = lambda_D_dagger,
    kappa: float = 1.0,      # Gradient coupling
    Phi_dag: float = 1e-10   # Tidal reference
) -> DecoherenceRates:
    """
    Compute all decoherence rate contributions.
    
    Total rate: 1/λ_D = γ_g + γ_σ + γ_∇ + γ_tidal
    
    Each contribution:
        γ_g = g / (g† · λ_D†)         # High g → rapid phase evolution
        γ_σ = σ_v / (σ† · λ_D†)       # Random motions scramble phases
        γ_∇ = |∇ρ| / (ρ · κ)          # Sharp gradients disrupt wavefronts
        γ_tidal = |∇²Φ_ext| / Φ†      # External perturbations
    """
    # Acceleration contribution
    gamma_g = g / (g_dag * lambda_dag)
    
    # Velocity dispersion contribution
    gamma_sigma = sigma_v / (sigma_dag * lambda_dag)
    
    # Density gradient contribution
    rho_safe = np.maximum(rho, 1e-30)
    gamma_grad = grad_rho / (rho_safe * kappa)
    
    # Tidal field contribution
    gamma_tidal = tidal / Phi_dag
    
    # Total rate
    total = gamma_g + gamma_sigma + gamma_grad + gamma_tidal
    
    return DecoherenceRates(
        gamma_g=gamma_g,
        gamma_sigma=gamma_sigma,
        gamma_grad=gamma_grad,
        gamma_tidal=gamma_tidal,
        total=total
    )


def decoherence_mean_free_path(rates: DecoherenceRates) -> np.ndarray:
    """
    Compute decoherence mean free path: λ_D = 1 / (total rate)
    """
    return 1.0 / np.maximum(rates.total, 1e-30)


# =============================================================================
# SURVIVAL PROBABILITY AND ENHANCEMENT
# =============================================================================

def survival_probability(lambda_J: np.ndarray, lambda_D: np.ndarray) -> np.ndarray:
    """
    Coherence survival probability: P = exp(-λ_J / λ_D)
    
    This is the probability that coherence propagates the full Jeans length
    without disruption.
    
    - When λ_D >> λ_J: P ≈ 1 (coherence survives)
    - When λ_D << λ_J: P ≈ 0 (constant resets, never builds up)
    """
    ratio = lambda_J / np.maximum(lambda_D, 1e-30)
    return np.exp(-ratio)


def enhancement_factor(P_survive: np.ndarray, A: float = np.sqrt(3)) -> np.ndarray:
    """
    Gravitational enhancement factor: Σ = 1 + A × P_survive
    
    When P_survive = 1: Full enhancement (coherent regime)
    When P_survive = 0: No enhancement (GR recovered)
    """
    return 1.0 + A * P_survive


# =============================================================================
# PATH-INTEGRATED SURVIVAL (for source correlations)
# =============================================================================

def path_survival_probability(
    R: np.ndarray,           # Radial positions [m]
    lambda_D: np.ndarray,    # Decoherence mean free path at each R [m]
    R_source: float          # Source radius [m]
) -> np.ndarray:
    """
    Path-integrated survival probability from source to each radius.
    
    P_path(R) = exp( -∫_{R_source}^{R} ds / λ_D(s) )
    
    This accumulates "disruption risk" along the path - if total exceeds ~1,
    coherence is killed.
    
    Uses trapezoidal integration.
    """
    # Sort by radius for integration
    sort_idx = np.argsort(R)
    R_sorted = R[sort_idx]
    lambda_D_sorted = lambda_D[sort_idx]
    
    # Find starting point (closest to R_source)
    start_idx = np.searchsorted(R_sorted, R_source)
    if start_idx >= len(R_sorted):
        start_idx = len(R_sorted) - 1
    
    # Integrate outward from source
    P_path = np.ones_like(R_sorted)
    cumulative_integral = 0.0
    
    for i in range(start_idx + 1, len(R_sorted)):
        dr = R_sorted[i] - R_sorted[i-1]
        # Trapezoidal rule
        avg_rate = 0.5 * (1.0/lambda_D_sorted[i] + 1.0/lambda_D_sorted[i-1])
        cumulative_integral += dr * avg_rate
        P_path[i] = np.exp(-cumulative_integral)
    
    # Also integrate inward from source (for inner radii)
    cumulative_integral = 0.0
    for i in range(start_idx - 1, -1, -1):
        dr = R_sorted[i+1] - R_sorted[i]
        avg_rate = 0.5 * (1.0/lambda_D_sorted[i] + 1.0/lambda_D_sorted[i+1])
        cumulative_integral += dr * avg_rate
        P_path[i] = np.exp(-cumulative_integral)
    
    # Unsort to match original order
    unsort_idx = np.argsort(sort_idx)
    return P_path[unsort_idx]


# =============================================================================
# SIMPLIFIED MODEL FOR SPARC FITTING
# =============================================================================

def h_function_original(g: np.ndarray, g_dag: float = g_dagger) -> np.ndarray:
    """Original h(g) function for comparison."""
    g = np.maximum(g, 1e-15)
    return np.sqrt(g_dag / g) * g_dag / (g_dag + g)


def survival_model_simple(
    R_kpc: np.ndarray,
    V_bar: np.ndarray,
    sigma_v_kms: float = 30.0,  # Velocity dispersion [km/s]
    A: float = np.sqrt(3),
    alpha: float = 1.0,         # Acceleration weight in λ_D
    beta: float = 0.5,          # Dispersion weight in λ_D
    lambda_D0_kpc: float = 10.0 # Reference decoherence scale [kpc]
) -> np.ndarray:
    """
    Simplified survival model for SPARC fitting.
    
    Uses:
        λ_J ≈ σ_v × R / V_bar²  (simplified Jeans-like scale)
        λ_D = λ_D0 / (α × g/g† + β × σ/σ†)
        P_survive = exp(-λ_J / λ_D)
        V_pred = V_bar × √(1 + A × P_survive × h(g))
    """
    # Convert to SI
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    sigma_v_ms = sigma_v_kms * 1000
    
    # Baryonic acceleration
    g_bar = V_bar_ms**2 / R_m
    
    # Simplified Jeans-like coherence length
    # λ_J ≈ σ_v × R / V_bar (dimensional analysis)
    lambda_J = sigma_v_ms * R_m / np.maximum(V_bar_ms, 1.0)
    
    # Decoherence mean free path
    lambda_D0 = lambda_D0_kpc * kpc_to_m
    decoherence_factor = alpha * g_bar / g_dagger + beta * sigma_v_ms / sigma_dagger
    lambda_D = lambda_D0 / np.maximum(decoherence_factor, 0.01)
    
    # Survival probability
    P_survive = survival_probability(lambda_J, lambda_D)
    
    # Enhancement (combine with h(g) for acceleration dependence)
    h = h_function_original(g_bar)
    Sigma = 1.0 + A * P_survive * h
    
    return V_bar * np.sqrt(Sigma)


def survival_model_threshold(
    R_kpc: np.ndarray,
    V_bar: np.ndarray,
    r_char_kpc: float = 5.0,   # Characteristic radius for threshold
    A: float = np.sqrt(3),
    alpha: float = 1.0,        # g/g† exponent
    beta: float = 0.5          # r_char/r exponent
) -> np.ndarray:
    """
    Single-parameter threshold model from the original formulation:
    
        Σ = 1 + A × exp( -(r_char/r)^β × (g/g†)^α )
    
    Where:
        - r_char/r captures "have you gone far enough" (λ_J piece)
        - g/g† captures "is disruption slow enough" (λ_D piece)
    """
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    # Threshold exponent
    r_ratio = r_char_kpc / np.maximum(R_kpc, 0.01)
    g_ratio = g_bar / g_dagger
    
    exponent = -np.power(r_ratio, beta) * np.power(g_ratio, alpha)
    P_survive = np.exp(exponent)
    
    # Also include original h(g) for smooth transition
    h = h_function_original(g_bar)
    
    Sigma = 1.0 + A * P_survive * h
    return V_bar * np.sqrt(Sigma)


# =============================================================================
# DATA LOADING (from existing code)
# =============================================================================

def find_sparc_data() -> Optional[Path]:
    """Find the SPARC data directory."""
    possible_paths = [
        Path("/Users/leonardspeiser/Projects/sigmagravity/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/sigmagravity/many_path_model/paper_release/data/Rotmod_LTG"),
        Path("/Users/leonardspeiser/Projects/GravityCalculator/data/Rotmod_LTG"),
    ]
    for p in possible_paths:
        if p.exists():
            return p
    return None


def load_galaxy_rotmod(rotmod_file: Path) -> Optional[Dict]:
    """Load a single galaxy rotation curve."""
    R, V_obs, V_err, V_gas, V_disk, V_bulge = [], [], [], [], [], []
    
    with open(rotmod_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    R.append(float(parts[0]))
                    V_obs.append(float(parts[1]))
                    V_err.append(float(parts[2]))
                    V_gas.append(float(parts[3]))
                    V_disk.append(float(parts[4]))
                    V_bulge.append(float(parts[5]) if len(parts) > 5 else 0.0)
                except ValueError:
                    continue
    
    if len(R) < 3:
        return None
    
    R = np.array(R)
    V_obs = np.array(V_obs)
    V_err = np.array(V_err)
    V_gas = np.array(V_gas)
    V_disk = np.array(V_disk)
    V_bulge = np.array(V_bulge)
    
    # Compute V_bar
    V_bar_sq = np.sign(V_gas) * V_gas**2 + np.sign(V_disk) * V_disk**2 + V_bulge**2
    if np.any(V_bar_sq < 0):
        return None
    V_bar = np.sqrt(V_bar_sq)
    
    return {
        'R': R,
        'V_obs': V_obs,
        'V_err': V_err,
        'V_bar': V_bar,
        'V_gas': V_gas,
        'V_disk': V_disk,
        'V_bulge': V_bulge
    }


def load_all_galaxies(sparc_dir: Path) -> Dict[str, Dict]:
    """Load all SPARC galaxies."""
    galaxies = {}
    for rotmod_file in sorted(sparc_dir.glob('*_rotmod.dat')):
        name = rotmod_file.stem.replace('_rotmod', '')
        data = load_galaxy_rotmod(rotmod_file)
        if data is not None:
            galaxies[name] = data
    return galaxies


# =============================================================================
# METRICS
# =============================================================================

def compute_rms(V_obs: np.ndarray, V_pred: np.ndarray) -> float:
    """RMS velocity error in km/s."""
    return np.sqrt(np.mean((V_obs - V_pred)**2))


def compute_chi2_red(V_obs: np.ndarray, V_pred: np.ndarray, V_err: np.ndarray) -> float:
    """Reduced chi-squared."""
    V_err_safe = np.maximum(V_err, 1.0)
    chi2 = np.sum(((V_obs - V_pred) / V_err_safe)**2)
    dof = len(V_obs) - 1
    return chi2 / max(dof, 1)


# =============================================================================
# COMPARISON MODELS
# =============================================================================

def predict_mond(R_kpc: np.ndarray, V_bar: np.ndarray) -> np.ndarray:
    """MOND simple interpolation prediction."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    x = g_bar / a0_mond
    x = np.maximum(x, 1e-10)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x)))
    
    g_obs = g_bar * nu
    return np.sqrt(g_obs * R_m) / 1000


def predict_original_sigma(R_kpc: np.ndarray, V_bar: np.ndarray, 
                           R_d: float = 3.0, A: float = np.sqrt(3)) -> np.ndarray:
    """Original Σ-Gravity with W(r) window."""
    R_m = R_kpc * kpc_to_m
    V_bar_ms = V_bar * 1000
    g_bar = V_bar_ms**2 / R_m
    
    xi = (2/3) * R_d
    W = 1 - (xi / (xi + R_kpc)) ** 0.5
    h = h_function_original(g_bar)
    
    Sigma = 1 + A * W * h
    return V_bar * np.sqrt(Sigma)


# =============================================================================
# MAIN TEST
# =============================================================================

def run_survival_model_test():
    """Test the coherence survival model on SPARC galaxies."""
    
    print("\n" + "=" * 80)
    print("LOADING SPARC DATA")
    print("=" * 80)
    
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("ERROR: SPARC data not found!")
        return
    
    print(f"Found SPARC data: {sparc_dir}")
    galaxies = load_all_galaxies(sparc_dir)
    print(f"Loaded {len(galaxies)} galaxies")
    
    # =========================================================================
    # TEST 1: Parameter scan for threshold model
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 1: THRESHOLD MODEL PARAMETER SCAN")
    print("=" * 80)
    print("\nModel: Σ = 1 + A × exp(-(r_char/r)^β × (g/g†)^α) × h(g)")
    
    # Scan parameters
    r_char_values = np.linspace(1.0, 20.0, 20)
    alpha_values = np.linspace(0.1, 2.0, 20)
    beta_values = [0.3, 0.5, 0.7, 1.0]
    
    best_params = {'r_char': 5.0, 'alpha': 1.0, 'beta': 0.5}
    best_mean_rms = np.inf
    
    for beta in beta_values:
        for r_char in r_char_values:
            for alpha in alpha_values:
                rms_list = []
                for name, data in galaxies.items():
                    try:
                        V_pred = survival_model_threshold(
                            data['R'], data['V_bar'],
                            r_char_kpc=r_char, alpha=alpha, beta=beta
                        )
                        rms = compute_rms(data['V_obs'], V_pred)
                        if np.isfinite(rms):
                            rms_list.append(rms)
                    except:
                        continue
                
                if len(rms_list) > 0:
                    mean_rms = np.mean(rms_list)
                    if mean_rms < best_mean_rms:
                        best_mean_rms = mean_rms
                        best_params = {'r_char': r_char, 'alpha': alpha, 'beta': beta}
    
    print(f"\nBest parameters:")
    print(f"  r_char = {best_params['r_char']:.2f} kpc")
    print(f"  α = {best_params['alpha']:.3f}")
    print(f"  β = {best_params['beta']:.3f}")
    print(f"  Mean RMS = {best_mean_rms:.2f} km/s")
    
    # =========================================================================
    # TEST 2: Compare models
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 2: MODEL COMPARISON")
    print("=" * 80)
    
    models = {
        'Survival Threshold': lambda R, V: survival_model_threshold(
            R, V, r_char_kpc=best_params['r_char'], 
            alpha=best_params['alpha'], beta=best_params['beta']
        ),
        'Original Σ-Gravity': lambda R, V: predict_original_sigma(R, V, R_d=3.0),
        'MOND': predict_mond
    }
    
    results = {name: {'rms': [], 'chi2': []} for name in models.keys()}
    
    for gal_name, data in galaxies.items():
        R = data['R']
        V_obs = data['V_obs']
        V_err = data['V_err']
        V_bar = data['V_bar']
        
        for model_name, model_func in models.items():
            try:
                V_pred = model_func(R, V_bar)
                results[model_name]['rms'].append(compute_rms(V_obs, V_pred))
                results[model_name]['chi2'].append(compute_chi2_red(V_obs, V_pred, V_err))
            except:
                continue
    
    print(f"\n{'Model':<25} {'Mean RMS':<12} {'Med RMS':<12} {'Med χ²':<12}")
    print("-" * 60)
    
    for model_name, metrics in results.items():
        rms_arr = np.array([x for x in metrics['rms'] if np.isfinite(x)])
        chi2_arr = np.array([x for x in metrics['chi2'] if np.isfinite(x)])
        
        mean_rms = np.mean(rms_arr) if len(rms_arr) > 0 else np.nan
        med_rms = np.median(rms_arr) if len(rms_arr) > 0 else np.nan
        med_chi2 = np.median(chi2_arr) if len(chi2_arr) > 0 else np.nan
        
        print(f"{model_name:<25} {mean_rms:<12.2f} {med_rms:<12.2f} {med_chi2:<12.2f}")
    
    # =========================================================================
    # TEST 3: Head-to-head comparison
    # =========================================================================
    print("\n" + "-" * 60)
    print("HEAD-TO-HEAD: Survival vs MOND")
    print("-" * 60)
    
    survival_rms = np.array(results['Survival Threshold']['rms'])
    mond_rms = np.array(results['MOND']['rms'])
    
    valid = np.isfinite(survival_rms) & np.isfinite(mond_rms)
    survival_wins = np.sum(survival_rms[valid] < mond_rms[valid])
    mond_wins = np.sum(mond_rms[valid] < survival_rms[valid])
    
    print(f"Survival model wins: {survival_wins} ({100*survival_wins/valid.sum():.1f}%)")
    print(f"MOND wins: {mond_wins} ({100*mond_wins/valid.sum():.1f}%)")
    
    # =========================================================================
    # TEST 4: Physical interpretation
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 4: PHYSICAL INTERPRETATION")
    print("=" * 80)
    
    # Pick a representative galaxy
    test_galaxy = 'NGC2403'
    if test_galaxy in galaxies:
        data = galaxies[test_galaxy]
        R = data['R']
        V_bar = data['V_bar']
        V_obs = data['V_obs']
        
        print(f"\nExample: {test_galaxy}")
        print("-" * 40)
        
        # Compute survival parameters at each radius
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        # Threshold model components
        r_ratio = best_params['r_char'] / np.maximum(R, 0.01)
        g_ratio = g_bar / g_dagger
        
        exponent = -np.power(r_ratio, best_params['beta']) * np.power(g_ratio, best_params['alpha'])
        P_survive = np.exp(exponent)
        
        print(f"\n{'R (kpc)':<10} {'g/g†':<12} {'P_survive':<12} {'V_bar':<10} {'V_obs':<10}")
        print("-" * 60)
        
        for i in range(0, len(R), max(1, len(R)//10)):
            print(f"{R[i]:<10.2f} {g_ratio[i]:<12.2f} {P_survive[i]:<12.4f} {V_bar[i]:<10.1f} {V_obs[i]:<10.1f}")
        
        print(f"""
Physical interpretation:
- At small R: g/g† is large → disruption is frequent → P_survive ≈ 0
- At large R: g/g† is small AND r > r_char → P_survive → 1
- The threshold is SHARP: enhancement kicks in only when both conditions are met

This is different from MOND where enhancement depends only on local g.
The survival model requires coherence to "build up" from inner regions.
""")
    
    # =========================================================================
    # TEST 5: Testable prediction - disturbed vs relaxed
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 5: TESTABLE PREDICTION")
    print("=" * 80)
    print("""
KEY PREDICTION:
    Two galaxies with identical acceleration profiles but different radial 
    coherence paths should show different outer enhancements.

SPECIFIC TEST:
    Galaxies with bars/rings/warps in the middle should have REDUCED outer 
    enhancement compared to smooth disk galaxies, because the disruption 
    "breaks the chain" and forces coherence to restart.

SPARC galaxies to compare:
    - Smooth disks: NGC2403, NGC3198, NGC7331
    - Barred/disturbed: NGC1300, NGC4548, NGC5383

This prediction is UNIQUE to the survival model and not present in MOND.
""")
    
    # =========================================================================
    # TEST 6: Simplified full Jeans model
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST 6: FULL JEANS-BASED SURVIVAL MODEL")
    print("=" * 80)
    
    # Scan σ_v and λ_D0
    sigma_values = np.linspace(10, 80, 15)  # km/s
    lambda_D0_values = np.linspace(2, 30, 15)  # kpc
    
    best_sigma = 30.0
    best_lambda = 10.0
    best_rms_jeans = np.inf
    
    for sigma_v in sigma_values:
        for lambda_D0 in lambda_D0_values:
            rms_list = []
            for name, data in galaxies.items():
                try:
                    V_pred = survival_model_simple(
                        data['R'], data['V_bar'],
                        sigma_v_kms=sigma_v, lambda_D0_kpc=lambda_D0
                    )
                    rms = compute_rms(data['V_obs'], V_pred)
                    if np.isfinite(rms):
                        rms_list.append(rms)
                except:
                    continue
            
            if len(rms_list) > 0:
                mean_rms = np.mean(rms_list)
                if mean_rms < best_rms_jeans:
                    best_rms_jeans = mean_rms
                    best_sigma = sigma_v
                    best_lambda = lambda_D0
    
    print(f"\nJeans-based model best parameters:")
    print(f"  σ_v = {best_sigma:.1f} km/s")
    print(f"  λ_D0 = {best_lambda:.1f} kpc")
    print(f"  Mean RMS = {best_rms_jeans:.2f} km/s")
    
    # Compare with threshold model
    print(f"\nComparison:")
    print(f"  Threshold model: {best_mean_rms:.2f} km/s")
    print(f"  Jeans model:     {best_rms_jeans:.2f} km/s")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: COHERENCE SURVIVAL MODEL")
    print("=" * 80)
    print(f"""
The coherence survival model provides a first-principles framework where:

1. MECHANISM: Coherence must propagate distance λ_J without disruption
   - λ_J (Jeans length) = "finish line" for coherence activation
   - λ_D (mean free path) = how far before disruption resets the counter

2. THRESHOLD BEHAVIOR: P_survive = exp(-λ_J/λ_D)
   - Sharp transition from GR (P≈0) to enhanced (P≈1)
   - Not simple attenuation - disruption RESETS the process

3. BEST FIT PARAMETERS:
   Threshold model: r_char = {best_params['r_char']:.2f} kpc, α = {best_params['alpha']:.3f}, β = {best_params['beta']:.3f}
   Jeans model: σ_v = {best_sigma:.1f} km/s, λ_D0 = {best_lambda:.1f} kpc

4. UNIQUE PREDICTION:
   Galaxies with internal disruptions (bars, warps) should show REDUCED
   outer enhancement compared to smooth disks at same acceleration.

5. SELF-REGULATING:
   - Dense, cold regions: λ_J small, λ_D large → coherence survives
   - Hot, high-g regions: λ_D tiny → constant resets → GR recovered
""")


def test_disturbed_vs_smooth_prediction():
    """
    Test the key unique prediction: disturbed galaxies should show less enhancement.
    
    The survival model predicts that a disruption in the middle of a galaxy
    "breaks the chain" and forces coherence to restart from that radius.
    
    This is NOT predicted by MOND or standard Σ-Gravity.
    """
    print("\n" + "=" * 80)
    print("DISTURBED vs SMOOTH GALAXY TEST")
    print("=" * 80)
    
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("ERROR: SPARC data not found!")
        return
    
    galaxies = load_all_galaxies(sparc_dir)
    
    # Classify galaxies by morphology (manual classification based on known properties)
    # These are approximate - in a real analysis you'd use proper morphological catalogs
    smooth_disks = ['NGC2403', 'NGC3198', 'NGC6946', 'NGC2841', 'NGC7331', 'NGC5055']
    barred_galaxies = ['NGC1300', 'NGC4548', 'NGC5383', 'NGC3992', 'NGC4321', 'NGC2903']
    
    print("\nComparing residuals for smooth vs barred galaxies...")
    print("(Survival model predicts barred galaxies should have LARGER residuals)")
    
    # Best parameters from earlier scan
    r_char = 20.0
    alpha = 0.1
    beta = 0.3
    
    def compute_outer_residual(data: Dict, model_func) -> float:
        """Compute residual in outer region (R > 0.5 R_max)."""
        R = data['R']
        V_obs = data['V_obs']
        V_bar = data['V_bar']
        
        R_half = 0.5 * R.max()
        outer_mask = R > R_half
        
        if outer_mask.sum() < 2:
            return np.nan
        
        V_pred = model_func(R, V_bar)
        return np.sqrt(np.mean((V_obs[outer_mask] - V_pred[outer_mask])**2))
    
    # Survival model
    survival_model = lambda R, V: survival_model_threshold(R, V, r_char, alpha=alpha, beta=beta)
    
    print(f"\n{'Galaxy':<15} {'Type':<10} {'Survival RMS':<15} {'MOND RMS':<15} {'Δ(Surv-MOND)':<15}")
    print("-" * 70)
    
    smooth_survival_residuals = []
    smooth_mond_residuals = []
    barred_survival_residuals = []
    barred_mond_residuals = []
    
    for name in smooth_disks:
        if name in galaxies:
            data = galaxies[name]
            surv_res = compute_outer_residual(data, survival_model)
            mond_res = compute_outer_residual(data, predict_mond)
            if np.isfinite(surv_res) and np.isfinite(mond_res):
                smooth_survival_residuals.append(surv_res)
                smooth_mond_residuals.append(mond_res)
                print(f"{name:<15} {'Smooth':<10} {surv_res:<15.2f} {mond_res:<15.2f} {surv_res-mond_res:<15.2f}")
    
    for name in barred_galaxies:
        if name in galaxies:
            data = galaxies[name]
            surv_res = compute_outer_residual(data, survival_model)
            mond_res = compute_outer_residual(data, predict_mond)
            if np.isfinite(surv_res) and np.isfinite(mond_res):
                barred_survival_residuals.append(surv_res)
                barred_mond_residuals.append(mond_res)
                print(f"{name:<15} {'Barred':<10} {surv_res:<15.2f} {mond_res:<15.2f} {surv_res-mond_res:<15.2f}")
    
    print("\n" + "-" * 70)
    print("SUMMARY:")
    print(f"  Smooth galaxies - Survival mean RMS: {np.mean(smooth_survival_residuals):.2f} km/s")
    print(f"  Smooth galaxies - MOND mean RMS:     {np.mean(smooth_mond_residuals):.2f} km/s")
    print(f"  Barred galaxies - Survival mean RMS: {np.mean(barred_survival_residuals):.2f} km/s")
    print(f"  Barred galaxies - MOND mean RMS:     {np.mean(barred_mond_residuals):.2f} km/s")
    
    print(f"""
INTERPRETATION:
    The survival model should fit smooth disks BETTER than barred galaxies
    (relative to MOND), because bars disrupt coherence propagation.
    
    If Δ(Surv-MOND) is more negative for smooth disks than barred disks,
    this supports the coherence survival mechanism.
""")


def test_path_integrated_survival():
    """
    Test the full path-integrated survival model.
    
    This computes the survival probability by integrating disruption risk
    along the entire path from the source region outward.
    """
    print("\n" + "=" * 80)
    print("PATH-INTEGRATED SURVIVAL MODEL TEST")
    print("=" * 80)
    
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("ERROR: SPARC data not found!")
        return
    
    galaxies = load_all_galaxies(sparc_dir)
    
    # Test on a few well-measured galaxies
    test_galaxies = ['NGC2403', 'NGC3198', 'NGC6946', 'NGC2841']
    
    print("\nPath-integrated model: P(R) = exp(-∫ ds/λ_D(s))")
    print("This accumulates disruption risk along the entire path.\n")
    
    for gal_name in test_galaxies:
        if gal_name not in galaxies:
            continue
            
        data = galaxies[gal_name]
        R = data['R']
        V_bar = data['V_bar']
        V_obs = data['V_obs']
        
        # Convert to SI
        R_m = R * kpc_to_m
        V_bar_ms = V_bar * 1000
        g_bar = V_bar_ms**2 / R_m
        
        # Compute local decoherence mean free path
        # λ_D = λ_D0 / (g/g† + σ/σ†)
        lambda_D0 = 5.0 * kpc_to_m  # 5 kpc reference scale
        sigma_v = 30e3  # 30 km/s typical dispersion
        
        decoherence_factor = g_bar / g_dagger + sigma_v / sigma_dagger
        lambda_D = lambda_D0 / np.maximum(decoherence_factor, 0.01)
        
        # Compute path-integrated survival from inner region
        R_source = R_m[0]  # Start from innermost measured point
        P_path = path_survival_probability(R_m, lambda_D, R_source)
        
        # Enhancement with path-integrated survival
        h = h_function_original(g_bar)
        A = np.sqrt(3)
        Sigma = 1.0 + A * P_path * h
        V_pred = V_bar * np.sqrt(Sigma)
        
        rms = compute_rms(V_obs, V_pred)
        
        print(f"\n{gal_name}:")
        print(f"  RMS = {rms:.2f} km/s")
        print(f"  Inner P_path = {P_path[0]:.4f}")
        print(f"  Outer P_path = {P_path[-1]:.4f}")
        print(f"  λ_D range: {lambda_D.min()/kpc_to_m:.1f} - {lambda_D.max()/kpc_to_m:.1f} kpc")


def test_jeans_length_correlation():
    """
    Test whether the Jeans length correlates with enhancement.
    
    The theory predicts that galaxies with smaller λ_J (easier target)
    should show stronger enhancement.
    """
    print("\n" + "=" * 80)
    print("JEANS LENGTH CORRELATION TEST")
    print("=" * 80)
    
    sparc_dir = find_sparc_data()
    if sparc_dir is None:
        print("ERROR: SPARC data not found!")
        return
    
    galaxies = load_all_galaxies(sparc_dir)
    
    # For each galaxy, compute:
    # 1. Characteristic Jeans length
    # 2. Observed enhancement (V_obs/V_bar at outer radii)
    
    jeans_lengths = []
    observed_enhancements = []
    galaxy_names = []
    
    for name, data in galaxies.items():
        R = data['R']
        V_bar = data['V_bar']
        V_obs = data['V_obs']
        
        # Skip if too few points
        if len(R) < 5:
            continue
        
        # Outer region (R > 0.7 R_max)
        R_outer = 0.7 * R.max()
        outer_mask = R > R_outer
        
        if outer_mask.sum() < 2:
            continue
        
        # Observed enhancement at outer radii
        V_bar_outer = V_bar[outer_mask].mean()
        V_obs_outer = V_obs[outer_mask].mean()
        
        if V_bar_outer < 10:  # Skip if V_bar too small
            continue
            
        enhancement = V_obs_outer / V_bar_outer
        
        # Estimate Jeans length
        # λ_J ≈ σ_v / √(4πGρ)
        # Use σ_v ≈ 0.2 × V_bar (typical for disks)
        # ρ ≈ V²/(4πGR²) for flat rotation
        
        R_m = R.mean() * kpc_to_m
        V_bar_ms = V_bar.mean() * 1000
        sigma_v = 0.2 * V_bar_ms
        
        # Rough density estimate
        rho = V_bar_ms**2 / (4 * np.pi * G_SI * R_m**2)
        lambda_J = sigma_v / np.sqrt(4 * np.pi * G_SI * rho)
        lambda_J_kpc = lambda_J / kpc_to_m
        
        jeans_lengths.append(lambda_J_kpc)
        observed_enhancements.append(enhancement)
        galaxy_names.append(name)
    
    jeans_lengths = np.array(jeans_lengths)
    observed_enhancements = np.array(observed_enhancements)
    
    # Compute correlation
    valid = np.isfinite(jeans_lengths) & np.isfinite(observed_enhancements)
    if valid.sum() > 5:
        corr = np.corrcoef(jeans_lengths[valid], observed_enhancements[valid])[0, 1]
        
        print(f"\nCorrelation between λ_J and observed enhancement: {corr:.3f}")
        print(f"Number of galaxies: {valid.sum()}")
        
        # Sort by Jeans length and show extremes
        sort_idx = np.argsort(jeans_lengths[valid])
        
        print(f"\nGalaxies with SMALLEST λ_J (should have MOST enhancement):")
        for i in sort_idx[:5]:
            idx = np.where(valid)[0][i]
            print(f"  {galaxy_names[idx]:<15} λ_J = {jeans_lengths[idx]:.1f} kpc, V_obs/V_bar = {observed_enhancements[idx]:.2f}")
        
        print(f"\nGalaxies with LARGEST λ_J (should have LEAST enhancement):")
        for i in sort_idx[-5:]:
            idx = np.where(valid)[0][i]
            print(f"  {galaxy_names[idx]:<15} λ_J = {jeans_lengths[idx]:.1f} kpc, V_obs/V_bar = {observed_enhancements[idx]:.2f}")
        
        print(f"""
INTERPRETATION:
    If the survival model is correct, smaller λ_J should correlate with
    stronger enhancement (easier to "make it to the finish line").
    
    Correlation = {corr:.3f}
    {'SUPPORTS' if corr < -0.1 else 'DOES NOT SUPPORT'} the survival model prediction.
""")


if __name__ == "__main__":
    run_survival_model_test()
    test_disturbed_vs_smooth_prediction()
    test_path_integrated_survival()
    test_jeans_length_correlation()

