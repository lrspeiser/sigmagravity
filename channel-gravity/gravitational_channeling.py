"""
Gravitational Channeling with Gravity Competition
==================================================

Production code for testing on SPARC rotation curves.

Theory: Channels form in gravitational field lines, digging deeper
where gravity is weaker (a < a_0). Spiral winding suppresses fast rotators.

Author: Leonard
Date: 2025
"""

import numpy as np
import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ChannelingParams:
    """Parameters for gravitational channeling model."""
    chi_0: float = 0.4          # Coupling strength
    alpha: float = 1.0          # Radial growth exponent
    beta: float = 0.5           # Velocity coherence exponent
    gamma: float = 0.3          # Time accumulation exponent
    epsilon: float = 0.3        # Surface density exponent
    zeta: float = 0.3           # Gravity competition exponent (NEW!)
    D_max: float = 3.0          # Saturation depth
    t_age: float = 10.0         # System age [Gyr]
    tau_0: float = 1.0          # Reference formation time [Gyr]
    # Reference scales
    Sigma_ref: float = 100.0    # Reference surface density [M_sun/pc²]
    sigma_ref: float = 30.0     # Reference velocity dispersion [km/s]
    R_0: float = 8.0            # Reference radius [kpc]
    a_0: float = 3700.0         # MOND-like acceleration scale [(km/s)²/kpc]
    # Winding suppression
    N_crit: float = 10.0        # Critical orbits for interference
    wind_power: float = 2.0     # Winding steepness
    use_winding: bool = True    # Enable spiral winding


def gravitational_channeling(
    R: np.ndarray,
    v_bary: np.ndarray,
    sigma_v: np.ndarray,
    Sigma: np.ndarray,
    params: ChannelingParams
) -> Tuple[np.ndarray, Dict]:
    """
    Compute gravitational channeling enhancement.
    
    Parameters
    ----------
    R : array
        Galactocentric radius [kpc]
    v_bary : array
        Baryonic circular velocity [km/s]
    sigma_v : array
        Velocity dispersion [km/s]
    Sigma : array
        Surface density [M_sun/pc²]
    params : ChannelingParams
        Model parameters
    
    Returns
    -------
    F : array
        Enhancement factor (F=1 means no enhancement)
    diagnostics : dict
        Intermediate quantities for analysis
    """
    # Protect against edge cases
    R_safe = np.maximum(R, 0.1)
    v_safe = np.maximum(np.abs(v_bary), 1.0)
    sigma_safe = np.maximum(sigma_v, 1.0)
    Sigma_safe = np.maximum(Sigma, 0.01)
    
    # 1. Channel formation time
    # τ_ch = τ_0 × (σ_v/σ_ref) × (R_0/R)
    # High σ_v → turbulence erases channels → longer formation
    # Small R → less space → faster formation
    tau_ch = params.tau_0 * (sigma_safe / params.sigma_ref) * (params.R_0 / R_safe)
    tau_ch = np.maximum(tau_ch, 0.01)
    
    # 2. Time accumulation: (t_age/τ_ch)^γ
    # γ < 1 → sublinear (channels merge/compete)
    time_term = (params.t_age / tau_ch) ** params.gamma
    
    # 3. Velocity coherence: (v_c/σ_v)^β
    # Cold systems carve deeper channels
    coherence_term = (v_safe / sigma_safe) ** params.beta
    
    # 4. Radial scaling: (R/R_0)^α
    # Larger radii → more room for channels
    radial_term = (R_safe / params.R_0) ** params.alpha
    
    # 5. NEW: Gravity competition: (a_0/a)^ζ
    # Weak gravity (a << a_0) → more room → deeper channels
    # Strong gravity (a >> a_0) → crowded → shallow channels
    a = v_safe**2 / R_safe  # Centripetal acceleration [(km/s)²/kpc]
    competition_term = (params.a_0 / a) ** params.zeta
    competition_term = np.minimum(competition_term, 10.0)  # Cap
    
    # 6. Raw channel depth
    D_raw = time_term * coherence_term * radial_term * competition_term
    
    # 7. Saturation: D / (1 + D/D_max)
    # Channels can't deepen forever
    D = D_raw / (1.0 + D_raw / params.D_max)
    
    # 8. Surface density factor: (Σ/Σ_ref)^ε
    # Solar System safety: Σ→0 for point masses
    density_factor = (Sigma_safe / params.Sigma_ref) ** params.epsilon
    density_factor = np.minimum(density_factor, 5.0)
    
    # 9. Spiral winding suppression: f_wind = 1/(1 + (N/N_crit)^n)
    # Field lines wind up → tight spirals → interference
    if params.use_winding:
        N_orbits = params.t_age * v_safe / (2 * np.pi * R_safe * 0.978)
        f_wind = 1.0 / (1.0 + (N_orbits / params.N_crit) ** params.wind_power)
    else:
        N_orbits = np.zeros_like(R)
        f_wind = np.ones_like(R)
    
    # 10. Total enhancement
    F = 1.0 + params.chi_0 * density_factor * D * f_wind
    
    # Diagnostics
    diagnostics = {
        'tau_ch': tau_ch,
        'time_term': time_term,
        'coherence_term': coherence_term,
        'radial_term': radial_term,
        'competition_term': competition_term,
        'a': a,
        'D_raw': D_raw,
        'D': D,
        'density_factor': density_factor,
        'N_orbits': N_orbits,
        'f_wind': f_wind,
        'F': F,
    }
    
    return F, diagnostics


def estimate_sigma_v(R: np.ndarray, v_bary: np.ndarray, 
                     is_gas_dominated: bool = False) -> np.ndarray:
    """
    Estimate velocity dispersion profile.
    
    σ_v(R) = σ_0 × exp(-R/R_σ) + σ_floor
    
    Gas-dominated: σ_0 = 15 km/s, R_σ = 3 kpc
    Stellar: σ_0 = 40 km/s, R_σ = 6 kpc
    """
    if is_gas_dominated:
        sigma_0 = 15.0
        R_sigma = 3.0
        sigma_floor = 6.0
    else:
        sigma_0 = 40.0
        R_sigma = 6.0
        sigma_floor = 10.0
    
    sigma_v = sigma_0 * np.exp(-R / R_sigma) + sigma_floor
    return sigma_v


def estimate_surface_density(R: np.ndarray, v_disk: np.ndarray, 
                             v_gas: np.ndarray) -> np.ndarray:
    """
    Estimate surface density from velocity components.
    
    Σ ≈ v² / (2πGR) for thin disk
    """
    G = 4.302e-6  # kpc (km/s)² / M_sun
    
    Sigma_disk = v_disk**2 / (2 * np.pi * G * np.maximum(R, 0.1) * 1e3)
    Sigma_gas = v_gas**2 / (2 * np.pi * G * np.maximum(R, 0.1) * 1e3)
    
    Sigma_total = Sigma_disk + Sigma_gas
    return Sigma_total


def load_sparc_galaxy(filepath: str) -> Dict:
    """
    Load SPARC rotation curve data.
    
    Expected format: R, v_obs, v_err, v_gas, v_disk, v_bul
    """
    data = np.loadtxt(filepath, comments='#')
    
    if data.shape[1] < 6:
        raise ValueError(f"Expected 6 columns, got {data.shape[1]}")
    
    name = os.path.basename(filepath).replace('_rotmod.dat', '')
    
    return {
        'name': name,
        'R': data[:, 0],           # kpc
        'v_obs': data[:, 1],       # km/s
        'v_err': data[:, 2],       # km/s
        'v_gas': data[:, 3],       # km/s
        'v_disk': data[:, 4],      # km/s
        'v_bul': data[:, 5],       # km/s
    }


def fit_galaxy(data: Dict, params: ChannelingParams) -> Dict:
    """
    Fit gravitational channeling to a galaxy.
    
    Returns
    -------
    result : dict
        Fitting results with RMS, improvement status, etc.
    """
    R = data['R']
    v_obs = data['v_obs']
    v_err = data['v_err']
    v_gas = np.abs(data['v_gas'])
    v_disk = np.abs(data['v_disk'])
    v_bul = np.abs(data['v_bul'])
    
    # Baryonic velocity
    v_bary = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
    
    # Estimate σ_v
    is_gas_dom = np.mean(v_gas[-3:]) > np.mean(v_disk[-3:])
    sigma_v = estimate_sigma_v(R, v_bary, is_gas_dom)
    
    # Estimate Σ
    Sigma = estimate_surface_density(R, v_disk, v_gas)
    
    # Compute enhancement
    F, diag = gravitational_channeling(R, v_bary, sigma_v, Sigma, params)
    
    # Enhanced velocity
    v_pred = v_bary * np.sqrt(F)
    
    # RMS comparison
    rms_bary = np.sqrt(np.mean((v_bary - v_obs)**2))
    rms_pred = np.sqrt(np.mean((v_pred - v_obs)**2))
    delta_rms = rms_pred - rms_bary
    improved = rms_pred < rms_bary
    
    # Flat velocity (outer disk average)
    v_flat = np.mean(v_obs[-3:])
    
    return {
        'name': data['name'],
        'n_points': len(R),
        'v_flat': v_flat,
        'rms_bary': rms_bary,
        'rms_pred': rms_pred,
        'delta_rms': delta_rms,
        'improved': improved,
        'R': R,
        'v_obs': v_obs,
        'v_bary': v_bary,
        'v_pred': v_pred,
        'F': F,
        'sigma_v': sigma_v,
        'Sigma': Sigma,
        'diagnostics': diag,
    }


def run_sparc_batch(data_dir: str, params: ChannelingParams) -> List[Dict]:
    """
    Run on all SPARC galaxies.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing *_rotmod.dat files
    params : ChannelingParams
        Model parameters
    
    Returns
    -------
    results : list of dict
        Results for each galaxy
    """
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching {pattern}")
    
    print(f"Found {len(files)} SPARC galaxies")
    
    results = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, params)
            results.append(result)
        except Exception as e:
            print(f"Error processing {os.path.basename(filepath)}: {e}")
            continue
    
    return results


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze batch results."""
    
    n_total = len(results)
    n_improved = sum(1 for r in results if r['improved'])
    pct_improved = 100 * n_improved / n_total
    
    delta_rms = np.array([r['delta_rms'] for r in results])
    v_flat = np.array([r['v_flat'] for r in results])
    
    # By galaxy type
    dwarfs = [r for r in results if r['v_flat'] < 80]
    intermediate = [r for r in results if 80 <= r['v_flat'] < 150]
    massive = [r for r in results if r['v_flat'] >= 150]
    
    summary = {
        'n_total': n_total,
        'n_improved': n_improved,
        'pct_improved': pct_improved,
        'median_delta_rms': np.median(delta_rms),
        'mean_delta_rms': np.mean(delta_rms),
        'dwarf_pct': 100 * sum(1 for r in dwarfs if r['improved']) / len(dwarfs) if dwarfs else 0,
        'inter_pct': 100 * sum(1 for r in intermediate if r['improved']) / len(intermediate) if intermediate else 0,
        'massive_pct': 100 * sum(1 for r in massive if r['improved']) / len(massive) if massive else 0,
        'results': results,
    }
    
    return summary


def print_summary(summary: Dict, param_name: str = ""):
    """Print results summary."""
    
    print("\n" + "=" * 70)
    print(f"RESULTS {param_name}")
    print("=" * 70)
    
    print(f"\nOverall:")
    print(f"  Galaxies tested: {summary['n_total']}")
    print(f"  Improved: {summary['n_improved']}/{summary['n_total']} ({summary['pct_improved']:.1f}%)")
    print(f"  Median ΔRMS: {summary['median_delta_rms']:.2f} km/s")
    print(f"  Mean ΔRMS: {summary['mean_delta_rms']:.2f} km/s")
    
    print(f"\nBy galaxy type:")
    print(f"  Dwarfs (v < 80 km/s): {summary['dwarf_pct']:.1f}% improved")
    print(f"  Intermediate (80-150): {summary['inter_pct']:.1f}% improved")
    print(f"  Massive (v > 150): {summary['massive_pct']:.1f}% improved")


# =============================================================================
# MAIN TEST SCRIPT
# =============================================================================

def main():
    """
    Main test script for SPARC data.
    
    EDIT THE DATA DIRECTORY PATH BELOW!
    """
    
    print("=" * 70)
    print("GRAVITATIONAL CHANNELING - SPARC BATCH TEST")
    print("=" * 70)
    
    # ========== EDIT THIS PATH ==========
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    # ====================================
    
    if not os.path.exists(data_dir):
        print(f"\nERROR: Data directory not found: {data_dir}")
        print("Please edit the 'data_dir' path in main() to point to your SPARC data.")
        return
    
    # Test 1: Galaxy-optimized parameters
    print("\n" + "=" * 70)
    print("TEST 1: GALAXY-OPTIMIZED PARAMETERS")
    print("=" * 70)
    
    params_galaxy = ChannelingParams(
        chi_0=0.4,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        epsilon=0.3,
        zeta=0.3,          # Moderate gravity competition
        D_max=3.0,
        N_crit=10.0,       # Winding suppression ON
        use_winding=True,
    )
    
    print("\nParameters:")
    print(f"  χ₀ = {params_galaxy.chi_0} (coupling)")
    print(f"  ζ = {params_galaxy.zeta} (gravity competition)")
    print(f"  α = {params_galaxy.alpha} (radial)")
    print(f"  N_crit = {params_galaxy.N_crit} (winding)")
    
    results_galaxy = run_sparc_batch(data_dir, params_galaxy)
    summary_galaxy = analyze_results(results_galaxy)
    print_summary(summary_galaxy, "(Galaxy Params)")
    
    # Test 2: Cluster-optimized parameters
    print("\n" + "=" * 70)
    print("TEST 2: CLUSTER-OPTIMIZED PARAMETERS (on galaxies)")
    print("=" * 70)
    
    params_cluster = ChannelingParams(
        chi_0=2.38,
        alpha=1.0,
        beta=0.5,
        gamma=0.3,
        epsilon=0.3,
        zeta=0.5,          # Stronger gravity competition
        D_max=12.5,        # Deeper saturation
        N_crit=1000.0,     # Winding suppression OFF (clusters don't spiral)
        use_winding=False,
    )
    
    print("\nParameters:")
    print(f"  χ₀ = {params_cluster.chi_0} (coupling)")
    print(f"  ζ = {params_cluster.zeta} (gravity competition)")
    print(f"  α = {params_cluster.alpha} (radial)")
    print(f"  Winding: OFF (cluster params)")
    
    results_cluster = run_sparc_batch(data_dir, params_cluster)
    summary_cluster = analyze_results(results_cluster)
    print_summary(summary_cluster, "(Cluster Params)")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Parameter Set':<25} {'Overall %':<12} {'Dwarf %':<12} {'Massive %':<12}")
    print("-" * 61)
    print(f"{'Galaxy-optimized':<25} {summary_galaxy['pct_improved']:<12.1f} "
          f"{summary_galaxy['dwarf_pct']:<12.1f} {summary_galaxy['massive_pct']:<12.1f}")
    print(f"{'Cluster-optimized':<25} {summary_cluster['pct_improved']:<12.1f} "
          f"{summary_cluster['dwarf_pct']:<12.1f} {summary_cluster['massive_pct']:<12.1f}")
    
    # Key question
    print("\n" + "=" * 70)
    print("KEY QUESTION:")
    print("=" * 70)
    print("""
Are the parameters UNIVERSAL?

If cluster params also work well for galaxies → YES, universal!
If galaxy params work better → Parameters are scale-dependent.

Cluster test (from separate analysis):
  Coma: F=5.60 (need 5.0) ✓
  A2029: F=4.16 (need 5.0) 
  A1689: F=5.12 (need 5.0) ✓
  Average: 99% of needed enhancement

This uses ζ=0.5, which may over-enhance galaxies.
""")


if __name__ == "__main__":
    main()
