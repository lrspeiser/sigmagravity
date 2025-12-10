"""
Test Spiral Winding on Real SPARC Data
======================================

Does the N_crit winding suppression improve massive spiral success rate?
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cooperative_channeling import load_sparc_galaxy, estimate_sigma_v


def compute_orbital_periods(R, v_c, t_age=10.0):
    """Number of orbits in t_age Gyr."""
    T_orbital_gyr = 2 * np.pi * R / v_c * 0.978
    return t_age / T_orbital_gyr


def spiral_winding_factor(N_orbits, N_crit=30.0, steepness=2.0):
    """Winding suppression: tight winding → interference."""
    return 1.0 / (1.0 + (N_orbits / N_crit) ** steepness)


def winding_channeling_enhancement(
    R, v_c, Sigma, sigma_v,
    chi_0=0.4, alpha=1.0, beta=0.5, gamma=0.3, epsilon=0.3,
    D_max=3.0, t_age=10.0, tau_0=1.0,
    N_crit=30.0, wind_steepness=2.0,
    Sigma_ref=100.0, sigma_ref=30.0, R_0=8.0,
):
    """Channeling with spiral winding suppression."""
    R_safe = np.maximum(R, 0.1)
    sigma_safe = np.maximum(sigma_v, 1.0)
    Sigma_safe = np.maximum(Sigma, 0.01)
    
    # Channel depth
    tau_ch = tau_0 * (sigma_safe / sigma_ref) * (R_0 / R_safe)
    tau_ch = np.maximum(tau_ch, 0.01)
    
    time_term = (t_age / tau_ch) ** gamma
    coherence_term = (v_c / sigma_safe) ** beta
    radial_term = (R_safe / R_0) ** alpha
    
    D_raw = time_term * coherence_term * radial_term
    D = D_raw / (1.0 + D_raw / D_max)
    
    # Surface density factor
    density_factor = (Sigma_safe / Sigma_ref) ** epsilon
    density_factor = np.minimum(density_factor, 5.0)
    
    # Winding suppression
    N_orbits = compute_orbital_periods(R_safe, v_c, t_age)
    f_wind = spiral_winding_factor(N_orbits, N_crit, wind_steepness)
    
    F = 1.0 + chi_0 * density_factor * D * f_wind
    
    return F, {'N_orbits': N_orbits, 'f_wind': f_wind, 'D': D}


def fit_galaxy(data, N_crit=30.0, chi_0=0.4):
    """Fit winding model to a galaxy."""
    R = data['R']
    v_obs = data['v_obs']
    v_bary = np.abs(data['v_bary'])
    Sigma = data['Sigma']
    
    is_gas_dom = np.mean(np.abs(data['v_gas'][-3:])) > np.mean(np.abs(data['v_disk'][-3:]))
    sigma_v = estimate_sigma_v(R, v_bary, is_gas_dominated=is_gas_dom)
    
    F, diag = winding_channeling_enhancement(R, v_bary, Sigma, sigma_v, N_crit=N_crit, chi_0=chi_0)
    v_pred = v_bary * np.sqrt(F)
    
    rms_pred = np.sqrt(np.mean((v_pred - v_obs)**2))
    rms_bary = np.sqrt(np.mean((v_bary - v_obs)**2))
    
    return {
        'name': data['name'],
        'v_flat': np.mean(v_obs[-3:]),
        'rms_pred': rms_pred,
        'rms_bary': rms_bary,
        'delta_rms': rms_pred - rms_bary,
        'improved': rms_pred < rms_bary,
        'mean_N_orbits': np.mean(diag['N_orbits']),
        'mean_f_wind': np.mean(diag['f_wind']),
    }


def run_batch(data_dir, N_crit=30.0, chi_0=0.4):
    """Run on all SPARC."""
    pattern = os.path.join(data_dir, "*_rotmod.dat")
    files = sorted(glob.glob(pattern))
    
    results = []
    for filepath in files:
        try:
            data = load_sparc_galaxy(filepath)
            if len(data['R']) < 5:
                continue
            result = fit_galaxy(data, N_crit=N_crit, chi_0=chi_0)
            results.append(result)
        except:
            continue
    return results


def analyze(results):
    """Analyze by galaxy type."""
    dwarfs = [r for r in results if r['v_flat'] < 80]
    intermediate = [r for r in results if 80 <= r['v_flat'] < 150]
    massive = [r for r in results if r['v_flat'] >= 150]
    
    def stats(subset, name):
        if not subset:
            return
        improved = sum(1 for r in subset if r['improved'])
        pct = 100 * improved / len(subset)
        med_delta = np.median([r['delta_rms'] for r in subset])
        mean_N = np.mean([r['mean_N_orbits'] for r in subset])
        mean_f = np.mean([r['mean_f_wind'] for r in subset])
        print(f"  {name}: {improved}/{len(subset)} ({pct:.1f}%), "
              f"med_Δ={med_delta:.1f}, N_orb={mean_N:.0f}, f_wind={mean_f:.2f}")
        return pct
    
    print(f"  All: {sum(1 for r in results if r['improved'])}/{len(results)} "
          f"({100*sum(1 for r in results if r['improved'])/len(results):.1f}%)")
    d_pct = stats(dwarfs, "Dwarf (<80)")
    i_pct = stats(intermediate, "Inter (80-150)")
    m_pct = stats(massive, "Massive (>150)")
    
    return d_pct or 0, i_pct or 0, m_pct or 0


def main():
    print("=" * 70)
    print("SPIRAL WINDING ON REAL SPARC DATA")
    print("=" * 70)
    
    data_dir = r"C:\Users\henry\dev\sigmagravity\data\Rotmod_LTG"
    
    # Test different N_crit values
    print("\n" + "=" * 70)
    print("N_crit SWEEP")
    print("=" * 70)
    
    results_summary = []
    
    for N_crit in [10, 20, 30, 50, 100, 200, 500, 1000]:
        print(f"\nN_crit = {N_crit}:")
        results = run_batch(data_dir, N_crit=N_crit, chi_0=0.4)
        d, i, m = analyze(results)
        results_summary.append((N_crit, d, i, m))
    
    # Also test with higher chi_0 to compensate for winding suppression
    print("\n" + "=" * 70)
    print("N_crit=30 WITH ADJUSTED chi_0")
    print("=" * 70)
    
    for chi_0 in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        print(f"\nchi_0 = {chi_0}, N_crit = 30:")
        results = run_batch(data_dir, N_crit=30, chi_0=chi_0)
        d, i, m = analyze(results)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: N_crit vs Success Rate")
    print("=" * 70)
    print(f"\n{'N_crit':>8} {'Dwarf %':>10} {'Inter %':>10} {'Massive %':>10}")
    print("-" * 42)
    for N_crit, d, i, m in results_summary:
        print(f"{N_crit:8.0f} {d:10.1f} {i:10.1f} {m:10.1f}")
    
    # Find optimal
    best_massive = max(results_summary, key=lambda x: x[3])
    print(f"\nBest for massive spirals: N_crit = {best_massive[0]} ({best_massive[3]:.1f}%)")


if __name__ == "__main__":
    main()
