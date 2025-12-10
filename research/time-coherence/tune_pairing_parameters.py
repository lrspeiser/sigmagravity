"""
Grid search to tune pairing microphysics parameters.

Goal: Find parameters that maximize improvement over GR while maintaining:
1. Solar System safety (K_pair at 1 AU << 10^-10)
2. Physical plausibility
3. Consistency across galaxy sample
"""

import argparse
from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import json

from microphysics_pairing import PairingParams, apply_pairing_boost
from sparc_utils import load_rotmod, rms_velocity


def evaluate_pairing_on_galaxy(rotmod_path, sigma_v, params: PairingParams):
    """Evaluate pairing model on a single galaxy."""
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    g_eff = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), params)
    V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
    
    rms_gr = rms_velocity(V_obs - V_gr)
    rms_pair = rms_velocity(V_obs - V_model)
    
    return rms_gr, rms_pair


def check_solar_system_safety(params: PairingParams):
    """
    Check if pairing model is Solar System safe.
    
    At R = 1 AU ≈ 5e-9 kpc, σ_v ~ 10 km/s (orbital velocities),
    we require K_pair < 10^-10.
    """
    from microphysics_pairing import K_pairing
    
    R_au = 5e-9  # 1 AU in kpc
    sigma_v_solar = 10.0  # km/s (typical planetary orbital velocity)
    
    K_solar = K_pairing(np.array([R_au]), sigma_v_solar, params)[0]
    
    return K_solar, K_solar < 1e-10


def evaluate_params_on_sample(summary, rotmod_dir, params: PairingParams):
    """Evaluate parameters on full SPARC sample."""
    rms_gr_list = []
    rms_pair_list = []
    
    for _, row in summary.iterrows():
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        try:
            rms_gr, rms_pair = evaluate_pairing_on_galaxy(
                rotmod_path, sigma_v, params
            )
            rms_gr_list.append(rms_gr)
            rms_pair_list.append(rms_pair)
        except:
            continue
    
    if len(rms_gr_list) == 0:
        return None
    
    rms_gr_mean = np.mean(rms_gr_list)
    rms_pair_mean = np.mean(rms_pair_list)
    improvement = (rms_gr_mean - rms_pair_mean) / rms_gr_mean * 100
    
    fraction_improved = np.sum(np.array(rms_pair_list) < np.array(rms_gr_list)) / len(rms_gr_list)
    
    # Check Solar System safety
    K_solar, is_safe = check_solar_system_safety(params)
    
    return {
        "rms_gr_mean": rms_gr_mean,
        "rms_pair_mean": rms_pair_mean,
        "improvement_pct": improvement,
        "fraction_improved": fraction_improved,
        "n_galaxies": len(rms_gr_list),
        "K_solar_system": K_solar,
        "solar_system_safe": is_safe,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune pairing microphysics parameters via grid search"
    )
    parser.add_argument(
        "--sparc-summary",
        type=str,
        default="data/sparc/sparc_combined.csv",
        help="Path to SPARC summary CSV",
    )
    parser.add_argument(
        "--rotmod-dir",
        type=str,
        default="data/Rotmod_LTG",
        help="Directory containing SPARC rotmod files",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="time-coherence/results/pairing_parameter_grid.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="time-coherence/results/pairing_best_params.json",
        help="Output JSON with best parameters",
    )
    args = parser.parse_args()
    
    # Load SPARC summary
    project_root = Path(__file__).parent.parent
    summary_path = project_root / args.sparc_summary
    rotmod_dir = project_root / args.rotmod_dir
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return
    
    summary = pd.read_csv(summary_path)
    print(f"Loaded {len(summary)} galaxies from {summary_path}")
    
    # Define parameter grid
    A_pair_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    sigma_c_values = [15.0, 20.0, 25.0, 30.0, 40.0]
    gamma_sigma_values = [1.5, 2.0, 2.5, 3.0]
    ell_pair_values = [2.0, 5.0, 10.0, 20.0]
    p_values = [0.5, 1.0, 1.5, 2.0, 3.0]
    
    print("\n" + "="*80)
    print("PAIRING PARAMETER GRID SEARCH")
    print("="*80)
    print(f"\nGrid size: {len(A_pair_values)} × {len(sigma_c_values)} × {len(gamma_sigma_values)} × {len(ell_pair_values)} × {len(p_values)}")
    print(f"Total combinations: {len(A_pair_values) * len(sigma_c_values) * len(gamma_sigma_values) * len(ell_pair_values) * len(p_values)}")
    
    # Grid search
    rows = []
    total = len(A_pair_values) * len(sigma_c_values) * len(gamma_sigma_values) * len(ell_pair_values) * len(p_values)
    count = 0
    
    for A_pair, sigma_c, gamma_sigma, ell_pair, p in itertools.product(
        A_pair_values, sigma_c_values, gamma_sigma_values, ell_pair_values, p_values
    ):
        count += 1
        if count % 100 == 0:
            print(f"  Progress: {count}/{total} ({count/total*100:.1f}%)")
        
        params = PairingParams(
            A_pair=A_pair,
            sigma_c=sigma_c,
            gamma_sigma=gamma_sigma,
            ell_pair_kpc=ell_pair,
            p=p,
        )
        
        result = evaluate_params_on_sample(summary, rotmod_dir, params)
        
        if result is not None:
            rows.append({
                "A_pair": A_pair,
                "sigma_c": sigma_c,
                "gamma_sigma": gamma_sigma,
                "ell_pair_kpc": ell_pair,
                "p": p,
                **result,
            })
    
    # Save results
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*80}")
    print(f"\nTested {len(out)} parameter combinations")
    print(f"Results saved to {out_path}")
    
    # Find best parameters (maximize improvement while maintaining safety)
    safe = out[out["solar_system_safe"] == True]
    
    if len(safe) > 0:
        best_idx = safe["improvement_pct"].idxmax()
        best = safe.loc[best_idx]
        
        print("\n" + "-"*80)
        print("BEST PARAMETERS (Solar System safe)")
        print("-"*80)
        print(f"  A_pair = {best['A_pair']:.2f}")
        print(f"  sigma_c = {best['sigma_c']:.1f} km/s")
        print(f"  gamma_sigma = {best['gamma_sigma']:.2f}")
        print(f"  ell_pair = {best['ell_pair_kpc']:.1f} kpc")
        print(f"  p = {best['p']:.2f}")
        print(f"\n  Improvement: {best['improvement_pct']:.2f}%")
        print(f"  Mean RMS: {best['rms_pair_mean']:.2f} km/s (vs GR: {best['rms_gr_mean']:.2f} km/s)")
        print(f"  Fraction improved: {best['fraction_improved']:.1%}")
        print(f"  K(Solar System): {best['K_solar_system']:.2e}")
        
        # Save best parameters
        best_params = {
            "A_pair": float(best['A_pair']),
            "sigma_c": float(best['sigma_c']),
            "gamma_sigma": float(best['gamma_sigma']),
            "ell_pair_kpc": float(best['ell_pair_kpc']),
            "p": float(best['p']),
            "performance": {
                "improvement_pct": float(best['improvement_pct']),
                "rms_pair_mean": float(best['rms_pair_mean']),
                "rms_gr_mean": float(best['rms_gr_mean']),
                "fraction_improved": float(best['fraction_improved']),
                "n_galaxies": int(best['n_galaxies']),
                "K_solar_system": float(best['K_solar_system']),
            }
        }
        
        out_json_path = Path(args.out_json)
        with open(out_json_path, "w") as f:
            json.dump(best_params, f, indent=2)
        
        print(f"\nBest parameters saved to {out_json_path}")
        
        # Top 10 safe configurations
        print("\n" + "-"*80)
        print("TOP 10 CONFIGURATIONS (Solar System safe)")
        print("-"*80)
        top10 = safe.nlargest(10, "improvement_pct")
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            print(f"{i:2d}. A={row['A_pair']:.1f}, σ_c={row['sigma_c']:.0f}, γ={row['gamma_sigma']:.1f}, ℓ={row['ell_pair_kpc']:.0f}, p={row['p']:.1f}  →  {row['improvement_pct']:+.1f}%")
    else:
        print("\nWARNING: No Solar System safe configurations found!")
        print("Need to adjust parameter ranges or radial envelope.")


if __name__ == "__main__":
    main()

