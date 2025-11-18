"""
Test the optimized pairing parameters on full SPARC sample.

Uses the best Solar System safe configuration from grid search.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_pairing import PairingParams, apply_pairing_boost
from sparc_utils import load_rotmod, rms_velocity


def main():
    # Load optimized parameters
    project_root = Path(__file__).parent.parent
    params_file = Path(__file__).parent / "results" / "pairing_best_params.json"
    
    if params_file.exists():
        with open(params_file, "r") as f:
            data = json.load(f)
            params_dict = data["parameters"]
            params = PairingParams(**params_dict)
        print("Loaded optimized parameters from grid search:")
    else:
        print("Using manually specified optimal parameters:")
        params = PairingParams(
            A_pair=5.0,
            sigma_c=15.0,
            gamma_sigma=3.0,
            ell_pair_kpc=20.0,
            p=1.5,
        )
    
    print(f"  A_pair = {params.A_pair}")
    print(f"  sigma_c = {params.sigma_c} km/s")
    print(f"  gamma_sigma = {params.gamma_sigma}")
    print(f"  ell_pair_kpc = {params.ell_pair_kpc} kpc")
    print(f"  p = {params.p}")
    
    # Load SPARC summary
    summary_path = project_root / "data" / "sparc" / "sparc_combined.csv"
    rotmod_dir = project_root / "data" / "Rotmod_LTG"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return
    
    summary = pd.read_csv(summary_path)
    print(f"\nLoaded {len(summary)} galaxies from SPARC")
    
    # Process galaxies
    rows = []
    for _, row in summary.iterrows():
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        try:
            df = load_rotmod(rotmod_path)
            R = df["R_kpc"].to_numpy()
            V_obs = df["V_obs"].to_numpy()
            V_gr = df["V_gr"].to_numpy()
            
            # Compute optimized pairing model
            g_gr = V_gr**2 / np.maximum(R, 1e-6)
            g_eff = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), params)
            V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
            
            rms_gr = rms_velocity(V_obs - V_gr)
            rms_pair = rms_velocity(V_obs - V_model)
            
            rows.append({
                "galaxy": name,
                "sigma_v": sigma_v,
                "rms_gr": rms_gr,
                "rms_pair": rms_pair,
                "delta_rms": rms_pair - rms_gr,
                "improvement_pct": (rms_gr - rms_pair) / rms_gr * 100 if rms_gr > 0 else 0.0,
            })
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue
    
    # Save results
    out_path = Path(__file__).parent / "results" / "optimized_pairing_sparc.csv"
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("OPTIMIZED PAIRING MODEL ON SPARC")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Results saved to {out_path}")
    
    if len(out) > 0:
        print("\nGlobal statistics:")
        print(f"  Mean RMS (GR):       {out['rms_gr'].mean():.2f} km/s")
        print(f"  Mean RMS (Pairing):  {out['rms_pair'].mean():.2f} km/s")
        print(f"  Mean improvement:    {out['improvement_pct'].mean():.2f}%")
        print(f"  Median improvement:  {out['improvement_pct'].median():.2f}%")
        print(f"  Fraction improved:   {(out['delta_rms'] < 0).sum()}/{len(out)} ({(out['delta_rms'] < 0).sum()/len(out)*100:.1f}%)")
        
        print("\nComparison with defaults:")
        print(f"  Default model:    +6.0% improvement")
        print(f"  Optimized model:  +{out['improvement_pct'].mean():.1f}% improvement")
        print(f"  Gain from tuning: +{out['improvement_pct'].mean() - 6.0:.1f}%")
        
        print("\nBest/worst performers:")
        best_idx = out['improvement_pct'].idxmax()
        worst_idx = out['improvement_pct'].idxmin()
        print(f"  Best:  {out.loc[best_idx, 'galaxy']} ({out.loc[best_idx, 'improvement_pct']:+.1f}%)")
        print(f"  Worst: {out.loc[worst_idx, 'galaxy']} ({out.loc[worst_idx, 'improvement_pct']:+.1f}%)")


if __name__ == "__main__":
    main()

