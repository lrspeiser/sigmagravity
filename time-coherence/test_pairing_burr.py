"""
Test pairing model with Burr-XII radial envelope on SPARC.

Compare exponential vs Burr-XII radial profiles.
"""

from pathlib import Path
import json

import pandas as pd
import numpy as np

from microphysics_pairing import PairingParams, apply_pairing_boost
from microphysics_pairing_burr import (
    PairingBurrParams,
    apply_pairing_burr_boost,
    check_solar_system_safety_burr,
)
from sparc_utils import load_rotmod, rms_velocity


def main():
    project_root = Path(__file__).parent.parent
    
    # Load optimized exponential envelope parameters
    params_file = Path(__file__).parent / "results" / "pairing_best_params.json"
    with open(params_file, "r") as f:
        data = json.load(f)
        exp_params = PairingParams(**data["parameters"])
    
    print("="*80)
    print("COMPARING EXPONENTIAL VS BURR-XII RADIAL ENVELOPES")
    print("="*80)
    
    print("\nExponential envelope (current best):")
    print(f"  A_pair = {exp_params.A_pair}")
    print(f"  sigma_c = {exp_params.sigma_c} km/s")
    print(f"  gamma_sigma = {exp_params.gamma_sigma}")
    print(f"  ell_pair = {exp_params.ell_pair_kpc} kpc")
    print(f"  p = {exp_params.p}")
    print(f"  Envelope: C = 1 - exp(-(R/ell)^p)")
    
    # Create Burr-XII version with same base parameters
    burr_params = PairingBurrParams(
        A_pair=exp_params.A_pair,
        sigma_c=exp_params.sigma_c,
        gamma_sigma=exp_params.gamma_sigma,
        ell_pair_kpc=exp_params.ell_pair_kpc,
        p=0.757,  # Empirical Burr-XII shape
        q=0.5,    # Empirical Burr-XII shape
    )
    
    print("\nBurr-XII envelope (test):")
    print(f"  A_pair = {burr_params.A_pair}")
    print(f"  sigma_c = {burr_params.sigma_c} km/s")
    print(f"  gamma_sigma = {burr_params.gamma_sigma}")
    print(f"  ell_pair = {burr_params.ell_pair_kpc} kpc")
    print(f"  p = {burr_params.p}")
    print(f"  q = {burr_params.q}")
    print(f"  Envelope: C = 1 - [1 + (R/ell)^p]^(-q)")
    
    # Check Solar System safety
    K_solar_burr, is_safe_burr = check_solar_system_safety_burr(burr_params)
    print(f"\nSolar System safety check (Burr-XII):")
    print(f"  K(1 AU) = {K_solar_burr:.2e}")
    print(f"  Safe: {is_safe_burr} (threshold: 1e-10)")
    
    # Load SPARC data
    summary_path = project_root / "data" / "sparc" / "sparc_combined.csv"
    rotmod_dir = project_root / "data" / "Rotmod_LTG"
    summary = pd.read_csv(summary_path)
    
    print(f"\nLoaded {len(summary)} galaxies from SPARC")
    print("\nProcessing...")
    
    # Test both models
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
            
            g_gr = V_gr**2 / np.maximum(R, 1e-6)
            
            # Exponential envelope
            g_exp = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), exp_params)
            V_exp = np.sqrt(np.clip(g_exp * R, 0.0, None))
            rms_exp = rms_velocity(V_obs - V_exp)
            
            # Burr-XII envelope
            g_burr = apply_pairing_burr_boost(g_gr, R, sigma_v * np.ones_like(R), burr_params)
            V_burr = np.sqrt(np.clip(g_burr * R, 0.0, None))
            rms_burr = rms_velocity(V_obs - V_burr)
            
            # GR baseline
            rms_gr = rms_velocity(V_obs - V_gr)
            
            rows.append({
                "galaxy": name,
                "sigma_v": sigma_v,
                "rms_gr": rms_gr,
                "rms_exp": rms_exp,
                "rms_burr": rms_burr,
                "improvement_exp": (rms_gr - rms_exp) / rms_gr * 100 if rms_gr > 0 else 0.0,
                "improvement_burr": (rms_gr - rms_burr) / rms_gr * 100 if rms_gr > 0 else 0.0,
                "delta_improvement": ((rms_gr - rms_burr) - (rms_gr - rms_exp)) / rms_gr * 100 if rms_gr > 0 else 0.0,
            })
        except Exception as e:
            continue
    
    out = pd.DataFrame(rows)
    
    # Save results
    out_path = Path(__file__).parent / "results" / "pairing_burr_comparison.csv"
    out.to_csv(out_path, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Results saved to {out_path}")
    
    if len(out) > 0:
        print("\nGlobal statistics:")
        print(f"{'Model':<25} {'Mean RMS':>12} {'Improvement':>12} {'Frac Improved':>15}")
        print("-"*80)
        print(f"{'GR (baseline)':<25} {out['rms_gr'].mean():>11.2f}  {'---':>12} {'---':>15}")
        print(f"{'Exponential envelope':<25} {out['rms_exp'].mean():>11.2f}  {out['improvement_exp'].mean():>11.1f}% {(out['improvement_exp'] > 0).sum()/len(out):>14.1%}")
        print(f"{'Burr-XII envelope':<25} {out['rms_burr'].mean():>11.2f}  {out['improvement_burr'].mean():>11.1f}% {(out['improvement_burr'] > 0).sum()/len(out):>14.1%}")
        
        print(f"\nDifference (Burr-XII - Exponential):")
        print(f"  Mean delta improvement: {out['delta_improvement'].mean():+.2f}%")
        print(f"  Median delta improvement: {out['delta_improvement'].median():+.2f}%")
        
        if out['improvement_burr'].mean() > out['improvement_exp'].mean():
            gain = out['improvement_burr'].mean() - out['improvement_exp'].mean()
            print(f"\n  Burr-XII is BETTER by {gain:.2f}%")
        else:
            loss = out['improvement_exp'].mean() - out['improvement_burr'].mean()
            print(f"\n  Exponential is BETTER by {loss:.2f}%")
        
        # Save summary
        summary_dict = {
            "exponential": {
                "mean_rms": float(out['rms_exp'].mean()),
                "improvement_pct": float(out['improvement_exp'].mean()),
                "fraction_improved": float((out['improvement_exp'] > 0).sum() / len(out)),
            },
            "burr_xii": {
                "mean_rms": float(out['rms_burr'].mean()),
                "improvement_pct": float(out['improvement_burr'].mean()),
                "fraction_improved": float((out['improvement_burr'] > 0).sum() / len(out)),
                "solar_system_safe": bool(is_safe_burr),
                "K_solar_system": float(K_solar_burr),
            },
            "delta_improvement": float(out['delta_improvement'].mean()),
        }
        
        summary_path = Path(__file__).parent / "results" / "pairing_burr_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_dict, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

