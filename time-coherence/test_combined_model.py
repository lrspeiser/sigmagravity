"""
Test combined roughness + pairing model on SPARC galaxies.

Tests both additive and multiplicative combination strategies.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_roughness import RoughnessParams
from microphysics_pairing import PairingParams
from microphysics_combined import CombinedParams, apply_combined_boost
from sparc_utils import load_rotmod, rms_velocity


def evaluate_combined_on_galaxy(rotmod_path, sigma_v, params: CombinedParams):
    """Evaluate combined model on a single galaxy."""
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    g_eff = apply_combined_boost(g_gr, R, V_gr, sigma_v * np.ones_like(R), params)
    V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
    
    rms_gr = rms_velocity(V_obs - V_gr)
    rms_combined = rms_velocity(V_obs - V_model)
    
    return rms_gr, rms_combined


def main():
    project_root = Path(__file__).parent.parent
    
    # Load optimized pairing parameters
    pairing_params_file = Path(__file__).parent / "results" / "pairing_best_params.json"
    if pairing_params_file.exists():
        with open(pairing_params_file, "r") as f:
            data = json.load(f)
            pairing_params = PairingParams(**data["parameters"])
        print("Loaded optimized pairing parameters from grid search")
    else:
        pairing_params = PairingParams(
            A_pair=5.0,
            sigma_c=15.0,
            gamma_sigma=3.0,
            ell_pair_kpc=20.0,
            p=1.5,
        )
        print("Using manually specified pairing parameters")
    
    # Use default roughness parameters (could optimize these too)
    roughness_params = RoughnessParams()
    print("\nUsing default roughness parameters:")
    print(f"  K0 = {roughness_params.K0}")
    print(f"  gamma = {roughness_params.gamma}")
    
    # Load SPARC summary
    summary_path = project_root / "data" / "sparc" / "sparc_combined.csv"
    rotmod_dir = project_root / "data" / "Rotmod_LTG"
    
    if not summary_path.exists():
        print(f"Error: {summary_path} not found")
        return
    
    summary = pd.read_csv(summary_path)
    print(f"\nLoaded {len(summary)} galaxies from SPARC")
    
    # Test both combination strategies
    for combination in ["additive", "multiplicative"]:
        print(f"\n{'='*80}")
        print(f"TESTING {combination.upper()} COMBINATION")
        print(f"{'='*80}")
        
        params = CombinedParams(
            roughness=roughness_params,
            pairing=pairing_params,
            combination=combination,
        )
        
        rows = []
        for _, row in summary.iterrows():
            name = row["galaxy_name"]
            sigma_v = row["sigma_velocity"]
            rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
            
            if not rotmod_path.exists():
                continue
            
            try:
                rms_gr, rms_combined = evaluate_combined_on_galaxy(
                    rotmod_path, sigma_v, params
                )
                rows.append({
                    "galaxy": name,
                    "sigma_v": sigma_v,
                    "rms_gr": rms_gr,
                    "rms_combined": rms_combined,
                    "delta_rms": rms_combined - rms_gr,
                    "improvement_pct": (rms_gr - rms_combined) / rms_gr * 100 if rms_gr > 0 else 0.0,
                })
            except Exception as e:
                print(f"  {name}: Failed - {e}")
                continue
        
        out = pd.DataFrame(rows)
        
        # Save results
        out_path = Path(__file__).parent / "results" / f"combined_{combination}_sparc.csv"
        out.to_csv(out_path, index=False)
        
        # Print summary
        print(f"\nProcessed {len(out)} galaxies")
        print(f"Results saved to {out_path}")
        
        if len(out) > 0:
            print("\nGlobal statistics:")
            print(f"  Mean RMS (GR):        {out['rms_gr'].mean():.2f} km/s")
            print(f"  Mean RMS (Combined):  {out['rms_combined'].mean():.2f} km/s")
            print(f"  Mean improvement:     {out['improvement_pct'].mean():.2f}%")
            print(f"  Median improvement:   {out['improvement_pct'].median():.2f}%")
            print(f"  Fraction improved:    {(out['delta_rms'] < 0).sum()}/{len(out)} ({(out['delta_rms'] < 0).sum()/len(out)*100:.1f}%)")
            
            print("\nComparison with single models:")
            print(f"  Roughness alone:  -16.6% (with default params)")
            print(f"  Pairing alone:    +11.6% (with optimized params)")
            print(f"  Combined ({combination}): {out['improvement_pct'].mean():+.1f}%")
            
            if out['improvement_pct'].mean() > 11.6:
                gain = out['improvement_pct'].mean() - 11.6
                print(f"  Gain from combination: +{gain:.1f}%")
            
            # Save summary
            summary_dict = {
                "combination": combination,
                "n_galaxies": len(out),
                "rms_gr_mean": float(out['rms_gr'].mean()),
                "rms_combined_mean": float(out['rms_combined'].mean()),
                "improvement_pct": float(out['improvement_pct'].mean()),
                "improvement_median": float(out['improvement_pct'].median()),
                "fraction_improved": float((out['delta_rms'] < 0).sum() / len(out)),
                "roughness_params": {
                    "K0": roughness_params.K0,
                    "gamma": roughness_params.gamma,
                    "alpha_length": roughness_params.alpha_length,
                    "beta_sigma": roughness_params.beta_sigma,
                },
                "pairing_params": {
                    "A_pair": pairing_params.A_pair,
                    "sigma_c": pairing_params.sigma_c,
                    "gamma_sigma": pairing_params.gamma_sigma,
                    "ell_pair_kpc": pairing_params.ell_pair_kpc,
                    "p": pairing_params.p,
                },
            }
            
            summary_path = Path(__file__).parent / "results" / f"combined_{combination}_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary_dict, f, indent=2)
            print(f"\nSummary saved to {summary_path}")
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    
    # Load both results
    additive = pd.read_csv(Path(__file__).parent / "results" / "combined_additive_sparc.csv")
    multiplicative = pd.read_csv(Path(__file__).parent / "results" / "combined_multiplicative_sparc.csv")
    
    print("\n{'Model':<30} {'Mean RMS':>12} {'Improvement':>12} {'Frac Improved':>15}")
    print("-" * 80)
    print(f"{'GR (baseline)':<30} {additive['rms_gr'].mean():>11.2f}  {'---':>12} {'---':>15}")
    print(f"{'Roughness (default)':<30} {'---':>12} {'-16.6%':>12} {'62.4%':>15}")
    print(f"{'Pairing (optimized)':<30} {'---':>12} {'+11.6%':>12} {'78.2%':>15}")
    print(f"{'Combined (additive)':<30} {additive['rms_combined'].mean():>11.2f}  {additive['improvement_pct'].mean():>11.1f}% {(additive['delta_rms'] < 0).sum()/len(additive)*100:>14.1f}%")
    print(f"{'Combined (multiplicative)':<30} {multiplicative['rms_combined'].mean():>11.2f}  {multiplicative['improvement_pct'].mean():>11.1f}% {(multiplicative['delta_rms'] < 0).sum()/len(multiplicative)*100:>14.1f}%")
    print(f"{'Target (empirical Sigma-G)':<30} {'~25-27':>12} {'~30%':>12} {'~85-90%':>15}")


if __name__ == "__main__":
    main()

