"""
Test combined model with tuned roughness amplitude.

The default K0=0.774 is too strong. Try treating roughness as a small
correction (10-20% effect) rather than dominant mechanism.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_roughness import RoughnessParams
from microphysics_pairing import PairingParams
from microphysics_combined import CombinedParams, apply_combined_boost
from sparc_utils import load_rotmod, rms_velocity


def test_roughness_amplitude(K0_value, combination="additive"):
    """Test a specific roughness amplitude."""
    project_root = Path(__file__).parent.parent
    
    # Load optimized pairing parameters
    pairing_params_file = Path(__file__).parent / "results" / "pairing_best_params.json"
    with open(pairing_params_file, "r") as f:
        data = json.load(f)
        pairing_params = PairingParams(**data["parameters"])
    
    # Use reduced roughness amplitude
    roughness_params = RoughnessParams(K0=K0_value)
    
    params = CombinedParams(
        roughness=roughness_params,
        pairing=pairing_params,
        combination=combination,
    )
    
    # Load SPARC summary
    summary_path = project_root / "data" / "sparc" / "sparc_combined.csv"
    rotmod_dir = project_root / "data" / "Rotmod_LTG"
    summary = pd.read_csv(summary_path)
    
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
            g_eff = apply_combined_boost(g_gr, R, V_gr, sigma_v * np.ones_like(R), params)
            V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
            
            rms_gr = rms_velocity(V_obs - V_gr)
            rms_combined = rms_velocity(V_obs - V_model)
            
            rows.append({
                "galaxy": name,
                "rms_gr": rms_gr,
                "rms_combined": rms_combined,
                "improvement_pct": (rms_gr - rms_combined) / rms_gr * 100 if rms_gr > 0 else 0.0,
            })
        except:
            continue
    
    out = pd.DataFrame(rows)
    
    if len(out) > 0:
        mean_improv = out['improvement_pct'].mean()
        frac_improved = (out['improvement_pct'] > 0).sum() / len(out)
        return mean_improv, frac_improved, len(out)
    return None, None, 0


def main():
    print("="*80)
    print("TUNING ROUGHNESS AMPLITUDE IN COMBINED MODEL")
    print("="*80)
    print("\nTesting various K0 values for roughness component...")
    print("(Pairing uses optimized params: A=5.0, sigma_c=15, etc.)")
    
    # Test a range of K0 values
    K0_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.774]
    
    results = []
    
    for combination in ["additive", "multiplicative"]:
        print(f"\n{'-'*80}")
        print(f"{combination.upper()} COMBINATION")
        print(f"{'-'*80}")
        print(f"{'K0':<8} {'Mean Improv':>12} {'Frac Improved':>15} {'N galaxies':>12}")
        print("-"*80)
        
        for K0 in K0_values:
            mean_improv, frac_improved, n_gal = test_roughness_amplitude(K0, combination)
            if mean_improv is not None:
                print(f"{K0:<8.3f} {mean_improv:>11.2f}% {frac_improved:>14.1%} {n_gal:>12}")
                results.append({
                    "combination": combination,
                    "K0": K0,
                    "mean_improvement": mean_improv,
                    "fraction_improved": frac_improved,
                    "n_galaxies": n_gal,
                })
    
    # Find best configurations
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*80}")
    
    for combination in ["additive", "multiplicative"]:
        subset = results_df[results_df["combination"] == combination]
        best_idx = subset["mean_improvement"].idxmax()
        best = subset.loc[best_idx]
        
        print(f"\n{combination.upper()}:")
        print(f"  Best K0 = {best['K0']:.3f}")
        print(f"  Mean improvement: {best['mean_improvement']:.2f}%")
        print(f"  Fraction improved: {best['fraction_improved']:.1%}")
        
        # Compare with pairing alone
        if best['mean_improvement'] > 11.6:
            gain = best['mean_improvement'] - 11.6
            print(f"  Gain over pairing alone: +{gain:.1f}%")
        else:
            loss = 11.6 - best['mean_improvement']
            print(f"  Loss vs pairing alone: -{loss:.1f}%")
    
    # Save results
    results_df.to_csv(Path(__file__).parent / "results" / "combined_k0_scan.csv", index=False)
    print(f"\nResults saved to results/combined_k0_scan.csv")
    
    # Key insight
    print(f"\n{'='*80}")
    print("KEY INSIGHT")
    print(f"{'='*80}")
    
    best_overall = results_df.loc[results_df["mean_improvement"].idxmax()]
    
    if best_overall['mean_improvement'] < 11.6:
        print("\nCombining roughness + pairing REDUCES performance!")
        print(f"Best combined: {best_overall['mean_improvement']:.1f}%")
        print(f"Pairing alone: 11.6%")
        print("\nConclusion: Roughness model interferes with pairing model.")
        print("Pairing alone is superior. Focus optimization there instead.")
    else:
        print(f"\nCombining improves performance!")
        print(f"Best combined: {best_overall['mean_improvement']:.1f}%")
        print(f"Configuration: {best_overall['combination']} with K0={best_overall['K0']:.3f}")


if __name__ == "__main__":
    main()

