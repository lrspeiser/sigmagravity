"""
Compare all three microphysics models on SPARC galaxies.

This script runs all three microphysics candidates:
1. Roughness / time-coherence (path-integral decoherence)
2. Graviton pairing / superfluid condensate
3. Metric resonance with fluctuation spectrum

And compares their performance against GR and the empirical Σ-Gravity kernel.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import json

from microphysics_roughness import RoughnessParams, system_level_exposure, apply_roughness_boost
from microphysics_pairing import PairingParams, apply_pairing_boost
from microphysics_resonance import ResonanceParams, apply_resonance_boost
from sparc_utils import load_rotmod, rms_velocity


def evaluate_all_models_on_galaxy(rotmod_path, sigma_v, 
                                   params_rough, params_pair, params_res):
    """
    Evaluate all three microphysics models on a single galaxy.
    
    Returns
    -------
    dict
        Results for this galaxy with RMS for each model
    """
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    
    # Model 1: Roughness
    Xi_sys = system_level_exposure(R, V_gr, sigma_v * np.ones_like(R), params_rough)
    g_rough = apply_roughness_boost(g_gr, Xi_sys, params_rough)
    V_rough = np.sqrt(np.clip(g_rough * R, 0.0, None))
    rms_rough = rms_velocity(V_obs - V_rough)
    
    # Model 2: Pairing
    g_pair = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), params_pair)
    V_pair = np.sqrt(np.clip(g_pair * R, 0.0, None))
    rms_pair = rms_velocity(V_obs - V_pair)
    
    # Model 3: Resonance
    g_res = apply_resonance_boost(g_gr, R, V_gr, sigma_v * np.ones_like(R), params_res)
    V_res = np.sqrt(np.clip(g_res * R, 0.0, None))
    rms_res = rms_velocity(V_obs - V_res)
    
    # GR baseline
    rms_gr = rms_velocity(V_obs - V_gr)
    
    return {
        "rms_gr": rms_gr,
        "rms_rough": rms_rough,
        "rms_pair": rms_pair,
        "rms_res": rms_res,
        "Xi": Xi_sys,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all microphysics models on SPARC galaxies"
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
        default="time-coherence/results/microphysics_comparison_sparc.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="time-coherence/results/microphysics_comparison_summary.json",
        help="Output JSON summary",
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
    
    # Initialize parameters
    params_rough = RoughnessParams()
    params_pair = PairingParams()
    params_res = ResonanceParams()
    
    print("\n" + "="*80)
    print("COMPARING THREE MICROPHYSICS MODELS")
    print("="*80)
    print("\nModel 1: Roughness / Time-Coherence")
    print(f"  K0 = {params_rough.K0:.3f}, gamma = {params_rough.gamma:.3f}")
    print("\nModel 2: Graviton Pairing / Superfluid")
    print(f"  A_pair = {params_pair.A_pair:.3f}, sigma_c = {params_pair.sigma_c:.1f} km/s")
    print("\nModel 3: Metric Resonance")
    print(f"  A_res = {params_res.A_res:.3f}, alpha = {params_res.alpha:.3f}")
    
    # Process galaxies
    rows = []
    for i, (_, row) in enumerate(summary.iterrows()):
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(summary)} galaxies...")
        
        try:
            res = evaluate_all_models_on_galaxy(
                rotmod_path, sigma_v, 
                params_rough, params_pair, params_res
            )
            rows.append({
                "galaxy": name,
                "sigma_v": sigma_v,
                **res,
            })
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue
    
    # Save detailed results
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    
    # Compute summary statistics
    print(f"\n{'='*80}")
    print("MICROPHYSICS MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Detailed results saved to {out_path}")
    
    if len(out) > 0:
        print("\n" + "-"*80)
        print("RMS VELOCITY COMPARISON")
        print("-"*80)
        print(f"{'Model':<30} {'Mean RMS':>12} {'Median RMS':>12} {'Improvement':>12}")
        print("-"*80)
        
        rms_gr_mean = out['rms_gr'].mean()
        rms_gr_med = out['rms_gr'].median()
        print(f"{'GR (baseline)':<30} {rms_gr_mean:>11.2f}  {rms_gr_med:>11.2f}  {'---':>12}")
        
        for model, col in [("Roughness", "rms_rough"), 
                           ("Pairing", "rms_pair"), 
                           ("Resonance", "rms_res")]:
            mean_rms = out[col].mean()
            med_rms = out[col].median()
            improvement = (rms_gr_mean - mean_rms) / rms_gr_mean * 100
            print(f"{model:<30} {mean_rms:>11.2f}  {med_rms:>11.2f}  {improvement:>11.1f}%")
        
        print("\n" + "-"*80)
        print("FRACTIONAL IMPROVEMENT (Fraction of galaxies improved)")
        print("-"*80)
        
        for model, col in [("Roughness", "rms_rough"), 
                           ("Pairing", "rms_pair"), 
                           ("Resonance", "rms_res")]:
            improved = (out[col] < out['rms_gr']).sum()
            frac = improved / len(out) * 100
            print(f"  {model:<20} {improved:>4}/{len(out):<4} ({frac:>5.1f}%)")
        
        # Save summary JSON
        summary_dict = {
            "n_galaxies": len(out),
            "gr_baseline": {
                "mean_rms": float(rms_gr_mean),
                "median_rms": float(rms_gr_med),
            },
            "roughness": {
                "mean_rms": float(out['rms_rough'].mean()),
                "median_rms": float(out['rms_rough'].median()),
                "improvement_pct": float((rms_gr_mean - out['rms_rough'].mean()) / rms_gr_mean * 100),
                "fraction_improved": float((out['rms_rough'] < out['rms_gr']).sum() / len(out)),
            },
            "pairing": {
                "mean_rms": float(out['rms_pair'].mean()),
                "median_rms": float(out['rms_pair'].median()),
                "improvement_pct": float((rms_gr_mean - out['rms_pair'].mean()) / rms_gr_mean * 100),
                "fraction_improved": float((out['rms_pair'] < out['rms_gr']).sum() / len(out)),
            },
            "resonance": {
                "mean_rms": float(out['rms_res'].mean()),
                "median_rms": float(out['rms_res'].median()),
                "improvement_pct": float((rms_gr_mean - out['rms_res'].mean()) / rms_gr_mean * 100),
                "fraction_improved": float((out['rms_res'] < out['rms_gr']).sum() / len(out)),
            },
        }
        
        out_json_path = Path(args.out_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(summary_dict, f, indent=2)
        
        print(f"\nSummary statistics saved to {out_json_path}")
        
        # Determine best model
        improvements = {
            "Roughness": summary_dict["roughness"]["improvement_pct"],
            "Pairing": summary_dict["pairing"]["improvement_pct"],
            "Resonance": summary_dict["resonance"]["improvement_pct"],
        }
        best_model = max(improvements, key=improvements.get)
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model} ({improvements[best_model]:.1f}% improvement)")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()

