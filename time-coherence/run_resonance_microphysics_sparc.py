"""
Test the metric resonance microphysics model on SPARC galaxies.

This evaluates a "resonance-only" model where:
    g_eff = g_GR * (1 + K_res(R))

with K_res computed from an explicit fluctuation spectrum P(λ) and resonance filter.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_resonance import (
    ResonanceParams,
    apply_resonance_boost,
)
from sparc_utils import load_rotmod, rms_velocity


def evaluate_resonance_on_galaxy(rotmod_path, sigma_v, params: ResonanceParams):
    """
    Evaluate resonance microphysics model on a single galaxy.
    
    Parameters
    ----------
    rotmod_path : str or Path
        Path to SPARC rotmod file
    sigma_v : float
        Velocity dispersion in km/s
    params : ResonanceParams
        Model parameters
        
    Returns
    -------
    tuple
        (rms_gr, rms_res)
    """
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    # Compute enhanced acceleration
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    g_eff = apply_resonance_boost(g_gr, R, V_gr, sigma_v * np.ones_like(R), params)
    
    # Convert to velocity
    V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
    
    # Compute RMS
    rms_gr = rms_velocity(V_obs - V_gr)
    rms_res = rms_velocity(V_obs - V_model)
    
    return rms_gr, rms_res


def main():
    parser = argparse.ArgumentParser(
        description="Test resonance microphysics on SPARC galaxies"
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
        default="time-coherence/results/resonance_microphysics_sparc.csv",
        help="Output CSV path",
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
    params = ResonanceParams()
    print("\nResonance parameters:")
    print(f"  A_res = {params.A_res:.3f}")
    print(f"  alpha = {params.alpha:.3f} (spectral slope)")
    print(f"  lam_coh = {params.lam_coh_kpc:.1f} kpc")
    print(f"  lam_cut = {params.lam_cut_kpc:.1f} kpc")
    print(f"  Q0 = {params.Q0:.3f}")
    print(f"  beta_Q = {params.beta_Q:.3f}")
    
    # Process galaxies
    rows = []
    for i, row in enumerate(summary.iterrows()):
        _, row = row
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(summary)} galaxies...")
        
        try:
            rms_gr, rms_res = evaluate_resonance_on_galaxy(
                rotmod_path, sigma_v, params
            )
            rows.append({
                "galaxy": name,
                "sigma_v": sigma_v,
                "rms_gr": rms_gr,
                "rms_res": rms_res,
                "delta_rms": rms_res - rms_gr,
                "improvement_pct": (rms_gr - rms_res) / rms_gr * 100 if rms_gr > 0 else 0.0,
            })
        except Exception as e:
            print(f"  {name}: Failed - {e}")
            continue
    
    # Save results
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("RESONANCE MICROPHYSICS ON SPARC")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Results saved to {out_path}")
    
    if len(out) > 0:
        print("\nGlobal statistics:")
        print(f"  Mean RMS (GR):       {out['rms_gr'].mean():.2f} km/s")
        print(f"  Mean RMS (Res):      {out['rms_res'].mean():.2f} km/s")
        print(f"  Mean improvement:    {out['improvement_pct'].mean():.1f}%")
        print(f"  Median improvement:  {out['improvement_pct'].median():.1f}%")
        print(f"  Fraction improved:   {(out['delta_rms'] < 0).sum()}/{len(out)} ({(out['delta_rms'] < 0).sum()/len(out)*100:.1f}%)")


if __name__ == "__main__":
    main()

