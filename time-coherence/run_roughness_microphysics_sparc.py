"""
Test the roughness/time-coherence microphysics model on SPARC galaxies.

This evaluates a "pure microphysics" model where:
    g_eff = g_GR * (1 + K_rough(Ξ_system))

with Ξ_system computed from the exposure factor τ_coh / T_orb,
and K_rough(Ξ) = K0 * Ξ^γ from empirical fits.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_roughness import (
    RoughnessParams,
    system_level_exposure,
    apply_roughness_boost,
)
from sparc_utils import load_rotmod, rms_velocity


def evaluate_roughness_on_galaxy(rotmod_path, sigma_v, params: RoughnessParams):
    """
    Evaluate roughness microphysics model on a single galaxy.
    
    Parameters
    ----------
    rotmod_path : str or Path
        Path to SPARC rotmod file
    sigma_v : float
        Velocity dispersion in km/s
    params : RoughnessParams
        Model parameters
        
    Returns
    -------
    tuple
        (rms_gr, rms_rough, Xi_sys)
    """
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    # Compute system-level exposure factor
    Xi_sys = system_level_exposure(R, V_gr, sigma_v * np.ones_like(R), params)
    
    # Compute enhanced acceleration
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    g_eff = apply_roughness_boost(g_gr, Xi_sys, params)
    
    # Convert to velocity
    V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
    
    # Compute RMS
    rms_gr = rms_velocity(V_obs - V_gr)
    rms_rough = rms_velocity(V_obs - V_model)
    
    return rms_gr, rms_rough, Xi_sys


def main():
    parser = argparse.ArgumentParser(
        description="Test roughness microphysics on SPARC galaxies"
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
        default="time-coherence/results/roughness_microphysics_sparc.csv",
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
    params = RoughnessParams()
    print("\nRoughness parameters:")
    print(f"  K0 = {params.K0:.3f}")
    print(f"  gamma = {params.gamma:.3f}")
    print(f"  alpha_length = {params.alpha_length:.4f}")
    print(f"  beta_sigma = {params.beta_sigma:.3f}")
    
    # Process galaxies
    rows = []
    for _, row in summary.iterrows():
        name = row["galaxy_name"]
        sigma_v = row["sigma_velocity"]
        rotmod_path = rotmod_dir / f"{name}_rotmod.dat"
        
        if not rotmod_path.exists():
            continue
        
        try:
            rms_gr, rms_rough, Xi_sys = evaluate_roughness_on_galaxy(
                rotmod_path, sigma_v, params
            )
            rows.append({
                "galaxy": name,
                "sigma_v": sigma_v,
                "Xi": Xi_sys,
                "rms_gr": rms_gr,
                "rms_rough": rms_rough,
                "delta_rms": rms_rough - rms_gr,
                "improvement_pct": (rms_gr - rms_rough) / rms_gr * 100 if rms_gr > 0 else 0.0,
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
    print("ROUGHNESS MICROPHYSICS ON SPARC")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Results saved to {out_path}")
    
    if len(out) > 0:
        print("\nGlobal statistics:")
        print(f"  Mean RMS (GR):       {out['rms_gr'].mean():.2f} km/s")
        print(f"  Mean RMS (Rough):    {out['rms_rough'].mean():.2f} km/s")
        print(f"  Mean improvement:    {out['improvement_pct'].mean():.1f}%")
        print(f"  Median improvement:  {out['improvement_pct'].median():.1f}%")
        print(f"  Fraction improved:   {(out['delta_rms'] < 0).sum()}/{len(out)} ({(out['delta_rms'] < 0).sum()/len(out)*100:.1f}%)")
        
        print("\nExposure factor statistics:")
        print(f"  Mean Xi:    {out['Xi'].mean():.3e}")
        print(f"  Median Xi:  {out['Xi'].median():.3e}")
        print(f"  Min Xi:     {out['Xi'].min():.3e}")
        print(f"  Max Xi:     {out['Xi'].max():.3e}")


if __name__ == "__main__":
    main()

