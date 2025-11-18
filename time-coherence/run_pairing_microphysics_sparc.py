"""
Test the graviton pairing/superfluid condensate microphysics model on SPARC galaxies.

This evaluates a "pairing-only" model where:
    g_eff = g_GR * (1 + K_pair(R, σ_v))

with K_pair strong in cold, extended systems and suppressed in hot, compact ones.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from microphysics_pairing import (
    PairingParams,
    apply_pairing_boost,
)
from sparc_utils import load_rotmod, rms_velocity


def evaluate_pairing_on_galaxy(rotmod_path, sigma_v, params: PairingParams):
    """
    Evaluate pairing microphysics model on a single galaxy.
    
    Parameters
    ----------
    rotmod_path : str or Path
        Path to SPARC rotmod file
    sigma_v : float
        Velocity dispersion in km/s
    params : PairingParams
        Model parameters
        
    Returns
    -------
    tuple
        (rms_gr, rms_pair)
    """
    df = load_rotmod(rotmod_path)
    R = df["R_kpc"].to_numpy()
    V_obs = df["V_obs"].to_numpy()
    V_gr = df["V_gr"].to_numpy()
    
    # Compute enhanced acceleration
    g_gr = V_gr**2 / np.maximum(R, 1e-6)
    g_eff = apply_pairing_boost(g_gr, R, sigma_v * np.ones_like(R), params)
    
    # Convert to velocity
    V_model = np.sqrt(np.clip(g_eff * R, 0.0, None))
    
    # Compute RMS
    rms_gr = rms_velocity(V_obs - V_gr)
    rms_pair = rms_velocity(V_obs - V_model)
    
    return rms_gr, rms_pair


def main():
    parser = argparse.ArgumentParser(
        description="Test pairing microphysics on SPARC galaxies"
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
        default="time-coherence/results/pairing_microphysics_sparc.csv",
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
    params = PairingParams()
    print("\nPairing parameters:")
    print(f"  A_pair = {params.A_pair:.3f}")
    print(f"  sigma_c = {params.sigma_c:.1f} km/s")
    print(f"  gamma_sigma = {params.gamma_sigma:.3f}")
    print(f"  ell_pair = {params.ell_pair_kpc:.1f} kpc")
    print(f"  p = {params.p:.3f}")
    
    # Process galaxies
    rows = []
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
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("PAIRING MICROPHYSICS ON SPARC")
    print(f"{'='*80}")
    print(f"\nProcessed {len(out)} galaxies")
    print(f"Results saved to {out_path}")
    
    if len(out) > 0:
        print("\nGlobal statistics:")
        print(f"  Mean RMS (GR):       {out['rms_gr'].mean():.2f} km/s")
        print(f"  Mean RMS (Pair):     {out['rms_pair'].mean():.2f} km/s")
        print(f"  Mean improvement:    {out['improvement_pct'].mean():.1f}%")
        print(f"  Median improvement:  {out['improvement_pct'].median():.1f}%")
        print(f"  Fraction improved:   {(out['delta_rms'] < 0).sum()}/{len(out)} ({(out['delta_rms'] < 0).sum()/len(out)*100:.1f}%)")
        
        print("\nSigma_v dependence:")
        print(f"  Cold galaxies (σ_v < 20 km/s): RMS = {out[out['sigma_v'] < 20]['rms_pair'].mean():.2f} km/s")
        print(f"  Warm galaxies (20-40 km/s):    RMS = {out[(out['sigma_v'] >= 20) & (out['sigma_v'] < 40)]['rms_pair'].mean():.2f} km/s")
        print(f"  Hot galaxies (σ_v > 40 km/s):  RMS = {out[out['sigma_v'] > 40]['rms_pair'].mean():.2f} km/s")


if __name__ == "__main__":
    main()

