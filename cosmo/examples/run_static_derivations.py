#!/usr/bin/env python3
"""
run_static_derivations.py
------------------------
Demonstration harness that uses your Σ kernel and runs all four derivations (A–D).
It can ingest a CSV with D_Mpc (distances) and write out z(D) predictions for comparison.
"""

import argparse
import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add cosmo to path
SCRIPT_DIR = Path(__file__).parent
COSMO_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(COSMO_DIR))

from sigma_redshift_derivations import SigmaKernel, bundle_models, z_curve

# Constants
Mpc = 3.0856775814913673e22  # m
c = 299792458.0  # m/s

def main():
    ap = argparse.ArgumentParser(description="Run static redshift derivations using Σ-Gravity kernel")
    ap.add_argument("--A", type=float, default=1.0, help="Amplitude for K=A*C")
    ap.add_argument("--ell0_kpc", type=float, default=200.0, help="Coherence length in kpc")
    ap.add_argument("--p", type=float, default=0.75, help="Burr-XII shape parameter")
    ap.add_argument("--ncoh", type=float, default=0.5, help="Burr-XII damping parameter")
    ap.add_argument("--H0", type=float, default=70.0, help="Hubble constant km/s/Mpc")
    ap.add_argument("--dist_csv", type=str, default="", help="CSV with D_Mpc column; if empty, uses 1..3000 Mpc")
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV file")
    args = ap.parse_args()

    # Set default output path if not provided
    if not args.out_csv:
        output_dir = COSMO_DIR / "outputs"
        output_dir.mkdir(exist_ok=True, parents=True)
        args.out_csv = str(output_dir / "static_derivations_curves.csv")

    print("="*80)
    print("STATIC REDSHIFT DERIVATIONS USING Σ-GRAVITY")
    print("="*80)
    print(f"\nParameters:")
    print(f"  Kernel: A={args.A}, ℓ₀={args.ell0_kpc} kpc, p={args.p}, n_coh={args.ncoh}")
    print(f"  H₀ = {args.H0} km/s/Mpc")
    print(f"  Output: {args.out_csv}")

    # Create kernel
    ker = SigmaKernel(A=args.A, ell0_kpc=args.ell0_kpc, p=args.p, ncoh=args.ncoh, metric="spherical")
    models = bundle_models(ker, H0_kms_Mpc=args.H0)

    # Load or generate distances
    if args.dist_csv:
        print(f"\nLoading distances from: {args.dist_csv}")
        D = pd.read_csv(args.dist_csv)["D_Mpc"].values.astype(float)
    else:
        print(f"\nGenerating synthetic distances: 1-3000 Mpc")
        D = np.linspace(1.0, 3000.0, 200)

    print(f"  Distance range: {D.min():.1f} - {D.max():.1f} Mpc ({len(D)} points)")

    # Compute redshift curves for all models
    print(f"\nComputing redshift curves...")
    out = pd.DataFrame({"D_Mpc": D})
    
    for name, model in models.items():
        print(f"  Computing {name} model...")
        out[f"z_{name}"] = z_curve(model, D)

    # Save results
    out.to_csv(args.out_csv, index=False)
    
    # Save metadata
    meta_file = args.out_csv.replace(".csv", "_meta.json")
    meta = {
        "kernel": dict(A=args.A, ell0_kpc=args.ell0_kpc, p=args.p, ncoh=args.ncoh),
        "H0": args.H0,
        "outfile": args.out_csv,
        "models": list(models.keys()),
        "distance_range": [float(D.min()), float(D.max())],
        "n_points": len(D)
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved:")
    print(f"  Data: {args.out_csv}")
    print(f"  Meta: {meta_file}")

    # Summary statistics
    print(f"\n" + "="*80)
    print("REDSHIFT AT KEY DISTANCES")
    print("="*80)
    
    test_distances = [10, 100, 500, 1000, 2000, 3000]
    print(f"\n{'D (Mpc)':>8} | ", end="")
    for name in models.keys():
        print(f"{'z_' + name:>10} | ", end="")
    print()
    print("-" * (8 + len(models) * 13))
    
    for D_test in test_distances:
        idx = np.argmin(np.abs(D - D_test))
        print(f"{D_test:8.0f} | ", end="")
        for name in models.keys():
            z_val = out.loc[idx, f"z_{name}"]
            print(f"{z_val:10.6f} | ", end="")
        print()

    # Compare to Hubble
    print(f"\n" + "="*80)
    print("COMPARISON TO HUBBLE LAW")
    print("="*80)
    
    # Compute reference Hubble law
    H0_SI = (args.H0 * 1000.0) / Mpc  # s^-1
    z_hubble = (H0_SI / c) * (D * Mpc)
    
    print(f"\nAt D = 1000 Mpc:")
    idx_1000 = np.argmin(np.abs(D - 1000))
    z_hub_ref = z_hubble[idx_1000]
    
    for name in models.keys():
        z_model = out.loc[idx_1000, f"z_{name}"]
        ratio = z_model / z_hub_ref
        print(f"  {name:>8}: z = {z_model:.6f} ({ratio*100:6.2f}% of Hubble)")

    print(f"\n  {'Hubble':>8}: z = {z_hub_ref:.6f} (reference)")

    print(f"\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
