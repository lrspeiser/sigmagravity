"""
Fit Burr-XII parameters from time-coherence kernel fits across SPARC.
Creates summary statistics for theory → empirical mapping.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from map_to_burr_xii import fit_burr_xii_to_theory, test_sparc_mapping

def main():
    sparc_csv = Path("time-coherence/sparc_coherence_test.csv")
    out_json = Path("time-coherence/burr_from_time_coherence_summary.json")
    
    if not sparc_csv.exists():
        print(f"Error: {sparc_csv} not found. Run test_sparc_coherence.py first.")
        return
    
    df = pd.read_csv(sparc_csv)
    
    # Load fiducial params
    fiducial_path = Path("time-coherence/time_coherence_fiducial.json")
    if fiducial_path.exists():
        with open(fiducial_path, "r") as f:
            fiducial = json.load(f)
    else:
        fiducial = {"alpha_length": 0.037, "beta_sigma": 1.5, "backreaction_cap": 10.0,
                   "A_global": 1.0, "p": 0.757, "n_coh": 0.5}
    
    ell0s = []
    As = []
    ps = []
    ns = []
    weights = []
    rms_errors = []
    
    print("=" * 80)
    print("FITTING BURR-XII TO TIME-COHERENCE KERNEL")
    print("=" * 80)
    print(f"\nProcessing {len(df)} galaxies...")
    
    # Sample a subset for speed (or process all)
    sample_size = min(50, len(df))  # Process up to 50 galaxies
    galaxies_to_process = df["galaxy"].head(sample_size).tolist()
    
    for i, galaxy in enumerate(galaxies_to_process):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{sample_size} galaxies...")
        
        try:
            result = test_sparc_mapping(galaxy)
            if result:
                ell0s.append(result["ell_0_kpc"])
                As.append(result["A"])
                ps.append(result["p"])
                ns.append(result["n"])
                weights.append(1.0)  # Equal weight for now
                rms_errors.append(result["relative_rms"])
        except Exception as e:
            continue
    
    if not ell0s:
        print("Error: No valid fits found")
        return
    
    ell0s = np.array(ell0s)
    As = np.array(As)
    ps = np.array(ps)
    ns = np.array(ns)
    w = np.array(weights)
    rms_errors = np.array(rms_errors)
    
    # Compute statistics
    summary = {
        "n_galaxies": int(len(ell0s)),
        "ell0": {
            "mean": float(np.mean(ell0s)),
            "median": float(np.median(ell0s)),
            "std": float(np.std(ell0s)),
            "min": float(np.min(ell0s)),
            "max": float(np.max(ell0s)),
            "percentiles": {
                "25": float(np.percentile(ell0s, 25)),
                "75": float(np.percentile(ell0s, 75)),
            }
        },
        "A": {
            "mean": float(np.mean(As)),
            "median": float(np.median(As)),
            "std": float(np.std(As)),
            "min": float(np.min(As)),
            "max": float(np.max(As)),
        },
        "p": {
            "mean": float(np.mean(ps)),
            "median": float(np.median(ps)),
            "std": float(np.std(ps)),
        },
        "n": {
            "mean": float(np.mean(ns)),
            "median": float(np.median(ns)),
            "std": float(np.std(ns)),
        },
        "fit_quality": {
            "mean_relative_rms": float(np.mean(rms_errors)),
            "median_relative_rms": float(np.median(rms_errors)),
            "max_relative_rms": float(np.max(rms_errors)),
        },
        "comparison": {
            "empirical_ell0_kpc": 5.0,  # From Σ-Gravity paper
            "empirical_A": 0.6,  # Approximate from Σ-Gravity
            "empirical_p": 0.757,  # From Σ-Gravity
            "empirical_n": 0.5,  # From Σ-Gravity
        }
    }
    
    # Save
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"\nBurr-XII parameters from time-coherence kernel:")
    print(f"  ell_0: {summary['ell0']['mean']:.2f} ± {summary['ell0']['std']:.2f} kpc")
    print(f"    (median: {summary['ell0']['median']:.2f} kpc)")
    print(f"    (range: {summary['ell0']['min']:.2f} - {summary['ell0']['max']:.2f} kpc)")
    print(f"  A: {summary['A']['mean']:.3f} ± {summary['A']['std']:.3f}")
    print(f"  p: {summary['p']['mean']:.3f} ± {summary['p']['std']:.3f}")
    print(f"  n: {summary['n']['mean']:.3f} ± {summary['n']['std']:.3f}")
    
    print(f"\nFit quality:")
    print(f"  Mean relative RMS: {summary['fit_quality']['mean_relative_rms']:.2%}")
    print(f"  Median relative RMS: {summary['fit_quality']['median_relative_rms']:.2%}")
    
    print(f"\nComparison to empirical Sigma-Gravity:")
    print(f"  ell_0: {summary['ell0']['mean']:.2f} kpc (theory) vs {summary['comparison']['empirical_ell0_kpc']:.2f} kpc (empirical)")
    print(f"  A: {summary['A']['mean']:.3f} (theory) vs {summary['comparison']['empirical_A']:.3f} (empirical)")
    print(f"  p: {summary['p']['mean']:.3f} (theory) vs {summary['comparison']['empirical_p']:.3f} (empirical)")
    
    print(f"\nSaved to {out_json}")

if __name__ == "__main__":
    main()

