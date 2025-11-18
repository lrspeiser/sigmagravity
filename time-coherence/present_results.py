"""
Present comprehensive results from all tests.
"""

import json
import pandas as pd
from pathlib import Path

def main():
    print("=" * 80)
    print("TIME-COHERENCE KERNEL: COMPREHENSIVE RESULTS")
    print("=" * 80)
    
    # 1. MW Results
    print("\n1. MILKY WAY RESULTS")
    print("-" * 80)
    mw_path = Path("time-coherence/mw_coherence_canonical.json")
    if mw_path.exists():
        with open(mw_path, "r") as f:
            mw = json.load(f)
        print(f"  ell_coh_mean: {mw.get('ell_coh_mean_kpc', 'N/A'):.3f} kpc")
        print(f"  K_max: {mw.get('K_max', 'N/A'):.3f}")
        if mw.get('rms') is not None:
            print(f"  RMS: {mw['rms']:.2f} km/s")
            print(f"  RMS_GR: {mw['rms_gr']:.2f} km/s")
            print(f"  Improvement: {mw['rms_gr'] - mw['rms']:.2f} km/s")
    else:
        print("  MW results not found")
    
    # 2. SPARC Results
    print("\n2. SPARC GALAXY RESULTS")
    print("-" * 80)
    sparc_path = Path("time-coherence/sparc_coherence_canonical.csv")
    if sparc_path.exists():
        df = pd.read_csv(sparc_path)
        print(f"  Total galaxies: {len(df)}")
        print(f"  Mean delta_rms: {df['delta_rms'].mean():.3f} km/s")
        print(f"  Median delta_rms: {df['delta_rms'].median():.3f} km/s")
        improved = (df['delta_rms'] < 0).sum()
        print(f"  Improved: {improved}/{len(df)} ({improved/len(df)*100:.1f}%)")
        if 'ell_coh_mean_kpc' in df.columns:
            print(f"  Mean ell_coh: {df['ell_coh_mean_kpc'].mean():.2f} kpc")
            print(f"  Median ell_coh: {df['ell_coh_mean_kpc'].median():.2f} kpc")
    else:
        print("  SPARC results not found")
    
    # 3. Burr-XII Mapping
    print("\n3. BURR-XII MAPPING")
    print("-" * 80)
    burr_path = Path("time-coherence/burr_from_time_coherence_summary.json")
    if burr_path.exists():
        with open(burr_path, "r") as f:
            burr = json.load(f)
        print(f"  N galaxies: {burr['n_galaxies']}")
        print(f"  ell_0: {burr['ell0']['mean']:.2f} ± {burr['ell0']['std']:.2f} kpc")
        print(f"    (median: {burr['ell0']['median']:.2f} kpc)")
        print(f"  A: {burr['A']['mean']:.3f} ± {burr['A']['std']:.3f}")
        print(f"  Mean relative RMS: {burr['fit_quality']['mean_relative_rms']:.2%}")
        print(f"\n  Comparison to empirical:")
        print(f"    ell_0: {burr['ell0']['mean']:.2f} kpc (theory) vs {burr['comparison']['empirical_ell0_kpc']:.2f} kpc (empirical)")
        print(f"    A: {burr['A']['mean']:.3f} (theory) vs {burr['comparison']['empirical_A']:.3f} (empirical)")
    else:
        print("  Burr-XII summary not found")
    
    # 4. Solar System
    print("\n4. SOLAR SYSTEM SAFETY")
    print("-" * 80)
    ss_path = Path("time-coherence/solar_system_coherence_test.json")
    if ss_path.exists():
        with open(ss_path, "r") as f:
            ss = json.load(f)
        print(f"  Max K: {ss['max_K']:.3e}")
        print(f"  K at 1 AU: {ss['K_at_1AU']:.3e}")
        print(f"  K at 100 AU: {ss['K_at_100AU']:.3e}")
        print(f"  Status: {'PASS' if ss['safety_check']['passed'] else 'FAIL'}")
    else:
        print("  Solar System results not found")
    
    # 5. Clusters
    print("\n5. CLUSTER RESULTS")
    print("-" * 80)
    cluster_path = Path("time-coherence/cluster_coherence_summary.csv")
    if cluster_path.exists():
        df_cluster = pd.read_csv(cluster_path)
        print(f"  N clusters: {len(df_cluster)}")
        print(f"  Mean K_Einstein: {df_cluster['K_Einstein'].mean():.3f}")
        print(f"  Mean mass boost: {df_cluster['mass_boost'].mean():.2f}x")
        print(f"  Range: {df_cluster['mass_boost'].min():.2f}x - {df_cluster['mass_boost'].max():.2f}x")
    else:
        print("  Cluster summary not found")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nStatus: All phases complete")
    print("Next: Test morphology gates on SPARC to improve mean delta_RMS")

if __name__ == "__main__":
    main()

