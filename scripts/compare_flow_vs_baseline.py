#!/usr/bin/env python3
"""Compare C_flow vs C_baseline values to understand the difference.

This script runs both baseline and flow modes, exports diagnostics,
and compares the coherence values.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def run_with_export(mode: str, output_csv: str) -> bool:
    """Run regression with pointwise export."""
    cmd = [
        sys.executable,
        'scripts/run_regression_experimental.py',
        '--core',
        '--export-sparc-points=' + output_csv,
    ]
    
    if mode == 'flow':
        cmd.extend(['--coherence=flow'])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent,
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {mode}: {e}")
        return False


def compare_coherence_values(baseline_csv: str, flow_csv: str) -> None:
    """Compare C_baseline vs C_flow values."""
    df_base = pd.read_csv(baseline_csv)
    df_flow = pd.read_csv(flow_csv)
    
    # Merge on galaxy and R_kpc
    df = df_base.merge(
        df_flow,
        on=['galaxy', 'R_kpc'],
        suffixes=('_base', '_flow'),
        how='inner'
    )
    
    print("\n" + "="*80)
    print("COHERENCE COMPARISON: C_baseline vs C_flow")
    print("="*80)
    
    # Overall statistics
    C_base = df['C_term_base'].values
    C_flow = df['C_term_flow'].values
    
    print(f"\nOverall (N={len(df)} points):")
    print(f"  C_baseline: mean={np.mean(C_base):.4f}, median={np.median(C_base):.4f}, std={np.std(C_base):.4f}")
    print(f"  C_flow:     mean={np.mean(C_flow):.4f}, median={np.median(C_flow):.4f}, std={np.std(C_flow):.4f}")
    print(f"  Difference: mean={np.mean(C_flow - C_base):.4f}, median={np.median(C_flow - C_base):.4f}")
    
    # By bulge fraction
    print(f"\nBy local bulge fraction:")
    for f_bin in [0.0, 0.3, 0.6, 0.9]:
        mask = (df['f_bulge_r_base'] >= f_bin) & (df['f_bulge_r_base'] < f_bin + 0.3)
        if mask.sum() > 0:
            C_base_bin = C_base[mask]
            C_flow_bin = C_flow[mask]
            print(f"  f_bulge_r ∈ [{f_bin:.1f}, {f_bin+0.3:.1f}] (N={mask.sum()}):")
            print(f"    C_baseline: {np.mean(C_base_bin):.4f} ± {np.std(C_base_bin):.4f}")
            print(f"    C_flow:     {np.mean(C_flow_bin):.4f} ± {np.std(C_flow_bin):.4f}")
            print(f"    ΔC:         {np.mean(C_flow_bin - C_base_bin):.4f}")
    
    # Correlation
    corr = np.corrcoef(C_base, C_flow)[0, 1]
    print(f"\nCorrelation: {corr:.4f}")
    
    # Save comparison
    out_csv = Path(__file__).parent / 'regression_results' / 'coherence_comparison.csv'
    out_csv.parent.mkdir(exist_ok=True)
    
    df_comp = pd.DataFrame({
        'galaxy': df['galaxy'],
        'R_kpc': df['R_kpc'],
        'f_bulge_r': df['f_bulge_r_base'],
        'C_baseline': C_base,
        'C_flow': C_flow,
        'dC': C_flow - C_base,
        'V_obs_kms': df['V_obs_kms_base'],
        'V_pred_baseline': df['V_pred_kms_base'],
        'V_pred_flow': df['V_pred_kms_flow'],
    })
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved comparison to: {out_csv}")


def main():
    """Run comparison."""
    outdir = Path(__file__).parent / 'regression_results'
    outdir.mkdir(exist_ok=True)
    
    baseline_csv = outdir / 'sparc_pointwise_baseline.csv'
    flow_csv = outdir / 'sparc_pointwise_flow.csv'
    
    print("Running baseline mode...")
    if not run_with_export('baseline', str(baseline_csv)):
        print("ERROR: Baseline run failed")
        return
    
    print("\nRunning flow mode...")
    if not run_with_export('flow', str(flow_csv)):
        print("ERROR: Flow run failed")
        return
    
    print("\nComparing coherence values...")
    compare_coherence_values(str(baseline_csv), str(flow_csv))


if __name__ == '__main__':
    main()



