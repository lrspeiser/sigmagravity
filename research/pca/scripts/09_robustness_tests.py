#!/usr/bin/env python3
"""
Robustness testing: Compare PCA subspaces across different normalizations and weightings.

Tests:
1. Radius normalization: R/Rd vs R/Re
2. Velocity normalization: with vs without V/Vf  
3. Weighting: weighted vs unweighted PCA

Outputs principal angles for subspace stability assessment.
"""
import argparse
import numpy as np
import pandas as pd
import subprocess
import sys
from pathlib import Path

def principal_angles(A, B):
    """Compute principal angles between subspaces spanned by A and B columns."""
    Qa, _ = np.linalg.qr(A, mode='reduced')
    Qb, _ = np.linalg.qr(B, mode='reduced')
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0, 1)
    return np.arccos(s)  # radians

def run_pipeline(suffix, radius_norm, velocity_norm):
    """Run complete PCA pipeline with specified normalization."""
    print(f"\n{'='*70}")
    print(f"Running pipeline: {suffix}")
    print(f"  Radius norm: {radius_norm}")
    print(f"  Velocity norm: {velocity_norm}")
    print(f"{'='*70}\n")
    
    base_dir = Path(__file__).parent.parent
    
    # Step 1: Ingest
    cmd1 = [
        sys.executable, str(base_dir / 'scripts' / '01_ingest_sparc.py'),
        '--curves_dir', str(base_dir / 'data' / 'raw' / 'sparc_curves'),
        '--meta_csv', str(base_dir / 'data' / 'raw' / 'metadata' / 'sparc_meta.csv'),
        '--out_npz', str(base_dir / 'data' / 'processed' / f'sparc_curvematrix_{suffix}.npz'),
        '--grid_min', '0.2', '--grid_max', '6.0', '--grid_k', '50',
        '--norm_radius', radius_norm,
        '--norm_velocity', velocity_norm
    ]
    print("Step 1/4: Ingest...")
    subprocess.run(cmd1, check=True, capture_output=True)
    
    # Step 2: Build features
    cmd2 = [
        sys.executable, str(base_dir / 'scripts' / '02_build_curve_matrix.py'),
        '--npz', str(base_dir / 'data' / 'processed' / f'sparc_curvematrix_{suffix}.npz'),
        '--scalars_json', str(base_dir / 'configs' / 'scalars_sparc.json'),
        '--out_prefix', str(base_dir / 'data' / 'processed' / f'sparc_features_{suffix}')
    ]
    print("Step 2/4: Build features...")
    subprocess.run(cmd2, check=True, capture_output=True)
    
    # Step 3: Run PCA
    cmd3 = [
        sys.executable, str(base_dir / 'scripts' / '03_run_weighted_pca.py'),
        '--features_npz', str(base_dir / 'data' / 'processed' / f'sparc_features_{suffix}_curve_only.npz'),
        '--n_components', '10',
        '--out_dir', str(base_dir / 'outputs' / 'robustness')
    ]
    # Rename output for clarity
    print("Step 3/4: Run PCA...")
    subprocess.run(cmd3, check=True, capture_output=True)
    
    # Move/rename the output
    src = base_dir / 'outputs' / 'robustness' / 'pca_results_curve_only.npz'
    dst = base_dir / 'outputs' / 'robustness' / f'pca_results_{suffix}.npz'
    if src.exists():
        src.rename(dst)
    
    print(f"✓ Pipeline complete: {suffix}\n")
    return dst

def compare_subspaces(pca1_path, pca2_path, label1, label2, n_components=3):
    """Compare PC subspaces and print principal angles."""
    print(f"\n{'='*70}")
    print(f"Comparing {label1} vs {label2}")
    print(f"{'='*70}")
    
    # Load PCA results
    pca1 = np.load(pca1_path, allow_pickle=True)
    pca2 = np.load(pca2_path, allow_pickle=True)
    
    # Get names and align
    names1 = pca1['names']
    names2 = pca2['names']
    
    # Find common galaxies
    common = set(names1) & set(names2)
    idx1 = [i for i, n in enumerate(names1) if n in common]
    idx2 = [i for i, n in enumerate(names2) if n in common]
    
    # Extract aligned scores
    scores1 = pca1['scores'][idx1, :n_components]
    scores2 = pca2['scores'][idx2, :n_components]
    
    # Compute principal angles
    angles = principal_angles(scores1, scores2)
    
    print(f"\nPrincipal angles (first {n_components} PCs):")
    for i, angle in enumerate(angles):
        deg = np.degrees(angle)
        print(f"  Angle {i+1}: {angle:.4f} rad = {deg:.2f}°")
    
    # Interpretation
    print(f"\nInterpretation:")
    if angles[0] < np.radians(10):
        print(f"  ✓ PC1 is STABLE (angle < 10°)")
    else:
        print(f"  ⚠ PC1 differs significantly (angle = {np.degrees(angles[0]):.1f}°)")
    
    if angles[1] < np.radians(20):
        print(f"  ✓ PC2 is stable (angle < 20°)")
    else:
        print(f"  ⚠ PC2 differs (angle = {np.degrees(angles[1]):.1f}°)")
    
    return angles

def main():
    parser = argparse.ArgumentParser(description='Test PCA robustness across normalizations')
    parser.add_argument('--baseline', default='Rd', help='Baseline already computed (Rd/Vf)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / 'outputs' / 'robustness'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline (already exists)
    baseline_path = base_dir / 'outputs' / 'pca_results_curve_only.npz'
    
    results = {}
    
    # Test 1: R/Re normalization
    print("\n" + "="*70)
    print("TEST 1: RADIUS NORMALIZATION (R/Rd vs R/Re)")
    print("="*70)
    
    re_path = run_pipeline('Re', 'Re', 'Vf')
    angles_radius = compare_subspaces(baseline_path, re_path, 'R/Rd', 'R/Re')
    results['radius_norm'] = angles_radius
    
    # Test 2: Velocity normalization  
    print("\n" + "="*70)
    print("TEST 2: VELOCITY NORMALIZATION (V/Vf vs unnormalized V)")
    print("="*70)
    
    noVf_path = run_pipeline('noVf', 'Rd', 'none')
    angles_velocity = compare_subspaces(baseline_path, noVf_path, 'V/Vf', 'V (unnormalized)')
    results['velocity_norm'] = angles_velocity
    
    # Summary
    print("\n" + "="*70)
    print("ROBUSTNESS SUMMARY")
    print("="*70)
    
    print("\nTest 1: R/Rd vs R/Re")
    print(f"  PC1 angle: {np.degrees(results['radius_norm'][0]):.2f}° → {'STABLE' if results['radius_norm'][0] < np.radians(10) else 'VARIABLE'}")
    print(f"  PC2 angle: {np.degrees(results['radius_norm'][1]):.2f}° → {'STABLE' if results['radius_norm'][1] < np.radians(20) else 'VARIABLE'}")
    
    print("\nTest 2: V/Vf vs unnormalized V")
    print(f"  PC1 angle: {np.degrees(results['velocity_norm'][0]):.2f}° → {'STABLE' if results['velocity_norm'][0] < np.radians(10) else 'VARIABLE'}")
    print(f"  PC2 angle: {np.degrees(results['velocity_norm'][1]):.2f}° → {'STABLE' if results['velocity_norm'][1] < np.radians(20) else 'VARIABLE'}")
    
    # Save results
    summary_file = output_dir / 'robustness_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("PCA ROBUSTNESS TESTS\n")
        f.write("="*70 + "\n\n")
        
        f.write("Test 1: Radius Normalization (R/Rd vs R/Re)\n")
        for i, angle in enumerate(results['radius_norm']):
            f.write(f"  PC{i+1} angle: {angle:.4f} rad = {np.degrees(angle):.2f}°\n")
        
        f.write("\nTest 2: Velocity Normalization (V/Vf vs unnormalized)\n")
        for i, angle in enumerate(results['velocity_norm']):
            f.write(f"  PC{i+1} angle: {angle:.4f} rad = {np.degrees(angle):.2f}°\n")
        
        f.write("\nConclusion:\n")
        if all(a < np.radians(10) for a in [results['radius_norm'][0], results['velocity_norm'][0]]):
            f.write("  ✓ PC1 is ROBUST across normalization choices\n")
        if all(a < np.radians(20) for a in [results['radius_norm'][1], results['velocity_norm'][1]]):
            f.write("  ✓ PC2 is STABLE across normalization choices\n")
    
    print(f"\n✓ Results saved to {summary_file}")

if __name__ == '__main__':
    main()

