#!/usr/bin/env python3
"""
Tune guided-gravity parameters to optimize SPARC performance.

Tests different combinations of:
- GUIDED_KAPPA: guidance strength (0.1 to 2.0)
- GUIDED_C_DEFAULT: default coherence when no proxy (0.0, 0.5, 1.0)

Reports best combination based on SPARC RMS and cluster ratio.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

def run_test_with_params(kappa: float, c_default: float) -> Dict:
    """Run experimental test with given parameters and return results."""
    script_path = Path(__file__).parent / "run_regression_experimental.py"
    
    try:
        # Run the test with command-line parameters
        result = subprocess.run(
            [
                sys.executable, str(script_path), 
                '--core', 
                '--guided',
                f'--guided-kappa={kappa}',
                f'--guided-c-default={c_default}'
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output for SPARC and cluster results
        output = result.stdout
        sparc_rms = None
        sparc_win_rate = None
        cluster_ratio = None
        
        for line in output.split('\n'):
            if 'SPARC Galaxies:' in line:
                # Extract RMS
                if 'RMS=' in line:
                    parts = line.split('RMS=')
                    if len(parts) > 1:
                        rms_str = parts[1].split()[0]
                        try:
                            sparc_rms = float(rms_str)
                        except:
                            pass
                # Extract win rate
                if 'Win=' in line:
                    parts = line.split('Win=')
                    if len(parts) > 1:
                        win_str = parts[1].split('%')[0]
                        try:
                            sparc_win_rate = float(win_str) / 100.0
                        except:
                            pass
            if 'Clusters:' in line and 'Median ratio=' in line:
                parts = line.split('Median ratio=')
                if len(parts) > 1:
                    ratio_str = parts[1].split()[0]
                    try:
                        cluster_ratio = float(ratio_str)
                    except:
                        pass
        
        # Also try to read JSON report
        report_path = script_path.parent / "regression_results" / "experimental_report_C_guided.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
                for result in report.get('results', []):
                    if result.get('name') == 'SPARC Galaxies':
                        details = result.get('details', {})
                        sparc_rms = details.get('mean_rms', sparc_rms)
                        sparc_win_rate = details.get('win_rate', sparc_win_rate)
                    elif result.get('name') == 'Clusters':
                        details = result.get('details', {})
                        cluster_ratio = details.get('median_ratio', cluster_ratio)
        
        return {
            'kappa': kappa,
            'c_default': c_default,
            'sparc_rms': sparc_rms,
            'sparc_win_rate': sparc_win_rate,
            'cluster_ratio': cluster_ratio,
            'success': sparc_rms is not None and cluster_ratio is not None
        }
    except subprocess.TimeoutExpired:
        return {
            'kappa': kappa,
            'c_default': c_default,
            'sparc_rms': None,
            'sparc_win_rate': None,
            'cluster_ratio': None,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'kappa': kappa,
            'c_default': c_default,
            'sparc_rms': None,
            'sparc_win_rate': None,
            'cluster_ratio': None,
            'success': False,
            'error': str(e)
        }


def main():
    """Test parameter combinations and find optimal values."""
    print("=" * 80)
    print("GUIDED-GRAVITY PARAMETER TUNING")
    print("=" * 80)
    print()
    
    # Parameter ranges to test
    kappa_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    c_default_values = [0.0, 0.5, 1.0]
    
    print(f"Testing {len(kappa_values)} × {len(c_default_values)} = {len(kappa_values) * len(c_default_values)} combinations")
    print(f"  GUIDED_KAPPA: {kappa_values}")
    print(f"  GUIDED_C_DEFAULT: {c_default_values}")
    print()
    
    results = []
    best_rms = float('inf')
    best_params = None
    baseline_rms = 17.42  # From previous runs
    
    total = len(kappa_values) * len(c_default_values)
    current = 0
    
    for kappa in kappa_values:
        for c_default in c_default_values:
            current += 1
            print(f"[{current}/{total}] Testing κ={kappa:.1f}, C_default={c_default:.1f}...", end=' ', flush=True)
            
            result = run_test_with_params(kappa, c_default)
            results.append(result)
            
            if result['success'] and result['sparc_rms'] is not None:
                cluster_ok = 0.8 < result['cluster_ratio'] < 1.3 if result['cluster_ratio'] else False
                print(f"SPARC RMS={result['sparc_rms']:.2f} km/s, Clusters={result['cluster_ratio']:.3f} {'✓' if cluster_ok else '✗'}")
                if result['sparc_rms'] < best_rms and cluster_ok:
                    best_rms = result['sparc_rms']
                    best_params = (kappa, c_default)
            else:
                print(f"FAILED: {result.get('error', 'unknown error')}")
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Sort by SPARC RMS, filter valid cluster ratios
    successful = [r for r in results if r['success'] and r['sparc_rms'] is not None and 
                  r['cluster_ratio'] is not None and 0.8 < r['cluster_ratio'] < 1.3]
    successful.sort(key=lambda x: x['sparc_rms'])
    
    print("Top 15 combinations (by SPARC RMS, cluster ratio 0.8-1.3):")
    print()
    print(f"{'κ':<8} {'C_default':<12} {'SPARC RMS':<12} {'Win Rate':<12} {'Cluster Ratio':<12} {'vs Baseline':<12}")
    print("-" * 70)
    
    for r in successful[:15]:
        vs_baseline = ((r['sparc_rms'] - baseline_rms) / baseline_rms) * 100
        vs_baseline_str = f"{vs_baseline:+.1f}%"
        print(f"{r['kappa']:<8.1f} {r['c_default']:<12.1f} {r['sparc_rms']:<12.2f} {r['sparc_win_rate']*100:<12.1f} {r['cluster_ratio']:<12.3f} {vs_baseline_str:<12}")
    
    print()
    if best_params:
        print(f"BEST COMBINATION:")
        print(f"  GUIDED_KAPPA = {best_params[0]:.2f}")
        print(f"  GUIDED_C_DEFAULT = {best_params[1]:.2f}")
        print(f"  SPARC RMS = {best_rms:.2f} km/s")
        
        best_result = next(r for r in successful if r['kappa'] == best_params[0] and r['c_default'] == best_params[1])
        print(f"  Win Rate = {best_result['sparc_win_rate']*100:.1f}%")
        print(f"  Cluster Ratio = {best_result['cluster_ratio']:.3f}")
        
        # Compare to baseline
        improvement = ((baseline_rms - best_rms) / baseline_rms) * 100
        if improvement > 0:
            print(f"  vs Baseline: {improvement:.1f}% improvement")
        else:
            print(f"  vs Baseline: {abs(improvement):.1f}% worse")
    else:
        print("No valid combination found (all cluster ratios out of range)")
    
    # Also find best for clusters
    cluster_best = None
    cluster_best_ratio = None
    for r in successful:
        if r['cluster_ratio'] is not None:
            cluster_error = abs(r['cluster_ratio'] - 1.0)
            if cluster_best is None or cluster_error < abs(cluster_best_ratio - 1.0):
                cluster_best = (r['kappa'], r['c_default'])
                cluster_best_ratio = r['cluster_ratio']
    
    if cluster_best:
        print()
        print(f"BEST FOR CLUSTERS (closest to 1.0):")
        print(f"  GUIDED_KAPPA = {cluster_best[0]:.2f}")
        print(f"  GUIDED_C_DEFAULT = {cluster_best[1]:.2f}")
        print(f"  Cluster Ratio = {cluster_best_ratio:.3f}")
    
    # Save results
    output_file = Path(__file__).parent / "regression_results" / "guided_tuning_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': {
                'kappa': best_params[0] if best_params else None,
                'c_default': best_params[1] if best_params else None,
                'sparc_rms': best_rms if best_params else None
            },
            'best_cluster_params': {
                'kappa': cluster_best[0] if cluster_best else None,
                'c_default': cluster_best[1] if cluster_best else None,
                'cluster_ratio': cluster_best_ratio if cluster_best else None
            },
            'all_results': results,
            'top_15': successful[:15]
        }, f, indent=2, default=float)
    
    print()
    print(f"Full results saved to: {output_file}")
    
    return 0 if best_params else 1


if __name__ == "__main__":
    sys.exit(main())

