#!/usr/bin/env python3
"""
Tune dynamical path-length parameter v_coh to optimize SPARC performance.

Tests different values of v_coh (coherence transport speed) to find optimal
L_eff(r) = v_coh * r / V_pred(r) that improves SPARC predictions.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

def run_test_with_vcoh(vcoh: float) -> Dict:
    """Run experimental test with given v_coh and return results."""
    script_path = Path(__file__).parent / "run_regression_experimental.py"
    
    try:
        result = subprocess.run(
            [
                sys.executable, str(script_path), 
                '--core', 
                '--ldyn',
                f'--vcoh-kms={vcoh}'
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout
        sparc_rms = None
        sparc_win_rate = None
        cluster_ratio = None
        
        for line in output.split('\n'):
            if 'SPARC Galaxies:' in line:
                if 'RMS=' in line:
                    parts = line.split('RMS=')
                    if len(parts) > 1:
                        rms_str = parts[1].split()[0]
                        try:
                            sparc_rms = float(rms_str)
                        except:
                            pass
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
        report_path = script_path.parent / "regression_results" / "experimental_report_C.json"
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
            'vcoh': vcoh,
            'sparc_rms': sparc_rms,
            'sparc_win_rate': sparc_win_rate,
            'cluster_ratio': cluster_ratio,
            'success': sparc_rms is not None and cluster_ratio is not None
        }
    except subprocess.TimeoutExpired:
        return {
            'vcoh': vcoh,
            'sparc_rms': None,
            'sparc_win_rate': None,
            'cluster_ratio': None,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'vcoh': vcoh,
            'sparc_rms': None,
            'sparc_win_rate': None,
            'cluster_ratio': None,
            'success': False,
            'error': str(e)
        }


def main():
    """Test v_coh values and find optimal."""
    print("=" * 80)
    print("DYNAMICAL PATH-LENGTH PARAMETER TUNING")
    print("=" * 80)
    print()
    
    # Parameter range to test
    vcoh_values = [5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 60, 80, 100]
    
    print(f"Testing {len(vcoh_values)} values of v_coh")
    print(f"  v_coh (km/s): {vcoh_values}")
    print()
    
    results = []
    best_rms = float('inf')
    best_params = None
    baseline_rms = 17.42  # From previous runs
    
    total = len(vcoh_values)
    current = 0
    
    for vcoh in vcoh_values:
        current += 1
        print(f"[{current}/{total}] Testing v_coh={vcoh:.0f} km/s...", end=' ', flush=True)
        
        result = run_test_with_vcoh(vcoh)
        results.append(result)
        
        if result['success'] and result['sparc_rms'] is not None:
            cluster_ok = 0.8 < result['cluster_ratio'] < 1.3 if result['cluster_ratio'] else False
            print(f"SPARC RMS={result['sparc_rms']:.2f} km/s, Clusters={result['cluster_ratio']:.3f} {'✓' if cluster_ok else '✗'}")
            if result['sparc_rms'] < best_rms and cluster_ok:
                best_rms = result['sparc_rms']
                best_params = vcoh
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
    
    print("Top 10 combinations (by SPARC RMS, cluster ratio 0.8-1.3):")
    print()
    print(f"{'v_coh (km/s)':<15} {'SPARC RMS':<12} {'Win Rate':<12} {'Cluster Ratio':<12} {'vs Baseline':<12}")
    print("-" * 65)
    
    for r in successful[:10]:
        vs_baseline = ((r['sparc_rms'] - baseline_rms) / baseline_rms) * 100
        vs_baseline_str = f"{vs_baseline:+.1f}%"
        print(f"{r['vcoh']:<15.0f} {r['sparc_rms']:<12.2f} {r['sparc_win_rate']*100:<12.1f} {r['cluster_ratio']:<12.3f} {vs_baseline_str:<12}")
    
    print()
    if best_params:
        print(f"BEST COMBINATION:")
        print(f"  v_coh = {best_params:.0f} km/s")
        print(f"  SPARC RMS = {best_rms:.2f} km/s")
        
        best_result = next(r for r in successful if r['vcoh'] == best_params)
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
    
    # Save results
    output_file = Path(__file__).parent / "regression_results" / "dyn_l_tuning_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': {
                'vcoh': best_params if best_params else None,
                'sparc_rms': best_rms if best_params else None
            },
            'all_results': results,
            'top_10': successful[:10]
        }, f, indent=2, default=float)
    
    print()
    print(f"Full results saved to: {output_file}")
    
    return 0 if best_params else 1


if __name__ == "__main__":
    sys.exit(main())

