#!/usr/bin/env python3
"""
Tune JJ coherence model parameters to optimize SPARC performance.

Tests different combinations of:
- JJ_XI_MULT: correlation scale multiplier (0.3 to 3.0)
- JJ_SMOOTH_M_POINTS: smoothing window for density proxy (3 to 9)

Reports best combination based on SPARC RMS.
"""

import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np

def run_test_with_params(xi_mult: float, smooth_points: int) -> Dict:
    """Run experimental test with given parameters and return SPARC results."""
    script_path = Path(__file__).parent / "run_regression_experimental.py"
    
    try:
        # Run the test with command-line parameters
        result = subprocess.run(
            [
                sys.executable, str(script_path), 
                '--core', 
                '--coherence=jj',
                f'--jj-xi-mult={xi_mult}',
                f'--jj-smooth={smooth_points}'
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse output for SPARC results
        output = result.stdout
        sparc_rms = None
        win_rate = None
        scatter = None
        
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
                            win_rate = float(win_str) / 100.0
                        except:
                            pass
                # Extract scatter
                if 'Scatter=' in line:
                    parts = line.split('Scatter=')
                    if len(parts) > 1:
                        scatter_str = parts[1].split()[0]
                        try:
                            scatter = float(scatter_str)
                        except:
                            pass
        
        # Also try to read JSON report
        report_path = script_path.parent / "regression_results" / "experimental_report_JJ.json"
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
                for result in report.get('results', []):
                    if result.get('name') == 'SPARC Galaxies':
                        details = result.get('details', {})
                        sparc_rms = details.get('mean_rms', sparc_rms)
                        win_rate = details.get('win_rate', win_rate)
                        scatter = details.get('rar_scatter_dex', scatter)
                        break
        
        return {
            'xi_mult': xi_mult,
            'smooth_points': smooth_points,
            'rms': sparc_rms,
            'win_rate': win_rate,
            'scatter': scatter,
            'success': sparc_rms is not None
        }
    except subprocess.TimeoutExpired:
        return {
            'xi_mult': xi_mult,
            'smooth_points': smooth_points,
            'rms': None,
            'win_rate': None,
            'scatter': None,
            'success': False,
            'error': 'timeout'
        }
    except Exception as e:
        return {
            'xi_mult': xi_mult,
            'smooth_points': smooth_points,
            'rms': None,
            'win_rate': None,
            'scatter': None,
            'success': False,
            'error': str(e)
        }


def main():
    """Test parameter combinations and find optimal values."""
    print("=" * 80)
    print("JJ COHERENCE PARAMETER TUNING")
    print("=" * 80)
    print()
    
    # Parameter ranges to test
    # Focus on smaller xi_mult values first (larger correlation scale)
    xi_mult_values = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    smooth_points_values = [3, 5, 7]
    
    print(f"Testing {len(xi_mult_values)} × {len(smooth_points_values)} = {len(xi_mult_values) * len(smooth_points_values)} combinations")
    print(f"  JJ_XI_MULT: {xi_mult_values}")
    print(f"  JJ_SMOOTH_M_POINTS: {smooth_points_values}")
    print()
    
    results = []
    best_rms = float('inf')
    best_params = None
    
    total = len(xi_mult_values) * len(smooth_points_values)
    current = 0
    
    for xi_mult in xi_mult_values:
        for smooth_points in smooth_points_values:
            current += 1
            print(f"[{current}/{total}] Testing JJ_XI_MULT={xi_mult:.1f}, JJ_SMOOTH_M_POINTS={smooth_points}...", end=' ', flush=True)
            
            result = run_test_with_params(xi_mult, smooth_points)
            results.append(result)
            
            if result['success'] and result['rms'] is not None:
                print(f"RMS={result['rms']:.2f} km/s, Win={result['win_rate']*100:.1f}%, Scatter={result['scatter']:.3f} dex")
                if result['rms'] < best_rms:
                    best_rms = result['rms']
                    best_params = (xi_mult, smooth_points)
            else:
                print(f"FAILED: {result.get('error', 'unknown error')}")
    
    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    # Sort by RMS
    successful = [r for r in results if r['success'] and r['rms'] is not None]
    successful.sort(key=lambda x: x['rms'])
    
    print("Top 10 combinations (by RMS):")
    print()
    print(f"{'JJ_XI_MULT':<12} {'JJ_SMOOTH':<12} {'RMS (km/s)':<12} {'Win Rate':<12} {'Scatter (dex)':<12}")
    print("-" * 60)
    
    for r in successful[:10]:
        print(f"{r['xi_mult']:<12.1f} {r['smooth_points']:<12} {r['rms']:<12.2f} {r['win_rate']*100:<12.1f} {r['scatter']:<12.3f}")
    
    print()
    if best_params:
        print(f"BEST COMBINATION:")
        print(f"  JJ_XI_MULT = {best_params[0]:.2f}")
        print(f"  JJ_SMOOTH_M_POINTS = {best_params[1]}")
        print(f"  RMS = {best_rms:.2f} km/s")
        
        best_result = next(r for r in successful if r['xi_mult'] == best_params[0] and r['smooth_points'] == best_params[1])
        print(f"  Win Rate = {best_result['win_rate']*100:.1f}%")
        print(f"  Scatter = {best_result['scatter']:.3f} dex")
        
        # Compare to baseline
        baseline_rms = 17.42
        improvement = ((baseline_rms - best_rms) / baseline_rms) * 100
        if improvement > 0:
            print(f"  vs Baseline (C model): {improvement:.1f}% improvement")
        else:
            print(f"  vs Baseline (C model): {abs(improvement):.1f}% worse")
    
    # Save results
    output_file = Path(__file__).parent / "regression_results" / "jj_tuning_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': {
                'xi_mult': best_params[0] if best_params else None,
                'smooth_points': best_params[1] if best_params else None,
                'rms': best_rms if best_params else None
            },
            'all_results': results,
            'top_10': successful[:10]
        }, f, indent=2, default=float)
    
    print()
    print(f"Full results saved to: {output_file}")
    
    return 0 if best_params else 1


if __name__ == "__main__":
    sys.exit(main())

