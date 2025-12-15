#!/usr/bin/env python3
"""
Sweep harness for geometry path-length suppression parameter (GEO_L_BULGE_MULT).

Tests the OPPOSITE direction: suppressing A for bulge galaxies instead of enhancing it.
"""

import subprocess
import re
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "run_regression_experimental.py"

def run_test(bulge_mult: float) -> dict:
    """Run regression test with given bulge_mult parameter."""
    cmd = [
        "python",
        str(SCRIPT_PATH),
        "--core",
        "--coherence=c",
        "--lgeom",
        f"--geo-l-bulge-mult={bulge_mult}",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    # Parse SPARC results
    sparc_line = None
    for line in result.stdout.split('\n'):
        if 'SPARC Galaxies' in line and 'RMS=' in line:
            sparc_line = line
            break
    
    if not sparc_line:
        return {
            "bulge_mult": bulge_mult,
            "overall_rms": None,
            "bulge_rms": None,
            "disk_rms": None,
            "error": "Could not parse SPARC results",
        }
    
    # Extract overall RMS
    rms_match = re.search(r'RMS=([\d.]+)', sparc_line)
    overall_rms = float(rms_match.group(1)) if rms_match else None
    
    # Extract bulge and disk RMS
    bulge_match = re.search(r'Bulge:\s*([\d.]+)', sparc_line)
    disk_match = re.search(r'Disk:\s*([\d.]+)', sparc_line)
    
    bulge_rms = float(bulge_match.group(1)) if bulge_match else None
    disk_rms = float(disk_match.group(1)) if disk_match else None
    
    return {
        "bulge_mult": bulge_mult,
        "overall_rms": overall_rms,
        "bulge_rms": bulge_rms,
        "disk_rms": disk_rms,
        "output": result.stdout,
        "error": result.stderr if result.returncode != 0 else None,
    }

def main():
    """Sweep GEO_L_BULGE_MULT parameter (suppression mode)."""
    # Parameter grid: positive values = gradual suppression, negative = threshold mode
    bulge_mults = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print("=" * 80)
    print("GEOMETRY PATH-LENGTH SUPPRESSION SWEEP (OPPOSITE DIRECTION)")
    print("=" * 80)
    print("Suppression formula: L_bulge = L_0 / (1 + mult × f_bulge)")
    print("Higher values = more suppression (smaller A for bulge galaxies)")
    print()
    print(f"{'α (mult)':<15} {'Overall RMS':<15} {'Bulge RMS':<15} {'Disk RMS':<15} {'Change':<15}")
    print("-" * 80)
    
    baseline_result = None
    results = []
    
    for bulge_mult in bulge_mults:
        result = run_test(bulge_mult)
        results.append(result)
        
        if bulge_mult == 0.0:
            baseline_result = result
        
        change_str = ""
        if baseline_result and result['overall_rms']:
            change = result['overall_rms'] - baseline_result['overall_rms']
            change_str = f"{change:+.2f}" if change != 0 else "0.00"
        
        print(
            f"{bulge_mult:<15.1f} "
            f"{result['overall_rms']:<15.2f} "
            f"{result['bulge_rms']:<15.2f} "
            f"{result['disk_rms']:<15.2f} "
            f"{change_str:<15}"
        )
    
    print("-" * 80)
    print()
    
    # Find best result
    valid_results = [r for r in results if r['overall_rms'] is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['overall_rms'])
        print(f"Best overall RMS: {best['overall_rms']:.2f} km/s at α = {best['bulge_mult']:.1f}")
        print(f"  Bulge RMS: {best['bulge_rms']:.2f} km/s")
        print(f"  Disk RMS: {best['disk_rms']:.2f} km/s")
        print()
        
        if baseline_result:
            improvement = baseline_result['overall_rms'] - best['overall_rms']
            bulge_improvement = baseline_result['bulge_rms'] - best['bulge_rms'] if baseline_result['bulge_rms'] else None
            print(f"Improvement vs baseline (α=0):")
            print(f"  Overall: {improvement:+.2f} km/s")
            if bulge_improvement:
                print(f"  Bulge: {bulge_improvement:+.2f} km/s")

if __name__ == "__main__":
    main()


