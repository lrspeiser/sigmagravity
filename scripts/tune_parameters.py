#!/usr/bin/env python3
"""
Parameter tuning script for Σ-Gravity extended phase coherence model.
"""

import subprocess
import re
from itertools import product

def parse_output(output):
    """Extract key metrics from regression output."""
    results = {}
    
    lines = output.split('\n')
    for line in lines:
        # Parse SPARC line
        if 'SPARC Galaxies' in line and 'km/s' in line and 'LCDM' in line:
            # Extract all numbers from line
            nums = re.findall(r'\d+\.?\d*', line)
            nums = [float(n) for n in nums if float(n) > 0]
            if len(nums) >= 4:
                results['sparc_observed'] = nums[0]
                results['sparc_baseline'] = nums[1]
                results['sparc_new'] = nums[2]
                results['sparc_mond'] = nums[3]
        
        # Parse Wide Binaries line - "1.35+/-0.10" format
        if 'Wide Binaries' in line and '+/-' in line:
            # Handle "1.35+/-0.10" format by extracting all numbers
            nums = re.findall(r'\d+\.?\d*', line)
            nums = [float(n) for n in nums if float(n) > 0]
            # Should be: [1.35, 0.10, 1.63, 1.58, 1.73, 1.00] 
            if len(nums) >= 5:
                results['wb_observed'] = nums[0]
                # nums[1] is error (0.10)
                results['wb_baseline'] = nums[2]
                results['wb_new'] = nums[3]
                results['wb_mond'] = nums[4]
        
        # Parse DF2 line - "8.50+/-2.30" format  
        if 'DF2 (UDG)' in line and 'km/s' in line:
            nums = re.findall(r'\d+\.?\d*', line)
            nums = [float(n) for n in nums if float(n) > 0]
            # Should be: [8.50, 2.30, 20.77, 19.82, 20.00, 8.84]
            if len(nums) >= 4:
                results['df2_observed'] = nums[0]
                # nums[1] is error (2.30)
                results['df2_baseline'] = nums[2]
                results['df2_new'] = nums[3]
        
        # Parse improvement percentages
        if 'SPARC Galaxies' in line and '%' in line and ('IMPROVED' in line or 'WORSENED' in line or 'UNCHANGED' in line):
            match = re.search(r'([+-]?\d+\.?\d*)%', line)
            if match:
                results['sparc_improv'] = float(match.group(1))
        
        if 'Wide Binaries' in line and '%' in line and ('IMPROVED' in line or 'WORSENED' in line or 'UNCHANGED' in line):
            match = re.search(r'([+-]?\d+\.?\d*)%', line)
            if match:
                results['wb_improv'] = float(match.group(1))
        
        if 'DF2' in line and '%' in line and ('IMPROVED' in line or 'WORSENED' in line or 'UNCHANGED' in line):
            match = re.search(r'([+-]?\d+\.?\d*)%', line)
            if match:
                results['df2_improv'] = float(match.group(1))
    
    return results


def run_with_params(d_asym, d_tidal, d_wb):
    """Run regression with given parameters."""
    cmd = f'python scripts/run_regression_extended.py --extended-phi --d-asymmetry={d_asym} --d-tidal={d_tidal} --d-wb={d_wb}'
    try:
        output = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
        return parse_output(output)
    except Exception as e:
        return {'error': str(e)}


def main():
    print("=" * 100)
    print("Sigma-GRAVITY PARAMETER TUNING - Extended Phase Coherence Model")
    print("=" * 100)
    print()
    
    # Focused parameter grid - reduced for faster iteration
    d_asym_values = [0.8, 1.0, 1.5, 2.0]
    d_tidal_values = [3.0, 5.0]
    d_wb_values = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    results = []
    
    print("Parameter sweep:")
    print(f"  D_ASYMMETRY_SCALE: {d_asym_values}")
    print(f"  D_TIDAL_THRESHOLD: {d_tidal_values}")
    print(f"  D_WIDE_BINARIES: {d_wb_values}")
    print()
    
    total = len(d_asym_values) * len(d_tidal_values) * len(d_wb_values)
    print(f"Running {total} configurations...")
    print()
    
    print(f"{'D_asym':>8} {'D_tidal':>8} {'D_wb':>8} | {'SPARC':>8} {'Imp%':>8} | {'WB':>8} {'Imp%':>8} | {'DF2':>8} {'Imp%':>8}")
    print("-" * 100)
    
    for d_asym in d_asym_values:
        for d_tidal in d_tidal_values:
            for d_wb in d_wb_values:
                r = run_with_params(d_asym, d_tidal, d_wb)
                r['d_asym'] = d_asym
                r['d_tidal'] = d_tidal
                r['d_wb'] = d_wb
                results.append(r)
                
                sparc = r.get('sparc_new', 99)
                sparc_imp = r.get('sparc_improv', 0)
                wb = r.get('wb_new', 99)
                wb_imp = r.get('wb_improv', 0)
                df2 = r.get('df2_new', 99)
                df2_imp = r.get('df2_improv', 0)
                
                print(f"{d_asym:>8.1f} {d_tidal:>8.1f} {d_wb:>8.2f} | {sparc:>8.2f} {sparc_imp:>+8.1f}% | {wb:>8.2f} {wb_imp:>+8.1f}% | {df2:>8.2f} {df2_imp:>+8.1f}%")
    
    print()
    print("=" * 100)
    print("OPTIMAL PARAMETERS")
    print("=" * 100)
    
    # Filter valid results
    valid = [r for r in results if 'sparc_new' in r and r.get('sparc_new', 99) < 50]
    
    if not valid:
        print("ERROR: No valid results!")
        return None
    
    # Best for SPARC
    best_sparc = min(valid, key=lambda x: abs(x['sparc_new'] - 17.15))
    print(f"\n1. BEST FOR SPARC (target: 17.15 km/s):")
    print(f"   D_ASYMMETRY_SCALE = {best_sparc['d_asym']}")
    print(f"   D_WIDE_BINARIES = {best_sparc['d_wb']}")
    print(f"   SPARC = {best_sparc['sparc_new']:.2f} km/s ({best_sparc.get('sparc_improv', 0):+.1f}%)")
    
    # Best for Wide Binaries (closest to 1.35)
    valid_wb = [r for r in valid if 'wb_new' in r and r['wb_new'] < 50]
    if valid_wb:
        best_wb = min(valid_wb, key=lambda x: abs(x.get('wb_new', 99) - 1.35))
        print(f"\n2. BEST FOR WIDE BINARIES (target: 1.35x):")
        print(f"   D_WIDE_BINARIES = {best_wb['d_wb']}")
        print(f"   Wide Binaries = {best_wb.get('wb_new', 0):.2f}x ({best_wb.get('wb_improv', 0):+.1f}%)")
    
    # Best balanced
    def score(r):
        sparc_err = abs(r.get('sparc_new', 99) - 17.15)
        wb_err = abs(r.get('wb_new', 99) - 1.35) if r.get('wb_new', 99) < 50 else 10
        return sparc_err + wb_err * 5
    
    best = min(valid, key=score)
    print(f"\n" + "=" * 100)
    print(f"RECOMMENDED SETTINGS:")
    print(f"=" * 100)
    print(f"   D_ASYMMETRY_SCALE = {best['d_asym']}")
    print(f"   D_TIDAL_THRESHOLD = {best['d_tidal']}")
    print(f"   D_WIDE_BINARIES = {best['d_wb']}")
    print(f"\n   Results:")
    print(f"   - SPARC = {best['sparc_new']:.2f} km/s ({best.get('sparc_improv', 0):+.1f}%)")
    print(f"   - WB = {best.get('wb_new', 0):.2f}x ({best.get('wb_improv', 0):+.1f}%)")
    print(f"   - DF2 = {best.get('df2_new', 0):.2f} km/s ({best.get('df2_improv', 0):+.1f}%)")
    
    return best


if __name__ == "__main__":
    best = main()
