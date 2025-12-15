#!/usr/bin/env python3
"""Sweep flow coherence parameters (FLOW_ALPHA, FLOW_BETA, FLOW_GAMMA) to find optimal values.

Based on residual discovery findings that vorticity is the #1 driver in Gaia residuals.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class FlowResult:
    alpha: float
    beta: float
    gamma: float
    sparc_rms: float
    sparc_bulge_rms: Optional[float] = None
    sparc_disk_rms: Optional[float] = None
    clusters_ratio: Optional[float] = None
    solar_gamma: Optional[float] = None


def parse_regression_output(output: str) -> FlowResult:
    """Parse output from run_regression_experimental.py."""
    lines = output.split('\n')
    
    sparc_rms = None
    sparc_bulge_rms = None
    sparc_disk_rms = None
    clusters_ratio = None
    solar_gamma = None
    
    for line in lines:
        if 'SPARC Galaxies: RMS=' in line:
            # Format: [✓] SPARC Galaxies: RMS=17.42 km/s (MOND=17.15, ΛCDM~15), Scatter=0.100 dex, Win=42.7% | Bulge: 28.93, Disk: 16.06
            import re
            rms_match = re.search(r'RMS=([\d.]+)', line)
            if rms_match:
                sparc_rms = float(rms_match.group(1))
            
            bulge_match = re.search(r'Bulge:\s*([\d.]+)', line)
            if bulge_match:
                sparc_bulge_rms = float(bulge_match.group(1))
            
            disk_match = re.search(r'Disk:\s*([\d.]+)', line)
            if disk_match:
                sparc_disk_rms = float(disk_match.group(1))
        
        elif 'Clusters: Median ratio=' in line:
            # Format: [✓] Clusters: Median ratio=0.987 (MOND~0.33, ΛCDM~1.0), Scatter=0.132 dex (42 clusters)
            import re
            ratio_match = re.search(r'ratio=([\d.]+)', line)
            if ratio_match:
                clusters_ratio = float(ratio_match.group(1))
        
        elif 'Solar System' in line and '|γ-1|' in line:
            # Format: [✓] Solar System: |γ-1| = 0.000123
            import re
            gamma_match = re.search(r'\|γ-1\|\s*=\s*([\d.e-]+)', line)
            if gamma_match:
                solar_gamma = float(gamma_match.group(1))
    
    return FlowResult(
        alpha=0.0,  # Will be set by caller
        beta=0.0,
        gamma=0.0,
        sparc_rms=sparc_rms or float('nan'),
        sparc_bulge_rms=sparc_bulge_rms,
        sparc_disk_rms=sparc_disk_rms,
        clusters_ratio=clusters_ratio,
        solar_gamma=solar_gamma,
    )


def run_one(alpha: float, beta: float, gamma: float) -> FlowResult:
    """Run regression with given flow parameters."""
    cmd = [
        sys.executable,
        'scripts/run_regression_experimental.py',
        '--core',
        '--coherence=flow',
        f'--flow-alpha={alpha:g}',
        f'--flow-beta={beta:g}',
        f'--flow-gamma={gamma:g}',
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(__file__).parent.parent,
        )
        
        if result.returncode != 0:
            print(f"ERROR: Command failed with return code {result.returncode}")
            print(result.stderr)
            return FlowResult(alpha=alpha, beta=beta, gamma=gamma, sparc_rms=float('nan'))
        
        parsed = parse_regression_output(result.stdout)
        parsed.alpha = alpha
        parsed.beta = beta
        parsed.gamma = gamma
        return parsed
    
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {alpha:g}, {beta:g}, {gamma:g}")
        return FlowResult(alpha=alpha, beta=beta, gamma=gamma, sparc_rms=float('nan'))
    except Exception as e:
        print(f"ERROR: {alpha:g}, {beta:g}, {gamma:g}: {e}")
        return FlowResult(alpha=alpha, beta=beta, gamma=gamma, sparc_rms=float('nan'))


def to_markdown(results: List[FlowResult]) -> str:
    """Format results as markdown table."""
    lines = [
        "# Flow Coherence Parameter Sweep Results",
        "",
        "| α | β | γ | SPARC RMS | Bulge RMS | Disk RMS | Clusters Ratio | Solar |γ-1| |",
        "|---:|---:|---:|----------:|----------:|---------:|---------------:|----------------:|",
    ]
    
    for r in sorted(results, key=lambda x: (x.sparc_rms if not np.isnan(x.sparc_rms) else 999, x.sparc_bulge_rms or 999)):
        bulge_str = f"{r.sparc_bulge_rms:.3f}" if r.sparc_bulge_rms is not None else "N/A"
        disk_str = f"{r.sparc_disk_rms:.3f}" if r.sparc_disk_rms is not None else "N/A"
        clusters_str = f"{r.clusters_ratio:.3f}" if r.clusters_ratio is not None else "N/A"
        solar_str = f"{r.solar_gamma:.2e}" if r.solar_gamma is not None else "N/A"
        rms_str = f"{r.sparc_rms:.3f}" if not np.isnan(r.sparc_rms) else "N/A"
        
        lines.append(
            f"| {r.alpha:g} | {r.beta:g} | {r.gamma:g} | {rms_str} | {bulge_str} | {disk_str} | {clusters_str} | {solar_str} |"
        )
    
    return "\n".join(lines)


def main():
    """Run parameter sweep."""
    # Grid search: focus on alpha (shear weight) first, then beta/gamma
    # Based on discovery: vorticity is #1, shear is #4, so alpha should be significant
    
    results: List[FlowResult] = []
    
    # Baseline: current defaults
    print("Running baseline (α=1.0, β=0.1, γ=0.5)...")
    baseline = run_one(1.0, 0.1, 0.5)
    results.append(baseline)
    bulge_str = f"{baseline.sparc_bulge_rms:.3f}" if baseline.sparc_bulge_rms is not None else "N/A"
    print(f"  SPARC RMS: {baseline.sparc_rms:.3f}, Bulge RMS: {bulge_str}")
    
    # Sweep alpha (shear weight) - most important based on discovery
    print("\nSweeping alpha (shear weight)...")
    for alpha in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        print(f"  α={alpha:g}...")
        r = run_one(alpha, 0.1, 0.5)
        results.append(r)
        bulge_str = f"{r.sparc_bulge_rms:.3f}" if r.sparc_bulge_rms is not None else "N/A"
        print(f"    SPARC RMS: {r.sparc_rms:.3f}, Bulge RMS: {bulge_str}")
    
    # Sweep beta (divergence weight) - typically small for rotation curves
    print("\nSweeping beta (divergence weight)...")
    for beta in [0.0, 0.05, 0.1, 0.2, 0.5]:
        print(f"  β={beta:g}...")
        r = run_one(1.0, beta, 0.5)
        results.append(r)
        bulge_str = f"{r.sparc_bulge_rms:.3f}" if r.sparc_bulge_rms is not None else "N/A"
        print(f"    SPARC RMS: {r.sparc_rms:.3f}, Bulge RMS: {bulge_str}")
    
    # Sweep gamma (tidal weight) - may be important for bulge suppression
    print("\nSweeping gamma (tidal weight)...")
    for gamma in [0.0, 0.1, 0.5, 1.0, 2.0]:
        print(f"  γ={gamma:g}...")
        r = run_one(1.0, 0.1, gamma)
        results.append(r)
        bulge_str = f"{r.sparc_bulge_rms:.3f}" if r.sparc_bulge_rms is not None else "N/A"
        print(f"    SPARC RMS: {r.sparc_rms:.3f}, Bulge RMS: {bulge_str}")
    
    # Find best by SPARC RMS
    valid_results = [r for r in results if not np.isnan(r.sparc_rms)]
    if valid_results:
        best = min(valid_results, key=lambda x: x.sparc_rms)
        print(f"\n{'='*80}")
        print(f"Best by SPARC RMS:")
        print(f"  α={best.alpha:g}, β={best.beta:g}, γ={best.gamma:g}")
        print(f"  SPARC RMS: {best.sparc_rms:.3f}")
        bulge_str = f"{best.sparc_bulge_rms:.3f}" if best.sparc_bulge_rms is not None else "N/A"
        print(f"  Bulge RMS: {bulge_str}")
        disk_str = f"{best.sparc_disk_rms:.3f}" if best.sparc_disk_rms is not None else "N/A"
        print(f"  Disk RMS: {disk_str}")
        clusters_str = f"{best.clusters_ratio:.3f}" if best.clusters_ratio is not None else "N/A"
        print(f"  Clusters Ratio: {clusters_str}")
        
        # Find best by bulge RMS
        bulge_results = [r for r in valid_results if r.sparc_bulge_rms is not None]
        if bulge_results:
            best_bulge = min(bulge_results, key=lambda x: x.sparc_bulge_rms or 999)
            print(f"\nBest by Bulge RMS:")
            print(f"  α={best_bulge.alpha:g}, β={best_bulge.beta:g}, γ={best_bulge.gamma:g}")
            print(f"  SPARC RMS: {best_bulge.sparc_rms:.3f}")
            print(f"  Bulge RMS: {best_bulge.sparc_bulge_rms:.3f}")
            disk_str = f"{best_bulge.sparc_disk_rms:.3f}" if best_bulge.sparc_disk_rms is not None else "N/A"
            print(f"  Disk RMS: {disk_str}")
    
    # Save results
    outdir = Path(__file__).parent / 'regression_results'
    outdir.mkdir(exist_ok=True)
    
    md_path = outdir / 'FLOW_COHERENCE_SWEEP.md'
    md_path.write_text(to_markdown(results))
    print(f"\nSaved results to: {md_path}")
    
    # Also save JSON for programmatic access
    import json
    json_path = outdir / 'FLOW_COHERENCE_SWEEP.json'
    json_path.write_text(json.dumps([
        {
            'alpha': r.alpha,
            'beta': r.beta,
            'gamma': r.gamma,
            'sparc_rms': r.sparc_rms,
            'sparc_bulge_rms': r.sparc_bulge_rms,
            'sparc_disk_rms': r.sparc_disk_rms,
            'clusters_ratio': r.clusters_ratio,
            'solar_gamma': r.solar_gamma,
        }
        for r in results
    ], indent=2))
    print(f"Saved JSON to: {json_path}")


if __name__ == '__main__':
    main()

