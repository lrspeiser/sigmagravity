#!/usr/bin/env python3
"""
Analyze v/σ ratio across different astronomical systems.

Shows why dispersion should be used for:
- Bulges
- Dwarf spheroidals
- Ellipticals/UDGs
- Clusters

And rotation should be used for:
- Disk galaxies
- Spiral arms
"""

import numpy as np
from pathlib import Path

# System classifications by v/σ
SYSTEMS = {
    # Rotation-dominated (v/σ > 2)
    'rotation_dominated': {
        'Milky Way thin disk': {'v': 220, 'sigma': 30, 'source': 'Bland-Hawthorn+ 2016'},
        'SPARC disk (typical)': {'v': 150, 'sigma': 20, 'source': 'SPARC'},
        'M31 disk': {'v': 250, 'sigma': 35, 'source': 'Chemin+ 2009'},
    },
    
    # Intermediate (1 < v/σ < 2)
    'intermediate': {
        'Milky Way thick disk': {'v': 180, 'sigma': 60, 'source': 'Bland-Hawthorn+ 2016'},
        'S0 galaxy (typical)': {'v': 150, 'sigma': 100, 'source': 'Emsellem+ 2011'},
        'Barred galaxy center': {'v': 80, 'sigma': 100, 'source': 'Various'},
    },
    
    # Dispersion-dominated (v/σ < 1)
    'dispersion_dominated': {
        'Milky Way bulge': {'v': 60, 'sigma': 110, 'source': 'Zoccali+ 2014'},
        'Elliptical (M87)': {'v': 50, 'sigma': 350, 'source': 'Murphy+ 2011'},
        'Fornax dSph': {'v': 5, 'sigma': 11, 'source': 'Walker+ 2009'},
        'Draco dSph': {'v': 3, 'sigma': 9, 'source': 'Walker+ 2009'},
        'DF2 (UDG)': {'v': 4, 'sigma': 8.5, 'source': 'van Dokkum+ 2018'},
        'Coma Cluster': {'v': 0, 'sigma': 1000, 'source': 'Colless+ 1996'},
        'Globular cluster (typical)': {'v': 2, 'sigma': 8, 'source': 'Various'},
    }
}

def main():
    print("=" * 90)
    print("v/sigma ANALYSIS: When to Use Rotation vs Dispersion")
    print("=" * 90)
    print()
    
    print("Physical principle:")
    print("  - v/sigma >> 1: Use rotation velocity (cold, ordered orbits)")
    print("  - v/sigma ~ 1: Use both v and sigma (warm, intermediate)")
    print("  - v/sigma << 1: Use velocity dispersion (hot, random orbits)")
    print()
    
    for category, systems in SYSTEMS.items():
        title = category.replace('_', ' ').upper()
        print("-" * 90)
        print(f"{title}")
        print("-" * 90)
        print(f"{'System':<30} {'v (km/s)':>12} {'sigma (km/s)':>12} {'v/sigma':>10} {'Use':>15}")
        print("-" * 90)
        
        for name, data in systems.items():
            v = data['v']
            sigma = data['sigma']
            ratio = v / sigma
            
            if ratio > 2:
                use = "ROTATION"
            elif ratio > 1:
                use = "BOTH"
            else:
                use = "DISPERSION"
            
            print(f"{name:<30} {v:>12.0f} {sigma:>12.0f} {ratio:>10.2f} {use:>15}")
        print()
    
    print("=" * 90)
    print("IMPLICATIONS FOR Sigma-GRAVITY")
    print("=" * 90)
    print()
    print("1. SPARC ROTATION CURVES: Appropriate for disk-dominated galaxies (v/sigma > 2)")
    print("   - Current model works well")
    print("   - Extended phi improves high-asymmetry disks")
    print()
    print("2. BULGE-DOMINATED REGIONS: Need dispersion treatment (v/sigma < 1)")
    print("   - Don't use rotation curves for bulge points")
    print("   - Use sigma_tot from IFU data or Jeans modeling")
    print("   - We did this correctly for BRAVA")
    print()
    print("3. DWARF SPHEROIDALS: Already using dispersion (v/sigma ~ 0.3-0.5)")
    print("   - Current host-inheritance model uses sigma_obs")
    print("   - Correctly predicting sigma_pred / sigma_obs ratios")
    print()
    print("4. UDGs (DF2, Dragonfly 44): Dispersion-supported")
    print("   - Using sigma as observable is correct")
    print("   - The DF2 challenge is about Sigma value, not observable choice")
    print()
    print("5. GALAXY CLUSTERS: Pure dispersion (v/sigma = 0)")
    print("   - Already using mass ratios (M_pred / M_lens)")
    print("   - Dispersion of galaxies in cluster gives mass")
    print()
    print("CONCLUSION:")
    print("The current test suite is mostly using the RIGHT observables!")
    print("- SPARC: rotation for disks (correct)")
    print("- dSphs: dispersion (correct)")
    print("- UDGs: dispersion (correct)")
    print("- Clusters: mass from lensing (correct)")
    print()
    print("The ONE place we're using wrong observable: SPARC bulge points")
    print("These should be excluded from rotation curve fits or treated separately.")


if __name__ == "__main__":
    main()

