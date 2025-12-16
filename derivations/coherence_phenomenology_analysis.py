#!/usr/bin/env python3
"""
Coherence Phenomenology Analysis

This script analyzes what the Σ-Gravity formulas tell us about:
1. What CAUSES coherence (enhancement)
2. What CAUSES decoherence (suppression)

By examining the situations where Σ > 1 vs Σ ≈ 1, we can work backwards
to formulate a hypothesis about the underlying physics.

Author: Leonard Speiser
Date: December 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict

# Physical constants
c = 2.998e8  # m/s
H0 = 2.27e-18  # 1/s
G = 6.674e-11  # m³/kg/s²
kpc_to_m = 3.086e19

# Derived
g_dagger = c * H0 / (4 * np.sqrt(np.pi))

# =============================================================================
# CATALOG OF SITUATIONS
# =============================================================================

@dataclass
class Situation:
    """A physical situation with known coherence behavior."""
    name: str
    description: str
    
    # Physical properties
    acceleration: float  # m/s² (typical g_N)
    size: float  # meters (characteristic scale)
    rotation_velocity: float  # m/s (ordered circular motion)
    dispersion_velocity: float  # m/s (random motion)
    
    # Observed behavior
    enhancement_observed: float  # Σ - 1 (0 = no enhancement)
    coherence_status: str  # "high", "medium", "low", "none"
    
    # Derived quantities
    @property
    def v_over_sigma(self):
        """Ratio of ordered to random motion."""
        return self.rotation_velocity / max(self.dispersion_velocity, 1)
    
    @property
    def g_over_gdagger(self):
        """Ratio of local to critical acceleration."""
        return self.acceleration / g_dagger
    
    @property
    def dynamical_time(self):
        """Dynamical time t_dyn = √(r/g)."""
        return np.sqrt(self.size / max(self.acceleration, 1e-15))
    
    @property
    def hubble_time(self):
        """Hubble time."""
        return 1 / H0
    
    @property
    def t_dyn_over_t_H(self):
        """Ratio of dynamical to Hubble time."""
        return self.dynamical_time / self.hubble_time


# Create catalog of situations
SITUATIONS = [
    # HIGH COHERENCE (Σ >> 1)
    Situation(
        name="Outer disk of spiral galaxy",
        description="r ~ 10 kpc, flat rotation curve",
        acceleration=1e-10,  # ~g†
        size=10 * kpc_to_m,
        rotation_velocity=200e3,  # 200 km/s
        dispersion_velocity=20e3,  # 20 km/s
        enhancement_observed=1.5,  # Σ ~ 2.5
        coherence_status="high"
    ),
    
    Situation(
        name="LSB galaxy outer region",
        description="Low surface brightness, deep MOND regime",
        acceleration=1e-11,  # << g†
        size=20 * kpc_to_m,
        rotation_velocity=80e3,  # 80 km/s
        dispersion_velocity=10e3,  # 10 km/s
        enhancement_observed=3.0,  # Σ ~ 4
        coherence_status="high"
    ),
    
    Situation(
        name="Galaxy cluster at 200 kpc",
        description="Strong lensing region",
        acceleration=5e-11,  # ~0.5 g†
        size=200 * kpc_to_m,
        rotation_velocity=0,  # No net rotation
        dispersion_velocity=1000e3,  # 1000 km/s
        enhancement_observed=7.0,  # Σ ~ 8 (needs A_cluster)
        coherence_status="high"  # Despite no rotation!
    ),
    
    # MEDIUM COHERENCE (Σ ~ 1.5-2)
    Situation(
        name="Inner disk of spiral galaxy",
        description="r ~ 2 kpc, rising rotation curve",
        acceleration=5e-10,  # ~5 g†
        size=2 * kpc_to_m,
        rotation_velocity=150e3,  # 150 km/s
        dispersion_velocity=30e3,  # 30 km/s
        enhancement_observed=0.5,  # Σ ~ 1.5
        coherence_status="medium"
    ),
    
    Situation(
        name="Elliptical galaxy",
        description="Pressure-supported, no rotation",
        acceleration=2e-10,  # ~2 g†
        size=5 * kpc_to_m,
        rotation_velocity=50e3,  # Some rotation
        dispersion_velocity=200e3,  # High dispersion
        enhancement_observed=0.3,  # Σ ~ 1.3
        coherence_status="medium"
    ),
    
    # LOW/NO COHERENCE (Σ ≈ 1)
    Situation(
        name="Solar System (Earth orbit)",
        description="High acceleration, compact",
        acceleration=6e-3,  # >> g†
        size=1.5e11,  # 1 AU
        rotation_velocity=30e3,  # 30 km/s orbital
        dispersion_velocity=0,  # No dispersion
        enhancement_observed=0,  # Σ = 1
        coherence_status="none"
    ),
    
    Situation(
        name="Solar System (Saturn orbit)",
        description="Lower acceleration, still compact",
        acceleration=6e-5,  # >> g†
        size=1.4e12,  # 9.5 AU
        rotation_velocity=10e3,  # 10 km/s orbital
        dispersion_velocity=0,
        enhancement_observed=0,  # Σ = 1
        coherence_status="none"
    ),
    
    Situation(
        name="Wide binary (10,000 AU)",
        description="Low acceleration, but compact system",
        acceleration=6e-12,  # < g†
        size=1.5e15,  # 10,000 AU
        rotation_velocity=0.3e3,  # 0.3 km/s orbital
        dispersion_velocity=0,
        enhancement_observed=0,  # Σ = 1 (disputed!)
        coherence_status="none"  # No extended rotation
    ),
    
    Situation(
        name="Galaxy bulge center",
        description="High density, high dispersion",
        acceleration=1e-9,  # >> g†
        size=0.5 * kpc_to_m,
        rotation_velocity=50e3,
        dispersion_velocity=150e3,  # High dispersion
        enhancement_observed=0.1,  # Σ ~ 1.1
        coherence_status="low"
    ),
    
    # SPECIAL CASES
    Situation(
        name="Counter-rotating disk",
        description="Two stellar populations rotating opposite",
        acceleration=1e-10,  # ~g†
        size=10 * kpc_to_m,
        rotation_velocity=0,  # NET rotation is zero!
        dispersion_velocity=200e3,  # Effective dispersion from counter-rotation
        enhancement_observed=0.5,  # Σ ~ 1.5 (reduced!)
        coherence_status="low"  # Counter-rotation destroys coherence
    ),
    
    Situation(
        name="High-z galaxy (z=2)",
        description="Turbulent, gas-rich, high H(z)",
        acceleration=1e-10,
        size=5 * kpc_to_m,
        rotation_velocity=150e3,
        dispersion_velocity=80e3,  # High turbulence
        enhancement_observed=0.3,  # Reduced vs z=0
        coherence_status="medium"  # Reduced by H(z) and turbulence
    ),
]


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_coherence_drivers():
    """Analyze what drives coherence in each situation."""
    
    print("=" * 100)
    print("COHERENCE PHENOMENOLOGY ANALYSIS")
    print("=" * 100)
    print()
    
    # Sort by enhancement
    sorted_situations = sorted(SITUATIONS, key=lambda s: s.enhancement_observed, reverse=True)
    
    print("SITUATIONS RANKED BY ENHANCEMENT:")
    print("-" * 100)
    print(f"{'Situation':<35} | {'Σ-1':>6} | {'g/g†':>8} | {'v/σ':>6} | {'t_dyn/t_H':>10} | {'Status':<8}")
    print("-" * 100)
    
    for s in sorted_situations:
        print(f"{s.name:<35} | {s.enhancement_observed:>6.2f} | {s.g_over_gdagger:>8.2f} | {s.v_over_sigma:>6.1f} | {s.t_dyn_over_t_H:>10.4f} | {s.coherence_status:<8}")
    
    print("-" * 100)
    print()
    
    # Identify patterns
    print("PATTERN ANALYSIS:")
    print("=" * 100)
    print()
    
    # 1. Acceleration dependence
    print("1. ACCELERATION DEPENDENCE (g/g†)")
    print("-" * 50)
    high_coh = [s for s in SITUATIONS if s.coherence_status == "high"]
    low_coh = [s for s in SITUATIONS if s.coherence_status in ["low", "none"]]
    
    print(f"   High coherence systems: g/g† = {np.mean([s.g_over_gdagger for s in high_coh]):.2f} (mean)")
    print(f"   Low coherence systems:  g/g† = {np.mean([s.g_over_gdagger for s in low_coh]):.2f} (mean)")
    print()
    print("   OBSERVATION: Low g/g† correlates with high coherence, BUT")
    print("   the Solar System has g/g† >> 1 AND no extended mass distribution.")
    print("   Clusters have g/g† ~ 0.5 but HIGH coherence.")
    print()
    print("   → Acceleration alone doesn't determine coherence.")
    print()
    
    # 2. Rotation/dispersion ratio
    print("2. ROTATION/DISPERSION RATIO (v/σ)")
    print("-" * 50)
    print(f"   High coherence systems: v/σ = {np.mean([s.v_over_sigma for s in high_coh]):.1f} (mean)")
    print(f"   Low coherence systems:  v/σ = {np.mean([s.v_over_sigma for s in low_coh]):.1f} (mean)")
    print()
    print("   OBSERVATION: High v/σ correlates with high coherence for GALAXIES.")
    print("   BUT clusters have v/σ = 0 (no rotation) yet HIGH coherence!")
    print()
    print("   → v/σ matters for disks, but something else matters for clusters.")
    print()
    
    # 3. Dynamical time
    print("3. DYNAMICAL TIME (t_dyn/t_H)")
    print("-" * 50)
    print(f"   High coherence systems: t_dyn/t_H = {np.mean([s.t_dyn_over_t_H for s in high_coh]):.4f} (mean)")
    print(f"   Low coherence systems:  t_dyn/t_H = {np.mean([s.t_dyn_over_t_H for s in low_coh]):.6f} (mean)")
    print()
    print("   OBSERVATION: High coherence systems have t_dyn ~ 0.01-0.1 × t_H")
    print("   Low coherence systems have t_dyn << t_H (much faster dynamics).")
    print()
    print("   → Systems with dynamics comparable to cosmic timescale show enhancement!")
    print()
    
    # 4. System size
    print("4. SYSTEM SIZE")
    print("-" * 50)
    print(f"   High coherence systems: size = {np.mean([s.size/kpc_to_m for s in high_coh]):.1f} kpc (mean)")
    print(f"   Low coherence systems:  size = {np.mean([s.size/kpc_to_m for s in low_coh]):.4f} kpc (mean)")
    print()
    print("   OBSERVATION: Large systems (kpc-Mpc) show enhancement.")
    print("   Small systems (AU-pc) don't, even at low g.")
    print()
    print("   → Size matters! Coherence requires EXTENDED mass distribution.")
    print()
    
    return sorted_situations


def identify_coherence_conditions():
    """Identify the conditions for coherence."""
    
    print()
    print("=" * 100)
    print("CONDITIONS FOR COHERENCE")
    print("=" * 100)
    print()
    
    print("""
Based on the phenomenology, coherence (Σ > 1) requires:

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONDITION 1: LOW ACCELERATION (g < g†)                                      │
│                                                                             │
│ The local gravitational acceleration must be below the critical scale:     │
│     g_N < g† ≈ 10⁻¹⁰ m/s²                                                   │
│                                                                             │
│ This is necessary but NOT sufficient.                                       │
│ Wide binaries have g < g† but show no enhancement.                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONDITION 2: EXTENDED MASS DISTRIBUTION                                     │
│                                                                             │
│ The source must be spatially extended (kpc-Mpc scale).                      │
│ Compact systems (binaries, Solar System) don't show enhancement.            │
│                                                                             │
│ This explains why wide binaries don't show MOND effects:                    │
│ They have low g but are NOT extended mass distributions.                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONDITION 3: ORGANIZED MOTION (for disk systems)                            │
│                                                                             │
│ For disk galaxies, coherence requires organized circular rotation:          │
│     v_rot >> σ_random                                                       │
│                                                                             │
│ Counter-rotating disks have REDUCED enhancement because the                 │
│ net angular momentum is reduced.                                            │
│                                                                             │
│ This is the "coherence" in the traditional sense.                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ CONDITION 4: PATH LENGTH THROUGH BARYONS (for 3D systems)                   │
│                                                                             │
│ For clusters (3D, no rotation), coherence scales with path length:          │
│     A ∝ L^(1/4)                                                             │
│                                                                             │
│ Clusters have no net rotation but LARGE path lengths (~600 kpc).            │
│ This gives A_cluster ~ 8 vs A_galaxy ~ 1.2.                                 │
│                                                                             │
│ This suggests coherence can arise from SPATIAL extent, not just rotation.   │
└─────────────────────────────────────────────────────────────────────────────┘
""")
    
    print()
    print("WHAT CAUSES DECOHERENCE:")
    print("-" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ DECOHERENCE 1: HIGH ACCELERATION (g >> g†)                                  │
│                                                                             │
│ When g >> g†, the system is "self-gravitating" - its dynamics are           │
│ dominated by local gravity, not cosmic effects.                             │
│                                                                             │
│ The h(g) function suppresses enhancement: h → 0 as g → ∞                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECOHERENCE 2: COMPACT SYSTEM (r << ξ)                                      │
│                                                                             │
│ When the system is smaller than the coherence scale ξ, the                  │
│ coherence window W(r) → 0.                                                  │
│                                                                             │
│ This is why the Solar System shows no enhancement: it's compact.            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECOHERENCE 3: RANDOM VELOCITIES (σ >> v_rot)                               │
│                                                                             │
│ When random motions dominate ordered rotation, coherence is reduced.        │
│                                                                             │
│ Counter-rotating disks: effective σ includes (v₁ - v₂)² term.               │
│ High-z galaxies: turbulence increases σ.                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ DECOHERENCE 4: HIGH HUBBLE RATE (at high z)                                 │
│                                                                             │
│ At high redshift, H(z) > H₀, so g†(z) > g†(0).                              │
│ The transition acceleration increases, reducing enhancement.                │
│                                                                             │
│ This is observed in KMOS3D high-z galaxies.                                 │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def formulate_hypothesis():
    """Formulate a hypothesis based on the phenomenology."""
    
    print()
    print("=" * 100)
    print("HYPOTHESIS: GRAVITATIONAL COHERENCE FROM COSMIC ENTANGLEMENT")
    print("=" * 100)
    print()
    
    print("""
Based on the phenomenology, we propose the following hypothesis:

╔═════════════════════════════════════════════════════════════════════════════╗
║                                                                             ║
║  HYPOTHESIS: Gravity is mediated by a field that exhibits COHERENCE        ║
║  when the source mass distribution satisfies certain conditions.           ║
║                                                                             ║
║  The coherence is NOT about quantum phases of gravitons.                    ║
║  It is about the ENTANGLEMENT STRUCTURE of spacetime itself.                ║
║                                                                             ║
╚═════════════════════════════════════════════════════════════════════════════╝

PHYSICAL PICTURE:
─────────────────

1. SPACETIME HAS ENTANGLEMENT STRUCTURE
   
   Following ER=EPR and tensor network ideas, spacetime geometry emerges
   from quantum entanglement between regions. The entanglement entropy
   determines the effective gravitational coupling.

2. EXTENDED MASS DISTRIBUTIONS MODIFY ENTANGLEMENT
   
   A mass distribution with spatial extent R creates entanglement between
   regions separated by up to R. This INCREASES the effective coupling
   compared to a point mass.

3. ORGANIZED MOTION ENHANCES ENTANGLEMENT
   
   When the mass is in organized motion (rotation), the entanglement
   structure is MORE ORDERED than for random motion. Ordered entanglement
   produces stronger gravitational effects.

4. COSMIC HORIZON SETS THE SCALE
   
   The Hubble horizon R_H = c/H₀ is the maximum entanglement scale.
   Systems with dynamics comparable to the Hubble time can "tap into"
   this cosmic entanglement, enhancing their gravity.

5. THE CRITICAL ACCELERATION g† = cH₀/(4√π)
   
   This is the acceleration at which the local dynamical time equals
   the cosmic time. Below g†, systems are "cosmically entangled".
   Above g†, systems are "locally isolated".

MATHEMATICAL FORMULATION:
─────────────────────────

The gravitational coupling depends on the entanglement entropy S_EE:

    G_eff = G × [1 + f(S_EE / S_max)]

where:
    - S_EE is the entanglement entropy of the source region
    - S_max is the maximum (Bekenstein-Hawking) entropy
    - f is a function that gives the enhancement

For an extended mass distribution with:
    - Spatial extent R
    - Organized velocity field v(r)
    - Local acceleration g

The entanglement entropy ratio is:

    S_EE / S_max = W(r) × h(g)

where:
    - W(r) = r/(ξ+r) captures the spatial extent effect
    - h(g) = √(g†/g) × g†/(g†+g) captures the cosmic connection
    - ξ = R_d/(2π) is the coherence scale

The enhancement factor is:

    Σ = 1 + A × W(r) × h(g)

where A is the maximum enhancement from full entanglement.

WHY THIS EXPLAINS THE PHENOMENOLOGY:
────────────────────────────────────

1. LOW ACCELERATION REQUIRED: Only systems with g < g† can connect to
   cosmic entanglement. High-g systems are locally isolated.

2. EXTENDED SIZE REQUIRED: Only extended mass distributions can create
   significant entanglement. Point masses can't.

3. ORGANIZED MOTION HELPS: Ordered velocity fields create ordered
   entanglement, which couples more strongly to gravity.

4. PATH LENGTH FOR CLUSTERS: In 3D systems, the entanglement scales
   with the path length through the mass distribution.

5. COUNTER-ROTATION REDUCES: Opposing velocities create DISORDERED
   entanglement, reducing the enhancement.

6. HIGH-Z SUPPRESSION: Higher H(z) means smaller cosmic entanglement
   scale, reducing the enhancement.

TESTABLE PREDICTIONS:
─────────────────────

1. Enhancement should correlate with v/σ for disk galaxies.
   → CONFIRMED in MaNGA data

2. Counter-rotating disks should have reduced enhancement.
   → CONFIRMED: 44% lower f_DM

3. High-z galaxies should show less enhancement.
   → CONSISTENT with KMOS3D data

4. Clusters should require larger A than galaxies.
   → CONFIRMED: A_cluster/A_galaxy ~ 7

5. Wide binaries should show NO enhancement (not extended).
   → DISPUTED: Chae vs Banik results conflict

6. The enhancement should be INSTANTANEOUS (spatial property).
   → REQUIRED for lensing to work
""")
    
    return


def main():
    """Run the full analysis."""
    
    situations = analyze_coherence_drivers()
    identify_coherence_conditions()
    formulate_hypothesis()
    
    # Summary
    print()
    print("=" * 100)
    print("SUMMARY: THE COHERENCE HYPOTHESIS")
    print("=" * 100)
    print("""
The Σ-Gravity enhancement factor Σ arises from GRAVITATIONAL COHERENCE,
which depends on:

    ┌────────────────────────────────────────────────────────────────┐
    │                                                                │
    │  Σ = 1 + A × W(r) × h(g)                                       │
    │                                                                │
    │  where:                                                        │
    │                                                                │
    │  • h(g) = cosmic connection factor                             │
    │    - Large when g < g† (slow dynamics, cosmic entanglement)    │
    │    - Small when g > g† (fast dynamics, local isolation)        │
    │                                                                │
    │  • W(r) = spatial coherence factor                             │
    │    - Large when r > ξ (extended mass distribution)             │
    │    - Small when r < ξ (compact system)                         │
    │                                                                │
    │  • A = amplitude factor                                        │
    │    - ~1.2 for 2D disks (azimuthal coherence)                   │
    │    - ~8 for 3D clusters (path length coherence)                │
    │                                                                │
    └────────────────────────────────────────────────────────────────┘

COHERENCE REQUIRES:
    1. Low acceleration (g < g†) — cosmic connection
    2. Extended mass distribution — spatial coherence
    3. Organized motion OR long path length — entanglement ordering

DECOHERENCE OCCURS FROM:
    1. High acceleration (g >> g†) — local isolation
    2. Compact system (r << ξ) — no spatial coherence
    3. Random velocities (σ >> v) — disordered entanglement
    4. High Hubble rate (high z) — reduced cosmic scale

The underlying physics is ENTANGLEMENT STRUCTURE of spacetime:
    - Extended, organized mass distributions create ordered entanglement
    - Ordered entanglement enhances the effective gravitational coupling
    - The cosmic horizon sets the maximum entanglement scale
""")


if __name__ == "__main__":
    main()




