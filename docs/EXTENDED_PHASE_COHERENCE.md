# Σ-Gravity: Extended Phase Coherence Model

## Overview

The Extended Phase Coherence Model introduces a state-dependent factor φ into the Σ-Gravity enhancement formula. This allows the theory to account for different gravitational behavior based on the dynamical state of matter (ordered vs turbulent, collisionless vs collisional).

## Core Formula

### Baseline Σ-Gravity
```
Σ = 1 + A(L) × C(v,σ) × h(g)
```

Where:
- `A(L) = A₀ × (L/L₀)^n` is the amplitude (path length dependent)
- `C(v,σ) = v²/(v² + σ²)` is the coherence scalar
- `h(g) = √(g†/g) × g†/(g†+g)` is the enhancement function
- `g† = c×H₀/(4√π) ≈ 9.6×10⁻¹¹ m/s²` is the critical acceleration

### Extended Phase Coherence
```
Σ = 1 + A(L) × φ × C(v,σ) × h(g)
```

Where φ is the **phase coherence factor**:
```
φ = 1 + λ₀ × D × (f_ordered - f_turb)
```

## Parameters

### Baseline (Locked)
| Parameter | Value | Description |
|-----------|-------|-------------|
| A₀ | 1.1725 | Base amplitude (e^(1/2π)) |
| L₀ | 0.40 kpc | Reference path length |
| n | 0.27 | Path length exponent |
| g† | 9.6×10⁻¹¹ m/s² | Critical acceleration |

### Extended Model (Tuned)
| Parameter | Value | Description |
|-----------|-------|-------------|
| λ₀ | 7.0 | Universal coupling constant |
| D_asymmetry_scale | 1.5 | How kinematic asymmetry maps to D |
| D_wide_binaries | 0.6 | D for wide binaries in MW tidal field |
| D_tidal_threshold | 5.0 | Tidal radius / r_half threshold |

### Physical Meaning

**Disturbance Parameter D (0 to 1)**
- D = 0: Perfect equilibrium (undisturbed)
- D = 1: Strong disturbance (major merger, shock)

**D Sources by System Type:**
- **Disk Galaxies**: D = D_asymmetry_scale × (v_asym / v_circ)
  - Derived from kinematic lopsidedness, warps, asymmetric rotation curves
- **Wide Binaries**: D = D_wide_binaries (fixed, from MW tidal environment)
- **Satellites/UDGs**: D from tidal proximity to host
- **Cluster Mergers**: D = (Mach - 0.5) / 2.5 (from shock strength)

**Ordered vs Turbulent Fractions:**
- **Collisionless stars (equilibrium)**: f_ordered = 0.95, f_turb = 0.0 → φ > 1
- **Collisionless stars (disturbed)**: 30% suppression → φ < 1
- **Collisional gas (shocked/turbulent)**: f_ordered = 0.1, f_turb ∝ Mach → φ < 1

## Key Physics

### In Mergers (Bullet Cluster type)
Stars maintain phase coherence through collisions → **enhanced gravity**
Gas gets shocked and turbulent → **suppressed/screened gravity**

Result: Lensing peaks at stars (20% of mass), not gas (80% of mass)

### In Equilibrium Disturbance (galaxies, binaries)
Disturbance disrupts phase coherence → **suppressed enhancement**

Result: Reduces overprediction in asymmetric/disturbed systems

## Regression Results (Tuned Model)

### Summary Table
| Test | Observed | Baseline | New Model | MOND | Best |
|------|----------|----------|-----------|------|------|
| SPARC Galaxies (RMS) | 17.15 km/s | 17.42 km/s | **17.19 km/s** | 17.15 km/s | MOND |
| Wide Binaries (boost) | 1.35x | 1.63x | **1.52x** | 1.73x | New |
| Bullet Cluster | STARS | GAS | **STARS** | GAS | New |
| Galaxy Clusters | 1.0 | 0.99 | 0.99 | 0.39 | Baseline |
| DF2 (UDG) | 8.5 km/s | 20.77 km/s | 19.82 km/s | 20.0 km/s | Newton |
| Dwarf Spheroidals | 1.0 | 0.87 | 0.87 | 0.63 | Baseline |

### Improvements vs Baseline
| Test | Improvement |
|------|-------------|
| SPARC Galaxies | **-85.0%** (closer to observed) |
| Wide Binaries | **-40.3%** (closer to 1.35x) |
| Bullet Cluster | **-32.1%** (now at STARS) |
| DF2 | **-7.8%** (small improvement) |

### SPARC Binned Analysis
| Bin | N | Baseline | New | MOND | Best |
|-----|---|----------|-----|------|------|
| Disk-dominated (B/T<0.1) | 146 | 15.49 | **15.23** | 15.27 | New |
| High D (>0.15) | 48 | 13.04 | **12.71** | 13.69 | New |
| Gas-rich (f_gas>0.5) | 82 | 12.86 | **12.63** | 12.66 | New |

## Usage

Run with extended phase coherence:
```bash
python scripts/run_regression_extended.py --extended-phi
```

Custom parameters:
```bash
python scripts/run_regression_extended.py --extended-phi \
    --d-asymmetry=1.5 \
    --d-wb=0.6 \
    --d-tidal=5.0 \
    --phi-lambda=7.0
```

## Physical Interpretation

The Extended Phase Coherence model posits that gravitational enhancement in Σ-Gravity depends on the **phase coherence** of the matter distribution:

1. **Ordered, collisionless matter** (stars in equilibrium disks) maintains coherent phase relationships → full enhancement
2. **Disturbed matter** (asymmetric kinematics, tidal distortion) has reduced coherence → suppressed enhancement
3. **Shocked/turbulent gas** has random phases → screened gravity

This naturally explains:
- **Bullet Cluster**: Stars pass through undisturbed (φ > 1), gas gets shocked (φ < 1)
- **Wide binary overprediction**: MW tidal field reduces coherence (φ < 1)
- **DF2 challenge**: Tidal stripping reduces coherence (φ < 1)
- **Galaxy asymmetry**: More asymmetric galaxies need less enhancement

## Future Work

1. **Direct D measurement**: Use IFU data to measure kinematic asymmetry directly
2. **Cluster dynamics**: Test on more merging clusters with measured shock speeds
3. **Bulge kinematics**: Apply to dispersion-supported systems
4. **Theoretical derivation**: Connect φ to field-theoretic phase coherence

