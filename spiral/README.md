# Σ-Gravity with Spiral Winding Extension

This folder explores adding a **morphology-dependent spiral winding gate** to the original Σ-Gravity theory.

## Background

The original Σ-Gravity kernel is:
```
K(R) = A × C(R; ℓ₀, p, n_coh) × Π_j G_j
```

where:
- `A` = amplitude
- `C(R)` = Burr-XII coherence window
- `G_j` = geometry gates (bulge, bar, etc.)

## The Winding Extension

We add a new gate based on differential rotation:

```
K(R) = A × C(R) × Π_j G_j × G_winding(R, v_c)
```

where:
```
G_winding = 1 / (1 + (N_orbits/N_crit)²)

N_orbits = t_age × v_c / (2πR × 0.978)
N_crit ≈ 10  (derived from v_c/σ_v)
```

## Physical Motivation

1. **Coherent paths form in galactic disks** over cosmic time
2. **Differential rotation winds these paths** into tighter spirals
3. **After ~10 orbits**, adjacent paths interfere destructively
4. **Fast rotators (massive spirals)** have more orbits → more winding → more suppression
5. **Slow rotators (dwarfs)** have fewer orbits → less winding → less suppression

This provides **morphology-dependent enhancement without ad-hoc classification**!

## Key Insight: N_crit Derivation

The critical winding number is **derivable from coherence geometry**:

```
Azimuthal coherence length: ℓ_azimuthal ~ (σ_v/v_c) × 2πR
After N orbits, wound spacing: λ_wound ~ 2πR/N

Destructive interference when: λ_wound ~ ℓ_azimuthal
→ N_crit ~ v_c/σ_v ~ 200/20 = 10 ✓
```

This is **not a free parameter** - it emerges from the physics!

## Files

```
spiral/
├── README.md                      # This file
├── winding_gate.py                # Core winding gate implementation
├── sigma_gravity_winding.py       # Σ-Gravity with winding integrated
└── tests/
    └── test_sparc_winding.py      # SPARC batch comparison test
```

## Usage

### Test the winding gate:
```bash
python winding_gate.py
```

### Test on SPARC:
```bash
python tests/test_sparc_winding.py
```

## Expected Results

| Galaxy Type | N_orbits | G_winding | Effect |
|-------------|----------|-----------|--------|
| Dwarf | ~10-20 | 0.2-0.5 | Moderate suppression |
| Intermediate | ~20-30 | 0.1-0.2 | Stronger suppression |
| Massive | ~30-50 | 0.04-0.1 | Strong suppression |

**Key prediction**: Massive spirals get **less enhancement** than dwarfs, matching observations.

## Comparison with Original Σ-Gravity

| Metric | Original | With Winding |
|--------|----------|--------------|
| Massive spiral fit | Weak point | Should improve |
| Morphology dependence | Via gates | Natural from orbits |
| Free parameters | G_j params | N_crit (derived!) |

## Theoretical Relationship

The winding gate is **compatible with** the original Σ-Gravity:
- It's just another G_j gate
- Multiplies the existing kernel
- Preserves Solar System safety (N_orbits → ∞ at small R)
- Preserves cluster behavior (pressure-supported, no winding)

## Author

Leonard Speiser, 2025-11-25

## Note

This is an **extension exploration**. It does not modify the original Σ-Gravity paper or root files.
