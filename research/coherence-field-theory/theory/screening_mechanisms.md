# Screening Mechanisms for Coherence Scalar Field

## Why Screening is Necessary

Solar system tests constrain deviations from GR to extremely high precision:
- |γ - 1| < 2.3×10⁻⁵ (Cassini spacecraft)
- |β - 1| < 8×10⁻⁵ (lunar laser ranging)

Yet we need the coherence field to have significant effects at:
- Galaxy scales (flat rotation curves)
- Cluster scales (enhanced lensing)
- Cosmological scales (dark energy)

**Solution**: Environment-dependent screening mechanisms that suppress the field's effects in dense regions (Solar System) while allowing it to operate in diffuse regions (galaxy outskirts, cosmic voids).

## 1. Chameleon Mechanism

### Concept
The field acquires an environment-dependent effective mass through coupling to matter density.

### Modified Potential
```
V_eff(φ) = V(φ) + ρ_matter φ / M_Pl
```

Where M_Pl is the Planck mass and ρ_matter is the local matter density.

### Effective Mass
```
m_eff² = d²V_eff/dφ² = d²V/dφ² + ρ_matter / M_Pl
```

### Key Properties

In **high-density environments** (Sun, Earth):
- ρ_matter large → m_eff large
- Large mass → short range: λ = 1/m_eff small
- Field frozen at minimum of V_eff
- Fifth force range: r_force ~ λ << R_object
- **Result**: Field effectively screened, GR recovered

In **low-density environments** (galaxy halo, cosmic void):
- ρ_matter small → m_eff small  
- Small mass → long range: λ large
- Field free to vary and mediate forces
- **Result**: Fifth force active, modifies gravity

### Thin-Shell Condition

An object is screened if:
```
Δφ = φ_surface - φ_interior << M_Pl
```

This occurs when:
```
χ ≡ (3 Δφ M_Pl) / (M_Pl² / ρ R²) << 1
```

For the Sun: χ ~ 10⁻⁷ << 1 ✓ (screened)
For a galaxy halo: χ >> 1 ✗ (not screened, fifth force active)

### Implementation

1. **Potential Form**:
```
V(φ) = V₀ exp(-λφ) + (M⁴⁺ⁿ) / φⁿ
```

The inverse power term provides the chameleon behavior.

2. **Field Equation**:
```
∇²φ = dV/dφ + ρ_matter / M_Pl
    = -λ V₀ exp(-λφ) - n M⁴⁺ⁿ / φⁿ⁺¹ + ρ_matter / M_Pl
```

3. **Screening Radius**:
```
r_screen ~ (M_Pl² / ρ m_eff)^(1/3)
```

Inside r_screen: field suppressed
Outside r_screen: fifth force active

## 2. Symmetron Mechanism

### Concept
Spontaneous symmetry breaking driven by local matter density. Field is pinned at φ=0 in dense regions, but acquires VEV in voids.

### Potential
```
V(φ) = -μ²/2 φ² + λ/4 φ⁴ + (ρ_matter / M²) φ²
```

### Effective Mass
```
m_eff² = -μ² + λφ² + ρ_matter / M²
```

### Symmetry Breaking

**High density** (ρ_matter > ρ_crit):
- Effective mass² > 0 for all φ
- Minimum at φ = 0
- Field frozen at symmetric point
- **Result**: No fifth force, GR recovered

**Low density** (ρ_matter < ρ_crit):
- Effective mass² < 0 at φ = 0
- Symmetry spontaneously broken
- Field rolls to φ = ±φ_min ≠ 0
- Fifth force active with range λ = 1/m_eff
- **Result**: Gravity modification

### Critical Density
```
ρ_crit = μ² M²
```

### VEV in Vacuum
```
φ_min = μ / √λ
```

### Implementation

For Solar System:
- Choose μ, M such that ρ_☉ > ρ_crit (screened)
- For galaxy outskirts: ρ_halo < ρ_crit (not screened)

## 3. Vainshtein Mechanism

### Concept
Nonlinear derivative interactions become important at small scales, modifying the field's propagation and suppressing its coupling to matter.

### Lagrangian
```
L = -1/2 (∂φ)² - V(φ) + (∂φ)² □φ / Λ³ + φ T / M_Pl
```

The cubic term (∂φ)² □φ is the Vainshtein operator.

### Vainshtein Radius
```
r_V = (M_Pl R_Schwarzschild Λ⁻³)^(1/3)
```

For the Sun: r_V ~ 1000 R_☉

### Screening Behavior

**Inside Vainshtein radius** (r < r_V):
- Nonlinear terms dominate
- Kinetic energy suppresses coupling
- Fifth force weak: F_5th ~ (r/r_V)^(3/2) F_Newton
- **Result**: GR approximately recovered

**Outside Vainshtein radius** (r > r_V):
- Linear regime
- Standard scalar force mediation
- **Result**: Fifth force active

### Implementation

More complex to implement numerically due to nonlinearities. Requires solving:
```
∇²φ + 1/Λ³ [(∇²φ)² - (∂_i ∂_j φ)²] = ρ_matter / M_Pl
```

## 4. K-mouflage Mechanism

Similar to Vainshtein but with modified kinetic term:
```
L = K(X, φ) - V(φ) + φ T / M_Pl
```

Where X = -1/2 (∂φ)².

Screening through kinetic energy regulation.

## Comparison Table

| Mechanism  | Screening | Implementation | Solar System | Galaxy Halo |
|------------|-----------|----------------|--------------|-------------|
| Chameleon  | Mass      | Moderate       | χ << 1 ✓     | χ >> 1 ✓    |
| Symmetron  | SSB       | Moderate       | ρ > ρ_c ✓    | ρ < ρ_c ✓   |
| Vainshtein | Nonlinear | Hard           | r < r_V ✓    | r > r_V ✓   |
| K-mouflage | Kinetic   | Hard           | X large ✓    | X small ✓   |

## Recommended Approach for Coherence Field

### Stage 1: Chameleon (Easiest)
Start with chameleon-modified potential:
```
V(φ) = V₀ exp(-λφ) + M⁴ / φ
```

**Pros**: 
- Relatively simple to implement
- Well-studied in literature
- Clear physical picture

**Cons**:
- Adds one parameter (M⁴)

### Stage 2: Symmetron (Intermediate)
If chameleon doesn't fit well, try symmetron:
```
V(φ) = -μ²/2 φ² + λ/4 φ⁴
```
With matter coupling: ρφ/M_Pl

**Pros**:
- Natural Z₂ symmetry
- Clear phase transition

**Cons**:
- More parameters (μ, λ, M)

### Stage 3: Vainshtein (Advanced)
Only if simpler mechanisms fail:
```
L = -1/2 (∂φ)² - V(φ) + nonlinear terms
```

**Pros**:
- Can work with simpler potentials
- Natural from DGP/massive gravity

**Cons**:
- Numerically challenging
- Requires nonlinear PDE solver

## Testing Screening

### Solar System Tests

1. **PPN Parameters**:
   - Compute γ, β in screened region
   - Verify |γ-1|, |β-1| < 10⁻⁴

2. **Lunar Laser Ranging**:
   - Compute Nordtvedt effect
   - Verify Δ = |4β - γ - 3| < 10⁻³

3. **Cassini Tracking**:
   - Light deflection parameter γ
   - Verify |γ - 1| < 2×10⁻⁵

### Galaxy Tests

1. **Wide Binaries** (Gaia):
   - Test at ~0.1 kpc separations
   - Intermediate between screened and unscreened
   - Constrains screening transition

2. **Rotation Curves**:
   - Full fifth force should be active
   - Match flat rotation curves

3. **Tully-Fisher Relation**:
   - Should naturally emerge from coherence halos

### Dwarf Galaxies

Critical test: lowest density objects
- If field active in Milky Way halo but screened in Solar System
- Dwarfs are intermediate density
- Should show partial screening → unique signature

## Implementation Strategy

1. **Start simple**: Exponential potential, no screening
   - Verify it can't pass solar system tests
   - Establish need for screening

2. **Add chameleon term**: V(φ) = V₀ exp(-λφ) + M⁴/φ
   - Tune M⁴ for solar system compatibility
   - Check galaxy fits still work

3. **Optimize globally**:
   - Fit V₀, λ, M⁴ to all scales simultaneously
   - Verify no parameter tensions

4. **If unsuccessful, try symmetron**:
   - Different functional form may fit better

## Key Equations for Chameleon Implementation

### Field Equation
```
d²φ/dr² + 2/r dφ/dr = dV_eff/dφ
```

Where:
```
V_eff(φ, r) = V(φ) + ρ_matter(r) φ / M_Pl
```

### Boundary Conditions
- At r → ∞: φ → φ_∞ (cosmological value)
- At r → 0: dφ/dr → 0 (regularity)

### Force Law
Fifth force acceleration:
```
a_5th = -1/M_Pl ∇φ
```

Total acceleration:
```
a_total = -∇Φ_Newton - 1/M_Pl ∇φ
```

### Screening Test
Compute thin-shell parameter:
```
χ = |φ_surface - φ_center| M_Pl / (M_object / R_object²)
```

If χ << 1: screened ✓
If χ >> 1: not screened, full fifth force ✓

## Next Steps

1. Implement chameleon field solver in `solar_system/chameleon_solver.py`
2. Compute χ for Sun, Earth, Moon
3. Verify PPN parameters
4. Extend to galaxy scales
5. Check continuity of field profile across scales

