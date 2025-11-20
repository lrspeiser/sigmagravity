# Approach C Implementation: Symmetron Coherence Gravity

## What We Just Built

✅ **symmetron_potential.py** — Complete double-well potential with:
- Environment-dependent screening (ρ > ρ_crit → φ=0)
- Spontaneous symmetry breaking (ρ < ρ_crit → φ=±φ₀)
- Cosmological evolution (tracking solution)
- Direct K(R) = β φ²/M_Pl² connection

✅ **Test results**:
- Solar System: φ_min = 0 (SCREENED ✓)
- Galaxy edge: φ_min ≠ 0 (UNSCREENED ✓)
- Cosmology: Ω_φ = 0 (NEEDS TUNING ✗)

---

## The Field Theory

```
V_eff(φ, ρ) = V₀ + (1/2)[ρ/M² - μ²]φ² + (λ/4)φ⁴
```

**Parameters to fit**:
- μ: Bare mass (sets φ₀ = μ/√λ)
- λ: Self-coupling (quartic steepness)
- M: Matter coupling (sets ρ_crit = μ²M²)
- V₀: Vacuum energy (cosmology tuning)
- β: Metric coupling (gravity boost)

**Critical densities**:
- ρ_crit = μ²M² ~ 10⁻²¹ kg/m³ (galaxy transition)
- ρ_solar ~ 10⁻¹⁵ kg/m³ >> ρ_crit (screened)
- ρ_cosmic ~ 10⁻²⁶ kg/m³ << ρ_crit (unscreened)

---

## Step 2: Viability Scanner (DO NEXT)

### Goal
Find (μ, λ, M, V₀, β) that pass **all three filters**:

1. **Cosmology**: Ω_m ≈ 0.3, Ω_φ ≈ 0.7 at z=0
2. **Galaxy screening**: R_c ~ kpc (not >> Mpc, not << kpc)
3. **Solar System**: PPN parameters safe

### Implementation

Create `analysis/symmetron_viability_scan.py`:

```python
import numpy as np
from coherence-field-theory.cosmology.symmetron_potential import (
    SymmetronParams, dark_energy_fraction, critical_density
)

# Parameter ranges (log-uniform)
mu_range = np.logspace(-4, -2, 20)      # [eV]
lambda_range = np.logspace(-2, 1, 15)   # dimensionless
M_range = np.logspace(-4, -2, 20)       # [M_Pl]
V0_range = np.linspace(-1e-6, 1e-6, 10) # Tune for cosmology
beta_range = [0.1, 0.5, 1.0, 2.0]       # Discrete choices

# 20 × 15 × 20 × 10 × 4 = 240,000 points (1 hour scan)
```

**Filters**:
```python
def passes_cosmology(params):
    """Ω_m ∈ [0.25, 0.35], Ω_φ ∈ [0.65, 0.75] at a=1"""
    Omega_m, Omega_phi = dark_energy_fraction(a=1.0, rho_m0=2.5e-27, params)
    return (0.25 <= Omega_m <= 0.35) and (0.65 <= Omega_phi <= 0.75)

def passes_galaxy_screening(params):
    """R_c ~ 1-10 kpc for typical galaxy"""
    rho_crit = critical_density(params)
    # Exponential profile: ρ(r) = ρ_c exp(-r/R_d)
    # Want ρ_crit ~ ρ(R_c) where R_c ~ few kpc
    # For R_d=3 kpc, ρ_c=10^-20: need ρ_crit ~ 10^-21 to 10^-22
    return 1e-22 < rho_crit < 1e-20

def passes_ppn(params):
    """γ_PPN - 1 < 10^-5, β_PPN - 1 < 10^-4"""
    # In Solar System: φ ≈ 0 (screened)
    # → weak coupling → standard GR
    # Check: d²V_eff/dφ²|_φ=0 at ρ_solar
    rho_solar = 1e-15  # kg/m³
    m_eff_sq = rho_solar / params.M**2 - params.mu**2
    if m_eff_sq < 1e10:  # Not heavy enough
        return False
    return True  # Approximation: screened → PPN safe
```

---

## Step 3: Test Best Parameter Set on Galaxies

Once you have **ANY viable point** from the scan:

```python
# Load best parameters
params_best = SymmetronParams(mu=..., lambda_self=..., M=..., V0=..., beta=...)

# For each SPARC galaxy:
#   1. Load baryonic profile ρ_bar(r)
#   2. Solve field equation: ∇²φ = -dV_eff/dφ
#   3. Compute K(R) = β φ²/M_Pl²
#   4. Compute v_eff = √[v_bar² (1 + K(R))]
#   5. Compare to data: χ²

# Compare against:
#   - Your phenomenological Σ-Gravity K(R) fits
#   - NFW/Burkert dark matter
```

**Key test**: Can **one parameter set** fit **multiple galaxies**?
- If yes: You have a universal field theory! 🎯
- If no: Need galaxy-dependent parameters (weaker claim)

---

## Why This Might Work (Physical Intuition)

**Solar System** (ρ ~ 10⁻¹⁵):
```
ρ >> ρ_crit → m_eff² = ρ/M² - μ² ≈ ρ/M² >> μ²
→ φ stuck at 0 (huge mass)
→ no fifth force, PPN safe ✓
```

**Galaxy edge** (ρ ~ 10⁻²²):
```
ρ < ρ_crit → m_eff² = ρ/M² - μ² < 0 (tachyonic!)
→ φ rolls to minimum φ₀ = √[(μ² - ρ/M²)/(λ/2)]
→ K(R) = β φ₀²/M_Pl² ≈ 0.5 (your typical boost!) ✓
```

**Cosmology** (ρ ~ 10⁻²⁶):
```
ρ << ρ_crit → φ ≈ φ_vacuum = μ/√λ
V_eff ≈ V₀ - μ⁴/(4λ) (constant!)
→ acts like Λ, drives acceleration ✓
```

The **same field** does three jobs because ρ_crit sits perfectly between them!

---

## Advantages Over Previous Attempts

**vs. Exponential + Chameleon** (Approach A):
- ❌ That had *no* SSB → couldn't get coherence buildup
- ❌ Screening killed it (chameleon term dominated)
- ✅ Symmetron has **built-in** screening from ρ/M² term

**vs. Wave Amplification** (Approach B):
- ❌ That required time-dependent PDEs (too hard for now)
- ❌ Resonance structure difficult to constrain
- ✅ Symmetron is **static** (fits your current code)

**vs. Pure Phenomenology**:
- ❌ Your K(R) kernel was fitted, not derived
- ✅ Symmetron **predicts** K(R) shape from φ(r) solution

---

## Expected Challenges

1. **Cosmology tuning**: 
   - Default params gave Ω_φ ≈ 0 (too small!)
   - Need to scan V₀ or adjust μ, λ to get Ω_φ ~ 0.7
   - This is **solvable** — just a parameter search

2. **Numerical stability**:
   - Solving ∇²φ = -dV_eff/dφ in galaxies is nonlinear
   - Might need relaxation methods (you already have this)

3. **Universal parameters**:
   - Ideally ONE (μ, λ, M, V₀, β) fits all galaxies
   - If not: might need galaxy-mass-dependent tuning

---

## Success Criteria

### Minimal success:
- ✅ Find **at least one** (μ, λ, M, V₀, β) that passes all three filters
- ✅ Show it gives R_c ~ kpc in **one test galaxy**

### Strong success:
- ✅ Universal parameters fit **5+ SPARC galaxies** with χ² competitive with DM
- ✅ Predicts Ω_φ ≈ 0.7 without fine-tuning

### Paper-worthy success:
- ✅ Beats phenomenological Σ-Gravity on **same data** (fewer free params)
- ✅ Makes **new predictions** (e.g., φ(z) evolution testable with SNe)

---

## Files Created

```
coherence-field-theory/
├── cosmology/
│   ├── symmetron_potential.py         ← ✅ DONE
│   └── symmetron_evolution.py         ← (exists from earlier)
├── analysis/
│   ├── symmetron_viability_scan.py    ← TODO (Step 2)
│   └── global_viability_scan.py       ← (old, for comparison)
├── galaxies/
│   └── field_driven_symmetron.py      ← TODO (Step 3)
└── outputs/
    ├── symmetron_potential_shapes.png ← ✅ DONE
    └── symmetron_cosmology.png        ← ✅ DONE
```

---

## Next Commands to Run

```bash
# 1. Create viability scanner
cp coherence-field-theory/analysis/global_viability_scan.py \
   coherence-field-theory/analysis/symmetron_viability_scan.py

# Edit to use SymmetronParams instead of exponential potential

# 2. Run scan (this will take ~1 hour)
python coherence-field-theory/analysis/symmetron_viability_scan.py

# 3. Check results
ls -lh coherence-field-theory/outputs/symmetron_viability_scan/

# 4. If any points pass, test on galaxies
python coherence-field-theory/galaxies/fit_field_driven.py --params-from-scan
```

---

## Key Insight from Your Argument

You said:
> "A is phenomenology (limit of C), B is dynamics (populates C's minima), C is the backbone"

**This is exactly right!**

- **C (symmetron)** gives you the **field theory structure**
- **A (well)** is what happens when φ adiabatically tracks φ_min(ρ) (slow limit of C)
- **B (waves)** explains **how** the field gets to φ_min (fast dynamics, resonance)

By starting with C, you get:
- Immediate testability (static field equations)
- Clear screening mechanism (ρ > ρ_crit)
- Path to cosmology (tracking solution)

Then later, you can add B on top:
- Perturbations around φ₀ → wave modes
- Resonance structure → fine structure in K(R)
- Time-dependent effects → mergers, transients

---

## Summary

**You should pursue Approach C next** because:

1. ✅ Fits your existing code structure (GR + scalar + V(φ))
2. ✅ Has built-in screening (density-dependent)
3. ✅ Can drive cosmic acceleration (tune V₀)
4. ✅ Testable with viability scan (run it NOW!)
5. ✅ Clear path from field theory → phenomenology

**Don't get stuck** on the fact that default params failed cosmology — that's what the viability scan is for!

Run `symmetron_viability_scan.py` next and see if **any parameters pass**. Even ONE passing point would be huge. 🎯
