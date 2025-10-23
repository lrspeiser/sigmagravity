# Gate Quick Reference

## 🎯 When to Use Which Gate

```
Have measured R_bulge from imaging?
  ├─ YES → Use exponential gate
  └─ NO → Use distance or unified gate

Need to suppress in bars/non-circular regions?
  └─ Use distance gate with R_min = bar length

Have high-resolution g_bar(R)?
  └─ Use unified gate (distance × acceleration)

ALWAYS include:
  └─ Solar system safety gate (multiply everything)
```

## 📐 Core Equations

### 1. Distance Gate
```
G_dist(R) = [1 + (R_min/R)^α]^(-β)
```
- **When**: Suppress at small R regardless of acceleration
- **R_min**: Transition scale (kpc) - WHERE suppression turns off
- **α**: Steepness (1.5-3.0) - HOW FAST transition happens
- **β**: Strength (0.5-2.0) - HOW STRONG suppression is

**Example**: Inner disk (bulge region)
- R_min = 1.5 kpc (bulge radius)
- α = 2.0 (standard transition)
- β = 1.0 (moderate suppression)

### 2. Acceleration Gate
```
G_accel(g_bar) = [1 + (g_bar/g_crit)^α]^(-β)
```
- **When**: Suppress where g_bar is high (dense regions)
- **g_crit**: Critical acceleration (m/s²) - typically 10⁻¹⁰ to 10⁻⁹
- **α, β**: Same meaning as distance gate

**Example**: Bulge suppression
- g_crit = 1e-9 m/s² (high acceleration threshold)
- α = 2.0
- β = 1.0

### 3. Exponential Gate (for bulges)
```
G_bulge(R) = [1 - exp(-(R/R_bulge)^α)]^β
```
- **When**: You have measured R_bulge from Sérsic fit
- **R_bulge**: Effective radius from imaging (kpc)
- **α, β**: Shape parameters

**Example**: Measured bulge
- R_bulge = 2.0 kpc (from surface brightness fit)
- α = 2.0
- β = 1.0

### 4. Unified Gate
```
G_unified = G_dist(R) × G_accel(g_bar)
```
- **When**: Both effects matter (inner disk, cluster cores)
- Uses all parameters from #1 and #2

### 5. Solar System Safety (ALWAYS include!)
```
G_solar = [1 + (R_min_solar/R)^4]^(-2)
```
- **R_min_solar**: Fixed at 0.0001 kpc (≈ 100 AU)
- **α, β**: Fixed at (4, 2) for strong suppression
- **Purpose**: Ensures K(1 AU) < 10⁻¹⁴ (PPN constraint)

## 📊 Parameter Determination

### Observable Scales (MEASURE from data, don't fit!)

| Parameter | How to Measure | Typical Value |
|-----------|----------------|---------------|
| **R_bulge** | Sérsic fit to I(R) | 1-3 kpc |
| **g_crit** | From RAR or your g† | ~10⁻⁹ m/s² |
| **R_bar** | Bar length from imaging | 3-6 kpc |
| **ℓ₀** | Coherence length (paper) | 5 kpc (fixed) |

### Shape Parameters (FIT to data)

| Parameter | Range | Controls |
|-----------|-------|----------|
| **α** | 1.5 - 3.0 | Transition steepness |
| **β** | 0.5 - 2.0 | Suppression strength |

### Fixed by Physics (NON-NEGOTIABLE)

| Parameter | Value | Why |
|-----------|-------|-----|
| **R_min,solar** | 0.0001 kpc | PPN safety |
| **α_solar** | 4 | Steep suppression needed |
| **β_solar** | 2 | Full suppression required |

## ✅ Validation Checklist

After choosing parameters, verify:

```python
# 1. Mathematical constraints
assert np.all((G >= 0) & (G <= 1))  # Bounds
assert G[0] < 0.01  # Suppressed at R=0
assert G[-1] > 0.99  # Saturated at large R
assert np.all(np.gradient(G) >= 0)  # Monotonic

# 2. Physics constraints
K_1AU = compute_kernel(1*AU_in_kpc, ...)
assert K_1AU < 1e-14  # PPN safe

# 3. Data quality
chi2_reduced = compute_chi2(fit)
assert 0.8 < chi2_reduced < 1.2  # Good fit
```

## 🔧 Quick Start Examples

### Example 1: Simple Distance Gate
```python
from gate_core import G_distance, K_sigma_gravity

# Parameters
R_min = 1.0  # kpc, transition scale
alpha = 2.0  # standard steepness
beta = 1.0   # moderate strength

# Your data
R = np.logspace(-1, 2, 100)  # kpc
g_bar = ...  # from baryonic model

# Compute gate
G = G_distance(R, R_min, alpha, beta)

# Full kernel
K = K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0,
                    gate_type='distance',
                    gate_params={'R_min': R_min, 'alpha': alpha, 'beta': beta})
```

### Example 2: Bulge with Measured R_bulge
```python
from gate_core import G_bulge_exponential, K_sigma_gravity

# From imaging
R_bulge = 2.0  # kpc, from Sérsic fit

# Fit shape parameters
alpha = 2.0  # adjust to match residuals
beta = 1.0   # adjust strength

# Compute
G = G_bulge_exponential(R, R_bulge, alpha, beta)

# Full kernel
K = K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0,
                    gate_type='exponential',
                    gate_params={'R_bulge': R_bulge, 'alpha': alpha, 'beta': beta})
```

### Example 3: Unified Gate (Distance + Acceleration)
```python
from gate_core import G_unified, K_sigma_gravity

# Parameters for both components
gate_params = {
    'R_min': 1.0,      # distance scale
    'g_crit': 1e-10,   # accel scale
    'alpha_R': 2.0,    # distance steepness
    'beta_R': 1.0,     # distance strength
    'alpha_g': 2.0,    # accel steepness
    'beta_g': 1.0      # accel strength
}

# Compute
K = K_sigma_gravity(R, g_bar, A=0.6, ell0=5.0,
                    gate_type='unified',
                    gate_params=gate_params)
```

## 🎯 Physical Interpretation

### Why This Structure?

**Distance gate**: "Coherence needs room to develop"
- Small R → frequent interactions → rapid decoherence → G → 0
- Large R → extended structure → sustained coherence → G → 1

**Acceleration gate**: "Coherence needs low density"
- High g → compact object (bulge) → rapid decoherence → G → 0
- Low g → extended medium → sustained coherence → G → 1

**Product structure**: "Both conditions must be met"
```
G_total = 0  if  (R is small)  OR  (g is high)
G_total ≈ 1  if  (R is large)  AND  (g is low)
```

This naturally implements:
> "Coherence emerges in extended, low-density environments"

### Connection to Your Paper

Your galaxy kernel (Section 2.7):
```
K_gal(R) = A₀ · (g_bar/g†)^p · C(R; ℓ₀,p,n_coh) · G_bulge · G_shear · G_bar
```

**Mapping**:
- `(g_bar/g†)^p` → already an acceleration gate!
- `C(R)` → universal coherence window (Burr-XII)
- `G_bulge`, `G_shear`, `G_bar` → morphology-specific gates from this package
- **Solar system gate should multiply everything**

## 🚨 Common Pitfalls

### 1. Forgetting Solar System Safety
```python
# ❌ WRONG
K = A * C(R) * G_bulge(R)

# ✓ CORRECT
K = A * C(R) * G_bulge(R) * G_solar(R)
```

### 2. Fitting Observable Scales
```python
# ❌ WRONG - fitting measured quantity
fit_params = [R_bulge, alpha, beta]  # R_bulge should be from imaging!

# ✓ CORRECT - only fit shape parameters
fit_params = [alpha, beta]  # Use measured R_bulge = 2.0 kpc
```

### 3. Too Many Free Parameters
```python
# ❌ WRONG - 7 parameters!
fit_params = [A, ell0, p, n_coh, R_min, alpha, beta]

# ✓ CORRECT - Fix universal parameters from SPARC calibration
# Only fit: [A, alpha, beta] with measured R_min = R_bulge
```

### 4. Wrong Gate for Context
```python
# Have measured R_bulge but using distance gate?
# ❌ SUBOPTIMAL
G = G_distance(R, R_min=2.0, ...)  

# ✓ BETTER - use exponential gate
G = G_bulge_exponential(R, R_bulge=2.0, ...)
```

## 📈 Workflow

```
1. Measure observable scales
   ├─ R_bulge from imaging
   ├─ R_bar from morphology
   └─ g_crit from RAR or g†

2. Choose gate type
   └─ Decision tree at top of doc

3. Set observable scales
   └─ Use measured values, DON'T fit

4. Fit shape parameters (α, β)
   └─ Minimize χ² on rotation curve

5. Validate
   ├─ Check G ∈ [0,1], monotonic
   ├─ Check K(1 AU) < 10⁻¹⁴
   └─ Check χ²_reduced ≈ 1

6. Apply to full dataset
   └─ Same parameters across sample
```

## 🔗 See Also

- **Full theory**: `gate_mathematical_framework.md`
- **Visualization**: Run `python gate_modeling.py`
- **Fitting tool**: `python gate_fitting_tool.py --help`
- **Tests**: `pytest tests/test_section2_invariants.py`

---

**This is your one-stop reference for gate equations and usage!**

