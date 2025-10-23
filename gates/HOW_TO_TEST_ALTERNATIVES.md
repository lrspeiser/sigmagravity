# How to Test Alternative Formulas (The Right Way)

**Date:** 2025-10-22  
**Status:** Proper testing framework

---

## What We Learned

The baseline kernel (0.088 dex) has this structure:

```python
K = A_0 × RAR_shape × coherence_damping × small_gate
```

Where:
- **RAR_shape:** `(g†/g_bar)^p` - Controls low-acceleration behavior
- **coherence_damping:** `(L_coh/(L_coh+r))^n_coh` - Controls radial falloff
- **small_gate:** `1 - exp(-(r/r_gate)^2)` - Protects Solar System
- **L_coh:** `L_0 × f_bulge × f_shear × f_bar` - Physics gates

**This works beautifully!** But we can test alternatives systematically.

---

## The Right Way to Test

### ✅ DO: Test Individual Components

**Use `test_alternative_formulas.py` to swap ONE component at a time:**

```bash
python gates/test_alternative_formulas.py
```

**This tests:**

1. **Coherence damping alternatives** (keeping rest fixed):
   - Power-law (baseline): `(L_coh/(L_coh+r))^n_coh`
   - Exponential: `exp(-n_coh × r/L_coh)`
   - Stretched exponential: `exp(-(r/L_coh)^β)`
   - Gaussian: `exp(-(r/L_coh)^2)`
   - Burr-XII: `1 - (1 + (r/L_0)^p)^(-n_coh)` ⚠️ (grows to 1, probably wrong!)

2. **Small-radius gate alternatives** (keeping rest fixed):
   - Exponential (baseline): `1 - exp(-(r/r_gate)^2)`
   - Smoothstep: `x^2(3-2x)` where `x = r/(2×r_gate)`
   - Tanh: `0.5(1 + tanh(4(r-r_gate)/r_gate))`
   - Logistic: `1/(1 + exp(-k(r-r_gate)/r_gate))`

3. **RAR shape alternatives** (keeping rest fixed):
   - Power-law (baseline): `(g†/g_bar)^p`
   - Logarithmic: `p × log(1 + g†/g_bar)`
   - Tanh: `tanh(p × g†/g_bar)`
   - Exponential: `1 - exp(-p × g†/g_bar)`

**Each test measures RAR scatter and identifies best option.**

---

### ❌ DON'T: Add Extra Gates

**My original mistake:**
```python
# WRONG! Double-counts acceleration and uses incompatible formulas
K = A_0 × (g†/g_bar)^p × BURR_XII(R) × G_unified(R, g_bar)
                                      ^^^^^^^^^^^^^^^^^^^
                                      EXTRA GATE = BAD!
```

**Why this was wrong:**
- Burr-XII grows to 1 at large R (opposite of what we want!)
- G_unified double-counts acceleration dependence
- Incompatible with baseline structure

---

## Testing Like Cosmo Directory

The `cosmo/` directory tests alternatives systematically:

```
cosmo/
├── background.py       - Core ΛCDM computations
├── growth.py           - Growth factor alternatives
├── examples/
│   ├── make_outputs.py - Baseline results
│   └── score_vs_lcdm.py - Compare alternatives
```

**Same approach for kernel testing:**

```python
# gates/test_alternative_formulas.py

# 1. Define baseline
baseline_kernel = FlexibleKernel(hyperparams, 
                                 coherence_powerlaw,  # ← baseline
                                 gate_exponential,     # ← baseline
                                 rar_powerlaw)         # ← baseline

# 2. Test one alternative
alt_kernel = FlexibleKernel(hyperparams,
                           coherence_exponential,  # ← SWAP THIS
                           gate_exponential,        # ← keep baseline
                           rar_powerlaw)            # ← keep baseline

# 3. Compare
scatter_base = compute_rar_scatter(galaxies, baseline_kernel)
scatter_alt = compute_rar_scatter(galaxies, alt_kernel)
improvement = (scatter_base - scatter_alt) / scatter_base * 100
```

---

## Expected Results

**Baseline:** 0.088 dex (verified)

**If alternatives are similar:**
- Shows baseline is near-optimal
- Validates the chosen functional forms
- Good for the paper!

**If alternatives are better:**
- Shows room for improvement
- Could be published as follow-up
- But need theoretical justification for new form!

**Most likely outcome:** Baseline is best or tied for best (it was optimized on this data)

---

## How to Run Tests

### Quick test (just coherence functions):
```python
from gates.test_alternative_formulas import test_coherence_functions
import json

# Load hyperparameters
with open('config/hyperparams_track2.json') as f:
    hp = json.load(f)

# Load SPARC
galaxies = load_sparc_data()

# Test
results = test_coherence_functions(galaxies, hp)
print(f"Best: {min(results, key=results.get)}")
```

### Full test suite:
```bash
python gates/test_alternative_formulas.py
```

**Output:**
- Console: Scatter for each alternative
- `gates/outputs/alternative_tests/alternative_test_results.json` - Numeric results
- `gates/outputs/alternative_tests/alternative_functions.png` - Visual comparison

---

## Advanced: Test Combinations

**If you want to test combinations (e.g., exponential coherence + smoothstep gate):**

```python
kernel_combo = FlexibleKernel(
    hyperparams,
    coherence_func=coherence_exponential,  # Alternative 1
    gate_func=gate_smoothstep,              # Alternative 2
    rar_func=rar_powerlaw                   # Keep baseline
)

scatter = compute_rar_scatter(galaxies, kernel_combo)
```

**Caution:** 
- Testing all combinations = 5 × 4 × 4 = 80 tests
- Risk of overfitting
- Need validation set

**Better approach:**
- Test components individually first
- Only combine if both show improvement
- Always validate on held-out data

---

## Theoretical Justification

**If an alternative IS better, you need to explain WHY:**

### Good justification:
"Stretched exponential damping `exp(-(r/L_coh)^β)` with β < 1 represents anomalous diffusion in a turbulent medium, consistent with observed ISM structure."

### Bad justification:
"We tried 20 functions and this one fit best."

**The baseline's power-law form `(L_coh/(L_coh+r))^n_coh` has nice properties:**
- Smooth at r=0
- Power-law tail at large r
- Single parameter n_coh controls falloff rate
- Connects to Lorentzian profile in Fourier space

**Any alternative needs equally good motivation!**

---

## What About Physics Gates?

The baseline already has sophisticated physics gates:
- `f_bulge(B/T, r)` - Suppresses coherence near bulges
- `f_shear(∂Ω/∂r)` - Suppresses in high-shear regions
- `f_bar(bar_strength, r)` - Suppresses along bars

**Could test alternatives for these too:**

```python
# Current bulge gate
f_bulge = 1 / (1 + (BT^β) / (r/r_bulge + 0.1))

# Alternative: Gaussian profile
f_bulge_alt = exp(-BT^β × exp(-(r/r_bulge)^2))

# Test both, see which fits better
```

**But:** These are harder to test because they require galaxy-specific morphology data (B/T, bar strength, etc.)

---

## Key Principles

1. **Change one thing at a time** - Isolate what matters
2. **Always compare to baseline** - Know your reference
3. **Use held-out data** - Don't overfit
4. **Have theoretical justification** - Don't just data-mine
5. **Check all regimes** - Solar system, galaxies, clusters

---

## Summary

**Your paper's baseline is excellent!** (0.088 dex)

**To test alternatives properly:**
1. Use `test_alternative_formulas.py` 
2. Swap ONE component at a time
3. Measure scatter on held-out set
4. If improvement found, understand why
5. Validate on different dataset (MW, clusters)

**Most likely:** Baseline will remain best or tied. That's good! It means the formulation is robust.

**This is the right way to do it** - systematic, controlled, scientific. ✅

---

**Remember:** The goal isn't to beat the baseline at any cost. The goal is to understand which functional forms are physically motivated and robust across datasets. If the baseline wins, that validates your choices!

