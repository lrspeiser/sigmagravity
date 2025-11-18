# F_missing Analysis: Identifying the Second Mechanism

## Key Finding

**F_missing shows strong negative correlations** with system properties, revealing what the second mechanism must explain.

---

## Correlation Results

### 1. Velocity Dispersion (σ_v) ⭐ STRONGEST

- **Spearman r = -0.585** (p < 1e-15) - **HIGHLY SIGNIFICANT**
- **Pearson r = -0.483** (p < 1e-10)
- **Interpretation**: Higher σ_v → **Lower F_missing**

**Physical Meaning**:
- Galaxies with **higher velocity dispersion** need **less additional enhancement**
- This suggests the second mechanism is **suppressed** in high-σ_v systems
- Or roughness is **more effective** in high-σ_v galaxies

**Implication**: Second mechanism likely **anti-correlates with σ_v**

---

### 2. Disc Scale Length (R_d) ⭐ STRONG

- **Spearman r = -0.489** (p < 1e-10) - **HIGHLY SIGNIFICANT**
- **Pearson r = -0.386** (p < 1e-6)
- **Interpretation**: Larger discs → **Lower F_missing**

**Physical Meaning**:
- Galaxies with **larger scale lengths** need **less additional enhancement**
- This suggests the second mechanism depends on **disc compactness**
- Or it's suppressed in extended discs

**Implication**: Second mechanism likely **anti-correlates with R_d**

---

### 3. Bulge Fraction ⭐ MODERATE

- **Spearman r = -0.442** (p < 1e-8) - **SIGNIFICANT**
- **Pearson r = -0.255** (p < 0.001)
- **Interpretation**: More bulge → **Lower F_missing**

**Physical Meaning**:
- Galaxies with **larger bulges** need **less additional enhancement**
- This suggests the second mechanism is **suppressed** in bulge-dominated systems
- Or it's geometry-dependent

**Implication**: Second mechanism likely **anti-correlates with bulge fraction**

---

## Physical Interpretation

### What These Correlations Tell Us:

1. **F_missing is NOT random** - It has clear physical correlates
2. **All correlations are NEGATIVE** - Higher property → Lower F_missing
3. **Strongest with σ_v** - Velocity dispersion is the key factor

### Possible Explanations:

#### Hypothesis A: Second Mechanism Suppressed by Dispersion

The second mechanism (e.g., metric resonance, graviton pairing) is **suppressed** in high-σ_v systems because:
- High dispersion → **decoherence** → less coherent enhancement
- Roughness already handles high-σ_v systems well
- Second mechanism only needed for **low-σ_v** systems

#### Hypothesis B: Roughness More Effective at High σ_v

Roughness itself might be **more effective** in high-σ_v systems:
- High σ_v → longer τ_noise → longer τ_coh → higher Ξ → higher K_rough
- But we're using **mean Ξ**, so this should already be accounted for
- Unless there's a **non-linear** effect

#### Hypothesis C: Second Mechanism is Velocity-Dependent

The second mechanism might be:
- **Inversely proportional to σ_v**: F_missing ∝ 1/σ_v^α
- **Inversely proportional to R_d**: F_missing ∝ 1/R_d^β
- **Combined**: F_missing ∝ (σ_ref / σ_v)^α × (R_ref / R_d)^β

---

## Next Steps

### Step 1: Fit Functional Form

Try fitting:
```
F_missing = A × (σ_ref / σ_v)^α × (R_ref / R_d)^β
```

Where σ_ref and R_ref are reference values (e.g., median).

### Step 2: Test Microphysics Models

Modify coherence model fits to target **F_missing** instead of K_total:
- Which model best reproduces the σ_v and R_d scaling?
- Does it naturally give F_missing ∝ 1/σ_v^α?

### Step 3: Unified Kernel

Once second mechanism identified:
```
K_total(R) = K_rough(Ξ) × K_resonant(σ_v, R_d) × C(R/ℓ₀)
```

Where K_resonant explains F_missing.

---

## Key Insight

**The negative correlations are GOOD NEWS**:

1. They show F_missing is **not arbitrary** - it has clear physics
2. They point to **specific mechanisms** (velocity-dependent, scale-dependent)
3. They provide **constraints** for microphysics models

The fact that **σ_v is strongest** suggests the second mechanism is fundamentally about **velocity/dispersion**, not just geometry.

---

## Files

- `F_missing_correlations.json` - Full correlation results
- `sparc_roughness_amplitude.csv` - Galaxy-by-galaxy F_missing values
- `F_MISSING_ANALYSIS.md` - This document

---

## Conclusion

F_missing analysis reveals:
- **Strong negative correlation with σ_v** (r = -0.585)
- **Strong negative correlation with R_d** (r = -0.489)
- **Moderate negative correlation with bulge_frac** (r = -0.442)

This points to a **velocity/dispersion-dependent second mechanism** that:
- Is **suppressed** in high-σ_v systems
- Is **suppressed** in large discs
- Is **suppressed** in bulge-dominated systems

**Next**: Fit functional form and test microphysics models against F_missing.

