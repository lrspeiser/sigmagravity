# Σ-Gravity Scale-Finding Test Suite

Comprehensive Python testing suite to find the physical scales behind the Σ-Gravity coherence length.

## Overview

This suite systematically tests dimensional combinations against SPARC galaxy rotation curves to determine:

1. **Does the coherence length scale with galaxy properties?** (Tully-Fisher test)
2. **What is the best power-law fit?** (Optimization)
3. **Which physical scales match observations?** (Hypothesis library)

## Quick Start

### Step 1: Prepare SPARC Data

```bash
python GravityWaveTest/prepare_sparc_data.py
```

This creates `data/sparc/sparc_combined.csv` with all required columns:
- `M_baryon` (total baryonic mass, M_sun)
- `v_flat` (flat rotation velocity, km/s)
- `R_disk` (disk scale length, kpc)
- `sigma_velocity` (velocity dispersion, km/s)
- Additional metadata

### Step 2: Run All Tests

```bash
python GravityWaveTest/run_all_tests.py
```

This runs all three test suites in optimal order (~5-10 minutes total).

**Or run individual tests:**

```bash
# Test 1: Tully-Fisher scaling (~30 seconds)
python GravityWaveTest/test_tully_fisher_scaling.py

# Test 2: Power-law optimization (~1-2 minutes)
python GravityWaveTest/optimize_power_law.py

# Test 3: Scale hypothesis library (~2-5 minutes)
python GravityWaveTest/scale_finder.py
```

## Test Descriptions

### Test 1: Tully-Fisher Scaling

**File**: `test_tully_fisher_scaling.py`

**Question**: Does λ_g ∝ √M_b as predicted by the baryonic Tully-Fisher relation?

**Method**:
1. Compute implied λ_g = αGM_b/v² for each galaxy
2. Fit power law: λ_g = k × M_b^γ
3. Check if γ ≈ 0.5 (Tully-Fisher prediction)
4. Check if v⁴ ∝ M_b (equivalent test)

**Key Metrics**:
- `gamma_fit`: Power-law index (expect 0.5)
- `TF_slope`: Tully-Fisher slope (expect 1.0)
- `scatter_dex`: Scatter in log(λ_g)

**Output**:
- `tully_fisher_results.json`
- `tully_fisher_scaling_test.png`

### Test 2: Power-Law Optimization

**File**: `optimize_power_law.py`

**Question**: What is the best-fit power law λ ~ M_b^α × v^β × R^γ?

**Method**:
1. Define general power law with 3 exponents + scale factor
2. Optimize to minimize scatter around fitted ℓ₀ = 4.993 kpc
3. Use global optimization (differential evolution)
4. Interpret physical meaning of exponents

**Key Metrics**:
- `alpha_M`, `alpha_v`, `alpha_R`: Exponents
- `scale`: Overall normalization
- `scatter_dex`: Residual scatter

**Output**:
- `power_law_fits/optimized_params.json`
- `power_law_fits/optimized_power_law.png`

**Physical Interpretation**:
- α_M ≈ 0.5, α_v ≈ -2 → Tully-Fisher scaling
- α_M ≈ 1, α_v ≈ -2 → GM/v² (gravitational radius)
- α_R ≈ 1 → Disk scale dominates
- α_R ≈ 0 → Universal scale

### Test 3: Scale Hypothesis Library

**File**: `scale_finder.py`

**Question**: Which physical scales from dimensional analysis match observations?

**Method**:
1. Define library of ~15 scale hypotheses
2. Compute predicted ℓ₀ for each galaxy under each hypothesis
3. Compare to fitted value (4.993 kpc)
4. Rank by scatter, BIC, correlation

**Hypothesis Categories**:
1. **Simple density scales** (baseline)
   - Virial density: ℓ ~ c/√(Gρ_virial)
   
2. **Tully-Fisher inspired**
   - Direct TF: ℓ ~ GM_b/v²
   
3. **Multi-scale combinations**
   - Disk scale height: ℓ ~ σ²/(πGΣ)
   - Crossing time: ℓ ~ σ_v × (R_disk/v_circ)
   - Geometric mean: ℓ ~ √(R_disk × h_disk)
   
4. **Dynamical time scales**
   - Orbital period: ℓ ~ 2πR_disk
   - Jeans length: ℓ ~ σ/√(Gρ)
   
5. **Gravitational radius scales**
   - Gravitational: ℓ ~ GM/v²
   - Schwarzschild (scaled): ℓ ~ β(GM/c²)
   
6. **Hybrid scales**
   - Hybrid: ℓ ~ (GM/v²)^0.5 × R_disk^0.5
   - Dispersion-modulated: ℓ ~ R_disk × (σ/v_circ)
   
7. **Empirical power laws**
   - Various combinations with fixed exponents

**Key Metrics**:
- `scatter_dex`: Scatter around fitted value
- `BIC`: Bayesian Information Criterion
- `correlation_with_Mb`: Correlation with baryonic mass

**Output**:
- `scale_tests/scale_test_results.json`
- `scale_tests/<hypothesis>_diagnostic.png` (one per hypothesis)

## Understanding the Results

### Smoking Gun: γ ≈ 0.5

If the Tully-Fisher test finds **γ ≈ 0.5**, this is strong evidence that:
- λ_g ∝ √M_b (mass-dependent scale)
- Σ-Gravity naturally reproduces BTFR
- The coherence length is NOT universal

### Best-Fit Power Law

The optimized exponents tell you:
- **α_M**: How λ scales with mass
- **α_v**: How λ scales with velocity
- **α_R**: How λ scales with disk size

Compare to theoretical expectations:
- Tully-Fisher: (0.5, -2, 0)
- Gravitational radius: (1, -2, 0)
- Disk-scale: (0, 0, 1)

### Winning Physical Scale

The hypothesis with **lowest scatter** is the best match. Check:
1. Is it physically motivated?
2. Does it match the power-law exponents?
3. Does it have predictive power for new systems?

## Expected Outcomes

### Scenario A: Tully-Fisher Confirmed (γ ≈ 0.5)

**Interpretation**: The coherence length is intrinsically mass-dependent.

**Implications**:
- v⁴ ∝ M_b emerges naturally
- Universal RAR from non-universal λ
- Need to explain why λ ∝ √M_b

**Next Steps**: Connect to fundamental theory (why √M_b?)

### Scenario B: Universal Scale (γ ≈ 0)

**Interpretation**: Single coherence length for all galaxies.

**Implications**:
- λ ≈ constant ≈ 4.993 kpc
- Tully-Fisher emerges differently
- Easier to connect to fundamental physics

**Next Steps**: Identify which universal scale (disk height? crossing time?)

### Scenario C: Complex Scaling

**Interpretation**: λ depends on multiple parameters.

**Implications**:
- More complex than simple dimensional analysis
- May need additional physics
- Check hybrid models

**Next Steps**: Look at hypothesis library for multi-parameter scales

## Files Generated

```
GravityWaveTest/
├── README.md (this file)
├── prepare_sparc_data.py
├── test_tully_fisher_scaling.py
├── optimize_power_law.py
├── scale_finder.py
├── run_all_tests.py
├── tully_fisher_results.json
├── tully_fisher_scaling_test.png
├── power_law_fits/
│   ├── optimized_params.json
│   └── optimized_power_law.png
├── scale_tests/
│   ├── scale_test_results.json
│   └── <hypothesis>_diagnostic.png (×15)
└── SCALE_FINDING_REPORT.md
```

## Requirements

```bash
pip install numpy pandas scipy matplotlib
```

All packages should already be installed if you're running the main Σ-Gravity codebase.

## Troubleshooting

### "Could not find SPARC data file"

Run `prepare_sparc_data.py` first. If it still fails, manually check:
- Do you have `many_path_model/data/sparc_masses.csv`?
- Or any file with galaxy masses and velocities?

Edit `prepare_sparc_data.py` line 25 to point to your data.

### "Missing required columns"

The SPARC data needs at minimum:
- Baryonic mass (M_sun)
- Flat rotation velocity (km/s)
- Disk scale length (kpc)

Edit `prepare_sparc_data.py` to map your column names.

### Tests run but results are nonsense

Check:
1. Are your fitted values correct? (ℓ₀ = 4.993 kpc, A = 0.591)
2. Are units correct in SPARC data? (M_sun, km/s, kpc)
3. Are there NaN values in critical columns?

## Comparison to Period-Counter Approach

This scale-finding suite answers: **"What is λ?"**

The period-counter approach (from your documents) answers: **"Does M(N) = 1 + μN work?"**

### Recommended Strategy

1. **Week 1**: Run period-counter test (faster, more decisive)
   - Test if enhancement scales with N = R/λ
   - Check BTFR slope prediction
   
2. **Week 2**: Run this scale-finding suite (if period-counter works)
   - Understand why λ has the fitted value
   - Connect to physical mechanisms

See the "Head-to-Head Comparison" section in the original document for details.

## Citation

If you use this suite in your research, please cite:

```
Σ-Gravity Scale-Finding Test Suite
Part of the Σ-Gravity Project
https://github.com/[your-repo]/sigmagravity
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

---

**Good luck with your scale hunting!** 🔍✨

