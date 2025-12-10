# GravityWaveTest Quick Start Guide

## 🚀 Get Results in 3 Steps

### Step 1: Prepare Your Data (1 minute)

```bash
cd GravityWaveTest
python prepare_sparc_data.py
```

This will look for your existing SPARC data and create `data/sparc/sparc_combined.csv`.

**If it fails**: Edit `prepare_sparc_data.py` line 25-28 to point to your SPARC files.

### Step 2: Run All Tests (5-10 minutes)

```bash
python run_all_tests.py
```

This runs all three test suites and generates a comprehensive report.

**Or run tests individually:**

```bash
# Fastest first (30 seconds) - Most important!
python test_tully_fisher_scaling.py

# Optimization (1-2 minutes)
python optimize_power_law.py

# Full library (2-5 minutes)
python scale_finder.py
```

### Step 3: Check Results

Look at:
1. **`SCALE_FINDING_REPORT.md`** - Executive summary
2. **`tully_fisher_results.json`** - Is γ ≈ 0.5?
3. **`tully_fisher_scaling_test.png`** - Visual diagnostics
4. **`power_law_fits/optimized_params.json`** - Best power law
5. **`scale_tests/scale_test_results.json`** - All hypotheses ranked

## 🎯 What to Look For

### The Smoking Gun: γ ≈ 0.5

Open `tully_fisher_results.json` and check:

```json
{
  "gamma_fit": 0.4823,  // ← Is this close to 0.5?
  "TF_slope": 0.9876,   // ← Is this close to 1.0?
  ...
}
```

**If γ ≈ 0.5 and TF_slope ≈ 1.0**:
- ✓✓ **STRONG EVIDENCE** that λ_g ∝ √M_b
- Your theory naturally reproduces Tully-Fisher!
- The coherence length is mass-dependent

**If γ ≈ 0 (and small scatter)**:
- ✓ **Universal scale** - same λ for all galaxies
- Easier to connect to fundamental physics
- Check which physical scale in hypothesis library matches

**If γ is something else**:
- More complex scaling
- Check power-law optimization results
- Look at hybrid scales in hypothesis library

### Best Power Law

Open `power_law_fits/optimized_params.json`:

```json
{
  "optimized_params": {
    "alpha_M": 0.487,   // Mass exponent
    "alpha_v": -1.973,  // Velocity exponent
    "alpha_R": 0.023,   // Radius exponent
    "scale": 5.124
  }
}
```

**Interpretation guide**:
- α_M ≈ 0.5, α_v ≈ -2, α_R ≈ 0 → **Tully-Fisher scaling** (λ ~ √M/v²)
- α_M ≈ 1, α_v ≈ -2, α_R ≈ 0 → **Gravitational radius** (λ ~ M/v²)
- α_M ≈ 0, α_v ≈ 0, α_R ≈ 1 → **Disk scale** (λ ~ R_disk)
- α_M ≈ 0, α_v ≈ 0, α_R ≈ 0 → **Universal constant**

### Winner from Hypothesis Library

Open `scale_tests/scale_test_results.json` and check the top-ranked hypotheses:

```json
{
  "results": [
    {
      "rank": 1,
      "name": "tully_fisher_direct",
      "scatter_dex": 0.0823,  // ← Lower is better
      "bic": 234.5,           // ← Lower is better
      ...
    },
    ...
  ]
}
```

The **rank 1** hypothesis is the best match to your fitted ℓ₀ = 4.993 kpc.

## 📊 Understanding the Plots

### `tully_fisher_scaling_test.png`

**Top row**:
- **Left**: λ_g vs M_baryon - should follow power law
- **Center**: Residuals - should scatter around zero
- **Right**: Histogram - shows distribution of λ_g values

**Bottom row**:
- **Left**: Baryonic Tully-Fisher (v⁴ vs M) - should be linear
- **Center**: λ_g vs velocity - check for trends
- **Right**: Consistency check - predicted vs observed velocities

**What you want to see**:
- Left plot: Data follows red fit line (not scattered)
- Residuals: Random scatter, no systematic trend with mass
- Histogram: Peaked distribution (not too broad)

### `power_law_fits/optimized_power_law.png`

Six diagnostic plots showing:
1. How predicted ℓ₀ varies with M, v, R
2. Residuals vs mass (should be random)
3. Histogram of predictions (should be peaked near target)
4. Q-Q plot (should be linear if residuals are normal)

**What you want to see**:
- Predictions cluster near horizontal red line (target)
- Residuals have no systematic trends
- Small scatter in histogram

### `scale_tests/<hypothesis>_diagnostic.png`

For each hypothesis, 6 panels showing:
- How predicted ℓ₀ varies with galaxy properties
- Residuals and distributions
- Tully-Fisher consistency check

Compare across hypotheses to see which looks most consistent.

## 🤔 Common Questions

### Q: My γ is 0.23, not 0.5. What does this mean?

**A**: Your coherence length has **weaker** mass-dependence than Tully-Fisher predicts (λ ~ M^0.23 instead of M^0.5). This could mean:
1. There's a nearly universal scale with small corrections
2. The relationship is more complex (check hybrid models)
3. There might be systematics in the data

Check the scatter - if it's small, the fit is still good even if γ ≠ 0.5.

### Q: Multiple hypotheses have similar scatter. How do I choose?

**A**: Use these criteria:
1. **Physical motivation**: Does it make sense?
2. **Simplicity**: Prefer simpler models (fewer parameters)
3. **BIC**: Lower is better (balances fit quality vs complexity)
4. **Predictive power**: Does it work on other systems (MW, clusters)?

### Q: The optimized power law has weird exponents (e.g., α_M = 1.7)

**A**: This suggests:
1. No simple dimensional analysis works perfectly
2. May need hybrid or multi-scale model
3. Could be fitting noise - check scatter and N_valid

If scatter is large (>0.5 dex), the model might be overfitting.

### Q: Can I test on my own fitted ℓ₀?

**A**: Yes! Edit these lines:

```python
# In test_tully_fisher_scaling.py, line 19:
fitted_ell0: float = 4.993  # ← Change this

# In optimize_power_law.py, line 163:
target_ell0=4.993  # ← Change this

# In scale_finder.py, line 553:
fitted_ell0=4.993  # ← Change this
```

Or pass as command-line argument (would need to add argparse).

### Q: How do I add my own hypothesis?

**A**: Edit `scale_finder.py`, add to `create_hypothesis_library()`:

```python
def my_custom_scale(row):
    """Your scale formula."""
    M_b = row['M_baryon'] * Msun_to_kg
    # ... your calculation ...
    return ell0_kpc

hypotheses.append(ScaleHypothesis(
    name="my_custom_scale",
    formula=my_custom_scale,
    description="What this scale represents",
    expected_range=(min_kpc, max_kpc)
))
```

Then re-run `python scale_finder.py`.

## 📝 Next Steps After Testing

### If Tully-Fisher is confirmed (γ ≈ 0.5):

1. **Paper**: Emphasize λ ∝ √M_b → v⁴ ∝ M naturally
2. **Theory**: Explain why λ has this scaling (dimensional analysis?)
3. **Prediction**: Test on other systems (clusters, MW satellites)

### If universal scale is found (γ ≈ 0):

1. **Paper**: Identify the physical origin of λ ≈ 5 kpc
2. **Theory**: Connect to disk scale height, crossing time, etc.
3. **Universality**: Explain why same λ works for all galaxies

### If complex scaling:

1. **Paper**: Present best-fit power law or hybrid model
2. **Theory**: Look for multi-parameter physical mechanism
3. **Validation**: Test on independent datasets

## 🐛 Troubleshooting

### Error: "No module named 'scipy'"

```bash
pip install scipy matplotlib pandas numpy
```

### Error: "Could not find SPARC data file"

Edit `prepare_sparc_data.py` line 25-28 to point to your data files.

### Error: "ZeroDivisionError" or "RuntimeWarning: invalid value"

Some galaxies have missing/invalid data. The tests should handle this gracefully. If tests crash:
1. Check for NaN values in SPARC data
2. Make sure M_baryon, v_flat, R_disk are all positive
3. Remove problematic galaxies manually if needed

### Plots look weird / scatter is huge

1. Check if ℓ₀ = 4.993 kpc is correct for your fit
2. Verify SPARC data units (M_sun, km/s, kpc)
3. Make sure you're using outer v_flat, not inner velocities

---

**Happy scale hunting!** If you find γ ≈ 0.5, you've got a **killer result**! 🎯

