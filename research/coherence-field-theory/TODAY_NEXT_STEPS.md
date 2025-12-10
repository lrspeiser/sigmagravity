# Today / Next 3 Days: Explicit Checklist

**Date**: November 19, 2025  
**Status**: Phase 0 Complete ✅  
**Goal**: Scale up to statistically meaningful sample

## ✅ Phase 0 Complete

**Achievement**: Coherence wins 5/7 galaxies (71%), mean ratio = 0.584

### Completed Today
- [x] Enhanced SPARC fitter with NFW comparison
- [x] Fit 7 diverse galaxies (DDO154, DDO168, NGC2403, NGC6946, ESO079-G014, DDO064, CamB)
- [x] CSV export with all parameters
- [x] Publication-quality comparison plots (4 figures)
- [x] Results analysis and summary

**Files Created**:
- `outputs/sparc_fit_summary.csv` - Results table
- `outputs/chi_squared_scatter.png` - Scatter plot
- `outputs/chi_squared_ratio_histogram.png` - Ratio distribution
- `outputs/parameter_trends.png` - Parameter trends
- `outputs/sparc_comparison_summary.png` - Comprehensive summary
- `analysis/plot_sparc_comparison.py` - Plotting code

---

## Next 3 Days: Explicit Checklist

### Day 1: Expand Sample

- [ ] **Run fitter on 10-15 galaxies**
  ```bash
  python -c "from galaxies.fit_sparc_enhanced import fit_multiple_galaxies; \
  fit_multiple_galaxies(['DDO154','DDO168','NGC2403','NGC6946','ESO079-G014','DDO064','CamB', \
  'DDO161','DDO170','NGC925','NGC3198','NGC6503'], 'outputs')"
  ```
  
- [ ] **Regenerate comparison plots**
  ```bash
  python analysis/plot_sparc_comparison.py
  ```

- [ ] **Inspect CSV results**
  - Check for outliers
  - Identify galaxies with poor fits (χ²_red > 10)
  - Note parameter trends

### Day 2: Analyze and Improve

- [ ] **Analyze NGC2403/NGC6946**
  - Why does NFW fit better? (1.5-1.7×)
  - Try adding bulge component explicitly
  - Try different coherence halo profiles
  - Document findings

- [ ] **Try Burkert halo baseline**
  - Add Burkert halo option to fitter
  - Compare: Coherence vs NFW vs Burkert
  - Focus on LSB galaxies (where Burkert typically better)

- [ ] **Improve NFW implementation** (if needed)
  - Try even wider bounds
  - Test different r_s calculation methods
  - Verify fairness

### Day 3: Statistics and Documentation

- [ ] **Statistical tests**
  - Kolmogorov-Smirnov test: coherence vs NFW χ²_red distributions
  - T-test or Wilcoxon rank-sum test
  - Compute confidence intervals

- [ ] **Write 1-2 page results note**
  - Summary of findings
  - Table: all galaxy fits
  - Key plots: scatter, histogram, trends
  - Statistical analysis
  - Conclusions and next steps

---

## Immediate Commands

### Run Fits on Expanded Sample

```bash
cd coherence-field-theory

# Fit 12 galaxies
python -c "
from galaxies.fit_sparc_enhanced import fit_multiple_galaxies
galaxies = ['DDO154','DDO168','NGC2403','NGC6946','ESO079-G014','DDO064','CamB',
            'DDO161','DDO170','NGC925','NGC3198','NGC6503']
results = fit_multiple_galaxies(galaxies, 'outputs')
print(f'Fitted {len(results)} galaxies')
"
```

### Generate Updated Plots

```bash
python analysis/plot_sparc_comparison.py
```

### Check Results

```python
import pandas as pd
df = pd.read_csv('outputs/sparc_fit_summary.csv')
print(df[['galaxy', 'chi2_red_coherence', 'chi2_red_nfw', 'ratio', 'winner']])
print(f"\nCoherence wins: {(df['winner'] == 'Coherence').sum()}/{len(df)}")
print(f"Mean ratio: {df['ratio'].mean():.3f}")
```

---

## Success Criteria (Next 3 Days)

### Minimum
- [ ] 10+ galaxies fitted
- [ ] Coherence wins ≥60% (currently 71%)
- [ ] Mean ratio < 1.0 (currently 0.584)
- [ ] Updated plots generated

### Good
- [ ] 15+ galaxies fitted
- [ ] Coherence wins ≥65%
- [ ] Statistical tests complete
- [ ] Results note written

### Excellent
- [ ] 20+ galaxies fitted
- [ ] Coherence wins ≥70%
- [ ] All outliers understood
- [ ] Publication-ready summary

---

## Files to Create/Update

### Code
- [ ] `galaxies/fit_burkert.py` - Add Burkert halo option
- [ ] `analysis/statistical_tests.py` - Statistical analysis

### Data
- [ ] `outputs/sparc_fit_summary.csv` - Updated with more galaxies
- [ ] `outputs/statistical_analysis.csv` - Test results

### Plots
- [ ] Updated comparison plots with larger sample
- [ ] Statistical test visualizations

### Documentation
- [ ] `INITIAL_RESULTS_NOTE.md` - 1-2 page summary
- [ ] Update `RESEARCH_PLAN_2-4_WEEKS.md` with progress

---

## Key Questions to Answer

1. **Why does NFW fit NGC2403/NGC6946 better?**
   - Missing bulge component?
   - Wrong coherence halo profile?
   - Need multi-component halo?

2. **Does coherence consistently win for dwarfs?**
   - Current: 4/4 dwarfs (100%)
   - Need: More dwarfs to confirm

3. **Are parameter trends physical?**
   - R_c / R_disk ≈ 1-2 (confirmed so far)
   - ρ_c0 scales with mass? (check)

4. **Is NFW being treated fairly?**
   - Improved r_s calculation
   - Wider c bounds (1-30)
   - Still need: Burkert comparison

---

## Notes

**Current Status**: Very promising!
- 71% win rate is above 60% target
- Mean ratio 0.584 is well below 1.0
- Several excellent fits (χ²_red < 1)

**Next Priority**: Scale up to confirm trend, then implement screening

**Timeline**: Phase 1 complete by end of Week 2

