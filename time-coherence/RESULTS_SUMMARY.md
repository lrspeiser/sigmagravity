# Time-Coherence Kernel: Results Summary

## Phase 1: Fiducial Kernel Baseline ✅

### MW Results (Canonical)
- **ell_coh_mean**: 0.945 kpc
- **K_max**: 0.661
- **K_mean**: 0.661
- **RMS**: (needs MW data fix)

### SPARC Results (Canonical - Running)
- Processing with fiducial parameters
- Morphology gates: ENABLED

---

## Phase 2: Outlier Analysis ✅

### Worst 30 Galaxies:
- **Mean delta_RMS**: 56.21 km/s
- **Range**: 27.17 - 91.74 km/s
- **Mean sigma_v**: 30.92 km/s (1.68× overall mean)
- **Mean bulge_frac**: 0.126

### Key Outliers:
1. NGC5005: 91.74 km/s, σ_v=39.6 km/s
2. UGC11914: 88.65 km/s, σ_v=42.6 km/s
3. NGC6195: 82.15 km/s, σ_v=37.7 km/s

**Pattern**: High σ_v galaxies dominate outliers (1.68× average)

---

## Phase 3: Burr-XII Mapping ✅

### Results from 50 SPARC Galaxies:

**Theory Kernel → Burr-XII Fit:**
- **ell_0**: 1.69 ± 1.50 kpc (median: 1.25 kpc)
- **Range**: 0.18 - 7.11 kpc
- **A**: 0.647 ± 0.171
- **p**: 0.100 (hitting lower bound)
- **n**: 2.000 (hitting upper bound)

**Fit Quality:**
- **Mean relative RMS**: 2.63%
- **Median relative RMS**: 2.74%
- **Max relative RMS**: ~4.5%

**Comparison to Empirical Σ-Gravity:**
- **ell_0**: 1.69 kpc (theory) vs 5.0 kpc (empirical) - **smaller by factor of ~3**
- **A**: 0.647 (theory) vs 0.6 (empirical) - **excellent match**
- **p**: 0.100 (theory) vs 0.757 (empirical) - **different** (hitting fit boundary)
- **n**: 2.000 (theory) vs 0.5 (empirical) - **different** (hitting fit boundary)

**Interpretation:**
- The time-coherence kernel produces Burr-XII-like shapes
- Fit quality is excellent (<3% RMS)
- ell_0 is systematically smaller than empirical (1.7 vs 5 kpc)
- p and n parameters hitting bounds suggests the functional form may need adjustment

---

## Phase 4: Solar System Safety ⚠️

### Before Small-Scale Suppression:
- **Max K**: 9.919e-02 (too large!)
- **K at 1 AU**: 8.734e-03
- **K at 100 AU**: 7.459e-02
- **Status**: FAILED

### After Small-Scale Suppression (R < 1 pc):
- Testing now...

**Note**: Added suppression factor `(R/R_min)²` for R < 0.001 kpc (1 pc)

---

## Phase 5: Cluster Analysis ⏳

### Status: Running in background
- Testing clusters with fiducial parameters
- Will generate summary table

---

## Key Findings So Far

### ✅ Strengths:
1. **Burr-XII mapping works**: Theory kernel fits Burr-XII with <3% RMS
2. **Amplitude matches**: A ≈ 0.65 matches empirical A ≈ 0.6
3. **Outliers identified**: High σ_v galaxies are the problem
4. **Morphology gates implemented**: Ready to suppress outliers

### ⚠️ Issues:
1. **ell_0 too small**: 1.7 kpc vs empirical 5 kpc (factor of ~3)
2. **Solar System**: K too large before suppression (needs verification)
3. **p, n parameters**: Hitting fit boundaries (may need functional form adjustment)

### 🔧 Next Steps:
1. Verify Solar System suppression works
2. Check SPARC results with morphology gates
3. Consider adjusting α_length to bring ell_0 closer to 5 kpc
4. Investigate why p, n hit boundaries in Burr-XII fits

