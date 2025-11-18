# Time-Coherence Kernel: Phase Completion Status

## Phase 1: Lock in Fiducial Kernel ✅ COMPLETE

### Completed:
- ✅ Added `--params-json` and `--out-*` arguments to all test scripts
- ✅ `test_mw_coherence.py` - accepts `--params-json` and `--out-json`
- ✅ `test_sparc_coherence.py` - accepts `--params-json` and `--out-csv`
- ✅ `test_cluster_coherence.py` - accepts `--params-json` and `--out-json`
- ✅ Regenerated canonical baseline:
  - `mw_coherence_canonical.json` - MW results with fiducial params
  - `sparc_coherence_canonical.csv` - SPARC results (running in background)

### Usage:
```bash
# MW
python time-coherence/test_mw_coherence.py \
  --params-json time-coherence/time_coherence_fiducial.json \
  --out-json time-coherence/mw_coherence_canonical.json

# SPARC
python time-coherence/test_sparc_coherence.py \
  --params-json time-coherence/time_coherence_fiducial.json \
  --out-csv time-coherence/sparc_coherence_canonical.csv

# Clusters
python time-coherence/test_cluster_coherence.py \
  --params-json time-coherence/time_coherence_fiducial.json \
  --out-json time-coherence/cluster_coherence_canonical.json
```

---

## Phase 2: Analyze SPARC Outliers ✅ COMPLETE

### Completed:
- ✅ Created `analyze_sparc_outliers.py` - analyzes worst-performing galaxies
- ✅ Generated `sparc_outlier_morphology.csv` with morphology flags

### Key Findings:
- Worst 30 galaxies: mean ΔRMS = 56.21 km/s
- Higher σ_v: mean 30.92 km/s vs overall 18.45 km/s (1.68×)
- Mean bulge fraction: 0.126 (moderate)

### Next Steps:
- Implement morphology gates in `coherence_time_kernel.py`
- Re-run SPARC tests with morphology suppression

---

## Phase 3: Burr-XII Summary ⏳ PENDING

### Created:
- ✅ `fit_burr_from_time_coherence.py` - fits Burr-XII across SPARC sample

### To Run:
```bash
python time-coherence/fit_burr_from_time_coherence.py
```

This will create `burr_from_time_coherence_summary.json` with:
- Mean ell_0, A, p, n across galaxies
- Comparison to empirical Σ-Gravity values

---

## Phase 4: Solar System Safety Test ✅ COMPLETE (with note)

### Completed:
- ✅ Created `test_solar_system_coherence.py`
- ⚠️ **Issue Found**: K ~ 0.1 at Solar System scales (target: < 1e-12)

### Result:
- Max K: 9.919e-02 (too large!)
- K at 1 AU: 8.734e-03
- K at 100 AU: 7.459e-02

### Action Required:
- Need additional suppression at small scales (R < ~0.1 kpc)
- May need to add scale-dependent gate: `K → 0 as R → 0`

---

## Phase 5: Cluster Analysis ⏳ PENDING

### Created:
- ✅ `analyze_cluster_coherence.py` - creates summary table

### To Run:
```bash
python time-coherence/test_cluster_coherence.py \
  --params-json time-coherence/time_coherence_fiducial.json \
  --out-json time-coherence/cluster_coherence_canonical.json

python time-coherence/analyze_cluster_coherence.py
```

---

## Summary

### ✅ Completed Phases:
1. **Phase 1**: Fiducial kernel locked in, CLI added to all tests
2. **Phase 2**: Outlier analysis complete, morphology data extracted
4. **Phase 4**: Solar System test created (issue identified)

### ⏳ Pending:
3. **Phase 3**: Run Burr-XII summary fit
5. **Phase 5**: Run cluster analysis

### 🔧 Issues to Address:
- **Solar System**: K too large at small scales - needs suppression
- **SPARC Outliers**: Need to implement morphology gates

### 📁 New Files Created:
1. `analyze_sparc_outliers.py` - Outlier morphology analysis
2. `fit_burr_from_time_coherence.py` - Burr-XII summary fitter
3. `test_solar_system_coherence.py` - Solar System safety test
4. `analyze_cluster_coherence.py` - Cluster summary generator
5. `PHASE_COMPLETION_STATUS.md` - This file

---

## Next Immediate Steps:

1. **Fix Solar System suppression** - Add R-dependent gate to kernel
2. **Implement morphology gates** - Add bar/warp/bulge suppression
3. **Run Burr-XII summary** - Complete Phase 3
4. **Run cluster analysis** - Complete Phase 5
5. **Re-run SPARC with morphology gates** - Verify improvement

