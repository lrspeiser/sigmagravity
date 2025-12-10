# Time-Coherence Kernel: Implementation Plan

## Status Summary

✅ **Completed:**
1. Added backreaction cap to `coherence_time_kernel.py`
2. Created outlier identification script (`identify_sparc_outliers.py`)
3. Created grid scan framework (`run_grid_scan.py`)
4. Created fiducial parameter creation script (`create_fiducial_params.py`)

🔧 **In Progress:**
- Grid scan needs fixes to MW/SPARC test functions
- Need to complete full grid scan with backreaction cap testing

📋 **Remaining Tasks:**

### 1. Fix Grid Scan Implementation
- Fix MW test function to properly load and process MW data
- Fix SPARC test function to properly process galaxies
- Run full grid scan with and without backreaction cap

### 2. Select Fiducial Parameters
- Run grid scan with backreaction_cap = 10.0
- Select best combination based on:
  - MW RMS < 75 km/s (target ~40-70 km/s)
  - SPARC mean ΔRMS ≤ 0 (ideally -1 to -3 km/s)
  - SPARC fraction improved ≥ 70%
  - SPARC worst ΔRMS < 60 km/s
  - Cluster boosts in [1.5×, 10×]

### 3. Update Test Scripts to Use Fiducial Parameters
- Modify `test_mw_coherence.py` to load fiducial params by default
- Modify `test_sparc_coherence.py` to load fiducial params by default
- Modify `test_cluster_coherence.py` to load fiducial params by default

### 4. Map Theory Kernel to Burr-XII
- Create script to fit Burr-XII to K_theory(R) for sample galaxies
- Compare fitted ℓ₀, p, n_coh to empirical values
- Document the mapping

### 5. Cluster Shape Analysis (Optional)
- Plot Σ_baryon(R), Σ_eff(R), κ(R) for fiducial kernel
- Compare to published lensing profiles

## Key Findings So Far

### Outlier Analysis
- Worst 20 galaxies have mean ΔRMS = 65.6 km/s
- They have higher σ_v (mean 33.11 vs overall 18.45 km/s, 1.79×)
- K_max values are reasonable (0.618-0.717), so problem isn't excessive K
- Worst galaxies: NGC5005, UGC11914, NGC6195, etc. (high-mass spirals)

### Current Parameters
- α_length = 0.037 (brings scales to ~1-2 kpc)
- β_sigma = 1.5 (stronger σ_v suppression)
- backreaction_cap = 10.0 (proposed universal limit)

## Next Steps

1. **Fix grid scan** - Ensure MW and SPARC tests work correctly
2. **Run comprehensive scan** - Test (α, β) combinations with cap=10.0
3. **Select fiducial** - Choose best combination
4. **Update defaults** - Make all tests use fiducial params
5. **Document mapping** - Show theory → empirical kernel connection

