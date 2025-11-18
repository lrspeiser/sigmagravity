# Time-Coherence Kernel Stabilization Plan

## Status: In Progress

### Step 1: Identify Outliers ✅ COMPLETE

**Results:**
- 32 outliers (18.3%) with ΔRMS > 25 km/s
- Worst: NGC5005 with 91.74 km/s
- **Key finding**: 30-50 km/s σ_v bin has 56.7% outliers (17/30 galaxies)
- Many outliers have very small ℓ_coh (0.15-0.68 kpc), suggesting kernel is too strong

**Outlier patterns:**
- High σ_v galaxies (30-50 km/s) are most problematic
- Extended galaxies (R_max > 70 kpc) also problematic
- Common offenders: NGC5005, UGC11914, NGC6195, NGC2955, NGC0891, NGC7331

### Step 2: Grid Scan (α, β, K_max) 🔄 IN PROGRESS

**Grid parameters:**
- `alpha_length`: [0.03, 0.037, 0.045] (coherence length prefactor)
- `beta_sigma`: [1.5, 1.8, 2.0] (σ_v suppression strength)
- `backreaction_cap`: [None, 5.0, 10.0, 15.0] (universal K_max limit)

**Total combinations:** 36

**Acceptance criteria:**
- MW RMS < 75 km/s (preserve MW improvement)
- Cluster mass boost: 1.5× to 12× (preserve cluster performance)
- SPARC worst ΔRMS < 80 km/s
- SPARC fraction improved ≥ 60%

**Ranking:** Minimize SPARC mean ΔRMS, maximize fraction improved

### Step 3: Backreaction Cap ✅ IMPLEMENTED

**Implementation:**
- Added `backreaction_cap` parameter to `compute_coherence_kernel()`
- Universal limit: `K = min(K_raw, backreaction_cap)`
- Physically motivated: metric fluctuations decohere when enhancement becomes too large
- Cluster tests show K_E ~ 0.5-9, so cap ~5-15 is reasonable

### Step 4: Fiducial Kernel (Next)

**Goal:** Lock in one parameter set as "the fiducial time-coherence kernel"

**Criteria:**
- MW outer disk RMS: 40-70 km/s (well below GR's 111 km/s)
- Cluster boosts at R_E: 1.5× to 10×
- SPARC:
  - Mean ΔRMS ≤ 0 (ideally -1 to -3 km/s)
  - ≥70% galaxies improved
  - Worst ΔRMS < 60 km/s

**Action:** Once grid scan completes, select best combination and save to `time_coherence_hyperparams_fiducial.json`

### Step 5: Map to Burr-XII (Next)

**Goal:** Show time-coherence kernel reproduces empirical Σ-Gravity Burr-XII kernel

**Method:**
1. Sample K_theory(R) from fiducial time-coherence kernel for:
   - MW outer disk
   - NGC2403 (canonical spiral)
   - NGC5055 (high-mass spiral)
   - DDO154 (LSB dwarf)

2. Fit Burr-XII + σ-gate template to K_theory(R):
   - Best-fit ℓ₀, p, n_coh
   - RMS between K_theory and K_empirical

3. If match is good (RMS < 10%), claim:
   > "The Burr-XII × Q-factor kernel used in §X is not ad-hoc; it emerges as an excellent approximation to the Green's function of a time-coherence governed metric with τ_geom ∼ R/v_circ and τ_noise ∼ R/σ_v^β."

## Current Baseline (Before Grid Scan)

**Default parameters** (α=0.037, β=1.5, no cap):
- MW: RMS = 66.40 km/s (improvement: 44.97 km/s)
- SPARC: Mean ΔRMS = +5.906 km/s, 64.0% improved
- Cluster: Mass boosts 1.6×-10×

**Fitted parameters** (from previous optimization):
- MW: RMS = 66.40 km/s
- SPARC: Mean ΔRMS = +12.038 km/s, 60.6% improved

**Target:** Find combination that brings SPARC mean ΔRMS ≤ 0 while preserving MW/cluster performance.

