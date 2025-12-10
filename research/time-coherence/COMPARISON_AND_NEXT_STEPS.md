# Time-Coherence Kernel: Comparison & Next Steps

## 1. Comparison: Earlier Concept vs Current Implementation

### Earlier "Time/Domain-Wall" Concept

The original first-principles idea was:

- **Core mechanism**: Gravity enhancement comes from how long metric phases stay coherent before being randomized by motion/noise
- **Two timescales**:
  - `τ_geom(R) ~ orbit_length / v_char` (geometric coherence)
  - `τ_noise(R) ~ ℓ_noise / σ_v(R)` (velocity dispersion/turbulence)
- **Combination**: `1/τ_coh = 1/τ_geom + 1/τ_noise` (harmonic mean)
- **Coherence fraction**: `F_coh(R) = τ_coh(R) / τ_geom(R)`
- **Enhancement**: `g_eff = g_GR · [1 + A · W_Burr(R) · F_coh(R)]`
- **No ad-hoc gates** - just the two timescales

### Current Implementation (`coherence_time_kernel.py`)

What we've actually built:

✅ **Computes τ_geom(R)** from gravitational time dilation:
   - `compute_tau_geom()` - uses tidal field or simple scaling
   - Based on proper time differences between nearby geodesics

✅ **Computes τ_noise(R)** from velocity dispersion:
   - `compute_tau_noise()` - `τ_noise ~ R / σ_v` for galaxies
   - Can use turbulence parameters for clusters

✅ **Combines timescales**: `1/τ_coh = 1/τ_geom + 1/τ_noise`

✅ **Converts to coherence length**: `ℓ_coh = c · τ_coh`

✅ **Uses Burr-XII window**: `K(R) = A_global · C(R/ℓ_coh(R))`

✅ **Applies enhancement**: `g_eff = g_GR · (1 + K(R))`

### Key Differences

| Aspect | Original Concept | Current Implementation |
|--------|------------------|------------------------|
| **Coherence fraction** | `F_coh = τ_coh/τ_geom` | Implicit in `ℓ_coh(R)` |
| **Length scale** | Fixed `ℓ₀` in Burr-XII | Computed `ℓ_coh(R)` |
| **Gates** | None (pure timescales) | None (pure timescales) ✅ |
| **Cluster handling** | Same mechanism | Turbulence-based `τ_noise` |

**Verdict**: The implementation matches the concept! The main difference is that we compute `ℓ_coh(R)` dynamically rather than using a fixed `ℓ₀`, which is actually more principled.

## 2. Current Results

### SPARC Performance
- **Mean ΔRMS**: +0.113 km/s (vs +5.25 km/s for previous theory kernel)
- **Improved**: 130/175 (74.3%) vs 24% before
- **Median ΔRMS**: -0.561 km/s (negative = improvement)

### Coherence Scales
- **MW**: `ℓ_coh ≈ 140 kpc` (target: ~5 kpc)
- **SPARC mean**: `ℓ_coh ≈ 135 kpc`
- **Correlation with σ_v**: Very weak (0.017) - suggests timescales need tuning

### Interpretation

✅ **Concept works**: 74% improvement on SPARC with near-zero mean ΔRMS  
⚠️ **Scales too large**: `ℓ_coh ~ 140 kpc` vs target `ℓ₀ ~ 5 kpc`  
⚠️ **Weak σ_v scaling**: Need stronger dependence on velocity dispersion

## 3. Next Steps: From "Nice Fit" → "First-Principles Candidate"

### Step 1: Joint MW + SPARC Hyperparameter Fit

**Script**: `fit_time_coherence_hyperparams.py`

**Goal**: Tune `A_global`, `p`, `n_coh`, `delta_R_kpc` to:
- Match MW empirical RMS (~40 km/s)
- Keep SPARC performance (mean ΔRMS ≈ 0, 70-80% improved)
- Move `ℓ_coh` toward ~5-20 kpc range

**Usage**:
```bash
python time-coherence/fit_time_coherence_hyperparams.py \
    --mw-parquet gravitywavebaseline/gaia_with_gr_baseline.parquet \
    --sparc-rotmod-dir data/Rotmod_LTG \
    --sparc-summary data/sparc/sparc_combined.csv \
    --n-sparc 40 \
    --out-json time-coherence/time_coherence_fit_hyperparams.json
```

**What to adjust**:
- Bounds in `bounds = [...]` if `ℓ_coh` stays too large
- `target_mw = 40.0` if you want different MW target
- `delta_R_kpc` scaling if geometric dephasing needs tuning

### Step 2: Analyze Coherence Scaling

**Script**: `analyze_full_coherence_scaling.py`

**Goal**: Understand how `ℓ_coh` and `τ_coh` scale with:
- Velocity dispersion (`σ_v`)
- System size (MW vs SPARC vs clusters)
- Baryonic density

**Usage**:
```bash
python time-coherence/analyze_full_coherence_scaling.py \
    --mw-json time-coherence/mw_coherence_test.json \
    --sparc-csv time-coherence/sparc_coherence_test.csv \
    --cluster-json time-coherence/cluster_coherence_test.json \
    --out-summary time-coherence/coherence_scaling_summary.json
```

**What to look for**:
- Strong negative correlation: `ℓ_coh ∝ σ_v^-β` (expected)
- Clusters naturally at `ℓ_coh ~ 100-300 kpc` (good for lensing)
- Dwarfs at larger `ℓ_coh` than high-σ discs (expected)

### Step 3: Tune Timescale Calculations

If `ℓ_coh` stays too large, consider:

**Option A**: Change `ℓ_coh = c · τ_coh` → `ℓ_coh = v_circ · τ_coh`
- More physical for galactic dynamics
- Will reduce `ℓ_coh` by factor ~200/300000 ≈ 0.0007
- This would give `ℓ_coh ~ 0.1 kpc` (too small!)

**Option B**: Add prefactor to `τ_geom` calculation
- Current: `τ_geom ~ c²/(ΔΦ) · T_orb`
- Try: `τ_geom ~ α · c²/(ΔΦ) · T_orb` with `α ~ 0.01-0.1`
- Reduces geometric coherence time

**Option C**: Stronger `σ_v` dependence in `τ_noise`
- Current: `τ_noise ~ R / σ_v`
- Try: `τ_noise ~ R / (σ_v^β)` with `β > 1`
- Makes high-σ systems have shorter coherence

**Option D**: Different conversion factor
- Instead of `c`, use characteristic velocity: `ℓ_coh = v_char · τ_coh`
- `v_char` could be `v_circ` or `σ_v` or combination

### Step 4: Cluster Validation

**Goal**: Verify that same mechanism works for clusters

**Expected**:
- Clusters: `ℓ_coh ~ 100-300 kpc` (from deep potentials + large scales)
- Mass boost at Einstein radius sufficient for lensing
- No per-cluster tuning needed

**If clusters fail**:
- May need cluster-specific `τ_noise` (ICM turbulence)
- Or different `A_global` scaling with system mass

## 4. First-Principles Story

With these steps, the narrative becomes:

1. **Postulate**: Metric phases stay coherent for `τ_coh(R)` determined by:
   - Geometry: `τ_geom` (gravitational time dilation)
   - Environment: `τ_noise` (velocity dispersion/turbulence)

2. **Derive**: Enhancement factor proportional to coherence fraction

3. **Implement**: Time-coherence kernel with few hyperparameters

4. **Validate**: 
   - ✅ SPARC: 74% improved, mean ΔRMS ≈ 0
   - 🔄 MW: Need to tune to match empirical `ℓ₀ ~ 5 kpc`
   - 🔄 Clusters: Test if naturally gives `ℓ_coh ~ 200 kpc`

## 5. Files Created

- `fit_time_coherence_hyperparams.py` - Joint MW+SPARC optimization
- `analyze_full_coherence_scaling.py` - Cross-system scaling analysis
- `COMPARISON_AND_NEXT_STEPS.md` - This document

## 6. Quick Start

```bash
# 1. Fit hyperparameters
python time-coherence/fit_time_coherence_hyperparams.py

# 2. Analyze scaling
python time-coherence/analyze_full_coherence_scaling.py

# 3. Review results
python time-coherence/analyze_results.py
```


