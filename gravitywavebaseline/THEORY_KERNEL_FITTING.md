# Theory Kernel Fitting: Phase-T2 Implementation

This document describes the three comprehensive fitting scripts for the first-principles metric-fluctuation resonance kernel, designed to test whether a simple QFT-like fluctuation spectrum can match the empirically calibrated Σ-Gravity kernel across MW, SPARC galaxies, and clusters.

## Overview

The theory kernel is based on a λ-integral over a fluctuation spectrum:
```
K_th(R) = A_global * ∫ d(ln λ) P(λ) * C_res(λ,R) * C_coh(λ) * W_geom(R,λ) / λ
```

where:
- `P(λ) ~ (λ_ref/λ)^α * exp(-λ/λ_cut)`: Power-law fluctuation spectrum
- `C_res`: Resonance filter (Lorentzian in log-λ space)
- `C_coh`: Coherence cutoff `exp(-(λ/λ_coh)^2)`
- `W_geom`: Geometric weight (1 for λ<R, (R/λ)² for λ>R)
- Optional Burr-XII radial envelope: `1 - (1 + (R/ℓ₀)^p)^(-n)`

## Scripts

### 1. `run_theory_kernel_mw_fit.py`

**Purpose**: Fit theory kernel parameters to match the empirical Σ-Gravity kernel shape on the Milky Way.

**Key Features**:
- Enforces positive correlation with empirical kernel (penalty function)
- Optionally fits `Q_ref` as a free parameter
- Multiple optimization runs with different seeds for robustness
- Broader parameter bounds (allows negative α for extended spectral models)

**Usage**:
```bash
python gravitywavebaseline/run_theory_kernel_mw_fit.py \
  --baseline-parquet gravitywavebaseline/gaia_with_gr_baseline.parquet \
  --mw-fit-json gravitywavebaseline/metric_resonance_mw_fit.json \
  --r-min 12.0 \
  --r-max 16.0 \
  --sigma-v 30.0 \
  --out-json gravitywavebaseline/theory_metric_resonance_mw_fit.json \
  --require-positive-corr \
  # --no-Q-ref  # Uncomment to disable Q_ref fitting
```

**Output**: JSON file with:
- `theory_fit_params`: Best-fit parameters (A_global, alpha, lam_coh_kpc, lam_cut_kpc, burr_ell0_kpc, Q_ref, phase_sign)
- `corr_K_emp_theory`: Correlation coefficient
- `chi2_K`: Chi-squared for kernel shape
- `optimization_success`: Whether optimization converged

**What it tells you**:
- If even with wide bounds you cannot get corr>0.95 and small χ², the simple metric-fluctuation ansatz is structurally incompatible with the empirical Σ-kernel on the MW.
- If you can get good agreement, you have a theory parameter vector θ* that reproduces the Σ-Gravity kernel in the MW band.

---

### 2. `fit_theory_kernel_clusters_amp.py`

**Purpose**: Fit a single cluster amplitude scaling factor `A_cluster` to match observed Einstein radius masses, given MW-fitted theory kernel parameters.

**Key Features**:
- Searches over `A_cluster` scaling factor (0.01 to 1000 by default)
- Computes normalized chi-squared across all clusters
- Reports `A_cluster / A_galaxy` ratio for comparison to empirical Σ-Gravity

**Usage**:
```bash
# First, ensure cluster summary exists (from run_theory_kernel_clusters.py)
python gravitywavebaseline/run_theory_kernel_clusters.py \
  --clusters MACSJ0416,MACSJ0717,ABELL_1689 \
  --theory-fit-json gravitywavebaseline/theory_metric_resonance_mw_fit.json \
  --out-csv gravitywavebaseline/theory_kernel_cluster_summary.csv

# Then fit A_cluster
python gravitywavebaseline/fit_theory_kernel_clusters_amp.py \
  --theory-fit-json gravitywavebaseline/theory_metric_resonance_mw_fit.json \
  --clusters MACSJ0416,MACSJ0717,ABELL_1689 \
  --sigma-v-default 1000.0 \
  --A-cluster-min 0.01 \
  --A-cluster-max 1000.0 \
  --out-json gravitywavebaseline/theory_kernel_cluster_amp_fit.json
```

**Output**: JSON file with:
- `A_cluster_best`: Best-fit cluster amplitude
- `A_cluster_A_gal_ratio`: Ratio to galaxy amplitude
- `chi2_norm`: Normalized chi-squared
- `cluster_results`: Per-cluster mass predictions and residuals

**What it tells you**:
- If you need `A_cluster ≫ 100` to hit M_required while MW/galaxies needed `A_gal ~ O(1-10)`, that's physically awkward.
- If a moderate `A_cluster` (say 5-20) gets you within factor ≲2 on M_required, the metric-resonance kernel structure is compatible with cluster lensing once you allow a separate domain amplitude (exactly what Σ-Gravity already does).

---

### 3. `fit_theory_kernel_joint.py`

**Purpose**: Joint fit of theory kernel parameters across MW, SPARC galaxies, and clusters simultaneously with a single parameter set θ*.

**Key Features**:
- Combined objective: `L_total = w_MW * L_MW + w_SPARC * L_SPARC + w_cluster * L_cluster`
- Fits `A_cluster` as part of the joint parameter vector
- Optional sigma gating for SPARC galaxies
- Configurable weights for balancing different domains

**Usage**:
```bash
python gravitywavebaseline/fit_theory_kernel_joint.py \
  --baseline-parquet gravitywavebaseline/gaia_with_gr_baseline.parquet \
  --r-min 12.0 \
  --r-max 16.0 \
  --sigma-v-mw 30.0 \
  --rotmod-dir data/Rotmod_LTG \
  --sparc-summary data/sparc/sparc_combined.csv \
  --summary-galaxy-col galaxy_name \
  --summary-sigma-col sigma_velocity \
  --max-sparc-galaxies 50 \
  --clusters MACSJ0416,MACSJ0717,ABELL_1689 \
  --sigma-v-cluster 1000.0 \
  --weight-mw 1.0 \
  --weight-sparc 1.0 \
  --weight-cluster 1.0 \
  --use-sigma-gating \
  --sigma-ref 25.0 \
  --beta-sigma 1.0 \
  --out-json gravitywavebaseline/theory_kernel_joint_fit.json
```

**Output**: JSON file with:
- `parameters`: Best-fit parameters (including A_cluster)
- `losses`: Individual and total loss values
- `metrics`: MW correlation, chi-squared
- `optimization`: Optimization status and statistics

**What it tells you**:
- Whether a single parameter set can simultaneously:
  - Match MW kernel shape (corr > 0.95, small χ²)
  - Improve SPARC RMS (mean ΔRMS < 0)
  - Reproduce cluster Einstein masses (within factor ~2)
- If all three domains fight you, that's still useful: you've shown this specific first-principles ansatz fails, justifying the stance that Σ-Gravity is a principled phenomenology, not yet a derived quantum-gravity theory.

---

## Workflow

### Recommended sequence:

1. **MW-only fit** (baseline):
   ```bash
   python gravitywavebaseline/run_theory_kernel_mw_fit.py \
     --require-positive-corr \
     --out-json gravitywavebaseline/theory_metric_resonance_mw_fit.json
   ```
   Check: `corr_K_emp_theory` should be > 0.9, ideally > 0.95.

2. **SPARC zero-shot test**:
   ```bash
   python gravitywavebaseline/run_theory_kernel_sparc_batch.py \
     --theory-fit-json gravitywavebaseline/theory_metric_resonance_mw_fit.json \
     --out-csv gravitywavebaseline/theory_kernel_sparc_from_mw.csv
   ```
   Check: Mean `delta_rms` and per-σ_v bin statistics.

3. **Cluster amplitude fit**:
   ```bash
   python gravitywavebaseline/fit_theory_kernel_clusters_amp.py \
     --theory-fit-json gravitywavebaseline/theory_metric_resonance_mw_fit.json \
     --out-json gravitywavebaseline/theory_kernel_cluster_amp_fit.json
   ```
   Check: `A_cluster_A_gal_ratio` and per-cluster `mass_ratio`.

4. **Joint fit** (if individual fits show promise):
   ```bash
   python gravitywavebaseline/fit_theory_kernel_joint.py \
     --weight-mw 1.0 \
     --weight-sparc 0.5 \
     --weight-cluster 0.5 \
     --out-json gravitywavebaseline/theory_kernel_joint_fit.json
   ```
   Check: All three losses and final correlation.

---

## Interpretation

### Success criteria:

- **MW**: `corr > 0.95`, `chi2_K < 1e-3`
- **SPARC**: Mean `delta_rms < -2 km/s`, > 60% galaxies improved
- **Clusters**: `A_cluster / A_galaxy` between 2-20, `mass_ratio` within 0.5-2.0

### Failure modes:

- **MW**: Negative correlation or `chi2_K > 0.1` → Theory kernel shape incompatible
- **SPARC**: Mean `delta_rms > +5 km/s` → Theory kernel systematically wrong for galaxies
- **Clusters**: `A_cluster > 100` or `mass_ratio < 0.1` → Theory kernel cannot scale to cluster regime

---

## Files

- `run_theory_kernel_mw_fit.py`: MW kernel shape fitting
- `fit_theory_kernel_clusters_amp.py`: Cluster amplitude scaling
- `fit_theory_kernel_joint.py`: Joint MW+SPARC+Cluster fit
- `theory_metric_resonance.py`: Core theory kernel implementation
- `run_theory_kernel_sparc_batch.py`: SPARC batch evaluation (existing)

---

## Dependencies

- `theory_metric_resonance.compute_theory_kernel`: Core kernel computation
- `metric_resonance_multiplier.metric_resonance_multiplier`: Empirical kernel
- `many_path_model.cluster_data_loader.ClusterDataLoader`: Cluster data loading
- `scipy.optimize.differential_evolution`: Global optimization
- `scipy.optimize.minimize_scalar`: 1D optimization (cluster amplitude)

---

## Notes

- The theory kernel uses a normalized form: `K = A_global * (K0 / max(|K0|))` to prevent amplitude from being absorbed into the spectrum parameters.
- `phase_sign` is computed post-fit to align sign with empirical kernel; if `phase_sign = -1`, the theory kernel is anti-correlated.
- Cluster fits assume a single `sigma_v_default` for all clusters; in reality, clusters have varying velocity dispersions, but this is a first-order approximation.
- Joint fitting is computationally expensive (300+ function evaluations); consider reducing `--max-sparc-galaxies` for faster iteration.

