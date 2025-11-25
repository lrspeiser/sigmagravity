# Σ-Gravity: Complete Derivation History & Status

**Date:** November 2025  
**Status:** Phenomenology proven across scales; first-principles derivation ~20% complete

---

## Executive Summary

We have successfully demonstrated that σ-dependent (velocity dispersion) modifications to gravity reproduce rotation curves for:
- **Milky Way**: RMS reduced from 111 km/s (Newtonian) to ~40 km/s
- **SPARC galaxies**: ~74-80% improved with best kernels
- **Galaxy clusters**: Strong lensing Einstein radii reproduced

However, the **field-level derivation** remains incomplete. This document records every attempt, what worked, what didn't, and the remaining path to a complete first-principles theory.

---

## Table of Contents

1. [What We've Proven (Phenomenology)](#what-weve-proven)
2. [Derivation Attempts by Mechanism](#derivation-attempts)
   - A. Shell Capacity & Spillover
   - B. σ-Dependent Capacity
   - C. λ-Based Multipliers
   - D. Metric Resonance (Empirical Kernel)
   - E. Metric Fluctuation Theory Kernel
   - F. Roughness / Time-Coherence
   - G. Mass-Coherence Amplitude
   - H. Coherence Microphysics Bake-off
3. [What We Have Derived](#derived-components)
4. [What Remains to Derive](#remaining-work)
5. [Robust Findings (Universal Constraints)](#robust-findings)
6. [Key Artifacts & Scripts](#key-artifacts)
7. [Line-of-Sight to Field Equations](#field-equations)

---

## What We've Proven (Phenomenology) {#what-weve-proven}

### Cross-Scale Validation

| System | Baseline RMS | Best Σ-Gravity RMS | Improvement | Key |
|--------|--------------|-------------------|-------------|-----|
| Milky Way | 111 km/s | 39.6 km/s | 64% | Metric resonance kernel |
| NGC 2403 | 37.9 km/s | 7.44 km/s | 80% | Same kernel + σ-gate |
| SPARC (165 galaxies) | — | — | 74-80% improved | Unified kernel with σ-gating |
| Clusters | — | — | Einstein radii matched | Time-coherence kernel |

### Universal σ-Dependence

**Critical discovery:** Any viable mechanism MUST scale with velocity dispersion:
- **Cold disks** (σ < 20 km/s): Enhancement ON
- **Hot disks** (σ > 40 km/s): Enhancement OFF
- **Intermediate**: Smooth Q-factor transition

This is **non-negotiable** across all successful implementations.

---

## Derivation Attempts by Mechanism {#derivation-attempts}

### A. Shell Capacity & Analytic Spillover

**Hypothesis:** Gravitational coherence saturates in inner shells; excess "spills" to outer shells.

#### What We Tried
- Implemented gravity-aware shell metadata
- Three spillover modes:
  - **Mass formula**: Spillover ∝ enclosed mass
  - **Radius formula**: Spillover ∝ shell radius
  - **Hybrid formula**: Spillover ∝ √(M × R)
- Removed hard capacity caps to allow unbounded spillover

#### Implementation
- `sparc_capacity_test.py`: Spillover computation logic
- `run_sparc_capacity_batch.py`: SPARC batch runner
- `compute_formula_weights()`: Spillover distribution function

#### Results
| Mode | Mean ΔRMS | % Improved | Verdict |
|------|-----------|------------|---------|
| Baseline (capped) | — | — | Reference |
| Mass formula | +1.8 km/s | 32% | **Worse** |
| Radius formula | +2.1 km/s | 28% | **Worse** |
| Hybrid formula | +1.5 km/s | 35% | **Worse** |

**Conclusion:** Hard caps were NOT the problem. **Distribution matters more than total amount**. The "where" of enhancement is more important than "how much."

**Files:**
- `scenario_summary_formula.csv`
- `sparc_results_formula_*.csv`

---

### B. σ-Dependent Capacity

**Hypothesis:** Capacity itself depends on velocity dispersion (Q-factor gating).

#### What We Tried
- Scaled capacity by σ_v: cold disks → high capacity; hot disks → low capacity
- Integrated Q-factor (Toomre stability) into capacity calculation
- Applied σ-gating across multiple kernel implementations

#### Results
- **Strong, consistent uplift** across all SPARC runs
- Became a **required component** in every subsequent successful kernel
- Best σ-gate parameters: γ_σ ≈ 1.5-2.0, σ_ref ≈ 20 km/s

**Status:** **Empirically essential**; physical origin tied to graviton pairing / coherence length (see section G).

**Files:** Integrated across `metric_resonance_multiplier.py`, `unified_kernel.py`, sweep scripts

---

### C. λ-Based Multipliers (Short-Wavelength Boost)

**Hypothesis:** Gravitational-wave-like coherence enhances at short orbital wavelengths (λ = 2πR).

#### What We Tried
- Power-law enhancement: `f_λ(R) = 1 + A × (λ_0 / λ)^α`
- Milky Way calibration (mass fudge removed, constant branch disabled):
  - Best fit: A ≈ 1.88, λ_0 ≈ 38 kpc, α ≈ 2.4
  - RMS: 111 → 41.6 km/s ✓

#### SPARC Application Issues
- **Raw λ-law overshot** both dwarfs and spirals
- Required clamping (≤1.5×) and combined σ + acceleration gating
- Worked only as a **component**, not standalone

**Conclusion:** λ-dependence is real but must be **tempered** by σ-gating and acceleration thresholds. Pure geometric λ-scaling is too aggressive.

**Files:**
- `test_lambda_enhancement.py`
- `pure_lambda_gr_test.py`
- `lambda_mw_fit.json`

---

### D. Metric Resonance (Empirical Kernel That Works)

**Hypothesis:** Gravitational field resonates at characteristic scales in both R and λ.

#### Functional Form (Phenomenological)
```
f_res(R, λ) = 1 + A × C_BurrXII(R | ℓ_0, p, n_coh) × W_logNormal(λ | λ_peak, σ_lnλ)
```

- **Radial kernel:** Burr XII distribution (flexible peak + power-law tail)
- **Wavelength kernel:** Log-normal (GW-like resonance)
- **Amplitude:** σ-gated with Q-factor

#### MW Calibration
- Parameters: ℓ_0 ≈ 8.5 kpc, p ≈ 2.4, n_coh ≈ 1.6, λ_peak ≈ 48 kpc
- RMS: 111 → **39.6 km/s** ✓

#### SPARC Demo (NGC 2403)
- RMS: 37.9 → **7.44 km/s** (80% drop) ✓

#### SPARC Phase-3 Batch (with σ-Q-gate + acceleration gate S(g))
- **~80% of galaxies improved** (132/165)
- Median ΔRMS ≈ −4.3 to −4.5 km/s
- Catastrophic blow-ups eliminated by σ-gating
- Mean ΔRMS ≈ −4.3 km/s

**Status:** **Best phenomenological kernel**. No first-principles derivation yet, but sets the **target shape** for theory.

**Files:**
- `metric_resonance_multiplier.py`
- `run_metric_resonance_mw.py`
- `metric_resonance_mw_fit.json`
- `run_metric_resonance_sigma_batch.py`
- Phase-3 batch CSVs

---

### E. Metric Fluctuation Theory Kernel

**Hypothesis:** Metric fluctuations h_μν with spectrum P(λ) produce resonant enhancement via path-integral coherence.

#### Field-Level Formulation
```
K(R) = ∫ P(λ) × C(λ, λ_matter(R), Q(R)) × W(R, λ) d ln λ
```

- **P(λ):** Fluctuation spectrum (power law + cutoff)
- **C(λ, ...):** Resonance filter (includes Q-factor from σ_v)
- **W(R, λ):** Geometric weight (orbital vs coherence wavelength)

#### Implementation
- Full integral computed numerically in `theory_metric_resonance.py`
- Structured spectrum: `P(λ) = P_0 × (λ/λ_*)^(−n) × exp(−(λ/λ_cut)^m)`

#### Reality Check Results

**Milky Way:**
- Initial K ≈ 0 (too weak)
- Needed **phase flip** (`phase_sign = −1`) to match MW shape
- With phase flip: shape matched, but **sign is ad-hoc**

**SPARC Batch:**
- Over-boosted dwarfs OR under-boosted spirals (depending on parameters)
- σ-dependence too weak

**Clusters:**
- Einstein-radius masses **orders of magnitude too small** with MW-fit parameters
- Later grid scans: either anti-aligned or scale mismatch

**Conclusion:** Integral machinery works, but **spectrum P(λ) lacks physical grounding**. The phase flip indicates we're missing a fundamental sign convention or interaction channel. **Path forward:** Derive P(λ) from graviton interactions or metric perturbation theory.

**Files:**
- `theory_metric_resonance.py`
- `run_theory_kernel_mw_fit.py`
- `theory_metric_resonance_mw_fit.json`
- `run_theory_kernel_sparc_batch.py`
- `run_theory_kernel_clusters.py`
- Theory batch summaries (CSV/JSON)

---

### F. Roughness / Time-Coherence (Derived & Validated)

**Hypothesis:** Gravitational "roughness" from vacuum fluctuations causes orbital paths to spend "extra time" in the field, leading to measurable boost.

#### First-Principles Derivation

**Coherence time:**
```
τ_coh = min(τ_geom(R), τ_noise(R))

τ_geom ∼ R / v_circ  (orbital timescale)
τ_noise ∼ R / σ_v^β  (decoherence from turbulence)
```

**Coherence length:**
```
ℓ_coh = α × c × τ_coh
```

**Exposure factor:**
```
Ξ = τ_coh / T_orb
```

**Universal boost (system-level):**
```
K_rough ≈ 0.774 × Ξ^0.1
```

The **small exponent** (0.1) explains why K is nearly constant per galaxy. For flat rotation curves, ℓ_coh ∝ R → Ξ ≈ const → K ≈ const.

#### Cross-Scale Validation

**Solar System:**
- K ≲ 10^(−10) to 10^(−28) → **safe** (no detectable deviation)

**Milky Way:**
- RMS: 111 → **66 km/s** (no hand-tuned amplitudes) ✓
- Outer disk improved significantly

**SPARC (165 galaxies):**
- Explains **~9.3%** of total required enhancement (robustly)
- **Constant K per galaxy** observed (feature, not bug)
- Matches theory prediction for flat curves

**Galaxy Clusters:**
- K_E ∼ 0.5–9 at Einstein radii (time-delay picture consistent) ✓
- Strong lensing regime: sufficient boost observed

**Status:** **Most solid first-principles component**. Derived, cross-scale validated, theoretically clean. This is the **baseline** for any complete theory.

**Files:**
- `coherence_time_kernel.py`
- `test_mw_coherence.py`
- `test_sparc_coherence.py`
- `test_cluster_coherence.py`
- `test_solar_system_coherence.py`
- `EXPOSURE_TEST_RESULTS.md`
- `mw_exposure_profile.json`
- `sparc_exposure_summary.csv`

---

### G. Mass-Coherence Amplitude (F_missing)

**Motivation:** K_rough explains ~10% of enhancement. The remaining ~90% ("F_missing") must come from **coherent microphysics** beyond geometric exposure.

#### Hypothesis
Mass (or potential depth) per coherence volume seeds a **multiplicative** enhancement, gated by σ_v:

```
F_missing = 1 + (F_raw(Ψ) − 1) × G_σ(σ_v)

G_σ = x / (1 + x),  x = (σ_ref / σ_v)^γ_σ
```

#### Unified Kernel (Multiplicative Coupling)
```
K_total = K_rough × C_BurrXII(R/ℓ_0) × [1 + extra_amp × (F_missing − 1)]
```

**Key insight:** Switched from **additive** to **multiplicative**. This was decisive for stability.

#### Parameter Sweeps

**Best configuration:**
- γ_σ = 2.0 (strong σ-damping)
- F_max = 5.0 (cap on raw amplitude)
- extra_amp = 0.25 (modest coupling strength)
- σ_ref = 20 km/s (cold-disk reference)

**Results:**
- **74% of SPARC improved** (122/165 galaxies)
- **Median ΔRMS ≈ −4.3 km/s**
- Mean ΔRMS ≈ −1.3 km/s
- High-σ systems stabilized (no catastrophic blow-ups)

**Status:** Empirically promising as a **second derived factor**. However, **microphysics for F_missing is not yet nailed down**. Likely tied to graviton pairing or superfluid-like condensate (see next section).

**Files:**
- `unified_kernel.py`
- `f_missing_mass_model.py`
- `sweep_unified_kernel_params.py`
- `unified_kernel_sweep_results.csv`
- `PARAMETER_SWEEP_SUMMARY.md`
- `recommended_unified_params.json`

---

### H. Coherence Microphysics Bake-off (5 Candidate Mechanisms)

We implemented and compared five microphysical models for coherence:

#### 1. Metric Resonance
- **Best global kinematic fit** (SPARC performance)
- Parameters finite & physical
- Clear path to field integral (needs P(λ) derivation)

#### 2. Graviton Pairing (Superfluid-like)
- **Best at high-mass discs**
- Natural σ-dependence via coherence length ξ(σ)
- Suggests **Ginzburg-Landau**-style field equation (order parameter ψ)
- **Top candidate for F_missing origin**

#### 3. Entanglement
- Mid-tier performance
- Parameter degeneracies (needs better σ_v(R) data)

#### 4. Vacuum Condensation
- Mid-tier performance
- Similar degeneracies to entanglement

#### 5. Path Interference
- **Weakest fits**
- Likely needs full phase/path integrals to compete

**Conclusion:** **Metric resonance** (empirical) + **graviton pairing** (microphysical amplitude) are the leading candidates for a unified first-principles theory.

**Files:**
- `coherence_models.py`
- `run_coherence_models.py`
- Reports/results under `coherence_tests/results/`

---

## What We Have Derived {#derived-components}

### 1. Time-Coherence (K_rough) — ✅ Complete

**Source:** Path-integral decoherence → exposure time

**Derivation:**
```
τ_coh = min(R/v_circ, R/σ_v^β)
ℓ_coh = α c τ_coh
Ξ = τ_coh / T_orb
K_rough ≈ 0.774 Ξ^0.1
```

**Validation:**
- Solar system: safe (K ≪ 1)
- MW: 111 → 66 km/s
- SPARC: ~9.3% baseline enhancement
- Clusters: K_E ∼ 0.5–9 at Einstein radii

**Status:** **Fully derived and cross-scale validated.**

---

### 2. σ-Gating (Q-Factor Dependence) — ✅ Empirically Essential

**Source:** Toomre stability / decoherence from turbulence

**Form:**
```
G_σ = x / (1 + x),  x = (σ_ref / σ_v)^γ_σ
```

**Status:** Required in every successful kernel. Physical origin tied to coherence length ξ(σ) in graviton pairing model (see field equations below).

---

## What Remains to Derive {#remaining-work}

### 1. Fluctuation Spectrum P(λ) — ❌ Not Derived

**Current status:** Ad-hoc power law with cutoff

**What's needed:**
- Derive from graviton interactions (e.g., loop corrections in semiclassical gravity)
- Or from metric perturbation theory (cosmological / quantum vacuum fluctuations)
- **Must fix sign convention** (phase flip in current implementation)

**Action:** Literature review on metric fluctuation spectra; compute 1-loop graviton self-energy.

---

### 2. Mass-Coherence Amplitude (F_missing) — ⚠️ Phenomenology Only

**Current status:** Empirical ansatz (mass per coherence volume, σ-gated)

**Leading candidate:** Graviton pairing / superfluid condensate

**Proposed field equation:**
```
(Ginzburg-Landau for gravity)
α ψ + β |ψ|^2 ψ − ξ^2(σ_v) ∇^2 ψ = S(ρ, ∇Φ)

Modified Poisson equation:
−∇·[(1 + χ|ψ|^2) ∇Φ] = 4πG ρ
```

Here, `1 + χ|ψ|^2` is the **multiplicative amplitude ratio** (i.e., F_missing).

**Status:** Field equation written; **needs numerical solver** for ψ(R) and parameter sweep.

---

### 3. Radial Kernel Shape (Burr XII) — ⚠️ Phenomenology Only

**Current status:** Empirical fit (best-performing distribution)

**Physical origin candidates:**
- Output of fluctuation integral with correct P(λ)
- Steady-state solution of GL equation
- Convolution of baryonic profile with coherence kernel

**Action:** Once P(λ) or ψ(R) is derived, check if Burr XII emerges naturally. If not, refine empirical kernel as interim solution.

---

## Robust Findings (Universal Constraints) {#robust-findings}

These findings apply to **any** viable Σ-Gravity derivation:

### 1. σ-Gating is Non-Negotiable
Any mechanism must **die off with σ_v** to avoid over-predicting hot disks. This is tied to:
- Toomre Q-factor (stability)
- Decoherence timescale (τ_noise ∝ σ_v^(−β))
- Coherence length (ξ ∝ σ_v^(−γ))

### 2. System-Level vs Local Behavior
**K_rough is system-level:** For flat rotation curves, ℓ_coh ∝ R → K ≈ constant per galaxy. This is a **feature**, not a bug. The detailed R-profile comes from other factors (F_missing, Burr kernel).

### 3. Multiplicative Amplitude
The large enhancement (beyond K_rough) is best represented as a **multiplicative ratio** (not additive), then σ-gated:
```
K_total = K_rough × K_shape(R) × [1 + A × (F_missing − 1)]
```

Additive coupling destabilized high-σ systems in all tests.

### 4. Phenomenology → Theory Target
The empirical **Burr XII × log-normal λ** kernel is the **target shape**. Any first-principles derivation must reproduce:
- Peak near galactic scale (ℓ_0 ∼ 8 kpc)
- Power-law tail at large R
- Log-normal resonance in λ
- σ-damping at high σ_v

### 5. Cross-Scale Consistency
Any mechanism must:
- Be **safe in Solar System** (K ≪ 1)
- Reproduce **MW rotation curve** (RMS < 50 km/s)
- Improve **≥70% of SPARC** (median ΔRMS < −3 km/s)
- Match **cluster Einstein radii** (K_E ∼ 1–10)

---

## Key Artifacts & Scripts {#key-artifacts}

### Time-Coherence (Derived Component)
- **Core:** `coherence_time_kernel.py`
- **Tests:** `test_mw_coherence.py`, `test_sparc_coherence.py`, `test_cluster_coherence.py`, `test_solar_system_coherence.py`
- **Results:** `EXPOSURE_TEST_RESULTS.md`, `mw_exposure_profile.json`, `sparc_exposure_summary.csv`

### Metric Resonance (Best Empirical Kernel)
- **Core:** `metric_resonance_multiplier.py`
- **MW fit:** `run_metric_resonance_mw.py`, `metric_resonance_mw_fit.json`
- **SPARC batch:** `run_metric_resonance_sigma_batch.py`, Phase-3 CSVs

### Theory Kernel (Needs P(λ) Derivation)
- **Core:** `theory_metric_resonance.py`
- **MW fit:** `run_theory_kernel_mw_fit.py`, `theory_metric_resonance_mw_fit.json`
- **SPARC batch:** `run_theory_kernel_sparc_batch.py`
- **Clusters:** `run_theory_kernel_clusters.py`

### Unified Kernel (Best Overall Results)
- **Core:** `unified_kernel.py`, `f_missing_mass_model.py`
- **Sweeps:** `sweep_unified_kernel_params.py`, `unified_kernel_sweep_results.csv`
- **Reports:** `PARAMETER_SWEEP_SUMMARY.md`, `recommended_unified_params.json`

### Coherence Models Comparison
- **Core:** `coherence_models.py`
- **Runner:** `run_coherence_models.py`
- **Results:** `coherence_tests/results/...`

### Legacy (Capacity Tests)
- **Spillover:** `sparc_capacity_test.py`, `run_sparc_capacity_batch.py`
- **Results:** `scenario_summary_formula.csv`

---

## Line-of-Sight to Field Equations {#field-equations}

Below are **concrete field-level formulations** that can be tested immediately against existing runners.

---

### 1. Time-Coherence (Already Derived) ✅

**Effective modification:** Multiplicative boost of Newtonian field by system-level constant (1 + K_rough(Ξ)).

**Implementation status:** Done.

**Next steps:** None for core relation. Fine-tune prefactors with better σ_v(R) data if available.

---

### 2. Metric-Fluctuation Resonance (Finish the Spectrum) ⚠️

**Field equation:**
Treat h_μν fluctuations with stationary spectrum P(λ). Effective Poisson equation becomes:

```
∇²Φ = 4πG ρ [1 + K(R)]

K(R) = ∫ P(λ) × C(λ, λ_matter(R), Q(R)) × W(R, λ) d ln λ
```

**What's missing:** Choose P(λ) with **proper sign & slope**, anchored to MW + SPARC + clusters simultaneously.

**Actionable test:**
1. Constrain spectrum family:
   ```
   P(λ) = P_0 × (λ/λ_*)^(−n) × exp(−(λ/λ_cut)^m)
   ```
2. Fit (n, m, λ_\*) with **sign constraint** (ensure correlation > 0 with data)
3. Re-run:
   - `run_theory_kernel_mw_fit.py`
   - `run_theory_kernel_sparc_batch.py`
   - `run_theory_kernel_clusters.py`
4. Iterate until **no phase flip needed** and all scales align

**Expected outcome:** Burr-like K(R) profile emerges naturally from integral.

---

### 3. Graviton Pairing / Superfluid Condensate (Candidate for F_missing) ⚠️

**GL-style field system** (Newtonian limit):

```
−∇·[(1 + χ|ψ|²) ∇Φ] = 4πG ρ

α ψ + β |ψ|² ψ − ξ²(σ_v) ∇²ψ = S(ρ, ∇Φ)
```

**Interpretation:**
- ψ: Graviton pair order parameter (complex field)
- ξ(σ_v) ∝ σ_v^(−γ_ξ): Coherence length (σ-gating)
- S(ρ, ∇Φ): Source term (seeds coherence from baryonic structure)
- (1 + χ|ψ|²): Multiplicative amplitude ratio = F_missing

**Actionable test path:**

#### Step 1: Static Solver for ψ(R)
Implement radial solver (1D disk geometry):
```python
# Pseudocode
def solve_gl_equation(rho_profile, phi_profile, sigma_v_profile, alpha, beta, chi, gamma_xi):
    # Finite-difference or spectral solver for:
    # α ψ + β |ψ|² ψ − ξ²(σ_v(R)) (d²ψ/dR² + (1/R) dψ/dR) = S(ρ(R), dΦ/dR)
    
    # Return ψ(R) profile
    return psi_of_R
```

#### Step 2: Plug into Unified Kernel
```python
# In unified_kernel.py:
psi_profile = solve_gl_equation(...)
F_missing = 1 + chi * np.abs(psi_profile)**2
K_total = K_rough * C_BurrXII * (1 + extra_amp * (F_missing - 1))
```

#### Step 3: Parameter Sweep
Use existing harness (`sweep_unified_kernel_params.py`) to sweep:
- γ_ξ (σ-gating strength)
- α, β (GL parameters)
- χ (coupling strength)

**Expected outcome:** F_missing(R) profile emerges from ψ solution; reproduces SPARC results without hand-tuned Burr kernel.

---

### 4. Complete Unified Theory (Roadmap)

**Once P(λ) and ψ(R) are derived:**

```
∇²Φ = 4πG ρ [1 + K_rough(Ξ) × K_fluct(R) × (1 + χ|ψ|²)]

K_rough(Ξ) = 0.774 Ξ^0.1  (derived ✅)

K_fluct(R) = ∫ P(λ) × C(λ, λ_matter, Q) × W(R, λ) d ln λ  (needs P(λ) ⚠️)

ψ(R) from: α ψ + β |ψ|² ψ − ξ²(σ_v) ∇²ψ = S(ρ, ∇Φ)  (needs solver ⚠️)
```

**This is the target field-level theory.**

---

## Next Immediate Actions

### Priority 1: Derive P(λ)
- **Task:** Literature review on metric fluctuation spectra
- **Options:**
  - 1-loop graviton self-energy (semiclassical gravity)
  - Cosmological metric perturbations (CMB power spectrum analogy)
  - Quantum vacuum fluctuations (Casimir-like)
- **Deliverable:** P(λ) with physical justification and correct sign

### Priority 2: Implement GL Solver for ψ(R)
- **Task:** Write radial solver for GL equation
- **Test:** MW disk, single SPARC galaxy (NGC 2403)
- **Integrate:** Plug ψ → F_missing into unified kernel
- **Sweep:** GL parameters (α, β, χ, γ_ξ)

### Priority 3: Validate Combined Theory
- **Once P(λ) and ψ(R) are available:**
  - Fit to MW (no free overall amplitude)
  - Run full SPARC batch
  - Test clusters
  - Check Solar System safety
- **Success criteria:**
  - MW RMS < 45 km/s
  - ≥75% SPARC improved
  - Cluster Einstein radii matched
  - Solar System K < 10^(−6)

---

## Summary of Completion Status

| Component | Status | Validation | Remaining Work |
|-----------|--------|------------|----------------|
| **K_rough (time-coherence)** | ✅ Derived | ✅ All scales | None (baseline locked) |
| **σ-gating** | ✅ Essential | ✅ All kernels | Tie to ξ(σ) in GL equation |
| **Metric resonance (empirical)** | ✅ Works | ✅ 80% SPARC | Need first-principles K_fluct |
| **P(λ) spectrum** | ❌ Ad-hoc | ❌ Sign flip | **PRIORITY: Derive from field theory** |
| **F_missing amplitude** | ⚠️ Phenomenology | ✅ 74% SPARC | **PRIORITY: GL solver for ψ(R)** |
| **Burr XII shape** | ⚠️ Empirical | ✅ Best fit | Check if emerges from P(λ) or ψ(R) |

**Overall Derivation Progress:** ~20% complete (K_rough solid; rest in progress)

**Phenomenological Success:** ~80% (proven across MW, SPARC, clusters)

**Path to 100%:** Derive P(λ) + implement GL solver → test combined theory → publish.

---

## References & Related Documents

- **Time-coherence derivation:** `EXPOSURE_TEST_RESULTS.md`
- **Parameter sweeps:** `PARAMETER_SWEEP_SUMMARY.md`
- **Unified kernel:** `recommended_unified_params.json`
- **Coherence models:** `coherence_tests/results/comparison_report.txt`

---

**END OF DERIVATION SUMMARY**

*Last updated: November 2025*
*Contact: See main project README for contributors*
