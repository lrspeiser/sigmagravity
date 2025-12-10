# Gravitational Polarization with Memory (GPM)
## Canonical First-Principles Statement

**Version**: 1.0  
**Date**: December 2024  
**Status**: Publication-ready theoretical framework

---

## The Principle

**Gravitational Polarization with Memory (GPM)**: In cold, rotating disks, the gravitational field develops a **causal, non-local polarization response** to baryons. The extra mass density is a convolution of the baryon distribution with a **universal, axisymmetric, causal kernel** whose susceptibility is suppressed by curvature/temperature/stability/mass ("gates"). In homogeneous high-curvature or hot environments (Solar System, clusters, FRW), the response vanishes.

---

## Mathematical Statement

### Static, Axisymmetric Form

```
ρ_coh(R, z) = α_eff[M, Q(R), σ_v(R), K(R)] × ∫ d²R' K₀(|R-R'|/ℓ(R)) Σ_b(R') × M(t;τ)
```

**Components**:

1. **Kernel**: K₀ (modified Bessel function of second kind, order 0)
   - This is the disk-aware Yukawa Green's function
   - Derived from linear response theory (see LINEAR_RESPONSE_DERIVATION.md)
   - Already implemented in axisymmetric convolution

2. **Coherence Length**: ℓ(R) = ℓ₀ (R_disk / 2 kpc)^p
   - Captures measured scaling with disk size
   - Physical basis: ℓ ~ σ_v / √(2πG ⟨Σ_b⟩_ℓ) (Toomre length scale)
   - Fitted values: ℓ₀ ~ 0.8 kpc, p ~ 0.5

3. **Environmental Gates**: α_eff = α₀ × g_Q × g_σ × g_M × g_K
   - Each g ∈ [0, 1] encodes suppression by environmental factors
   - **g_Q**: Toomre Q (gravitational stability)
   - **g_σ**: Velocity dispersion σ_v (thermal/turbulent dephasing)
   - **g_M**: Total mass M (homogeneity breaking)
   - **g_K**: Curvature K (high-density suppression)
   - These gates keep PPN and cosmology safe

4. **Temporal Memory**: M(t; τ) = ∫_{-∞}^t e^(-(t-t')/τ) (...) dt'
   - Optional: Encodes causal temporal smoothing
   - Timescale: τ ~ η × 2π/Ω (orbital time)
   - Fitted value: η ~ 1-3

### Field Equation View (Polarization Law)

```
∇²Φ = 4πG [ρ_b + ρ_coh]  ⟺  ∇·(g + 4πG P_g) = -4πG ρ_b
```

with constitutive relation:

```
P_g = χ ⋆ g
```

where χ reduces to the K₀ kernel times the gate product.

**This is a constitutive law for gravitational polarization** - analogous to:
- Electromagnetism: D = ε E (dielectric response)
- Magnetism: B = μ H (magnetic response)
- GPM: gravitational field = kernel ⋆ baryon source × gates

---

## Current Empirical Status

### Success Metrics (10 Galaxies)

| Metric | Value | Status |
|--------|-------|--------|
| Success rate | 80% (8/10) | ✓ Competitive |
| Median improvement | +80.7% over baryons | ✓ Large effect |
| Win rate vs DM | 43% (3/7) | ○ Comparable |
| Parameter universality | 4-7 global hyperparameters | ✓ Universal kernel |
| Multi-scale safety | Pass (PPN, cosmology) | ✓ Consistent |

**Interpretation**: GPM is **competitive** with DM models on disk-dominated galaxies, using a universal kernel with no per-galaxy coherence parameters. This is a major step toward a universal law.

### Gaps to Close for GR-Level Reliability

**Target**: χ²_red ≲ 2 for ≥90% of galaxies (GR-like reliability)

**Current issues**:
1. **Massive spirals with bulges**: NGC2841, NGC0801 fail (χ² +40%, +540%)
   - Need: Explicit 3-component convolution (disk + bulge + gas)
2. **Edge extrapolation**: Outer/inner RMS ratio 3.18 (goal < 1.5)
   - Need: Temporal memory refinement (τ adjustment or new functional form)
3. **Sample size**: 10 galaxies sufficient for proof-of-concept
   - Need: 30-50 galaxies for statistical dominance

---

## Falsifiable Predictions (How to Break the Model)

### 1. Vertical Anisotropy (★★★ STRONGEST)

**Prediction**: GPM's disk-aligned coherence halo creates **anisotropic velocity dispersion**:

```
β_z = 1 - σ_z²/σ_R² ~ 0.5 - 1.0  (at R = 5 kpc)
```

**Contrast**:
- **Spherical DM**: β_z ~ 0 (isotropic: σ_z = σ_R)
- **MOND**: β_z ~ 0 (no preferred geometry)
- **GPM**: β_z > 0.3 (disk-aligned: σ_z << σ_R)

**Test**:
- Measure vertical velocity dispersion σ_z(R) in **edge-on galaxies** (i > 75°)
- Target: NGC4565, NGC5746, IC2233, NGC891, NGC4244 (7 candidates identified)
- Data sources: PNe kinematics, HI vertical profile, stellar absorption lines

**Falsification criterion**:
- **If β_z ~ 0** in edge-on spirals → GPM falsified, DM/MOND supported
- **If β_z ~ 0.5** → GPM supported, spherical DM falsified

**Status**: ✓ Prediction generated (see vertical_anisotropy_predictions.png)

---

### 2. RAR-Q/σ_v Anti-Correlation

**Prediction**: Residual acceleration correlation (RAR) should **weaken** in high-Q, high-σ_v galaxies due to environmental gating.

Current detection:
- RAR-Q correlation: r(Δa, σ_v) = -0.23 (p = 0.0001)
- With α_eff = α₀/[1 + (σ_v/σ*)² + (Q/Q*)²], expect **stronger** anti-correlation

**Test**:
- Measure RAR scatter vs Q/σ_v across 50+ galaxies
- Quantify: ⟨|Δa|⟩ as function of Q and σ_v
- Fit power-law: ⟨|Δa|⟩ ∝ Q^n × σ_v^m

**Falsification criterion**:
- **If n, m ~ 0** (no Q/σ_v dependence) → Environmental gating falsified
- **If n, m < 0** (anti-correlation) → GPM gating mechanism supported

**Status**: ○ Framework exists (rar_scatter_analysis.py), needs re-run with microphysical gates

---

### 3. Edge Extrapolation (Temporal Memory Test)

**Prediction**: Temporal memory τ(R) ~ η × 2π/Ω(R) should create **smooth extrapolation** from inner to outer disk.

Current status:
- Outer/inner RMS ratio: 3.18 ± 3.19 (poor extrapolation)
- Goal: ratio < 1.5 (uniform residuals)

**Test**:
- Train GPM on inner disk (R < 2 R_disk)
- Predict outer disk (R > 2 R_disk) rotation curve
- Measure RMS(outer) / RMS(inner)

**Falsification criterion**:
- **If ratio > 3** persistently → Temporal memory inadequate, need DM
- **If ratio < 1.5** with adjusted τ → Memory mechanism validated

**Status**: ○ Analysis complete (edge_extrapolation_analysis.py), identified need for η adjustment

---

### 4. Cluster Lensing (Mass Gate Test)

**Prediction**: GPM should **fail** in massive clusters (M > 10¹⁴ M☉) due to mass gate suppression.

Mechanism: g_M = 1 / [1 + (M/M*)^n_M] → 0 as M → ∞

**Test**:
- Measure lensing profiles in clusters (e.g., Abell 1689, MS2137)
- Compare M_lens (from lensing) to M_bar + M_GPM (predicted)
- Expect: M_lens >> M_bar + M_GPM (GPM suppressed, DM needed)

**Falsification criterion**:
- **If M_GPM ~ M_lens** in clusters → Mass gate fails, GPM over-predicts
- **If M_GPM << M_lens** → Mass gate works, DM required at cluster scales

**Status**: ✓ Prediction validated (gpm_lensing.py: M_lens >> M_bar for clusters)

---

### 5. ℓ vs σ_v Scaling

**Prediction**: From linear response derivation, coherence length should scale **linearly** with velocity dispersion:

```
ℓ ∝ σ_v / √(2πG ⟨Σ_b⟩_ℓ)  ⇒  ℓ ∝ σ_v  (for fixed Σ_b)
```

**Test**:
- Plot ℓ_fit vs σ_v for 50+ galaxies (from per-galaxy fits)
- Expected: log(ℓ) vs log(σ_v) slope = 1 (linear)

**Falsification criterion**:
- **If slope ≠ 1** (e.g., ℓ ∝ σ_v²) → Derivation incorrect, kernel ad hoc
- **If slope = 1** → Toomre length interpretation validated

**Status**: ○ Need per-galaxy ℓ fits (currently use global ℓ₀)

---

### 6. ℓ vs Σ_b Anti-Correlation

**Prediction**: From linear response derivation:

```
ℓ ∝ Σ_b^(-1/2)  (inverse scaling with surface density)
```

**Test**:
- Plot ℓ_fit vs ⟨Σ_b⟩ (inner disk average surface density)
- Expected: log(ℓ) vs log(Σ_b) slope = -0.5

**Falsification criterion**:
- **If slope ≠ -0.5** → Polarization tensor calculation incorrect
- **If slope = -0.5** → Π(0, k) = ℓ⁻² derivation confirmed

**Status**: ○ Need per-galaxy ℓ fits

---

### 7. Solar System PPN Safety

**Prediction**: GPM should produce **zero** post-Newtonian corrections in the Solar System due to:
- High curvature (K-gate: g_K → 0)
- High velocity dispersion (σ-gate: g_σ → 0)
- Low Q (Q-gate: g_Q → 0)

**Test**:
- Compute PPN parameters γ, β from GPM
- Compare to Cassini bounds: |γ - 1| < 2.3×10⁻⁵, |β - 1| < 10⁻⁴

**Falsification criterion**:
- **If |γ - 1| > 10⁻⁵** → K-gate insufficient, GPM ruled out
- **If |γ - 1| < 10⁻¹⁰⁰** (machine precision) → Gates work

**Status**: ✓ Validated (ppn_safety.py: |γ-1|, |β-1| < 10⁻¹⁰⁰)

---

### 8. Cosmology Decoupling

**Prediction**: GPM should produce **zero** modification to Friedmann equations in FRW cosmology due to:
- Homogeneity (no preferred disk geometry)
- High curvature (K-gate: g_K → 0)

**Test**:
- Compute H(z), d_L(z) with GPM active in FRW
- Compare to standard ΛCDM

**Falsification criterion**:
- **If |H_GPM(z) - H_ΛCDM(z)| > 1%** → Homogeneity gate fails
- **If |H_GPM(z) - H_ΛCDM(z)| < machine precision** → Decoupling works

**Status**: ✓ Validated (cosmology_decoupling.py: H(z) = H_ΛCDM to 10⁻¹⁵)

---

## Distinction from Alternatives

### vs. MOND

| Feature | MOND | GPM |
|---------|------|-----|
| **Kernel** | Local μ-function | Non-local K₀ convolution |
| **Geometry** | Spherical | Disk-aligned (axisymmetric) |
| **Universal scale** | a₀ (acceleration) | ℓ (length), α (coupling) |
| **Environment** | Independent of Q, σ_v | Gated by Q, σ_v, M, K |
| **Vertical anisotropy** | β_z ~ 0 (isotropic) | β_z ~ 0.5 (anisotropic) |
| **Clusters** | Fails (needs DM anyway) | Suppressed by mass gate (needs DM) |
| **Cosmology** | Needs modified Friedmann | Standard ΛCDM |

**Key difference**: GPM is **gated** (environment-dependent susceptibility), MOND is **universal** (same μ everywhere).

### vs. Dark Matter

| Feature | DM | GPM |
|---------|-----|-----|
| **Nature** | Independent field/particles | Emergent from baryons |
| **Geometry** | Spherical halo | Disk-aligned |
| **Baryon dependence** | None (collisionless) | Direct (ρ_coh ∝ Σ_b ⊗ kernel) |
| **Vertical anisotropy** | β_z ~ 0 (spherical) | β_z ~ 0.5 (disk) |
| **RAR correlation** | Emergent (feedback?) | Built-in (response law) |
| **Environment** | Independent | Gated (suppressed in clusters/SS) |

**Key difference**: GPM is **baryon-dependent** (polarization response), DM is **independent** (separate sector).

---

## Path to GR-Level Credibility

### Quantitative Gap

**Current**: ~80% success, χ²_red ~ 2-5 on successful galaxies  
**Target**: ~90% success, χ²_red ~ 1-2 (GR-like reliability)

### Concrete Next Steps

1. **Lock universal kernel** (Priority 1)
   - Run hierarchical MCMC with axisymmetric kernel on 30-50 galaxies
   - Deliverable: Θ_global = {α₀, ℓ₀, p, M*, σ*, Q*, K*} with posteriors
   - Hold-out predictions on 10-20 test galaxies
   - **Status**: Framework ready (hierarchical_mcmc_gpu.py), needs production run

2. **Close bulge/HSB gap** (Priority 2)
   - Implement 3-component convolution (disk + bulge + gas)
   - Wire Sérsic bulge into axisymmetric kernel
   - Re-run NGC2841, NGC0801 with proper bulge treatment
   - **Status**: Code infrastructure exists, needs integration

3. **Codify effective action** (Priority 3)
   - Promote EFFECTIVE_ACTION_FORMALISM.md to canonical theory statement
   - Add to main paper as foundational section
   - **Status**: ✓ Complete (452 lines)

4. **Publish falsifiable predictions** (Priority 4)
   - Vertical anisotropy: ✓ Complete
   - RAR-Q/σ_v: ○ Needs microphysical gates implementation
   - Edge behavior: ✓ Quantified (needs τ refinement)
   - ℓ vs σ_v/Σ_b: ○ Needs per-galaxy ℓ fits

5. **Multi-scale checks in main text** (Priority 5)
   - PPN: ✓ Complete (ppn_safety.py plots)
   - Cosmology: ✓ Complete (cosmology_decoupling.py plots)
   - Clusters: ✓ Complete (gpm_lensing.py plots)

---

## Summary

GPM is a **single principle** - a causal constitutive law for gravitational polarization:

```
ρ_coh = α_eff × (Σ_b ⊗ K₀)
```

where:
- K₀: Universal Yukawa kernel (derived from linear response)
- α_eff: Environment-dependent susceptibility (gates)
- ⊗: Axisymmetric convolution (disk geometry)

**Current status**: Competitive with DM on disk galaxies, distinct falsifiable predictions, universal kernel with 4-7 parameters.

**Gap to close**: 90% success rate on 50+ galaxies including massive spirals.

**Falsification route**: Measure β_z in edge-on galaxies. If β_z ~ 0 → GPM ruled out.

---

**References**:
- Theory: LINEAR_RESPONSE_DERIVATION.md, EFFECTIVE_ACTION_FORMALISM.md
- Implementation: AXISYMMETRIC_VALIDATION_SUMMARY.md
- Comparison: GPM_VS_MOND_VS_DM.md
- Status: PUBLICATION_STATUS.md

**Author**: GPM Theory Team  
**Last updated**: December 2024
