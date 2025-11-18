# Microphysics Investigation: Field Equations for Σ-Gravity

## Overview

This directory contains an investigation into three candidate microphysical models that could provide **first-principles field equations** for Σ-Gravity, beyond the effective Burr-XII kernel.

**Goal**: Find which microphysics can reproduce the empirical Σ-Gravity kernel from a reasonably clean effective field theory, with fewer free parameters and clear physical interpretation.

## The Challenge

What we have so far:
- **Empirical kernel**: Burr-XII form fitted to SPARC+MW+clusters
- **Data-driven success**: Competitive with MOND/ΛCDM on rotation curves
- **But**: Still "effective theory", not yet GR-level clean field equations

What we want:
- Derive K(R) or K(system properties) from microphysics
- Match or beat empirical performance
- Solar System safe
- Works across domains (galaxies, clusters, MW)

## Three Candidate Models

### 1. Roughness / Time-Coherence (Path-Integral Decoherence)

**File**: `microphysics_roughness.py`

**Field-equation picture**:
- Start with GR + small stochastic metric fluctuations: g_μν = ḡ_μν + h_μν
- After coarse-graining: G_μν[ḡ] = 8πG(T_μν + T^(rough)_μν[⟨h h⟩])
- In weak-field limit: ∇²Φ = 4πG ρ + δS[ρ, σ_v, τ_coh]

**Effective law**:
```
g_eff(R) = g_GR(R) * [1 + K_rough(Ξ_system)]
```

where Ξ is the **system-level exposure factor**:
- Ξ = τ_coh / T_orb
- τ_coh combines geometric dephasing (τ_geom ~ R/v_circ) and noise decoherence (τ_noise ~ R/σ_v^β)

**Empirical law from Phase-2 fits**:
- K_rough(Ξ) = K0 * Ξ^γ
- K0 ≈ 0.774, γ ≈ 0.1

**Key feature**: K is **constant per system**, not R-dependent. This is a feature, not a bug: system-level properties renormalize the effective coupling.

**Strengths**:
- Minimally invasive: don't touch local field equations
- Natural connection to time-coherence we've already measured
- Explains why enhancement depends on system properties (σ_v, R, v_circ)

**Weaknesses**:
- Needs to explain where h_μν fluctuations come from
- γ = 0.1 is empirical, not derived

### 2. Graviton Pairing / Superfluid Condensate

**File**: `microphysics_pairing.py`

**Field-equation picture**:
- Add complex scalar condensate field ψ (Bose-condensed gravitational degree of freedom)
- Action includes coupling to baryonic trace: ∫ d⁴x √(-g) [...+ g|ψ|² T]
- Gross-Pitaevskii-like equation for ψ: □ψ - m²ψ - 2λ|ψ|²ψ - gTψ = 0

**Static weak-field limit**:
- Yukawa-corrected Poisson: ∇²Φ - μ²Φ = 4πG_eff(ρ,σ_v) ρ
- Condensate destroyed by velocity dispersion: K_pair ∝ (σ_c/σ_v)^γ for σ_v < σ_c

**Effective law**:
```
g_eff(R) = g_GR(R) * [1 + K_pair(R, σ_v(R))]
```

with:
- K_pair = A_pair * G_sigma(σ_v) * C_R(R)
- G_sigma ~ 1 for cold systems, falls sharply above critical σ_c (superfluid transition)
- C_R is radial envelope (Solar System suppression)

**Strengths**:
- Best bet for "missing 90%" factor
- Natural cold/hot dichotomy (galaxies vs clusters)
- Solar System safe (condensate doesn't form at high σ_v)
- Connects to condensed-matter analogs (superfluidity)

**Weaknesses**:
- Need to justify condensate formation in gravitational context
- Multiple parameters (A_pair, σ_c, γ_sigma, ℓ_pair)

### 3. Metric Resonance with Fluctuation Spectrum

**File**: `microphysics_resonance.py`

**Field-equation picture**:
- GR + stochastic background of metric fluctuations with power spectrum P(λ)
- P(λ) ∝ λ^(-α) exp(-λ/λ_cut)
- Matter resonantly couples to modes matching orbital wavelength λ_orb = 2πR

**Resonance filter**:
- C(λ,R) ~ Q² / [Q² + (λ/λ_orb - λ_orb/λ)²]
- Quality factor: Q ~ v_circ/σ_v

**Effective enhancement**:
```
K_res(R) = A_res ∫ d ln λ P(λ) C(λ,R) W(R,λ)
```

**Effective law**:
```
g_eff(R) = g_GR(R) * [1 + K_res(R)]
```

**Strengths**:
- Makes fluctuation spectrum explicit (can test different P(λ))
- Natural R-dependence from resonance condition
- Q-factor naturally ties to σ_v (coherence)

**Weaknesses**:
- Previous attempts showed over-boosting
- Need to fine-tune spectrum parameters
- Integral computation more expensive

## Test Harnesses

Each model has a SPARC test script:
- `run_roughness_microphysics_sparc.py`
- `run_pairing_microphysics_sparc.py`
- `run_resonance_microphysics_sparc.py`

Plus a comparison script:
- `compare_microphysics_models.py`

## Success Criteria

A model "wins" if it:
1. **Reproduces empirical performance**: Gets ≲0.10 dex RAR scatter on SPARC
2. **Fewer free parameters**: Or more physically interpretable ones
3. **Solar System safe**: K(1 AU, σ_v ~ km/s) ≪ 10^(-10)
4. **Works across domains**: MW + SPARC + clusters with same parameters
5. **Emerges from clean field theory**: Not just a fitted curve

## Comparison with MOND/ΛCDM Benchmarks

From existing Σ-Gravity results:
- **SPARC RAR scatter**: Σ-Gravity ≈0.087 dex (competitive with MOND)
- **MW star-level RAR**: Σ-Gravity bias +0.062 dex, σ=0.142 dex (vs MOND +0.166 dex)
- **Clusters**: ~8% median fractional error, 88.9% Einstein radius coverage

**Bar to beat**: Match or exceed these with a first-principles microphysics model.

## Parameter Tuning Strategy

Initial parameters are educated guesses. For optimization:
1. Run all three models with defaults
2. Identify which performs best
3. Grid search / Bayesian optimization on best candidate
4. Test across MW, SPARC, clusters
5. Check Solar System safety

## Next Steps

1. ✅ Implement three microphysics models
2. ✅ Create SPARC test harnesses
3. ⏳ Run comparison on SPARC sample
4. ⏳ Identify best performer
5. ⏳ Tune parameters for best model
6. ⏳ Test on MW and clusters
7. ⏳ Check Solar System constraints
8. ⏳ Write up as derivation section for paper

## Files in This Investigation

**Core microphysics modules**:
- `microphysics_roughness.py` - Time-coherence model
- `microphysics_pairing.py` - Graviton pairing model
- `microphysics_resonance.py` - Metric resonance model

**Test harnesses**:
- `run_roughness_microphysics_sparc.py`
- `run_pairing_microphysics_sparc.py`
- `run_resonance_microphysics_sparc.py`
- `compare_microphysics_models.py`

**Results** (to be generated):
- `results/roughness_microphysics_sparc.csv`
- `results/pairing_microphysics_sparc.csv`
- `results/resonance_microphysics_sparc.csv`
- `results/microphysics_comparison_sparc.csv`
- `results/microphysics_comparison_summary.json`

## Theory Context

This investigation addresses the question:

> "You've built a data-driven kernel that beats GR and is competitive with MOND/ΛCDM, 
> but the microphysics is still 'effective theory'. Can you derive it from first principles?"

The answer will determine whether Σ-Gravity is:
- **Option A**: A phenomenological effective theory (like Newtonian gravity before Einstein)
- **Option B**: Derivable from clean field equations (like GR's Einstein equations)

Option B is the goal. Let's find out which microphysics gets us there.

