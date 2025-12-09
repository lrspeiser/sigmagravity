# Coherence Cosmology: Next Steps

**Priority-ordered research agenda for developing the coherence cosmology framework**

---

## The Unified Microphysics (FOUNDATION)

**Before testing, we need to understand what we believe:**

### The Current-Current Correlator
The fundamental quantity is the current-current correlator:
```
G_jj(x,x') = ⟨j(x) · j(x')⟩_c    where j = ρv
```

Why this is compelling:
1. **It's what gravity couples to** - T_0i ~ ρv in the stress-energy tensor
2. **It naturally explains counter-rotation** - dot product is +/−/0 for co-/counter-/random motion
3. **It's measurable** - directly from IFU velocity maps
4. **It connects to coherence** - the "connected" correlator measures actual correlation

### The Fundamental Claim
**Gravity is not purely local.** The gravitational field depends on CORRELATIONS of T_μν, not just local values.

Analogies:
- EM in media: dielectric response depends on collective behavior
- Casimir effect: force from field correlations ⟨E(x)E(x')⟩
- GW emission: depends on mass quadrupole (non-local)

### The Coherence Field
1. A field φ_C permeates space with energy density ~ ρ_crit
2. It couples to matter's velocity correlations
3. It creates a potential Ψ_coh that grows with distance
4. This causes redshift, time dilation, AND the CMB
5. g† = cH₀/(4√π) emerges from this coupling

Files: `cosmology/fundamental_microphysics.py`

---

## Priority 1: CMB Power Spectrum (CRITICAL)

**Why it matters:** The CMB is the strongest evidence for the standard Big Bang + dark matter model. If coherence cosmology cannot reproduce the acoustic peaks, it fails.

### The T(z) = T₀(1+z) Solution (SOLVED!)

The coherence field has a local temperature that scales with potential:
```
T_coh(z) = T₀ × (1 + z)
```

This is NOT thermal equilibrium. The coherence field ACTIVELY maintains this temperature by converting potential energy to thermal radiation.

**Why it works:**
1. Molecules at z equilibrate with local coherence field at T_coh(z)
2. Photons emitted at T_coh(z) get redshifted by (1+z) traveling to us
3. We observe T₀ = T_coh(z)/(1+z) = T₀ ✓

**Physical picture:**
- CMB is continuously generated, not primordial
- CMB energy density scales as (1+z)^4
- At high z, CMB becomes significant fraction of total energy
- System is in steady state

File: `cosmology/tz_solution.py`

### Remaining CMB Tasks

1. **Understand what sets the acoustic peaks**
   - In ΛCDM: baryon-photon oscillations before recombination
   - In coherence: matter-coherence coupling? Coherence field oscillations?
   - The matter power spectrum has turnover at k ~ 0.01 h/Mpc (~ 100 Mpc)
   - This is close to the acoustic scale ~ 150 Mpc

2. **Identify what plays the role of "dark matter"**
   - The coherence field has energy density ~ ρ_critical
   - Does it cluster like dark matter?
   - Can it provide the potential wells for oscillations?

3. **Compute the power spectrum**
   - Modify CAMB or CLASS to include coherence effects
   - Compare with Planck 2018 data
   - File: `cosmology/cmb_analysis.py`

### Success Criteria
- Reproduce the first three acoustic peaks within 10%
- ✓ Explain T(z) scaling without expansion (DONE)
- Identify the "dark matter equivalent" in the coherence framework

---

## Priority 2: Environment-Dependent Effects (UNIQUE PREDICTION)

**Why it matters:** This is a unique prediction that ΛCDM doesn't make. If confirmed, it's strong evidence for coherence cosmology.

### Tasks

1. **Supernova environment analysis**
   - Cross-match Pantheon+ with galaxy density maps (SDSS, DES)
   - Compute local overdensity for each SN
   - Test: Is there a correlation between residuals and density?
   - File: `cosmology/sn_environment_test.py`

2. **Time dilation vs density**
   - Use light curve stretch parameter from Pantheon+
   - Correlate with local galaxy density
   - Prediction: Overdense regions → more time dilation
   - File: `cosmology/time_dilation_density.py`

3. **Redshift residuals**
   - At fixed distance, do overdense lines of sight show more redshift?
   - Use peculiar velocity-corrected redshifts
   - Look for systematic offsets

### Success Criteria
- Detect a correlation at > 2σ significance
- Effect size consistent with coherence prediction
- Rule out systematic explanations (selection effects, etc.)

---

## Priority 3: High-z Distance Measurements (DECISIVE TEST)

**Why it matters:** At z > 5, the models diverge by > 1 magnitude. JWST can test this.

### Tasks

1. **Identify standardizable candles at high z**
   - Gravitational lensing time delays (quasars)
   - Tip of the Red Giant Branch (TRGB) in lensed galaxies
   - Type II supernovae
   - Gamma-ray burst luminosity correlations

2. **Compile existing high-z distance data**
   - H0LiCOW lensing time delays
   - JWST early galaxy observations
   - File: `cosmology/highz_distances.py`

3. **Make specific predictions**
   - For each system, compute ΛCDM vs coherence distance modulus
   - Identify the best targets for discrimination

### Success Criteria
- Find 5+ systems at z > 3 with distance measurements
- Test whether they favor ΛCDM or coherence
- Identify future JWST targets

---

## Priority 4: Covariant Formulation

**Why it matters:** Without a proper action and field equations, the theory is incomplete.

### Tasks

1. **Write the coherence cosmology action**
   - Include the coherence field φ_C
   - Specify how it couples to matter and geometry
   - Ensure stress-energy conservation
   - File: `cosmology/covariant_action.py`

2. **Derive the field equations**
   - Vary the action with respect to metric and coherence field
   - Show how the metric modification emerges
   - Verify that the weak-field limit gives our formulas

3. **Check consistency**
   - Is the theory ghost-free?
   - Does it satisfy energy conditions?
   - Is it Lorentz invariant?

4. **Connect to teleparallel gravity**
   - How does the coherence field relate to torsion?
   - Can we derive the cosmological effects from the same framework as galaxy dynamics?

### Success Criteria
- Complete, self-consistent action
- Derived field equations that reproduce phenomenology
- Clear connection to Σ-Gravity framework

---

## Priority 5: Structure Formation

**Why it matters:** The matter power spectrum and galaxy clustering are strong constraints.

### Tasks

1. **Linear perturbation theory**
   - How do density perturbations grow in coherence cosmology?
   - What is the growth factor D(z)?
   - File: `cosmology/perturbation_growth.py`

2. **Matter power spectrum**
   - Compute P(k) at z = 0
   - Compare with SDSS/BOSS measurements
   - Identify scale-dependent differences from ΛCDM

3. **N-body simulations**
   - Modify an N-body code to include coherence effects
   - Run small-box simulations
   - Compare halo mass function with observations

### Success Criteria
- Reproduce the shape of P(k) at large scales
- Understand small-scale differences
- Identify testable predictions for galaxy clustering

---

## Priority 6: Nucleosynthesis

**Why it matters:** Big Bang nucleosynthesis predicts the correct abundances of H, He, Li.

### Tasks

1. **Understand the constraints**
   - Primordial He/H ~ 0.25
   - D/H ~ 2.5 × 10⁻⁵
   - Li/H ~ 10⁻¹⁰ (though there's a "lithium problem")

2. **Explore alternatives**
   - Steady-state nucleosynthesis in stars?
   - Primordial nucleosynthesis without Big Bang?
   - Some hybrid scenario?

3. **Quantitative predictions**
   - Can any static universe scenario reproduce the abundances?
   - What are the constraints on the coherence field from nucleosynthesis?

### Success Criteria
- Explain primordial abundances without Big Bang
- Or identify this as a fundamental limitation

---

## Priority 7: Gravitational Wave Propagation

**Why it matters:** GW170817 constrains c_GW = c to 10⁻¹⁵. Any modification must respect this.

### Tasks

1. **Derive GW propagation in coherence cosmology**
   - Does the coherence field affect GW speed?
   - Does it affect GW amplitude or phase?
   - File: `cosmology/gw_propagation.py`

2. **Compare GW and EM distances**
   - For neutron star mergers with EM counterparts
   - GW170817 gives d_L ~ 40 Mpc
   - Is this consistent with coherence cosmology?

3. **Make predictions for future events**
   - At higher z, do GW and EM distances diverge?
   - What would we expect to see?

### Success Criteria
- Show c_GW = c is preserved
- Consistent with GW170817
- Predictions for future events

---

## Quick Wins (Can Do Now)

### A. Improve Pantheon+ Analysis
- Include full covariance matrix (not just diagonal errors)
- Marginalize over systematic uncertainties
- Compare with SH0ES Cepheid calibration

### B. Add More BAO Data
- Include DESI Year 1 results
- Lyman-α BAO at z ~ 2.3
- Full shape analysis, not just BAO scale

### C. Test Tolman Surface Brightness
- Compile surface brightness data at various z
- Test (1+z)⁴ vs coherence prediction
- This is a classic test that has controversial results

### D. Quasar Variability
- Do quasars show (1+z) time dilation?
- Compile variability data from SDSS
- Test against coherence prediction

---

## Timeline Suggestion

| Phase | Duration | Focus |
|-------|----------|-------|
| **Phase 1** | 2 weeks | Environment effects (unique prediction) |
| **Phase 2** | 4 weeks | CMB power spectrum (critical test) |
| **Phase 3** | 2 weeks | High-z distances (decisive test) |
| **Phase 4** | 4 weeks | Covariant formulation (theory) |
| **Phase 5** | 4 weeks | Structure formation |
| **Phase 6** | 2 weeks | Nucleosynthesis & GW |

---

## Key Questions to Answer

1. ~~**What produces the CMB in a static universe?**~~ ✓ ANSWERED
   - The coherence field generates thermal EM radiation
   - Temperature scales with coherence potential: T_coh = T₀(1+z)
   - CMB is continuously generated, not primordial

2. ~~**Why does T_CMB scale as (1+z)?**~~ ✓ ANSWERED
   - Coherence field maintains T_coh ∝ (1+z)
   - Photons redshift by (1+z) traveling to us
   - We observe T₀ = T_coh/(1+z) = T₀

3. **What plays the role of dark matter in structure formation?**
   - Coherence field fluctuations?
   - Enhanced gravity from coherence?
   - The coherence field has ρ ~ ρ_crit — same as DM+DE combined

4. **Is the coherence field dynamical or static?**
   - It maintains T_coh ∝ (1+z) — this is ACTIVE, not passive
   - It converts potential energy to thermal radiation
   - What is its equation of state?

5. **How does coherence cosmology handle the horizon problem?**
   - In a static universe, there's no horizon problem!
   - The universe has always existed, so everything is causally connected
   - CMB uniformity is natural, not a problem

6. **NEW: What is the Lagrangian for the coherence field?**
   - How does it couple to gravity and EM?
   - What sets T_coh ∝ (1+z)?
   - How does it generate the acoustic peaks?

7. **NEW: How does coherence affect photon propagation?**
   - Redshift: ✓ (from coherence potential)
   - Time dilation: ✓ (from metric modification)
   - Polarization: ? (needs investigation)
   - Lensing: ? (should trace coherence, not just mass)

---

## Resources Needed

### Data
- [ ] Planck 2018 CMB power spectrum
- [ ] DESI Year 1 BAO
- [ ] Galaxy density maps (SDSS, DES)
- [ ] High-z distance measurements

### Code
- [ ] Modified CAMB/CLASS for coherence
- [ ] N-body code with coherence
- [ ] Statistical analysis pipeline

### Collaborations
- [ ] CMB expert for power spectrum analysis
- [ ] N-body simulation expert
- [ ] Observational cosmologist for data access

---

*This roadmap prioritizes tests that can confirm or rule out coherence cosmology. The CMB remains the biggest challenge, but environment-dependent effects offer a unique prediction that could provide early evidence.*

