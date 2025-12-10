# Remaining Challenges for Coherence Gravity

**Open problems requiring theoretical development**

---

## Challenge 1: CMB Power Spectrum

### The Problem

The CMB angular power spectrum has a specific structure:
- First peak at l ~ 220 (angular scale ~ 1°)
- Second peak at l ~ 540
- Third peak at l ~ 810
- Damping tail at l > 1000

In ΛCDM, these arise from baryon-photon acoustic oscillations before recombination (z ~ 1100).

### STATUS: PARTIALLY EXPLORED — FUNDAMENTAL ISSUE IDENTIFIED

**Files:** `current/derivations/cmb_power_spectrum.py`

### What We've Found

**WHAT WORKS:**
- ✓ Overall amplitude can be matched
- ✓ General shape (broad peak at l ~ few hundred)
- ✓ Damping at high l from matter power spectrum

**WHAT DOESN'T WORK:**
- ✗ Sharp acoustic peaks NOT reproduced
- ✗ Peak height ratios not explained
- ✗ Specific peak positions (220, 540, 810) not derived

### The Fundamental Issue

In ΛCDM, acoustic peaks come from a **TEMPORAL** process:
- Sound waves propagate for 400,000 years
- Modes fitting integer half-wavelengths in sound horizon are enhanced

In coherence cosmology (static universe):
- No "time since Big Bang"
- No natural timescale for oscillations
- Peaks must come from **SPATIAL** structure, not temporal evolution

### Approaches Explored

**1. Matter-Sourced CMB**
- Coherence field sourced by matter: δφ ∝ δ
- CMB traces coherence: δT/T ∝ δφ
- Result: C_l ∝ P(k) — gives SMOOTH spectrum, not peaks

**2. Coherence Field Oscillations**
- Characteristic wavelength λ_coh = 2πc/H₀ ~ 4000 Mpc
- Too large for acoustic peaks (need ~ 150 Mpc)

**3. Coupled Coherence-Matter System**
- Derived dispersion relation for coupled modes
- Two modes exist, but don't naturally produce BAO-scale peaks

### Possible Paths Forward

1. **Resonant amplification of BAO** — coherence field resonates with 150 Mpc scale
2. **Non-linear mode coupling** — creates harmonics at right scales
3. **Modified coherence dynamics** — add term that introduces 150 Mpc scale
4. **Alternative interpretation** — peaks from foreground/ISW effects

### Status: CRITICAL OPEN PROBLEM

This is the most challenging aspect of coherence cosmology.
The acoustic peaks are strong evidence for hot Big Bang + dark matter.

**Required work:**
1. [x] Derive coherence field equation of motion
2. [x] Compute power spectrum from matter coupling
3. [x] Calculate angular power spectrum C_l
4. [x] Compare with Planck 2018 data
5. [ ] **Find mechanism for sharp peaks** ← CRITICAL
6. [ ] Explain peak height ratios
7. [ ] Derive polarization patterns

---

## Challenge 2: CMB Polarization

### The Problem

The CMB shows polarization patterns:
- **E-modes:** Gradient patterns, detected at high significance
- **B-modes:** Curl patterns, from lensing and (possibly) primordial GWs

In ΛCDM, polarization arises from Thomson scattering:
1. Photons scatter off electrons at recombination
2. Quadrupole anisotropy in photon distribution creates polarization
3. E-modes from scalar perturbations
4. B-modes from tensor perturbations (GWs) or lensing

### The Challenge for Coherence Gravity

If there's no recombination epoch, how does polarization arise?

### Possible Approaches

**Approach 1: Coherence Field Polarization**

The coherence field might directly emit polarized radiation:
$$\mathbf{E}_{coh} = \nabla \phi_C \times \mathbf{something}$$

If the coherence field has a gradient, this could create polarization.

**Questions:**
- What is the "something"?
- Does this give E-modes or B-modes?
- What sets the amplitude?

**Approach 2: Foreground Scattering**

Photons from the coherence field scatter off free electrons in the IGM:
- IGM is ionized at z < 6
- Thomson scattering creates polarization
- This is similar to reionization in ΛCDM

**Questions:**
- Is there enough optical depth?
- Does this give the right angular pattern?
- How does it relate to the temperature pattern?

**Approach 3: Magnetic Fields**

If the coherence field couples to magnetic fields:
$$\mathbf{B}_{coh} = \nabla \times (\phi_C \mathbf{A})$$

Faraday rotation could create polarization patterns.

**Questions:**
- What is the coherence-EM coupling?
- Can this produce E-modes?
- What about B-modes?

### Status: PLACEHOLDER

**Required work:**
1. [ ] Identify polarization mechanism
2. [ ] Compute E-mode power spectrum C_l^{EE}
3. [ ] Compute B-mode power spectrum C_l^{BB}
4. [ ] Compute TE cross-correlation
5. [ ] Compare with Planck polarization data

---

## Challenge 3: Structure Formation

### The Problem

The universe has structure:
- Galaxies, clusters, filaments, voids
- Matter power spectrum P(k) well-measured
- BAO feature at k ~ 0.1 h/Mpc
- Growth rate f(z) measured from RSD

In ΛCDM, structure grows from primordial fluctuations:
1. Inflation creates δρ/ρ ~ 10⁻⁵ fluctuations
2. Dark matter starts growing at matter-radiation equality
3. Baryons fall into DM wells after recombination
4. Linear growth: δ ∝ D(z)
5. Non-linear collapse forms halos

### The Challenge for Coherence Gravity

How do density perturbations grow without dark matter?

### Possible Approaches

**Approach 1: Coherence-Enhanced Growth**

The coherence enhancement Σ might amplify gravitational instability:
$$\ddot{\delta} + 2H\dot{\delta} - 4\pi G \rho \Sigma \delta = 0$$

If Σ > 1 at low accelerations, growth is faster.

**Questions:**
- What is the effective growth factor D(z)?
- Does this match observations?
- What about the BAO feature?

**Approach 2: Coherence Field Perturbations**

The coherence field itself might have perturbations:
$$\phi_C = \bar{\phi}_C + \delta\phi_C$$

These perturbations could seed structure formation.

**Questions:**
- What sets the initial perturbations?
- How do they evolve?
- Do they match the observed P(k)?

**Approach 3: Eternal Universe**

In a static, eternal universe, structure might have always existed.
No need for "formation" — just equilibrium.

**Questions:**
- How do we explain the observed growth rate?
- What about high-z observations showing less structure?
- Is this consistent with galaxy evolution?

### Status: PLACEHOLDER

**Required work:**
1. [ ] Derive perturbation equations in coherence gravity
2. [ ] Compute growth factor D(z)
3. [ ] Compute matter power spectrum P(k)
4. [ ] Compare with SDSS/BOSS measurements
5. [ ] Run N-body simulations with coherence

---

## Challenge 4: Nucleosynthesis

### The Problem

Observed primordial abundances:
- H: ~75% by mass
- He-4: ~25% by mass
- D: ~2.5 × 10⁻⁵ by number
- He-3: ~10⁻⁵ by number
- Li-7: ~10⁻¹⁰ by number (though "lithium problem")

In ΛCDM, these come from Big Bang Nucleosynthesis:
1. Universe cools below ~1 MeV at t ~ 1 second
2. Neutrons freeze out at n/p ~ 1/7
3. Deuterium forms when T < 0.1 MeV
4. Helium-4 forms rapidly
5. Trace amounts of D, He-3, Li-7 remain

### The Challenge for Coherence Gravity

If there's no Big Bang, where do the light elements come from?

### Possible Approaches

**Approach 1: Stellar Nucleosynthesis**

All elements come from stars:
- H is primordial (always existed)
- He from hydrogen burning in stars
- D, Li from cosmic ray spallation

**Problems:**
- Stars destroy D, not create it
- He/H ratio requires specific star formation history
- Hard to get the exact abundances

**Approach 2: Coherence Field Nucleosynthesis**

The coherence field might catalyze nuclear reactions:
$$p + n + \phi_C \rightarrow D + \gamma$$

If the coherence field couples to nuclear processes, it could create light elements.

**Questions:**
- What is the coherence-nuclear coupling?
- Can this produce the right abundances?
- Where/when does this happen?

**Approach 3: Primordial Nucleosynthesis Without Big Bang**

Maybe there was a hot, dense phase without a singularity:
- Universe was hot and dense in the past
- Nucleosynthesis occurred normally
- But no expansion — just cooling

**Questions:**
- What caused the cooling?
- How is this different from ΛCDM?
- Is this consistent with the static universe?

**Approach 4: Accept as Limitation**

Perhaps nucleosynthesis is a genuine problem for coherence gravity.
The theory might need modification or might be incomplete.

### Status: PLACEHOLDER

**Required work:**
1. [ ] Calculate stellar production of He, D, Li
2. [ ] Compare with observed abundances
3. [ ] Explore coherence-nuclear coupling
4. [ ] Determine if this is a fundamental limitation

---

## Challenge 5: The Coherence Field Lagrangian

### The Problem

We have a phenomenological description:
- Enhancement factor Σ
- Coherence potential Ψ_coh
- CMB temperature T_coh

But we lack a fundamental action from which all this derives.

### STATUS: LARGELY SOLVED ✓

We have derived a Lagrangian that produces the phenomenology:

**THE COHERENCE GRAVITY ACTION:**
```
S = ∫ d⁴x √(-g) { (1 - αφ/M_P²) R/(16πG) 
                   - (1/2)(∂φ)² 
                   - Λ_C - (1/2)m₀²(g/g†)φ²
                   + λ φ [j² - ℓ²(∇j)²] / j₀² }
    + S_matter
```

**KEY FEATURES:**

1. **Non-minimal coupling** `(1 - αφ/M_P²)R` gives enhanced gravity:
   - G_eff = G(1 + αφ/M_P²)
   - Σ = 1 + αφ/M_P²

2. **Chameleon mass** `m²(g) = m₀²(g/g†)` produces h(g) screening:
   - At high g: m² large → φ screened → Σ ≈ 1
   - At low g: m² small → φ accumulates → Σ > 1
   - Characteristic scale λ_C = (1/m₀)√(g†/g)
   - Produces h(g) ~ (g†/g)^{3/2} at low accelerations ✓

3. **Coherence measure** `C = j² - ℓ²(∇j)²` couples to rotation:
   - Coherent rotation: C > 0 → φ sourced positively
   - Counter-rotation: C < 0 → φ suppressed
   - Explains counter-rotation suppression ✓

4. **Green's function** produces W(R) = R/(ξ+R):
   - For disk source, φ grows linearly at small R
   - Saturates at large R
   - ξ ~ R_d (disk scale length) ~ 1 kpc ✓

5. **Cosmological constant** Λ_C ~ ρ_crit c² gives dark energy

**PARAMETER RELATIONSHIPS:**
- g† = cH₀/(4√π) from m₀ ~ H₀/c
- ξ ~ R_d ~ 1 kpc from disk scale
- A ~ αλ × (ρv²/j₀²) × R_d / M_P²

**Files:** 
- `current/derivations/coherence_lagrangian.py`
- `current/derivations/lagrangian_to_phenomenology.py`

### Remaining Work

1. [x] Write candidate Lagrangian
2. [x] Derive field equations
3. [x] Check weak-field limit gives Σ-Gravity
4. [ ] Check cosmological limit gives correct d_L, d_A
5. [ ] Verify ghost-freedom and stability
6. [ ] Derive CMB properties from Lagrangian
7. [ ] Exact numerical relationship between α, λ and A

---

## Challenge 6: Time Dilation Derivation

### The Problem

We observe (1+z) time dilation in supernova light curves.
Our metric gives:
$$g_{tt} = -(1 + 2\Psi_{coh}) = -(1 + z)$$

This gives time dilation factor √(1+z), not (1+z).

### Possible Solutions

**Solution 1: Modified Metric**

Use:
$$g_{tt} = -(1 + z)^2$$

This gives:
$$\frac{d\tau}{dt} = (1+z)$$

But then Ψ_coh = [(1+z)² - 1]/2 ≠ z/2.

**Solution 2: Two Effects**

Time dilation comes from two sources:
1. Metric time dilation: √(1+z)
2. Photon travel time effect: √(1+z)

Combined: (1+z)

**Solution 3: Different Interpretation**

The (1+z) factor comes from comparing:
- Source frame time: Δτ_source
- Observer frame time: Δt_obs

In coherence cosmology:
$$\Delta t_{obs} = \Delta \tau_{source} \times \frac{1}{\sqrt{g_{tt}(source)}} \times (1+z)_{redshift}$$

If g_tt(source) = -(1+z) and redshift factor = 1:
$$\Delta t_{obs} = \Delta \tau_{source} \times \frac{1}{\sqrt{1+z}} \times 1 = \Delta \tau_{source} / \sqrt{1+z}$$

This is wrong. Need to reconsider.

### Status: PLACEHOLDER

**Required work:**
1. [ ] Carefully derive time dilation from metric
2. [ ] Account for all effects (metric, redshift, travel time)
3. [ ] Verify (1+z) factor emerges correctly
4. [ ] Or modify the metric ansatz

---

## Challenge 7: Horizon Problem (or Lack Thereof)

### The Problem

In ΛCDM, the horizon problem asks: why is the CMB so uniform when opposite sides of the sky were never in causal contact?

Inflation solves this by having a period of exponential expansion that brings causally connected regions to cosmological scales.

### In Coherence Gravity

If the universe is static and eternal:
- Everything has always been in causal contact
- There is no horizon problem
- CMB uniformity is natural

**But:** If the universe is eternal, why isn't it in thermal equilibrium?
Why do we see structure, temperature differences, etc.?

### Possible Resolution

The coherence field maintains a NON-equilibrium steady state:
- Energy flows through the coherence field
- Local fluctuations persist
- The system is driven, not equilibrated

This is like a river: it's in steady state but not equilibrium.

### Status: PLACEHOLDER

**Required work:**
1. [ ] Formalize the steady-state picture
2. [ ] Show how fluctuations are maintained
3. [ ] Explain why thermal equilibrium is avoided
4. [ ] Address the "heat death" concern

---

## Priority Order

1. **CMB Power Spectrum** — Critical test, highest priority
2. ~~**Coherence Field Lagrangian**~~ — ✓ LARGELY SOLVED
3. **Time Dilation Derivation** — Technical issue to resolve
4. **Structure Formation** — Important for completeness
5. **CMB Polarization** — Can be addressed after power spectrum
6. **Nucleosynthesis** — May be a fundamental limitation
7. **Horizon Problem** — Philosophical, lower priority

---

## Summary

| Challenge | Difficulty | Priority | Status |
|-----------|------------|----------|--------|
| CMB Power Spectrum | Hard | 1 | PLACEHOLDER |
| CMB Polarization | Hard | 5 | PLACEHOLDER |
| Structure Formation | Medium | 4 | PLACEHOLDER |
| Nucleosynthesis | Hard | 6 | PLACEHOLDER |
| Lagrangian | Hard | 2 | ✓ LARGELY SOLVED |
| Time Dilation | Medium | 3 | PLACEHOLDER |
| Horizon Problem | Easy | 7 | PLACEHOLDER |

---

## Notes for Future Work

### CMB Power Spectrum Approach

Most promising: Matter-coherence coupling

The coherence field couples to matter density fluctuations.
The matter power spectrum has structure at the right scales.
The CMB inherits this structure through the coherence field.

**Key calculation:** 
$$C_l^{TT} = \int \frac{dk}{k} \, P_\delta(k) \, |T_l(k)|^2$$

where T_l(k) is the transfer function from matter to CMB through coherence.

### Lagrangian Approach

Most promising: Scalar-tensor with correlation coupling

$$\mathcal{L}_{int} = \lambda \, \phi_C \int d^4x' \, K(x,x') \, T_{\mu\nu}(x) T^{\mu\nu}(x')$$

This directly encodes the correlation dependence.

**Key question:** What is K(x,x')?

### Nucleosynthesis Approach

Most promising: Accept partial limitation

The exact D/H ratio may be hard to explain.
Focus on He/H ratio, which is less constraining.
Acknowledge this as an area for future work.

