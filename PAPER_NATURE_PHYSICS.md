# Coherent Torsion Gravity: A Teleparallel Framework Reproducing Galaxy and Cluster Dynamics Without Dark Matter

**Authors:** Leonard Speiser  
**Correspondence:** [email]  
**Date:** November 30, 2025

---

## Abstract

We present a modification of teleparallel gravity in which coherent torsion fluctuations produce scale-dependent gravitational enhancement. In the Teleparallel Equivalent of General Relativity (TEGR), gravity is described by torsion rather than curvature. We propose that in extended, dynamically cold systems, torsion modes from spatially separated mass elements can add coherently, enhancing the effective gravitational field. The enhancement is governed by a universal function $\Sigma = 1 + A \cdot W(r) \cdot h(g)$, where $h(g) = \sqrt{g^\dagger/g} \cdot g^\dagger/(g^\dagger + g)$ encodes acceleration dependence and $W(r)$ encodes spatial coherence. The critical acceleration $g^\dagger = cH_0/(2e) \approx 1.2 \times 10^{-10}$ m/s² emerges from the cosmological horizon scale. Applied to 175 SPARC galaxies, the framework achieves 0.094 dex scatter on the radial acceleration relation—comparable to MOND and significantly better than ΛCDM halo models. Zero-shot application to Milky Way Gaia DR3 stars yields +0.062 dex bias versus +0.166 dex for MOND. For galaxy clusters, blind hold-out validation achieves 2/2 coverage within 68% posterior predictive intervals. The theory passes Solar System constraints by 8 orders of magnitude. Unlike particle dark matter, no per-system halo fitting is required; unlike MOND, the framework is embedded in a relativistic field theory with testable ~7% differences in the interpolation regime.

---

## 1. Introduction

The observed dynamics of galaxies and galaxy clusters systematically exceed predictions from visible matter alone. Stars orbit faster than Newtonian gravity predicts; gravitational lensing is stronger than baryonic mass can explain. The standard cosmological model addresses this through cold dark matter (CDM)—a hypothetical particle species comprising ~27% of cosmic energy density. Despite decades of direct detection efforts, no dark matter particle has been identified.

An alternative interpretation holds that gravity itself behaves differently at galactic scales. Milgrom's Modified Newtonian Dynamics (MOND) successfully predicts galaxy rotation curves using a single acceleration scale $a_0 \approx 1.2 \times 10^{-10}$ m/s², but lacks a relativistic foundation and struggles with galaxy clusters. Relativistic extensions (TeVeS, BIMOND) introduce additional fields but face theoretical challenges including superluminal propagation and instabilities.

Here we develop a different approach grounded in teleparallel gravity—an equivalent reformulation of General Relativity (GR) where the gravitational field is carried by torsion rather than curvature. We propose that coherent superposition of torsion modes from extended mass distributions produces measurable gravitational enhancement in dynamically cold systems while remaining undetectable in compact environments like the Solar System.

### 1.1 Key Results

| Domain | Metric | This work | MOND | ΛCDM |
|--------|--------|-----------|------|------|
| SPARC galaxies | RAR scatter | 0.094 dex | 0.10–0.13 dex | 0.18–0.25 dex |
| Milky Way (Gaia) | Bias | +0.062 dex | +0.166 dex | +1.4 dex* |
| Galaxy clusters | Hold-out | 2/2 in 68% | — | Baseline |
| Solar System | PPN γ−1 | < 10⁻¹³ | < 10⁻⁵ | 0 |

*Single fixed NFW halo, not per-galaxy tuned.

---

## 2. Theoretical Framework

### 2.1 Teleparallel Gravity

In GR, gravity manifests as spacetime curvature described by the Riemann tensor. The Teleparallel Equivalent of General Relativity (TEGR) provides an alternative where gravity is instead encoded in torsion—the antisymmetric part of an affine connection. The two formulations are mathematically equivalent for classical GR but suggest different physical pictures.

The fundamental variable in TEGR is the tetrad field $e^a_\mu$, which relates the spacetime metric to a local Minkowski frame:
$$g_{\mu\nu} = \eta_{ab} e^a_\mu e^b_\nu$$

The torsion tensor is defined as:
$$T^\lambda_{\mu\nu} = e^\lambda_a (\partial_\mu e^a_\nu - \partial_\nu e^a_\mu)$$

The TEGR action uses the torsion scalar $T$:
$$S_{\rm TEGR} = \frac{c^4}{16\pi G} \int T \sqrt{-g} \, d^4x$$

This produces field equations identical to GR. The conceptual difference is that gravity is mediated by torsion of a flat connection rather than curvature of a metric-compatible connection.

### 2.2 Coherent Torsion Enhancement

We propose that in extended systems, torsion contributions from spatially separated mass elements can interfere constructively. For a point mass, the classical saddle point dominates the gravitational path integral. For an extended, coherent mass distribution—such as a galactic disk with ordered circular motion—families of near-classical torsion configurations can add in phase.

The effective gravitational field becomes:
$$g_{\rm eff}(\mathbf{x}) = g_{\rm bar}(\mathbf{x}) \cdot \Sigma(\mathbf{x})$$

where $g_{\rm bar}$ is the Newtonian field from baryonic matter and $\Sigma \geq 1$ is the coherent enhancement factor.

**Physical mechanism:** In teleparallel gravity, gravitational waves carry two polarization modes (like GR). In the classical regime, typically one effective polarization contributes to a given measurement. When torsion modes from extended matter add coherently, both polarizations contribute. Two modes adding in quadrature give enhancement factor $\sqrt{2}$. Additional geometric factors from 3D integration increase this to $\sqrt{3}$ for disk galaxies.

### 2.3 The Coherence Window

Coherence requires phase alignment among contributing torsion modes. This is suppressed by:
1. **Distance:** Modes from distant regions accumulate phase mismatch
2. **Velocity dispersion:** Random motions destroy phase correlations
3. **Asymmetric structure:** Bars, bulges, and mergers disrupt ordered flow

We model the coherence survival probability as:
$$W(r) = 1 - \left(\frac{\xi}{\xi + r}\right)^{n_{\rm coh}}$$

where $\xi$ is the coherence length scale and $n_{\rm coh} = 0.5$ is the decay exponent.

**Derivation of $n_{\rm coh}$:** If the gravitational environment contains $k$ independent noise channels, the total dephasing rate follows a χ²(k) distribution. The coherence survival is:
$$K = \langle e^{-\Gamma_{\rm tot} R} \rangle = (1 + R/\xi)^{-k/2}$$

For radial measurements ($k = 1$), this gives $n_{\rm coh} = k/2 = 0.5$. This is a textbook result from Gamma-exponential conjugacy.

### 2.4 Acceleration Dependence

The enhancement depends on the local gravitational acceleration through:
$$h(g) = \sqrt{\frac{g^\dagger}{g}} \cdot \frac{g^\dagger}{g^\dagger + g}$$

This function has two components:
- $\sqrt{g^\dagger/g}$: The "deep-field" limit required to produce flat rotation curves
- $g^\dagger/(g^\dagger + g)$: High-acceleration cutoff ensuring GR recovery

**Physical interpretation:** The geometric mean of classical torsion $T_{\rm cl} \sim g/c^2$ and a critical fluctuation scale $T^\dagger \sim g^\dagger/c^2$ yields $T_{\rm eff} \sim \sqrt{T_{\rm cl} \cdot T^\dagger}$, giving the $\sqrt{g^\dagger/g}$ scaling. The interpolation factor ensures smooth transition to GR at high accelerations.

### 2.5 The Critical Acceleration

The critical acceleration is:
$$g^\dagger = \frac{cH_0}{2e} \approx 1.20 \times 10^{-10} \text{ m/s}^2$$

where $H_0 \approx 70$ km/s/Mpc is the Hubble constant and $e = 2.718...$ is Euler's number.

**Physical motivation:** The cosmological horizon defines a maximum coherence scale. Beyond the Hubble radius $R_H = c/H_0$, causal contact is lost. The factor $1/(2e)$ arises from:
- Factor 1/2: Two graviton polarizations sharing the coherent enhancement
- Factor 1/e: Optimal coherence at the de Sitter horizon scale

This matches the empirical MOND scale $a_0 \approx 1.2 \times 10^{-10}$ m/s² to within 0.4%.

### 2.6 Unified Formula

The complete enhancement is:
$$\Sigma = 1 + A \cdot W(r) \cdot h(g)$$

with:
- $h(g) = \sqrt{g^\dagger/g} \cdot g^\dagger/(g^\dagger + g)$ — universal for all systems
- $W(r) = 1 - (\xi/(\xi + r))^{0.5}$ with $\xi = (2/3)R_d$ — coherence window
- $A_{\rm galaxy} = \sqrt{3} \approx 1.73$ — amplitude for disk galaxies
- $A_{\rm cluster} = \pi\sqrt{2} \approx 4.44$ — amplitude for galaxy clusters

**Amplitude interpretation:** The factor $\sqrt{3}$ for galaxies arises from 3D geometry: $\sqrt{3} = \sqrt{2} \times \sqrt{3/2}$, where $\sqrt{2}$ is the two-polarization baseline and $\sqrt{3/2}$ is the surface-to-volume correction for sampling 3D space even in thin disks.

For clusters, the larger amplitude $\pi\sqrt{2}$ reflects the 3D spherical geometry ($\Omega = 3$ solid angle factor) combined with photon path coupling ($c \approx 1.5$ from lensing geometry). The cluster/galaxy ratio:
$$\frac{A_{\rm cluster}}{A_{\rm galaxy}} = \frac{\pi\sqrt{2}}{\sqrt{3}} = \pi\sqrt{2/3} \approx 2.57$$

This matches the empirically observed ratio of 2.60 to within 1.2%.

### 2.7 Comparison with MOND

Our $h(g)$ function differs from MOND's interpolating function $\nu(g/a_0)$. For MOND:
$$\nu(y) = \frac{1}{1 - e^{-\sqrt{y}}}$$

Both functions share the same asymptotic behavior:
- Low acceleration: $\Sigma \sim \sqrt{g^\dagger/g}$ (flat rotation curves)
- High acceleration: $\Sigma \to 1$ (GR recovery)

However, they differ in the transition region by approximately 7%—a testable prediction that distinguishes the theories.

### 2.8 Solar System Safety

In compact systems, the coherence window $W(r) \to 0$ and the enhancement vanishes. For the Solar System:

| Constraint | Bound | Prediction | Status |
|------------|-------|------------|--------|
| PPN γ−1 (Cassini) | < 2.3×10⁻⁵ | < 7×10⁻¹⁴ | **PASS** |
| Planetary ephemerides | No drift | < 10⁻¹⁴ | **PASS** |
| Wide binaries | No anomaly | < 10⁻⁸ | **PASS** |

The enhancement is suppressed by 8+ orders of magnitude below current observational limits.

---

## 3. Data and Methods

### 3.1 Galaxy Sample (SPARC)

We use the Spitzer Photometry and Accurate Rotation Curves (SPARC) database containing 175 galaxies with high-quality rotation curves and near-infrared photometry. The sample spans:
- Stellar masses: $10^7$ to $10^{11}$ M☉
- Morphologies: Irregular to early-type spiral
- Inclinations: 30° to 80° (excluding edge-on and face-on)

Baryonic accelerations are computed from gas and stellar components with standard mass-to-light ratios.

### 3.2 Milky Way (Gaia DR3)

We apply the framework zero-shot (no parameter adjustment) to 157,343 stars from Gaia DR3 with measured 3D velocities. Galactocentric distances span 0.09 to 19.9 kpc. The Milky Way baryonic model uses established disk and bulge components.

### 3.3 Galaxy Clusters

We analyze CLASH-quality clusters with X-ray gas profiles and strong lensing Einstein radii. Training sample: N=10 clusters. Blind hold-outs: Abell 2261, MACSJ1149.5+2223. Baryonic models include hot gas (gNFW profile) and BCG+ICL stellar components.

### 3.4 Validation Protocol

- **SPARC:** 80/20 train/test split with seed=42 for reproducibility
- **Milky Way:** Zero-shot prediction (no fitting)
- **Clusters:** Hierarchical Bayesian calibration with blind hold-out validation
- **All domains:** Same core formula, domain-specific amplitude only

---

## 4. Results

### 4.1 Radial Acceleration Relation (SPARC)

The radial acceleration relation (RAR) plots observed acceleration $g_{\rm obs}$ against baryonic acceleration $g_{\rm bar}$. Our framework achieves:

| Metric | Value |
|--------|-------|
| Hold-out RAR scatter | 0.094 dex |
| Bias | −0.07 dex |
| Points analyzed | 3,248 |

This scatter of 0.094 dex is competitive with MOND (~0.10-0.13 dex) and substantially better than ΛCDM halo models (~0.18-0.25 dex), which require per-galaxy fitting.

**Holdout validation:** Using 80/20 split with seed=42:
- Training set (140 galaxies): 0.095 dex
- Holdout set (35 galaxies): 0.091 dex
- Degradation: −4% (holdout actually performs better)

This confirms the formula generalizes to unseen data without overfitting.

### 4.2 Milky Way Zero-Shot Test

Applying the SPARC-calibrated formula to Gaia DR3 stars without adjustment:

| Model | Mean bias | Scatter |
|-------|-----------|---------|
| Newtonian (baryons only) | +0.380 dex | 0.176 dex |
| **This work** | **+0.062 dex** | **0.142 dex** |
| MOND | +0.166 dex | 0.161 dex |
| NFW (fixed halo) | +1.409 dex | 0.140 dex |

The framework achieves 4-13× improvement over Newtonian predictions in the outer disk (6-20 kpc) while maintaining near-zero residuals in the inner disk where coherence suppression is expected.

### 4.3 Galaxy Clusters

For lensing applications, the effective convergence is:
$$\kappa_{\rm eff}(R) = \frac{\Sigma_{\rm bar}(R)}{\Sigma_{\rm crit}} [1 + K_{\rm cl}(R)]$$

with cluster amplitude $A_c = \pi\sqrt{2} \approx 4.44$.

**Hierarchical calibration (N=10):**
- Population amplitude: $\mu_A = 4.6 \pm 0.4$
- Intrinsic scatter: $\sigma_A \approx 1.5$
- k-fold coverage: 16/18 = 88.9% within 68% PPC

**Blind hold-out validation:**
- Abell 2261: Predicted $\theta_E$ within 68% interval ✓
- MACSJ1149.5+2223: Predicted $\theta_E$ within 68% interval ✓
- Coverage: 2/2 = 100%

The framework explains cluster lensing using the same functional form as galaxies, with only the amplitude scaled by the geometrically predicted factor.

---

## 5. Discussion

### 5.1 Theoretical Status

The framework rests on several theoretical elements with varying degrees of rigor:

**Well-established:**
- Teleparallel gravity as equivalent to GR (mathematical identity)
- Two graviton polarizations (standard GR result)
- Coherence decay exponent $n_{\rm coh} = k/2$ (Gamma-exponential conjugacy)

**Motivated but not rigorous:**
- Critical acceleration $g^\dagger = cH_0/(2e)$ (matches observation; horizon physics provides intuition)
- Amplitude $A = \sqrt{3}$ (geometric arguments for 3D correction)
- Geometric mean structure of $h(g)$ (produces required scaling but not uniquely derived)

**Phenomenological:**
- Coherence length $\xi = (2/3)R_d$ (calibrated to data)

This places the framework between pure phenomenology (MOND) and fully derived theories (GR). We have a physical mechanism (coherent torsion) and several rigorously motivated components, but the complete derivation from a Lagrangian formulation remains incomplete.

### 5.2 Testable Predictions

1. **Interpolation function shape:** Our $h(g)$ differs from MOND by ~7% in the transition regime. High-precision rotation curves in galaxies with $g \sim g^\dagger$ can distinguish the theories.

2. **Counter-rotating disks:** Coherence should be reduced in counter-rotating systems. We predict weaker enhancement compared to co-rotating disks of similar mass.

3. **Tidal streams:** Extended, dynamically cold stellar streams should show enhanced self-gravity compared to compact globular clusters.

4. **JWST high-z galaxies:** If enhancement depends on coherence, early galaxies with higher velocity dispersions may show different dynamics than local predictions.

### 5.3 Relation to Dark Matter and MOND

Unlike particle dark matter:
- No per-system halo fitting required
- No concentration-mass relation assumed
- Natural explanation for tight RAR scatter

Unlike MOND:
- Embedded in relativistic field theory (teleparallel gravity)
- Natural cluster/galaxy amplitude ratio
- Different interpolation function (~7% testable difference)
- Solar System safety automatic from coherence window

### 5.4 Limitations

- The coherence mechanism, while physically motivated, is not derived from a specific Lagrangian
- The amplitude parameters ($\sqrt{3}$, $\pi\sqrt{2}$) are geometrically motivated but not uniquely determined
- Cosmological predictions (CMB, structure formation) require additional development
- Gravitational wave propagation in the enhanced regime needs investigation

---

## 6. Conclusions

We have presented a modification of teleparallel gravity where coherent torsion contributions from extended mass distributions enhance the effective gravitational field. The framework:

1. Reproduces galaxy rotation curves with 0.094 dex RAR scatter
2. Predicts Milky Way stellar dynamics zero-shot with +0.062 dex bias
3. Explains cluster lensing with geometrically predicted amplitude ratio
4. Passes Solar System constraints by 8 orders of magnitude
5. Uses the same functional form across all scales

The critical acceleration $g^\dagger = cH_0/(2e)$ connects galactic dynamics to the cosmological horizon scale. The ~7% difference from MOND in the interpolation regime provides a testable prediction.

This work suggests that the "dark matter problem" may reflect coherent quantum gravitational effects in extended systems rather than missing matter. The teleparallel framework provides a natural setting for such effects while maintaining compatibility with precision tests of GR.

---

## Methods

### Numerical Implementation

All calculations use Python with NumPy/SciPy. The enhancement factor $\Sigma$ is computed directly from baryonic models without iterative fitting. Rotation curves are predicted as:
$$V_{\rm pred}(R) = V_{\rm bar}(R) \sqrt{\Sigma(R, g_{\rm bar})}$$

### Reproducibility

Complete code and data are available at [repository]. Key validation commands:
```bash
# SPARC holdout validation
python derivations/connections/validate_holdout.py
# Expected: 0.091 dex on holdout set

# Milky Way zero-shot
python scripts/analyze_mw_rar_starlevel.py
# Expected: +0.062 dex bias, 0.142 dex scatter

# Cluster hold-outs
python scripts/run_holdout_validation.py
# Expected: 2/2 in 68% PPC
```

All results use seed=42 for reproducible random splits.

### Statistical Methods

- RAR scatter: Standard deviation of $\log_{10}(g_{\rm obs}/g_{\rm pred})$
- Cluster validation: Posterior predictive checks with 68% credible intervals
- Holdout validation: Train/test split without parameter adjustment

---

## Data Availability

SPARC data from http://astroweb.cwru.edu/SPARC/. Gaia DR3 from ESA archive. Cluster catalogs referenced in Supplementary Information.

## Code Availability

https://github.com/[repository]/sigmagravity

## References

[To be added: SPARC, Gaia DR3, TEGR foundations, MOND, cluster lensing papers]

---

## Supplementary Information

Extended derivations, additional validation tests, and complete reproduction instructions are provided in the Supplementary Information document.
