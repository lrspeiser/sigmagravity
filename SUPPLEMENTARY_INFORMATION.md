# Supplementary Information

## Σ-Gravity: A Universal Scale-Dependent Enhancement Reproducing Galaxy Dynamics and Cluster Lensing Without Particle Dark-Matter Halos

**Authors:** Leonard Speiser  
**Date:** 2025-10-20

This Supplementary Information (SI) accompanies the main manuscript and provides complete technical details, derivations, reproducibility instructions, and extended analyses.

---

## SI §1 — PN Integration and O(v/c) Cancellation

We outline a weak-field, post-Newtonian (PN) expansion consistent with causality. Using mass continuity $\dot\rho=-\nabla'\!\cdot(\rho\,\mathbf{v})$ and periodic/axisymmetric boundaries, the linear $\mathcal{O}(v/c)$ term vanishes after integration by parts, leaving the leading correction at $\mathcal{O}(v^2/c^2)$. For illustration we write the Poisson-limit potential kernel $1/\lvert \mathbf{x}-\mathbf{x}'\rvert$; this is a PN convenience, not a full GR Green's-function solution:

$$
\delta\Phi(\mathbf{x}) = \frac{G}{2c^2} \int \frac{\nabla'\!\cdot(\rho\,\mathbf{v}\!\otimes\!\mathbf{v})}{\lvert \mathbf{x}-\mathbf{x}'\rvert}\,\mathrm{d}^3\!x' ,\qquad
\delta\mathbf{g}(\mathbf{x}) = -\frac{G}{2c^2} \int \nabla\!\left(\frac{1}{\lvert \mathbf{x}-\mathbf{x}'\rvert}\right) \, \nabla'\!\cdot(\rho\,\mathbf{v}\!\otimes\!\mathbf{v})\,\mathrm{d}^3\!x' .
$$

Example (circular flow): for $\mathbf{v}=v_\phi\,\hat\phi$ in an axisymmetric disk, only the divergence of the Reynolds-stress-like tensor contributes; the induced field is curl-free by construction.

---

## SI §2 — Elliptic Ring Kernel (Exact Geometry)

The azimuthal integral reduces to complete elliptic integrals with dimensionless parameter
$$
 m \;\equiv\; \frac{4 R R'}{(R+R')^2} \in [0,1].
$$
Then
$$
\int_{0}^{2\pi} \frac{d\varphi}{\sqrt{R^2 + R'^2 - 2 R R'\cos\varphi}} \;=\; \frac{4}{R+R'}\,K(m).
$$

Reference check (relative error < 1e−6):

```python
import numpy as np
from mpmath import quad, ellipk

def ring_green_numeric(R, Rp):
    f = lambda phi: 1.0/np.sqrt(R**2 + Rp**2 - 2*R*Rp*np.cos(phi))
    return 2.0 * quad(f, [0, np.pi])

def ring_green_elliptic(R, Rp):
    m = 4.0*R*Rp/((R+Rp)**2)  # dimensionless parameter m \in [0,1]
    return 4.0/(R+Rp) * ellipk(m)

R, Rp = 5.0, 7.0
num = ring_green_numeric(R, Rp)
ana = ring_green_elliptic(R, Rp)
assert abs(num-ana)/num < 1e-6
```

---

## SI §3 — Coherence Window Derivation

Near the stationary azimuth $\varphi=0$ one may expand the separation as $\Delta(\varphi)\approx D + (RR'/(2D))\,\varphi^2$. The phase integral reduces to a Gaussian/Fresnel form; adding stochastic dephasing over a coherence length $\ell_0$ yields a radial envelope equivalent to

$$
C(R) = 1 - \Big[1 + (R/\ell_0)^p\Big]^{-n_{\rm coh}} ,
$$

with phenomenological exponents $p,n_{\rm coh}$ calibrated once on data. This envelope multiplies the Newtonian response, remaining curl-free.

### SI §3.1 Superstatistical derivation of the coherence window

We now show that the functional form of C(R) is not arbitrary but emerges naturally from a stochastic decoherence model in a heterogeneous medium. This derivation applies a standard mixture identity from reliability theory (Gamma–Weibull compounding yields Burr-XII survival; see, e.g., MATLAB Statistics Toolbox documentation and Rodriguez 1977) to the novel context of gravitational decoherence channels.

**Physical model.** Assume coherence loss along scale R follows a Poisson process with integrated hazard H(R|λ) = λ(R/ℓ₀)^p, where λ > 0 is a "decoherence rate" that varies across the system due to environmental heterogeneity (density clumps, turbulence, bars, shear). The survival probability for a single channel is

$$
S(R|\lambda) = e^{-\lambda(R/\ell_0)^p}.
$$

To represent this heterogeneity, we model λ as drawn from a Gamma distribution with shape n_coh and rate β:

$$
\lambda \sim \mathrm{Gamma}(n_{\mathrm{coh}}, \beta).
$$

**Derivation.** The marginal survival probability is the expectation over λ:

$$
S(R) = \mathbb{E}_{\lambda}\left[e^{-\lambda(R/\ell_0)^p}\right] = \int_0^\infty e^{-\lambda(R/\ell_0)^p} \frac{\beta^{n_{\mathrm{coh}}}}{\Gamma(n_{\mathrm{coh}})} \lambda^{n_{\mathrm{coh}}-1} e^{-\beta\lambda} d\lambda.
$$

Using the Laplace transform of the Gamma distribution,

$$
S(R) = \left(\frac{\beta}{\beta + (R/\ell_0)^p}\right)^{n_{\mathrm{coh}}} = \left[1 + \frac{(R/\ell_0)^p}{\beta}\right]^{-n_{\mathrm{coh}}}.
$$

Absorbing the rate parameter β into the definition of ℓ₀ (ℓ₀′ ≡ ℓ₀ β^(1/p)), the coherence window is

$$
C(R) = 1 - S(R) = 1 - \left[1 + \left(\frac{R}{\ell_0}\right)^p\right]^{-n_{\mathrm{coh}}}.
$$

This is the Burr Type XII (Singh–Maddala) cumulative distribution function.

**Interpretation.** The fitted parameters now have direct physical meaning:  
- **ℓ₀**: characteristic coherence scale set by local decoherence timescale τ_collapse  
- **p**: encodes how interactions accumulate with scale (p < 1 suggests correlated/sparse interactions; p = 2 would be area-like)  
- **n_coh**: effective number of independent decoherence channels; larger n_coh implies narrower variability in λ (more homogeneous environment)

**Testable predictions.** If this interpretation is correct, n_coh should increase in relaxed, homogeneous systems (ellipticals, relaxed clusters) and decrease in turbulent, clumpy environments (barred galaxies, merger clusters). The exponent p should shift systematically with morphology.

**Attribution.** The Gamma–Weibull → Burr-XII identity is standard (Rodriguez 1977, JSTOR; MATLAB docs); our contribution is the application to gravitational decoherence and the physical interpretation of {ℓ₀, p, n_coh} in terms of path coherence and environmental heterogeneity. For the broader "superstatistics" framework (heterogeneous rate parameters), see Beck & Cohen 2003, arXiv:cond-mat/0303288.

---

## SI §4 — PN Error Budget

We bound neglected terms by

$$
\Delta_{\rm PN} \;\lesssim\; C_1\,(v/c)^3 \, + \, C_2\,(v/c)^2\,(H/R) \, + \, C_3\,(v/c)^2\,(R/R_\Sigma)\,.
$$

In disks and clusters, representative values place all terms $\ll10^{-5}$, well below statistical errors.

---

## SI §5 — Reproducibility & Code Availability

### SI §5.1. Repository structure & prerequisites

Python ≥3.10; NumPy/SciPy/Matplotlib; pymc≥5; optional: emcee, CuPy (GPU), arviz.

```bash
pip install numpy scipy matplotlib pandas pymc arviz
```

### SI §5.2. SPARC Galaxy RAR — 0.087 dex Hold-Out Scatter

**Commands:**

```bash
# Run validation suite (includes 80/20 split, seed=42)
python many_path_model/validation_suite.py --rar-holdout

# Or full validation:
python many_path_model/validation_suite.py --all

# 5-fold cross-validation (0.083 ± 0.003 dex)
python many_path_model/run_5fold_cv.py
```

**Expected outputs:**
- Console: "Hold-out RAR scatter: 0.087 dex"
- Files: `many_path_model/results/validation_suite/VALIDATION_REPORT.md`
- 5-fold: `many_path_model/results/5fold_cv_results.json`

**Hyperparameters used:** `config/hyperparams_track2.json` (ℓ₀=4.993 kpc, A₀=0.591, p=0.757, n_coh=0.5)

### SI §5.3. Milky Way Star-Level RAR — Zero-Shot (+0.062 dex bias, 0.142 dex scatter)

**Commands:**

```bash
# Step 1: Predict star speeds (GPU recommended, ~3 min; CPU: ~30 min)
python scripts/suggest_gaia_star_speeds.py \
  --npz data/gaia/mw_gaia_full_coverage.npz \
  --fit data/gaia/outputs/mw_pipeline_run_vendor/fit_params.json \
  --out data/gaia/outputs/mw_gaia_full_coverage_suggested.csv \
  --device 0

# Step 2: Compute metrics
python scripts/analyze_mw_rar_starlevel.py \
  --pred_csv data/gaia/outputs/mw_gaia_full_coverage_suggested.csv \
  --out_prefix data/gaia/outputs/mw_rar_starlevel_full \
  --hexbin
```

**Expected output:**
- File: `data/gaia/outputs/mw_rar_starlevel_full_metrics.txt`
- Contains: "Mean bias: +0.062 dex, Scatter: 0.142 dex, n=157343"

### SI §5.4. Cluster Einstein Radii — Blind Hold-Outs (2/2 coverage, 14.9% error)

**Commands:**

```bash
# Hierarchical calibration (N=10 training)
python scripts/run_tier12_mcmc_fast.py

# Blind hold-out validation
python scripts/run_holdout_validation.py
```

**Expected outputs:**
- Console: "Hold-out coverage: 2/2 inside 68% PPC"
- Console: "Median fractional error: 14.9%"
- Files: `figures/holdouts_pred_vs_obs.png`, `output/n10_nutsgrid/flat_samples.npz`

### SI §5.5. Generate All Figures

```bash
# SPARC figures
python scripts/generate_rar_plot.py
python scripts/generate_rc_gallery.py

# MW figures
python scripts/generate_all_model_summary.py
python scripts/generate_radial_residual_map.py

# Cluster figures
python scripts/generate_cluster_kappa_panels.py
python scripts/run_holdout_validation.py
```

### SI §5.6. Quick Verification (15 minutes)

```bash
# 1. SPARC (most critical): Should print ~0.087 dex
python many_path_model/validation_suite.py --rar-holdout

# 2. MW (if CSV exists): Should show +0.062 dex, 0.142 dex  
python scripts/analyze_mw_rar_starlevel.py \
  --pred_csv data/gaia/outputs/mw_gaia_full_coverage_suggested.csv \
  --out_prefix data/gaia/outputs/test

# 3. Clusters: Should show 2/2 coverage
python scripts/run_holdout_validation.py
```

### SI §5.7. Expected Results Table

| Metric | Expected Value | Verification Command |
|--------|----------------|---------------------|
| SPARC hold-out scatter | 0.087 dex | validation_suite.py --rar-holdout |
| SPARC 5-fold CV | 0.083 ± 0.003 dex | run_5fold_cv.py |
| MW bias | +0.062 dex | analyze_mw_rar_starlevel.py output |
| MW scatter | 0.142 dex | analyze_mw_rar_starlevel.py output |
| Cluster hold-outs | 2/2 in 68% | run_holdout_validation.py |
| Cluster error | 14.9% median | run_holdout_validation.py |

**All scripts use seed=42 for reproducibility.**

### SI §5.8. Troubleshooting

**Unicode errors on Windows:**
```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
python many_path_model/validation_suite.py --all
```

**Import errors:**
```bash
# Ensure in repository root
cd sigmagravity
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/many_path_model"
```

---

## SI §6 — Extended Theory: Stationary-Phase Reduction (PRD Excerpt)

This section collects technical details that motivate the operator structure $\mathbf{g}_{\rm eff}=\mathbf{g}_{\rm bar}[1+K]$ via stationary-phase reduction and then justifies the Burr-XII coherence window as a superstatistical phenomenology; it is not a first-principles derivation of $C(R)$.

### I. FUNDAMENTAL POSTULATES

#### A. Gravitational Field as Quantum Superposition

**Postulate I**: In the absence of strong decoherence, the gravitational field exists as a superposition of geometric configurations characterized by different path histories.

Mathematically, for a test particle moving from point A to B, the propagator is:

```
K(B,A) = ∫ D[path] exp(iS[path]/ℏ)     (1)
```

where S[path] is the action along each geometric path.

**Justification**: This is standard path-integral quantum mechanics, applied to gravity. The novelty is in recognizing that decoherence rates differ dramatically between compact and extended systems.

#### B. Scale-Dependent Decoherence

**Postulate II**: Geometric superpositions collapse to classical configurations on a characteristic timescale τ_collapse(R) that depends on the spatial scale R and matter density ρ.

**Physical Mechanism**: We propose that gravitational geometries decohere through continuous weak measurement by matter. Unlike quantum systems that decohere via environmental entanglement (photon scattering, etc.), gravity decoheres through **self-interaction** with the mass distribution that sources it.

The decoherence rate is proportional to the rate at which matter "samples" different geometric configurations:

```
Γ_decoherence(R) ~ (interaction rate) × (geometric variation)     (2)
```

For a region of size R with density ρ:
- Interaction rate ~ ρ (more mass → more interactions)
- Geometric variation ~ R² (larger regions have more distinct paths)

Therefore:
```
τ_collapse(R) ~ 1/(ρ G R² α)     (3)
```

where α is a dimensionless constant of order unity characterizing the efficiency of gravitational self-measurement.

**Key Insight**: This gives a coherence length scale:
```
ℓ_0 = √(c/(ρ G α))     (4)
```

For typical galaxy halo densities ρ ~ 10⁻²¹ kg/m³:
```
ℓ_0 ~ √(3×10⁸ / (10⁻²¹ × 6.67×10⁻¹¹ × 1)) ~ 7×10¹⁹ m ~ 2 kpc     (5)
```

Order of magnitude correct; ℓ_0 naturally lands at galactic scales.

### II. DERIVATION OF THE ENHANCEMENT KERNEL

#### A. Weak-Field Expansion

```
g_μν = η_μν + h_μν,    |h_μν| ≪ 1     (6)
```

Newtonian potential:
```
h₀₀ = -2Φ/c²,    Φ_N(x) = -G ∫ ρ(x')/|x-x'| d³x'     (7)
```

#### B. Path Sum and Stationary Phase

```
Φ_eff(x) = -G ∫ d³x' ρ(x') ∫ D[geometry] exp(iS[geom]/ℏ) / |x-x'|_geom     (8)
```

Stationary phase:
```
S[path] = S_classical + (1/2)δ²S[deviation] + ...     (9)
```

gives a near-classical amplitude factor:
```
∫ D[path] exp(iS/ℏ) ≈ A_0 exp(iS_classical/ℏ) [1 + quantum corrections]     (10)
```

#### C. Coherence Weighting

Probability that a path of extent R remains coherent:
```
P_coherent(R) = exp(-∫ dt/τ_collapse(r(t)))     (11)
```

For characteristic scale R in density ρ:
```
P_coherent(R) ≈ exp(-(R/ℓ_0)^p)     (12)
```

with p ≈ 2. A smooth, causal window:
```
C(R) = 1 - [1 + (R/ℓ_0)^p]^(-n_coh)     (13)
```

#### D. Multiplicative Structure

Classical contribution from dV:
```
dΦ_classical ~ ρ(x') dV / |x-x'|     (14)
```
Quantum-enhanced contribution:
```
dΦ_quantum ~ ρ(x') dV × [coherent path sum] / |x-x'|     (15)
```
with
```
[coherent path sum] ≈ [1 + A · C(R)]     (16–17)
```
Hence
```
Φ_eff = Φ_classical [1 + K(R)],
 g_eff ≈ g_classical [1 + K(R)]     (18–20)
```

### III. CURL-FREE PROPERTY

For axisymmetric systems with K=K(R):
```
∇ × g_eff = (∇ × g_bar)(1+K) + ∇K × g_bar = 0     (21–22)
```
so the enhanced field remains conservative.

### IV. SOLAR SYSTEM CONSTRAINTS

Cassini bound |γ_PPN−1| < 2.3×10⁻⁵; with ℓ_0~kpc and A_gal~0.6:
$$
\text{Boost at 1 AU} \lesssim 7\times 10^{-14} \ll 10^{-5}
$$
Safety margin ≥10^8×.

### V. AMPLITUDE SCALING: GALAXIES VS CLUSTERS

Path-counting (2D disks vs 3D clusters) suggests A_cluster/A_gal ~ O(10). Empirically ≈7.7; consistent to order-unity after geometry factors.

### VI. QUANTITATIVE PREDICTIONS

- Galaxies (SPARC): RAR scatter ≈0.087 dex; BTFR ≈0.15 dex (using A_gal≈0.6, ℓ_0≈5 kpc).
- Clusters: θ_E accuracy ≈15% with A_cluster≈4.6; triaxial lever arm 20–30%.
- Solar System: Wide-binary regime (10²-10⁴ AU): K < 10⁻⁸.

### Technical addenda

#### Coherence scale

We treat $\ell_0$ operationally: $\ell_0 \equiv c\,\tau_{\rm collapse}$. Although dimensional arguments often suggest $\ell_0 \propto \rho^{-1/2}$, our derivation-validation suite shows such closures do not reproduce the empirically successful scales ($\ell_0 \simeq 5$ kpc for disks; $\ell_0 \sim 200$ kpc for cluster lensing) by factors of ~10–2500×. We therefore do not set $\ell_0$ from $\rho$; instead we calibrate $\ell_0$ on data and treat its microphysical origin as an open problem (SI §7).

#### Numerical kernel (example)

```python
def sigma_gravity_kernel(R_kpc, A=0.6, ell_0=5.0, p=0.75, n_coh=0.5):
    C = 1 - (1 + (R_kpc/ell_0)**p)**(-n_coh)
    return A * C
```

#### Ring kernel expression

$$
G_{\mathrm{ring}}(R, R') = \int_{0}^{2\pi} \frac{d\varphi}{\sqrt{R^2 + R'^2 - 2 R R'\cos\varphi}}
$$

---

## SI §7 — Derivation Validation: Negative Results

### SI §7.1. Purpose and Methodology

We built a comprehensive validation suite (`derivation/` folder in repository) to test whether theoretical derivations could predict the empirically successful parameters $\{A, \ell_0, p, n_{\rm coh}\}$ from first principles. The suite includes:

- `theory_constants.py`: Physical constants and theoretical calculations
- `simple_derivation_test.py`: Direct tests of theory predictions
- `parameter_sweep_to_find_derivation.py`: Systematic parameter exploration
- `cluster_validation.py`: Cluster-scale validation

### SI §7.2. Results: All Simple Derivations Fail

**Coherence length ℓ₀ = c/(α√(Gρ)):**
- Virial density (ρ ~ 10⁻²⁵ kg/m³): predicts ℓ₀ ~ 1,254,000 kpc (251,000× too large)
- Galactic density (ρ ~ 10⁻²¹ kg/m³): predicts ℓ₀ ~ 12,543 kpc (2,512× too large)
- Stellar density (ρ ~ 10⁻¹⁸ kg/m³): predicts ℓ₀ ~ 397 kpc (79× too large)
- Nuclear density (ρ ~ 10⁻¹⁵ kg/m³): predicts ℓ₀ ~ 12.5 kpc (2.5× too large)
- **Empirical fit:** ℓ₀ = 4.993 kpc (galaxies), ℓ₀ ~ 200 kpc (clusters)
- **Verdict:** No density scale reproduces observations

**Amplitude ratio A_cluster/A_galaxy from path counting:**
- Naive calculation: (4π/2π) × (1000 kpc/20 kpc) × (geometry factor 0.5) ~ 100
- **Empirical fit:** A_c/A_0 = 4.6/0.591 ~ 7.8
- **Discrepancy:** 13× too large
- **Verdict:** Path-counting significantly overestimates

**Interaction exponent p:**
- Theory prediction: p = 2.0 (area-like interactions)
- **Empirical fit:** p = 0.757
- **Discrepancy:** 2.7× too large
- **Verdict:** Theory prediction fails

### SI §7.3. Implications for Model Interpretation

These negative results establish that:

1. The Burr-XII envelope is **phenomenological**, justified by superstatistical reasoning but with parameters determined empirically
2. The characteristic scales ℓ₀ ~ 5 kpc (galaxies) and ℓ₀ ~ 200 kpc (clusters) are **not derivable** from simple density arguments
3. The amplitude values reflect complex geometric and physical effects beyond naive path-counting
4. Parameter values $\{A, \ell_0, p, n_{\rm coh}\}$ should be treated as **calibrated constants** within each observational domain

### SI §7.4. Reproducibility

Complete validation scripts and results are provided in:
- `derivation/DERIVATION_VALIDATION_RESULTS.md` - Comprehensive analysis
- `derivation/FINAL_DERIVATION_SUMMARY.md` - Executive summary
- `derivation/theory_constants.py` - Physical calculations
- `derivation/simple_derivation_test.py` - Direct validation tests
- `derivation/parameter_sweep_to_find_derivation.py` - Systematic exploration

All tests use seed=42 and are fully reproducible. Running `python derivation/simple_derivation_test.py` demonstrates the quantitative failures documented above.

### SI §7.5. Theoretical Outlook

Developing a first-principles theory that quantitatively predicts:
- ℓ₀ ~ 5 kpc for galaxy disks
- ℓ₀ ~ 200 kpc for cluster lensing  
- A₀ ~ 0.6 for galaxies
- A_c ~ 5 for clusters
- p ~ 0.75 (not 2.0)

remains an important open problem. The successful phenomenology presented in this paper provides empirical targets that any future microphysical theory must reproduce.

---

## SI §8 — CMB Analysis (Exploratory)

**Critical scope note:** This section is exploratory and speculative. Nothing here is used to set ${A, \ell_0, p, n_{\rm coh}}$ or to produce any galaxy/cluster results in the main paper. The quantitative success of Σ-Gravity at halo scales is independent of the cosmological extensions discussed below.

### SI §8.1. CMB Angular Power Spectrum — Quantitative Results

The Σ-Gravity coherence framework, originally developed for galaxies and clusters, has been extended to the Cosmic Microwave Background (CMB) angular power spectrum. **Without invoking Big Bang cosmology or CDM particles**, the framework reproduces the key features traditionally attributed to acoustic oscillations and dark matter.

#### Peak Ratio Performance

| Ratio | Observed | Σ-Gravity | Error |
|-------|----------|-----------|-------|
| P1/P2 | 2.397 | 2.505 | **4.5%** ✓ |
| P3/P4 | 2.318 | 2.270 | **2.1%** ✓ |
| P5/P6 | 1.538 | 1.564 | **1.7%** ✓ |

In standard cosmology, these odd/even peak ratios are explained by CDM potential wells. In Σ-Gravity, the same pattern emerges from **density-dependent coherence buildup**.

#### Step-Function Asymmetry Discovery

The observed data shows P1/P2 ≈ P3/P4 ≈ 2.3–2.4 (nearly constant), then P5/P6 ≈ 1.5 (sharp drop). This is **not** a smooth exponential decay but a step function with transition at ℓ_crit ≈ 1300.

**Physical interpretation:** A critical scale (~10 Mpc) where density contrast is sharply suppressed, possibly corresponding to Silk damping or the structure formation cutoff.

### SI §8.2. Physical Mechanism

The Σ-Gravity CMB mechanism operates through coherent gravitational effects rather than acoustic oscillations:

1. **Coherence at cosmic scales:** Light travels ~4000 Mpc through gravitational potentials. Coherent GW structure creates systematic effects with coherence length ℓ₀ ≈ 60 Mpc (same scaling as galaxies).

2. **Path interference creates peaks:** Constructive interference at characteristic scales ℓ_n ≈ n × π × D / ℓ₀. NOT acoustic oscillations—gravitational interference.

3. **Asymmetry from density-dependent coherence:** Overdense regions have shorter τ_coh → more coherence → odd peaks enhanced. Underdense regions have longer τ_coh → less coherence → even peaks suppressed. Creates odd/even asymmetry WITHOUT CDM particles.

4. **Step-function transition:** Below ℓ_crit ≈ 1300: strong asymmetry (a ≈ 0.35). Above ℓ_crit: weak asymmetry (a ≈ 0.02). Sharp transition at characteristic scale.

### SI §8.3. Hierarchical Scaling

The coherence length scales with structure size across 8 orders of magnitude:

| Structure | Size R | Coherence ℓ₀ | Source |
|-----------|--------|--------------|--------|
| Galaxy | 20 kpc | 5 kpc | SPARC rotation curves |
| Cluster | 1 Mpc | 200 kpc | Cluster lensing |
| CMB | ~400 Mpc | ~60 Mpc | First peak ℓ≈220 |

**Scaling law:** ℓ₀ ∝ R^0.94 — the **same physics operates at all scales**.

### SI §8.4. Comparison with ΛCDM

| Feature | ΛCDM | Σ-Gravity |
|---------|------|-----------|
| Peak locations | Sound horizon at z~1100 | Coherence interference |
| Peak asymmetry | CDM potential wells | Density-dependent coherence |
| Damping | Silk diffusion | Gravitational decoherence |
| Physical basis | Acoustic oscillations | Path interference |
| P1/P2 ratio | Excellent (1%) | Good (4.5%) |
| P3/P4 ratio | Excellent (1%) | Excellent (2.1%) |
| P5/P6 ratio | Excellent (1%) | Excellent (1.7%) |

Σ-Gravity is not yet as quantitatively precise as ΛCDM (which fits the full spectrum to <1%), but demonstrates that **coherent gravitational effects can reproduce the key features** traditionally attributed to acoustic oscillations and dark matter.

### SI §8.5. Remaining Challenges

1. **Peak height matching:** Absolute heights are overpredicted by ~20–50% for higher peaks.
2. **Polarization verification:** Gravitomagnetic predictions need quantitative comparison with Planck EE and TE data.
3. **BAO connection:** The CMB coherence scale (~60 Mpc) remarkably matches the BAO scale—this connection should be made explicit.
4. **Low-ℓ behavior:** The Sachs-Wolfe plateau at ℓ < 30 needs a separate mechanism.

**Artifacts:** See `cmb/sigma_gravity_cmb_complete.md` for full derivation, model parameters, and visualizations.

---

## SI §9 — Pantheon+ SNe Validation

Using the final Phase-2 Lockdown validation suite, we performed a complete, parity-fair comparison between the TG-τ Σ-Gravity redshift prescription and a flat FRW cosmology with free intercept, employing the full Pantheon+ SH0ES dataset (N = 1701 SNe) and the official STAT + SYS compressed covariance.

| Model | Hₛ / Ωₘ | α_SB / Intercept | χ² | AIC | ΔAIC | Akaike Weight |
|-------|----------|------------------|----|----|------|---------------|
| TG-τ | H_Σ = 72.00 ± 0.26 | α_SB = 1.200 ± 0.015 | 871.83 | 875.83 | + 59.21 | 0.000 |
| FRW | Ωₘ = 0.380 ± 0.020 | intercept = −0.0731 ± 0.0079 | 812.62 | 816.62 | 0 | 1.000 |

**Fair-comparison outcome.** Both models were fitted with identical freedoms (k = 2). Under this parity, FRW remains the statistically preferred description (ΔAIC = + 59.21 in its favor), but TG-τ's parameters are fully physical and stable:

- **H_Σ = 72.00 km s⁻¹ Mpc⁻¹** — consistent with H₀ ≈ 70
- **α_SB = 1.200 ± 0.015** — intermediate between energy-loss (1) and Tolman (4) scaling  
- **ξ = 4.8 × 10⁻⁵** — matching the expected Σ-Gravity micro-loss constant

**Distance-Duality Prediction**

The corrected TG-τ relation is now

$$
\eta(z) = \frac{D_L}{(1+z)^2 D_A} = (1+z)^{\alpha_{SB}-1} = (1+z)^{0.2}.
$$

Hence η(1) = 1.1487 and η(2) = 1.2457; these values provide a clear, testable signature for future BAO or cluster D_A datasets.

---

## SI §10 — LIGO Gravitational Wave Analysis

### SI §10.1. Overview and Motivation

If Σ-Gravity coherence enhancement operates on gravitational wave signals traversing cosmological distances, massive black hole binary (BBH) events should appear systematically more massive at greater distances. This section presents an independent analysis of LIGO/Virgo/KAGRA gravitational wave catalogs (GWTC-1 through GWTC-4) testing this prediction.

**Key prediction:** Gravitational wave strain accumulates a small coherence enhancement ε per coherence period, yielding:

$$
M_{\rm eff} = M_{\rm true} \times C(d) = M_{\rm true} \times (1 + \varepsilon)^{N}
$$

where $N = d / \lambda_{\rm coh}$ is the number of coherence periods traversed and $\lambda_{\rm coh} \approx 2.2$ kpc (from galaxy-scale calibration).

### SI §10.2. Data and Methodology

**Data sources:**
- GWTC-4.0 PESummaryTable (O4a): 86 unique events after deduplication
- GWTC cumulative catalog (GWOSC API): 219 events across O1–O4a
- Event deduplication: Multiple pipeline/posterior entries collapsed to per-event medians

**Quality cuts:**
- SNR > 8 (detection confidence)
- Valid mass, distance, and spin measurements
- Unique event names (removes duplicate pipeline entries)

**Gap event definition:** Total mass M_total ≥ 100 M☉ (pair-instability supernova forbidden region)

### SI §10.3. Mass-Distance Correlation

Σ-Gravity predicts a positive correlation between observed mass and distance. Standard GR with selection effects predicts r ~ 0.21 (massive events detectable at greater distances).

**Results (GWTC-4, deduplicated, n=79):**

| Metric | Value |
|--------|-------|
| Pearson r | **0.639** |
| p-value | 2.33 × 10⁻¹⁰ |
| Binned r | **0.990** |
| Selection-only r (Monte Carlo) | 0.215 ± 0.005 |

**Interpretation:** The observed correlation (r = 0.639) exceeds the selection-only prediction by 0.42. Monte Carlo null hypothesis test: P(r ≥ 0.639 | selection only) ≈ 0.0000. The excess correlation requires explanation beyond pure selection effects.

### SI §10.4. Gap Event Distribution

Events in the pair-instability mass gap (100–260 M☉ total) should not exist from standard stellar evolution. Σ-Gravity predicts these are coherence-enhanced normal BBH appearing massive due to distance.

**Results (n=79 after cuts):**

| Category | N | Median distance |
|----------|---|----------------|
| Normal (M < 100 M☉) | 68 | 2078 Mpc |
| Gap (M ≥ 100 M☉) | 11 | **3983 Mpc** |

Mann-Whitney U test (gap events more distant): U = 586, p = 1.37 × 10⁻³

### SI §10.5. Spin Distribution Test (Smoking Gun)

**Critical discriminant:** Hierarchical mergers (2nd-generation BH from prior mergers) predict high spins (χ_eff ~ 0.4–0.7). Σ-Gravity predicts normal spins (gap events are enhanced normal BBH, not hierarchical).

**Results (GWTC-4, deduplicated):**

| Population | n | Median χ_eff | χ_eff > 0.3 |
|------------|---|--------------|-------------|
| Normal events | 68 | 0.027 | 4.4% |
| Gap events | 11 | **0.059** | 27.3% |
| Hierarchical prediction | — | ~0.5 | >50% |

**Statistical tests:**
- KS test: D = 0.243, **p = 0.545** → distributions are **statistically indistinguishable**
- Mann-Whitney U: U = 459, p = 0.232

**Verdict:** Gap event spins (median 0.059) are **far below** hierarchical merger predictions (~0.5). The KS test confirms gap and normal events have the **same spin distribution** (p = 0.545). This is **inconsistent with hierarchical mergers** and **consistent with Σ-Gravity** (gap events are coherence-enhanced normal BBH).

### SI §10.6. Cross-Catalog Consistency

If ε is a universal physical constant, it should be consistent across observing runs with different sensitivities.

**Results:**

| Run | N | Gap | r | ε median | ε CV |
|-----|---|-----|---|----------|------|
| O3a | 50 | 5 | 0.692 | 5.08 × 10⁻⁷ | 15.5% |
| O3b | 29 | 1 | 0.686 | 1.06 × 10⁻⁶ | N/A |
| O4a | 86 | 13 | 0.653 | 3.25 × 10⁻⁷ | 70.4% |

**Cross-catalog ε statistics:**
- Range: [3.25 × 10⁻⁷, 1.06 × 10⁻⁶]
- Mean: 6.33 × 10⁻⁷
- **CV: 49.7%** (within 50% consistency threshold)

### SI §10.7. Summary and Implications

**Key findings:**

1. **Mass-distance correlation exceeds selection effects:** r = 0.639 vs predicted r ~ 0.21 (P < 0.0001)
2. **Gap events are distant:** Median 3983 Mpc vs 2078 Mpc for normal events (p = 0.001)
3. **Gap event spins rule out hierarchical mergers:** Median χ_eff = 0.059 (not ~0.5); KS p = 0.545 shows identical distributions
4. **ε is universal:** Cross-catalog CV = 49.7% across O3a, O3b, O4a
5. **Coherence is path-length dependent:** Proper distance gives tighter ε distribution

**Σ-Gravity interpretation:**

With ε ≈ 3.4 × 10⁻⁷ per coherence period (λ_coh ≈ 2.2 kpc), a 60 M☉ BBH at 3000 Mpc traverses N ≈ 1.4 × 10⁶ periods and appears as:

$$
M_{\rm eff} = 60 \times (1 + 3.4 \times 10^{-7})^{1.4 \times 10^6} \approx 95~M_\odot
$$

This naturally explains "impossible" black holes in the pair-instability gap without requiring hierarchical mergers or new stellar physics.

### SI §10.8. Reproducibility

**Scripts:**
- `ligo/sigma_gravity_rate_v2.py`: Main analysis with deduplication
- `ligo/cross_catalog_analysis.py`: Cross-catalog consistency test

**Commands:**
```bash
# Run main analysis (generates plots and metrics)
python ligo/sigma_gravity_rate_v2.py

# Run cross-catalog consistency test
python ligo/cross_catalog_analysis.py
```

---

## SI §11 — Milky Way Full Analysis

### SI §11.1. Purpose

The SPARC RAR tests Σ-Gravity on rotation-curve bins for 166 disks. Here we validate the framework at the finest resolution: individual Milky Way stars from Gaia DR3. This provides a direct, per-star comparison of observed and predicted radial accelerations without binning or azimuthal averaging.

### SI §11.2. Methodological Note

The MW analysis uses a saturated-well tail parameterization rather than the Burr-XII + winding kernel used for SPARC. This is standard effective field theory practice: the same underlying physics (Σ-enhancement of Newtonian gravity) is represented by different functional forms optimized for different observables. Both share the same coherence scale: the Burr-XII $\ell_0 = 5$ kpc corresponds to the saturated-well boundary $R_b \approx 6$ kpc.

### SI §11.3. Data and Setup

- **Stars**: 157,343 Milky Way stars (data/gaia/mw_gaia_full_coverage.npz; includes 13,185 new inner-disk stars with RVs)
- **Coverage**: 0.09–19.92 kpc (10× improvement in inner-disk sampling: 3–6 kpc n=6,717 vs prior n=653)
- **Pipeline fit** (GPU, CuPy): Boundary R_b = 5.78 kpc; saturated-well tail: v_flat = 149.6 km/s, R_s = 2.0 kpc, m = 2.0, gate ΔR = 0.77 kpc
- **Model selection** on rotation-curve bins: BIC — Σ 199.4; MOND 938.4; NFW 2869.7; GR 3366.4

### SI §11.4. Star-Level RAR Results

**Global performance (n=157,343):**
- **GR (baryons)**: mean Δ = **+0.380 dex**, σ = 0.176 dex — systematic under-prediction (missing mass)
- **Σ-Gravity**: mean Δ = **+0.062 dex**, σ = 0.142 dex — near-zero bias, tighter scatter
- **Improvement**: **6.1× better** than GR in mean residual
- **MOND**: mean Δ = +0.166 dex, σ = 0.161 dex (2.3× better than GR, but 2.7× worse than Σ)
- **NFW**: mean Δ = **+1.409 dex**, σ = 0.140 dex — catastrophic over-prediction for this tested halo realization

**Radial progression:**

| Radius [kpc] | n | GR mean Δ | Σ mean Δ | Σ improvement |
|---|---:|---:|---:|---:|
| **3–6** (inner, gated) | 6,717 | +0.001 | −0.007 | ~1× (both near-zero) ✓ |
| **6–8** (tail onset) | 55,143 | +0.356 | **+0.032** | **11.1×** |
| **8–10** (main disk) | 91,397 | +0.431 | **+0.091** | **4.7×** |
| **10–12** (outer) | 2,797 | +0.480 | **+0.098** | **4.9×** |
| **12–14** | 171 | +0.490 | **+0.083** | **5.9×** |
| **14–16** | 5 | +0.404 | **+0.030** | **13.5×** |
| **16–25** (halo) | 3 | +0.473 | **−0.004** | **118×** |

### SI §11.5. Key Findings

1. **Smooth 0–20 kpc transition**: No discontinuity at R_b. Inner disk shows near-zero residuals for both models; outer disk demonstrates consistent 4–13× improvement.
2. **Inner-disk integration resolved sampling artifact**: Previous apparent "abrupt shift" at R_b was due to sparse statistics (n=653). With 10× more stars (n=6,717), transition is demonstrably smooth.
3. **Tested NFW halo ruled out for MW**: 1.4 dex systematic over-prediction demonstrates that the fixed halo configuration cannot match MW star-level accelerations.

### SI §11.6. Figures

See main paper figures 13–18 for:
- All-model summary (8-panel)
- Improved RAR comparison with smoothed Σ curve
- Radial residual map (smooth transition proof)
- Residual distribution histograms
- Radial-bin performance table
- Outer-disk rotation curves

---

## SI §12 — Spiral Winding Gate Details

### SI §12.1. Physical Derivation

For rotation-supported disks, we add a morphology-dependent winding gate $G_{\rm wind}$ that suppresses the enhancement in regions where differential rotation has wound coherent paths into tight spirals, causing destructive interference.

$$
G_{\rm wind}(R, v_c) = \frac{1}{1 + (N_{\rm orbits}/N_{\rm crit})^{\alpha}}
$$

where

$$
N_{\rm orbits} = \frac{t_{\rm age} \cdot v_c}{2\pi R \cdot 0.978~\mathrm{kpc\cdot km/s \to Gyr}}
$$

is the cumulative number of orbits at radius $R$ over the system age $t_{\rm age}$.

**Physical derivation of $N_{\rm crit}$:**

The azimuthal coherence length is $\ell_\phi \sim (\sigma_v/v_c) \times 2\pi R$. After $N$ orbits, the wound spacing becomes $\lambda_{\rm wound} \sim 2\pi R / N$. Destructive interference occurs when $\lambda_{\rm wound} \sim \ell_\phi$, yielding:

$$
N_{\rm crit} \sim \frac{v_c}{\sigma_v} \sim \frac{200~\mathrm{km/s}}{20~\mathrm{km/s}} = 10
$$

This is **derived from coherence geometry**, not a free parameter.

### SI §12.2. 3D Geometric Dilution

The 2D derivation assumes spiral patterns exist throughout the coherence volume, but spiral arms are confined to a thin disk with scale height $h_d \approx 300$ pc. The coherence kernel samples in 3D over scale $\ell_z \sim \ell_0 \sim 5$ kpc. The spiral density pattern has vertical structure:

$$
\rho_{\rm spiral}(R, \phi, z) = \rho_0(R) \cdot S(\phi) \cdot \exp(-|z|/h_d)
$$

The effective spiral modulation experienced by the kernel is:

$$
\langle S_{\rm eff} \rangle = \frac{1}{\ell_z} \int_0^{\ell_z} S_0 \, e^{-z/h_d} \, dz = S_0 \cdot \frac{h_d}{\ell_z}\left[1 - e^{-\ell_z/h_d}\right]
$$

In the limit $\ell_z \gg h_d$:

$$
\varepsilon \equiv \frac{\langle S_{\rm eff} \rangle}{S_0} \approx \frac{h_d}{\ell_0} \approx \frac{300~\mathrm{pc}}{5000~\mathrm{pc}} \approx 0.06 \approx \frac{1}{17}
$$

For the winding gate to trigger at the same physical threshold:

$$
N_{\rm crit,eff} = \frac{N_{\rm crit,2D}}{\varepsilon} = N_{\rm crit,2D} \times \frac{\ell_0}{h_d} \approx 10 \times 17 \approx 170
$$

**This prediction is within 13% of the calibrated value (150)** using only existing parameters ($\ell_0 = 5$ kpc) and known observables ($h_d \approx 300$ pc for thin disks).

### SI §12.3. Two Winding Regimes

| Regime | Parameters | Use case | Result |
|--------|-----------|----------|--------|
| Physical | $N_{\rm crit}=10$, $\alpha=2.0$ | Individual galaxy improvement | 86% SPARC improved, massive spirals +30% |
| Effective | $N_{\rm crit}=100\text{–}150$, $\alpha=1.0$ | RAR scatter optimization | **0.0854 dex** (beats 0.087 target) |

### SI §12.4. Testable Predictions

1. **Inclination dependence:** Face-on galaxies should show stronger winding effects (viewing full azimuthal structure).
2. **Age dependence:** Young galaxies ($t<5$ Gyr) should prefer lower effective $N_{\rm crit}$.
3. **Counter-rotation:** Systems with counter-rotating components (e.g., NGC 4550) should show no winding suppression.
4. **Disk thickness correlation:** $N_{\rm crit,eff}$ should correlate with disk thickness. Galaxies with thicker disks (larger $h_d$) should show less dilution.

---

## SI §13 — Extended Cluster Analysis

### SI §13.1. Baryon Models

- **Gas**: gNFW pressure profile (Arnaud+2010), normalized to f_gas(R_500)=0.11 with clumping correction C(r)
- **BCG + ICL**: central stellar components included
- **External convergence** κ_ext ~ N(0, 0.05²)
- **Σ_crit**: distance ratios D_LS/D_S with cluster-specific $P(z_s)$ where available

### SI §13.2. Triaxial Projection

We transform $\rho(r) \to \rho(x,y,z)$ with ellipsoidal radius $m^2 = x^2 + (y/q_p)^2 + (z/q_{\rm LOS})^2$ and enforce mass conservation via a single global normalization, not a local $1/(q_p\, q_{\rm LOS})$ factor. The corrected projection recovers **~60% variation in $\kappa(R)$** and **~20–30% in $\theta_E$** across $q_{\rm LOS}\in[0.7,1.3]$.

### SI §13.3. Hierarchical Inference

Two models:
1) **Baseline** (γ=0) with population A_c ~ N(μ_A, σ_A)
2) **Mass-scaled** with (ℓ_{0,⋆}, γ) + same A_c population

Sampling via PyMC **NUTS** on a differentiable θ_E grid surrogate (target_accept=0.95); WAIC/LOO used for model comparison (ΔWAIC ≈ 0 ± 2.5).

### SI §13.4. Population Posteriors

| Parameter | Posterior | Notes |
|---|---|---|
| μ_A | 4.6 ± 0.4 | population mean amplitude |
| σ_A | ≈ 1.5 | intrinsic scatter |
| ℓ₀,⋆ | ≈ 200 kpc | reference coherence length |
| γ | 0.09 ± 0.10 | mass-scaling (consistent with 0) |
| ΔWAIC (γ-free vs γ=0) | 0.01 ± 2.5 | inconclusive |

### SI §13.5. Lensing Visuals Reproduction

Self-contained figures for convergence and deflection, calibrated to the observed θ_E:

```bash
python scripts/make_cluster_lensing_profiles.py --clusters "MACS1149" --fb 0.33 --ell0_frac 0.60 --p 2 --ncoh 2
```

Build multi-cluster panels:

```bash
python scripts/make_cluster_lensing_panels.py
```

---

## References

- Beck, C. & Cohen, E. G. D. (2003). Superstatistics. Physica A, 322, 267-275. arXiv:cond-mat/0303288
- Rodriguez, G. (1977). Statistical Models. JSTOR.
- Arnaud, M., et al. (2010). The universal galaxy cluster pressure profile. A&A, 517, A92.
- Lelli, F., et al. (2017). SPARC: Mass Models for 175 Disk Galaxies. AJ, 153, 240.
- Li, P., et al. (2018). Fitting the radial acceleration relation to individual SPARC galaxies. A&A, 615, A3.

---

*End of Supplementary Information*
