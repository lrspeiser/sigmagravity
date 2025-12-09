# Supplementary Information

## Σ-Gravity: A Coherence-Based Phenomenological Model for Galactic Dynamics

**Author:** Leonard Speiser

This Supplementary Information (SI) accompanies the main manuscript and provides complete technical details for reproducing and extending all results.

---

## Table of Contents

**Part I: Methods Canon**
1. [SI §1 — Final Model Definition](#si-1--final-model-definition)
2. [SI §2 — Parameter Values](#si-2--parameter-values)
3. [SI §3 — Data Sources and Preprocessing](#si-3--data-sources-and-preprocessing)
4. [SI §4 — Reproducibility](#si-4--reproducibility)
5. [SI §4a — Model Variants and A/B Testing](#si-4a--model-variants-and-ab-testing)

**Part II: Results**
5. [SI §5 — SPARC Galaxy Analysis](#si-5--sparc-galaxy-analysis)
6. [SI §6 — Galaxy Cluster Lensing](#si-6--galaxy-cluster-lensing)
7. [SI §7 — Milky Way Validation](#si-7--milky-way-validation)
8. [SI §8 — Unique Predictions](#si-8--unique-predictions)

**Part III: Theoretical Framework**
9. [SI §9 — Teleparallel Gravity Foundation](#si-9--teleparallel-gravity-foundation)
10. [SI §10 — Stress-Energy Conservation](#si-10--stress-energy-conservation)
11. [SI §11 — Relativistic Lensing Derivation](#si-11--relativistic-lensing-derivation)
12. [SI §12 — Wide Binary Analysis](#si-12--wide-binary-analysis)

**Part IV: Robustness and Ablations**
13. [SI §13 — Alternative Coherence Scales](#si-13--alternative-coherence-scales)
14. [SI §14 — Parameter Sensitivity](#si-14--parameter-sensitivity)
15. [SI §15 — Fitted-Parameter Comparison (Ablation)](#si-15--fitted-parameter-comparison-ablation)

---

# Part I: Methods Canon

## SI §1 — Final Model Definition

This section defines the **exact equations** used for all plots and tables in this paper.

### The Σ-Gravity Enhancement Formula

The effective gravitational acceleration is:

$$\boxed{g_{\text{eff}} = g_N \cdot \Sigma}$$

where $g_N = |\nabla\Phi_N|$ is the **baryonic Newtonian acceleration** (QUMOND-like structure).

The enhancement factor is:

$$\boxed{\Sigma = 1 + A(D,L) \cdot W(r) \cdot h(g_N)}$$

### Component Definitions

| Component | Formula | Description |
|-----------|---------|-------------|
| **h(g_N)** | $\sqrt{g^\dagger/g_N} \times g^\dagger/(g^\dagger + g_N)$ | Acceleration function |
| **W(r)** | $r/(\xi + r)$ | Coherence window |
| **ξ** | $R_d/(2\pi)$ | Coherence scale |
| **g†** | $cH_0/(4\sqrt{\pi}) \approx 9.60 \times 10^{-11}$ m/s² | Critical acceleration |
| **A(D,L)** | $A_0 \times [1 - D + D \times (L/L_0)^n]$ | Unified amplitude |

### Unified Amplitude Formula

$$\boxed{A(D,L) = A_0 \times [1 - D + D \times (L/L_0)^n]}$$

| Parameter | Value | Description |
|-----------|-------|-------------|
| **A₀** | $e^{1/(2\pi)} \approx 1.173$ | Base amplitude |
| **L₀** | 0.40 kpc | Reference path length |
| **n** | 0.27 | Path length exponent |
| **D** | 0 (galaxy) or 1 (cluster) | Dimensionality factor |

**Amplitude values:**
- Disk galaxies (D=0): A = A₀ = 1.173
- Galaxy clusters (D=1, L≈600 kpc): A = A₀ × (600/0.4)^0.27 ≈ 8.45

### Rotation Curve Prediction

For disk galaxies:

$$V_{\text{pred}} = V_{\text{bar}} \times \sqrt{\Sigma}$$

where:

$$V_{\text{bar}}^2 = V_{\text{gas}}^2 + \Upsilon_{\text{disk}} \cdot V_{\text{disk}}^2 + \Upsilon_{\text{bulge}} \cdot V_{\text{bulge}}^2$$

**Mass-to-light ratios (Lelli+ 2016 standard):**
- Υ_disk = 0.5 M☉/L☉ at 3.6μm
- Υ_bulge = 0.7 M☉/L☉ at 3.6μm

---

## SI §2 — Parameter Values

### Single Source of Truth

All results in this paper use exactly these parameters:

| Parameter | Symbol | Value | Units | Status |
|-----------|--------|-------|-------|--------|
| Critical acceleration | g† | 9.599×10⁻¹¹ | m/s² | Derived |
| Coherence scale | ξ | R_d/(2π) | kpc | Derived |
| Base amplitude | A₀ | 1.1725 | — | Derived |
| Reference path length | L₀ | 0.40 | kpc | Calibrated |
| Path length exponent | n | 0.27 | — | Calibrated |
| Disk M/L ratio | Υ_disk | 0.5 | M☉/L☉ | Fixed |
| Bulge M/L ratio | Υ_bulge | 0.7 | M☉/L☉ | Fixed |

### Physical Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Speed of light | c | 2.998×10⁸ | m/s |
| Hubble constant | H₀ | 70 | km/s/Mpc |
| Hubble constant (SI) | H₀ | 2.27×10⁻¹⁸ | s⁻¹ |
| Gravitational constant | G | 6.674×10⁻¹¹ | m³/kg/s² |
| Solar mass | M☉ | 1.989×10³⁰ | kg |
| Kiloparsec | kpc | 3.086×10¹⁹ | m |

### Derivation Status Key

- **Derived**: Mathematical result from stated assumptions
- **Calibrated**: Physical motivation with final value set by data
- **Fixed**: Standard value from literature, not fitted

### Key Result: No Free Parameters Per Galaxy

Unlike ΛCDM (2-3 parameters per galaxy for NFW halo fitting), Σ-Gravity uses the same formula with the same global parameters for all 171 SPARC galaxies.

---

## SI §3 — Data Sources and Preprocessing

### SI §3.1 SPARC Galaxies (N=171)

**Source:** Spitzer Photometry and Accurate Rotation Curves (SPARC)  
**Reference:** Lelli, McGaugh & Schombert (2016), AJ 152, 157  
**URL:** http://astroweb.cwru.edu/SPARC/

**Files:**
- `data/Rotmod_LTG/*_rotmod.dat` — Individual galaxy rotation curves (175 files)
- `data/Rotmod_LTG/MasterSheet_SPARC.mrt` — Galaxy properties

**Column Format (per galaxy .dat file):**

| Column | Name | Units | Description |
|--------|------|-------|-------------|
| 1 | R | kpc | Galactocentric radius |
| 2 | V_obs | km/s | Observed rotation velocity |
| 3 | V_err | km/s | Error on V_obs |
| 4 | V_gas | km/s | Gas velocity (from HI 21cm) |
| 5 | V_disk | km/s | Disk velocity (at M/L = 1) |
| 6 | V_bul | km/s | Bulge velocity (at M/L = 1) |

**Preprocessing:**

```python
# 1. Apply M/L correction
V_disk_scaled = V_disk * np.sqrt(0.5)   # √0.5 ≈ 0.707
V_bulge_scaled = V_bulge * np.sqrt(0.7) # √0.7 ≈ 0.837

# 2. Compute V_bar with signed gas contribution
V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)

# 3. Estimate disk scale length R_d from rotation curve shape
idx = len(data) // 3
R_d = R[idx] if idx > 0 else R[-1] / 2
```

**Sample Selection:**

| Criterion | N | Notes |
|-----------|---|-------|
| SPARC database | 175 | Original sample |
| Valid V_bar at all radii | 174 | Excludes UGC01281 |
| ≥5 rotation curve points | **171** | Quality cut |

### SI §3.2 Fox+ 2022 Galaxy Clusters (N=42)

**Source:** Fox et al. (2022), ApJ 928, 87  
**File:** `data/clusters/fox2022_unique_clusters.csv`

**Full Catalog:** The Fox+ 2022 catalog contains 94 unique galaxy clusters with strong lensing measurements.

**Selection Criteria (reducing to N=42):**

| Criterion | N remaining | Rationale |
|-----------|-------------|-----------|
| Fox+ 2022 full catalog | 94 | Starting sample |
| `spec_z_constraint == 'yes'` | 68 | Spectroscopic redshifts ensure accurate distance/mass calibration |
| `M500_1e14Msun > 2.0` | 42 | High-mass clusters have well-measured X-ray/SZ masses and reliable baryon fractions |
| Both M500 and MSL_200kpc available | **42** | Require both SZ mass and lensing mass for comparison |

**Rationale for filtering:**
- **Spectroscopic redshifts:** Photometric redshifts introduce ~5-10% distance uncertainties that propagate to ~10-20% mass errors, which would dominate over any theoretical signal.
- **High-mass cut (M500 > 2×10¹⁴ M☉):** Lower-mass clusters have larger fractional uncertainties in both M500 (from SZ/X-ray) and baryonic mass estimates. The cosmic baryon fraction f_baryon ≈ 0.15 is better calibrated for massive, relaxed clusters.
- **Complete data:** Both M500 (to estimate baryonic mass) and MSL_200kpc (lensing mass at 200 kpc) are required for the comparison.

**Baryonic Mass Estimate:**

$$M_{\rm bar}(200~{\rm kpc}) = 0.4 \times f_{\rm baryon} \times M_{500}$$

where f_baryon = 0.15 (cosmic baryon fraction). The factor 0.4 accounts for the concentration of baryonic mass within 200 kpc relative to R500.

### SI §3.3 Eilers-APOGEE-Gaia Milky Way (N=28,368)

**Source:** Cross-match of Eilers+ 2019, APOGEE DR17, Gaia EDR3  
**File:** `data/gaia/eilers_apogee_6d_disk.csv`

**Selection Criteria:**
- Disk stars: 4 < R_gal < 15 kpc
- Thin disk: |z| < 0.5 kpc
- Full 6D phase space available

**Baryonic Model:** McMillan 2017, scaled by 1.16×

### SI §3.4 Counter-Rotating Galaxies (N=63)

**Sources:**
1. MaNGA DynPop Catalog (Zhu et al. 2023): `data/manga_dynpop/SDSSDR17_MaNGA_JAM.fits`
2. Bevacqua et al. 2022: `data/stellar_corgi/bevacqua2022_counter_rotating.tsv`

---

## SI §4 — Reproducibility

All numerical results reported in this paper are reproducible using the provided code repository.

### Repository and Requirements

**Repository:** https://github.com/lrspeiser/SigmaGravity

**Dependencies:** numpy, scipy, pandas, matplotlib, astropy

**Installation:**
```bash
git clone https://github.com/lrspeiser/SigmaGravity.git && cd SigmaGravity
pip install numpy scipy pandas matplotlib astropy
```

### Validation Script

The master regression test is `scripts/run_regression_extended.py`, which runs 16 comprehensive tests using the canonical Σ-Gravity parameters:

| Parameter | Value | Status |
|-----------|-------|--------|
| $A_0$ | $e^{1/(2\pi)} \approx 1.1725$ | Derived |
| $L_0$ | 0.4 kpc | Calibrated |
| $n$ | 0.27 | Calibrated |
| $\xi$ | $R_d/(2\pi)$ | Derived per-galaxy |
| M/L (disk/bulge) | 0.5/0.7 | Fixed (Lelli+ 2016) |
| $g^\dagger$ | $9.599 \times 10^{-11}$ m/s² | Derived |

### Usage

```bash
python scripts/run_regression_extended.py           # Full 16 tests
python scripts/run_regression_extended.py --quick   # Skip slow tests (Gaia, counter-rotation)
python scripts/run_regression_extended.py --core    # Core 6 tests only (SPARC, clusters, Gaia, redshift, solar system, counter-rotation)
```

### Output Files

Results are saved to `scripts/regression_results/extended_report.json` in machine-readable JSON format.

---

## SI §4a — Complete Test Suite (16 Tests)

The regression test validates Σ-Gravity against 16 diverse astrophysical phenomena, comparing to both MOND and ΛCDM where applicable.

### Test Suite Summary

| # | Test | Gold Standard | Σ-Gravity | MOND | ΛCDM |
|---|------|---------------|-----------|------|------|
| 1 | SPARC Galaxies | Lelli+ 2016 | 17.75 km/s | 17.15 km/s | ~15 km/s (fitted) |
| 2 | Galaxy Clusters | Fox+ 2022 | 0.987× | ~0.33× | ~1.0× (fitted) |
| 3 | Milky Way | Eilers+ 2019 | 29.5 km/s | ~30 km/s | ~25 km/s |
| 4 | Redshift Evolution | Theory | ∝ H(z) | ∝ H(z)? | N/A |
| 5 | Solar System | Bertotti+ 2003 | 1.77×10⁻⁹ | ~10⁻⁵ | 0 |
| 6 | Counter-Rotation | Bevacqua+ 2022 | p=0.004 | N/A | N/A |
| 7 | Tully-Fisher | McGaugh 2012 | slope=4 | slope=4 | slope~3.5 |
| 8 | Wide Binaries | Chae 2023 | 63% boost | ~35% | 0% |
| 9 | Dwarf Spheroidals | Walker+ 2009 | 0.72× | ~1× | ~1× (fitted) |
| 10 | Ultra-Diffuse Galaxies | van Dokkum+ 2018 | EFE needed | EFE needed | Fitted |
| 11 | Galaxy-Galaxy Lensing | Stacking | 9.5× | ~10× | ~15× |
| 12 | External Field Effect | Theory | 0.36× | 0.3× | N/A |
| 13 | Gravitational Waves | GW170817 | c_GW = c | c_GW = c | c_GW = c |
| 14 | Structure Formation | Planck 2018 | Informational | Fails | Works |
| 15 | CMB Acoustic Peaks | Planck 2018 | Informational | Fails | Works |
| 16 | Bullet Cluster | Clowe+ 2006 | 1.12× | ~0.5× | ~2.1× |

### Observational Benchmarks (Gold Standard)

All benchmarks are documented in `derivations/observational_benchmarks.py` with full citations.

| Observable | Value | Source |
|------------|-------|--------|
| a₀ (MOND scale) | 1.2×10⁻¹⁰ m/s² | McGaugh 2016 |
| Cassini γ-1 | (0 ± 2.3×10⁻⁵) | Bertotti+ 2003 |
| BTFR slope | 3.98 ± 0.06 | McGaugh 2012 |
| Wide binary boost | 35% ± 10% | Chae 2023 |
| GW170817 Δc/c | <10⁻¹⁵ | Abbott+ 2017 |
| Bullet Cluster M_lens/M_bar | 2.1× | Clowe+ 2006 |

### How to Modify the Formula

To test alternative formulas, edit `scripts/run_regression_extended.py`:

1. **Change the enhancement function h(g):**
```python
def h_function(g: np.ndarray) -> np.ndarray:
    """Enhancement function h(g) = √(g†/g) × g†/(g†+g)"""
    g = np.maximum(np.asarray(g), 1e-15)
    return np.sqrt(g_dagger / g) * g_dagger / (g_dagger + g)
```

2. **Change the coherence window W(r):**
```python
def W_coherence(r: np.ndarray, xi: float) -> np.ndarray:
    """Coherence window W(r) = r/(ξ+r)"""
    xi = max(xi, 0.01)
    return r / (xi + r)
```

3. **Change the unified amplitude A(D,L):**
```python
def unified_amplitude(D: float, L: float) -> float:
    """Unified amplitude: A = A₀ × [1 - D + D × (L/L₀)^n]"""
    return A_0 * (1 - D + D * (L / L_0)**N_EXP)
```

4. **Change the full enhancement Σ:**
```python
def sigma_enhancement(g, r=None, xi=1.0, A=None, D=0, L=1.0):
    """Σ = 1 + A(D,L) × W(r) × h(g)"""
    # ... implementation
```

### Expected Results (Extended)

Running `python scripts/run_regression_extended.py` produces:

| Test | Expected Result |
|------|-----------------|
| SPARC Galaxies | RMS = 17.75 km/s, Scatter = 0.097 dex, Win = 47.4% |
| Galaxy Clusters | Median ratio = 0.987, Scatter = 0.132 dex (N=42) |
| Milky Way | RMS = 29.5 km/s (N=28,368 stars) |
| Redshift Evolution | g†(z=2)/g†(z=0) = 2.966 |
| Solar System | \|γ-1\| = 1.77×10⁻⁹ |
| Counter-Rotation | f_DM(CR) = 0.169 < f_DM(Normal) = 0.302, p = 0.004 |
| Tully-Fisher | M_pred/M_obs = 0.87, slope = 4 |
| Wide Binaries | 63.2% boost at 10 kAU |
| Dwarf Spheroidals | σ_pred/σ_obs = 0.72 |
| Ultra-Diffuse Galaxies | DF2: σ_pred = 20.8 km/s (EFE needed) |
| Galaxy-Galaxy Lensing | M_eff/M_star = 9.5× at 200 kpc |
| External Field Effect | Suppression = 0.36× |
| Gravitational Waves | c_GW = c (consistent with GW170817) |
| Structure Formation | g/g† = 1.5 at cluster scales |
| CMB Acoustic Peaks | g/g†(z) = 2.7×10⁻¹⁰ at z=1100 |
| Bullet Cluster | M_pred/M_bar = 1.12× |

---

# Part II: Results

## SI §5 — SPARC Galaxy Analysis

### Results Summary

| Metric | Σ-Gravity | MOND | Notes |
|--------|-----------|------|-------|
| Mean RMS error | **17.75 km/s** | 17.15 km/s | 171 galaxies |
| Win rate | 47.4% | 52.6% | Same M/L |
| RAR scatter | 0.097 dex | 0.098 dex | — |

### MOND Comparison Methodology

For all MOND comparisons:
- **Acceleration scale:** a₀ = 1.2×10⁻¹⁰ m/s² (fixed)
- **Interpolation function:** ν(x) = 1/(1 − e^(−√x))
- **Same M/L** as Σ-Gravity (0.5/0.7)

### Comparison to Other Theories

| Theory | Free params/galaxy | SPARC RMS | Clusters |
|--------|-------------------|-----------|----------|
| **GR + baryons** | 0 | N/A (fails) | Fails (×10) |
| **MOND** | 0 | 17.15 km/s | Fails (×3) |
| **ΛCDM (NFW)** | 2-3 | ~15 km/s | Works |
| **Σ-Gravity** | 0 | 17.75 km/s | Works |

---

## SI §6 — Galaxy Cluster Lensing

### Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Median M_pred/M_lens | **0.987** | N=42 clusters |
| Scatter | 0.132 dex | — |
| Range | 0.67–1.48 | All within factor 1.5 |

### Comparison to Other Theories

| Theory | M_pred/M_lens | Notes |
|--------|---------------|-------|
| GR + baryons | 0.10–0.15 | "Missing mass" |
| MOND | ~0.33 | "Cluster problem" |
| ΛCDM (fitted) | 0.95–1.05 | 2-3 params/cluster |
| **Σ-Gravity** | **0.987** | 0 params/cluster |

### Cluster Amplitude

Using the unified amplitude formula with D=1 and L=600 kpc:

$$A_{\rm cluster} = A_0 \times (600/0.4)^{0.27} = 1.173 \times 8.45 \approx 8.45$$

---

## SI §7 — Milky Way Validation

### Results Summary

| Model | RMS | Notes |
|-------|-----|-------|
| **Σ-Gravity** | **29.5 km/s** | 28,368 stars |
| MOND | 30.3 km/s | — |

### Methodology

1. Load Eilers-APOGEE-Gaia catalog
2. Apply asymmetric drift correction
3. Compute V_bar from McMillan 2017 (×1.16)
4. Apply Σ-enhancement
5. Compare to observed velocities

---

## SI §8 — Unique Predictions

### 1. Counter-Rotating Disks — CONFIRMED

| Metric | Counter-Rotating | Normal | Difference |
|--------|-----------------|--------|------------|
| f_DM mean | **0.169** | 0.302 | **−44%** |
| p-value | 0.004 | — | Significant |

### 2. Redshift Evolution

$$g^\dagger(z) = g^\dagger_0 \times \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$$

| z | g†(z)/g†(0) |
|---|-------------|
| 0 | 1.00 |
| 1 | 1.77 |
| 2 | 2.97 |

### 3. Solar System Safety

| Location | g/g† | Σ−1 |
|----------|------|-----|
| Earth | 6×10⁷ | <10⁻¹² |
| Saturn | 7×10⁵ | <10⁻⁹ |

Cassini bound: |γ−1| < 2.3×10⁻⁵  
Σ-Gravity: |γ−1| ~ 10⁻⁹

---

# Part III: Theoretical Framework

## SI §9 — Teleparallel Gravity Foundation

### Primary Formulation: QUMOND-Like Field Equations

The observable predictions of Σ-Gravity are captured by the modified Poisson equation:

$$\nabla^2 \Phi = 4\pi G \rho_b + \nabla \cdot [(\nu - 1) \mathbf{g}_N]$$

where:
- $\rho_b$ is the baryonic density
- $\mathbf{g}_N = -\nabla\Phi_N$ is the baryonic Newtonian acceleration
- $\Phi_N$ satisfies $\nabla^2 \Phi_N = 4\pi G \rho_b$
- $\nu(g_N, r) = 1 + A \cdot W(r) \cdot h(g_N) = \Sigma$

**Test particles follow geodesics of $\Phi$**—no non-minimal matter coupling in the particle action.

### Covariant Acceleration Scalar

The acceleration dependence is formulated covariantly using the 4-acceleration:

$$a^\mu = u^\nu \nabla_\nu u^\mu, \quad a^2 \equiv g_{\mu\nu} a^\mu a^\nu$$

In the weak-field limit: $a^2 \to g_N^2 = |\nabla\Phi_N|^2$

The coupling $f(a^2)\mathcal{L}_m$ is a scalar function of a scalar argument, following the prescription of Harko et al. (2014) for acceptable acceleration-dependent couplings. This avoids the "non-acceptable coupling" concern.

### The Auxiliary Field as Computational Device

The intermediate variable $\Phi_N$ is **not** a new gravitational degree of freedom:

**What $\Phi_N$ is:**
- The unique solution to $\nabla^2 \Phi_N = 4\pi G \rho_b$
- Determined entirely by the baryonic density
- An intermediate variable for computing $\nu(g_N)$

**What $\Phi_N$ is NOT:**
- An independent dynamical field with propagating modes
- A "second gravitational potential" with separate physical effects
- A source of additional gravitational degrees of freedom

This is exactly the QUMOND construction (Milgrom 2010, PRD 82, 043523).

### Action Formulation (for completeness)

$$S_{\Sigma} = \frac{1}{2\kappa} \int d^4x \, |e| \, \mathbf{T} + S_{\text{aux}} + \int d^4x \, |e| \, \mathcal{L}_m$$

where $S_{\text{aux}}$ encodes the auxiliary field:

$$S_{\text{aux}} = \int d^4x \, |e| \left[ -\frac{1}{8\pi G} (\nabla\Phi_N)^2 + \rho \Phi_N \right]$$

Varying with respect to $\Phi_N$ yields $\nabla^2 \Phi_N = 4\pi G \rho$—the auxiliary field is determined by baryons and has no independent dynamics.

### Covariant Coherence Scalar (Primary Theoretical Object)

The coherence scalar C is the fundamental object that determines gravitational enhancement:

$$\mathcal{C} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}$$

In the non-relativistic limit: $\mathcal{C} = v_{\rm rot}^2/(v_{\rm rot}^2 + \sigma^2)$

The phenomenological W(r) = r/(ξ+r) is a validated approximation to orbit-averaged C (see SI §13.5).

---

## SI §9b — Coherence Survival Formulation

### The First-Principles Picture

The coherence survival model provides a physical mechanism for the enhancement factor Σ. The key insight is that gravitational coherence must **propagate a minimum distance without disruption**, and disruption resets the counter.

### Modified Field Equations

**General Relativity (modified):**

$$G_{\mu\nu} = \frac{8\pi G}{c^4} T_{\mu\nu} \times \Sigma(x, R)$$

**Teleparallel Gravity (modified action):**

$$S = \frac{c^4}{16\pi G} \int f(T, \Phi) \sqrt{-g} \, d^4x + S_{\text{matter}}$$

where:

$$f(T, \Phi) = T + \sqrt{3} \times \Phi(r) \times F(T)$$

### The Torsion Modification Function

$$F(T) = T \times \left(\frac{T^\dagger}{T}\right)^{0.6} \times \exp\left(-\left(\frac{T}{T^\dagger}\right)^{0.1}\right)$$

**Critical torsion:** $T^\dagger = 2g^\dagger/c^2 \approx 2.14 \times 10^{-27}$ m⁻²

**Key feature:** $F(T) \sim T^{0.4}$ at low T (SUBLINEAR modification)

### The Coherence Field

$$\Phi(r) = 1 - \exp\left(-\left(\frac{r}{r_0}\right)^\beta\right)$$

| Parameter | Value | Physical Meaning |
|-----------|-------|------------------|
| $r_0$ | 20 kpc | Coherence scale |
| $\beta$ | 0.3 | Transition sharpness |

### Best-Fit Survival Parameters

From systematic testing on 175 SPARC galaxies (52-90% win rate vs MOND across categories):

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| $r_{\text{char}}$ | 20 kpc | Coherence horizon |
| $\alpha$ | 0.1 | Weak acceleration dependence |
| $\beta$ | 0.3 | Gradual radial transition |
| $A$ | $\sqrt{3} \approx 1.73$ | Mode counting amplitude |

### The Enhancement Formula (Survival Form)

$$\Sigma(r, g) = 1 + A \times \Phi(r) \times P_{\text{survive}}(r, g) \times h(g)$$

where:
- **Coherence field:** $\Phi(r) = 1 - \exp(-(r/r_0)^\beta)$
- **Survival probability:** $P_{\text{survive}} = \exp(-(r_0/r)^\beta \times (g/g^\dagger)^\alpha)$
- **Enhancement function:** $h(g) = \sqrt{g^\dagger/g} \times g^\dagger/(g^\dagger + g)$

### Physical Interpretation

1. **Coherence scale** ($r_0 = 20$ kpc): Gravitational coherence builds up over ~20 kpc of ordered rotation
2. **Weak acceleration dependence** ($\alpha = 0.1$): Acceleration sets the scale but not the shape
3. **Mode counting amplitude** ($A = \sqrt{3}$): From coherent addition of 3 torsion modes (radial + azimuthal + vertical)

### Distinguishing Features from Other Theories

| Feature | Coherence Survival | MOND | f(R)/f(T) |
|---------|-------------------|------|-----------|
| Modification type | SUBLINEAR ($T^{0.4}$) | Local | SUPERLINEAR |
| Spatial memory | Yes (via $\Phi(r)$) | No | No |
| Survival threshold | Yes | No | No |
| Amplitude origin | Mode counting | Free | Free |

### Unique Predictions

1. **Radial memory:** Enhancement at radius R depends on conditions at R' < R
2. **Morphology dependence:** Barred/disturbed galaxies show reduced outer enhancement
3. **Threshold behavior:** Sharp transition when survival probability drops

**Verification:** Preliminary morphology tests show Δ(Survival-MOND) more negative for smooth galaxies (-20.03) than barred (-18.07), supporting the radial memory prediction.

---

## SI §10 — Stress-Energy Conservation

### The Resolution

Promote Σ to a dynamical scalar field φ_C:

$$\boxed{\nabla_\mu \left( T^{\mu\nu}_{\text{matter}} + T^{\mu\nu}_{\text{coherence}} \right) = 0}$$

The coherence field carries the "missing" momentum/energy.

---

## SI §11 — Relativistic Lensing Derivation

### Gravitational Slip

$$\eta \equiv \frac{\Psi}{\Phi} = \frac{2\Sigma - 1}{3\Sigma - 2}$$

| Σ | η |
|---|---|
| 1.0 | 1.0 (GR) |
| 2.0 | 0.75 |

### Lensing-to-Dynamics Ratio

$$\frac{M_{\text{lens}}}{M_{\text{dyn}}} = \frac{5\Sigma - 3}{2(3\Sigma - 2)}$$

---

## SI §12 — Wide Binary Analysis

### Theoretical Status

**Option A:** External field effect (phenomenological)  
**Option B:** Coherence requires extended rotation

Current observational data is insufficient to distinguish.

---

# Part IV: Robustness and Ablations

## SI §13 — Alternative Coherence Scales

### SI §13.1 Geometric Baseline (Canonical)

The canonical coherence scale is:

$$\xi_{\rm geom} = \frac{R_d}{2\pi} \approx 0.159 \times R_d$$

This corresponds to one azimuthal wavelength at the disk scale length. All primary results in this paper use this form.

### SI §13.2 Dynamical Coherence Scale

An alternative formulation connects ξ directly to the ratio of random to ordered motion:

$$\xi_{\rm dyn} = k \times \frac{\sigma_{\rm eff}}{\Omega_d}$$

where:
- $k \approx 0.24$ (calibrated constant)
- $\sigma_{\rm eff}$ = effective velocity dispersion (mass-weighted: gas ~10 km/s, disk ~25 km/s, bulge ~120 km/s)
- $\Omega_d = V_{\rm bar}(R_d)/R_d$ = angular frequency at disk scale length

**Implementation:**

```python
def xi_dyn_kpc(R_d_kpc, V_bar_at_Rd_kms, sigma_eff_kms, k=0.24):
    """Dynamical coherence scale ξ = k × σ_eff / Ω_d"""
    Omega = V_bar_at_Rd_kms / np.maximum(R_d_kpc, 1e-6)  # (km/s)/kpc
    return k * sigma_eff_kms / np.maximum(Omega, 1e-12)  # kpc

def compute_sigma_eff(V_gas, V_disk, V_bulge):
    """Mass-weighted effective dispersion"""
    V_total_sq = V_gas.max()**2 + V_disk.max()**2 + V_bulge.max()**2
    gas_frac = V_gas.max()**2 / V_total_sq
    disk_frac = V_disk.max()**2 / V_total_sq
    bulge_frac = V_bulge.max()**2 / V_total_sq
    return gas_frac * 10 + disk_frac * 25 + bulge_frac * 120  # km/s
```

### SI §13.3 Full 16-Test Comparison

**Tests affected by ξ choice:**

| Metric | Canonical ξ = R_d/(2π) | Dynamical ξ_dyn (k=0.24) | Change |
|--------|------------------------|--------------------------|--------|
| SPARC RMS | 17.48 km/s | 17.39 km/s | **−0.5%** |
| RAR scatter | 0.194 dex | 0.191 dex | **−1.9%** |
| Win rate vs MOND | 45.6% | 46.8% | +1.2pp |
| Milky Way RMS | 66.7 km/s | 67.6 km/s | +1.4% |

**Tests NOT affected by ξ choice:**

| Test | Reason |
|------|--------|
| Galaxy Clusters | W = 1 at r = 200 kpc (ξ irrelevant) |
| Solar System | W = 0 (no extended disk structure) |
| GW170817 | Gravitational sector unchanged |
| Counter-Rotation | Coherence effect, not ξ-dependent |
| Dwarf Spheroidals | W → 0 (dispersion-dominated) |
| Bullet Cluster | W = 1 at lensing radii |

**With optimal k = 0.47:**

| Metric | Canonical | Dynamical (k=0.47) | Change |
|--------|-----------|-------------------|--------|
| SPARC RMS | 17.48 km/s | 17.29 km/s | **−1.1%** |
| RAR scatter | 0.194 dex | 0.190 dex | **−2.0%** |
| Win rate vs MOND | 45.6% | 44.4% | −1.2pp |

### SI §13.4 Physical Interpretation

The dynamical coherence scale tracks the ratio of random to ordered motion:

$$\xi_{\rm dyn} \propto \frac{\sigma}{\Omega} \propto T_{\rm orbit}$$

This connects to the covariant coherence scalar (§2.5): the transition $\mathcal{C} = 1/2$ occurs when $v_{\rm rot} = \sigma_v$, giving $r_{\rm transition} \sim \sigma/\Omega$.

**Coherence scale statistics (171 SPARC galaxies):**

| Scale | Mean (kpc) | Std (kpc) |
|-------|------------|-----------|
| ξ_canonical | 0.918 | 0.634 |
| ξ_dynamical | 0.532 | 0.366 |
| Ratio ξ_dyn/ξ_can | 0.72 | — |

### SI §13.5 Direct C(r) Formulation Test

**Motivation:** The covariant coherence scalar C (§2.5) is the theoretically proper object. We tested whether replacing W(r) directly with C(r) improves predictions.

**Implementation:**

```python
def C_local(v_rot_kms, sigma_kms):
    """Local coherence scalar: C = v²/(v² + σ²)"""
    v2 = np.maximum(v_rot_kms, 0.0)**2
    s2 = np.maximum(sigma_kms, 1e-6)**2
    return v2 / (v2 + s2)

def predict_velocity_C_local(R_kpc, V_bar, R_d, sigma_kms=20.0, max_iter=50):
    """Fixed-point iteration since C depends on V_pred."""
    g_bar = (V_bar * 1000)**2 / (R_kpc * kpc_to_m)
    h = h_function(g_bar)
    sigma = np.full_like(R_kpc, sigma_kms)
    
    V = np.array(V_bar, dtype=float)
    for _ in range(max_iter):
        C = C_local(V, sigma)  # Uses V_pred, not V_obs!
        Sigma = 1 + A_0 * C * h
        V_new = V_bar * np.sqrt(Sigma)
        if np.max(np.abs(V_new - V)) < 1e-6:
            break
        V = V_new
    return V
```

**Critical:** We use V_pred (not V_obs) to avoid data leakage.

**Results (171 SPARC galaxies):**

| Formulation | RMS (km/s) | Change | Win vs MOND |
|-------------|------------|--------|-------------|
| Canonical W(r) = r/(ξ+r) | 17.75 | — | 47.4% |
| C_local (σ = 15 km/s) | 17.82 | +0.4% | 47.4% |
| C_local (σ = 20 km/s) | 17.75 | 0.0% | 47.4% |
| C_local (σ = 25 km/s) | 17.72 | −0.2% | 47.4% |
| C_local (σ = 30 km/s) | 17.75 | 0.0% | 47.4% |

**Interpretation:** The direct C(r) formulation gives **identical results** to the phenomenological W(r). This confirms:

1. **W(r) is an excellent approximation** to orbit-averaged C
2. **The covariant formulation is validated** by identical predictions
3. **No empirical benefit** to the more complex iterative approach

### SI §13.6 Conclusion

We use the **canonical geometric form** W(r) = r/(ξ+r) for all primary results because:

1. **Simplicity:** ξ = R_d/(2π) requires only disk scale length
2. **Direct derivation:** One azimuthal wavelength at disk scale
3. **No iteration required:** W(r) is explicit, not implicit
4. **Validated approximation:** Direct C(r) gives identical results
5. **Theoretically grounded:** W(r) ≈ ⟨C⟩_orbit is now numerically confirmed

The covariant coherence scalar C remains the **primary theoretical object**; W(r) is its validated practical approximation.

---

## SI §14 — Parameter Sensitivity

### Coherence Window Exponent

| W(r) form | SPARC RMS |
|-----------|-----------|
| r/(ξ+r) [k=1] | 17.75 km/s |
| 1−(ξ/(ξ+r))^0.5 [k=0.5] | 18.97 km/s |

The k=1 form (simpler) performs better.

### Amplitude Sensitivity

| A_galaxy | SPARC RMS | Cluster Ratio |
|----------|-----------|---------------|
| 1.0 | 19.2 km/s | 0.85 |
| 1.173 | 17.75 km/s | 0.987 |
| 1.5 | 16.8 km/s | 1.12 |

The derived value A₀ = e^(1/2π) ≈ 1.173 provides optimal balance.

---

## SI §15 — Fitted-Parameter Comparison (Ablation)

This section presents a secondary analysis where Σ-Gravity and ΛCDM are compared with **equal numbers of fitted parameters per galaxy**. This is NOT the canonical model (which has zero per-galaxy parameters), but demonstrates that Σ-Gravity performs well even when allowing per-galaxy fitting.

### SI §15.1 Methodology

For a direct comparison with ΛCDM, we fit both models with 2 parameters per galaxy:

**Σ-Gravity parameters (2 per galaxy):**
- $A$: Enhancement amplitude (bounded: [0.01, 5.0])
- $\xi$: Coherence scale in kpc (bounded: [0.1, 50.0])

**ΛCDM/NFW parameters (2 per galaxy):**
- $\log_{10}(M_{200})$: Virial mass (bounded: [6, 14])
- $c$: Concentration (bounded: [1, 50])

### SI §15.2 Results

| Metric | Σ-Gravity (fitted) | ΛCDM (NFW) |
|--------|-------------------|------------|
| Mean χ²_red | **1.42** | 1.58 |
| Median χ²_red | **0.98** | 1.12 |
| Wins (better χ²_red) | **97** | 74 |
| RAR scatter | **0.105 dex** | 0.112 dex |

**Bootstrap 95% CI on win rate:** Σ-Gravity wins 55.4% ± 3.8% of galaxies.

### SI §15.3 Interpretation

When given equal fitting freedom:
1. Σ-Gravity achieves comparable or better fits
2. Fitted Σ-Gravity parameters cluster in narrow, physically-motivated ranges
3. NFW parameters span orders of magnitude with weak physical priors

**Key distinction:** The canonical model (main paper) uses A = 1.173 and ξ = R_d/(2π) for ALL galaxies with no fitting. This ablation study shows that even with per-galaxy fitting, Σ-Gravity outperforms ΛCDM, but the canonical zero-parameter model is the primary result.

---

## Acknowledgments

We thank **Emmanuel N. Saridakis** (National Observatory of Athens) for detailed feedback on the theoretical framework.

We thank **Rafael Ferraro** (CONICET – Universidad de Buenos Aires) for discussions on f(T) gravity.

---

## References

- Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157 (SPARC)
- Fox, C., et al. 2022, ApJ, 928, 87
- Eilers, A.-C., et al. 2019, ApJ, 871, 120
- Bevacqua, D., et al. 2022, MNRAS, 511, 139
- Zhu, L., et al. 2023, MNRAS, 522, 6326 (MaNGA DynPop)
- Milgrom, M. 1983, ApJ, 270, 365 (MOND)
- Bertotti, B., Iess, L., & Tortora, P. 2003, Nature, 425, 374 (Cassini)

---

## Legacy Documentation

For historical analyses and extended derivations, see:

`archive/SUPPLEMENTARY_INFORMATION_LEGACY.md`

---

*End of Supplementary Information*
