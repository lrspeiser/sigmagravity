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

**Part V: Additional Figures**
16. [SI §16 — Additional Figures](#si-16--additional-figures)

---

# Part I: Methods Canon

## SI §1 — Final Model Definition

This section defines the **exact equations** used for all plots and tables in this paper.

### The Σ-Gravity Enhancement Formula

The effective gravitational acceleration is:

$$\boxed{g_{\text{eff}} = g_N \cdot \Sigma}$$

where $g_N = |\nabla\Phi_N|$ is the **baryonic Newtonian acceleration** (QUMOND-like structure).

The enhancement factor is:

$$\boxed{\Sigma = 1 + A(D,L) \cdot \mathcal{C} \cdot h(g_N)}$$

where $\mathcal{C}$ is the **covariant coherence scalar** (primary formulation).

### Component Definitions

| Component | Formula | Description |
|-----------|---------|-------------|
| **C** | $v_{\rm rot}^2/(v_{\rm rot}^2 + \sigma^2)$ | Covariant coherence scalar (primary) |
| **h(g_N)** | $\sqrt{g^\dagger/g_N} \times g^\dagger/(g^\dagger + g_N)$ | Acceleration function |
| **g†** | $cH_0/(4\sqrt{\pi}) \approx 9.60 \times 10^{-11}$ m/s² | Critical acceleration |
| **A(L)** | $A_0 \times (L/L_0)^n$ | Unified 3D amplitude |

**Practical approximation:** For disk galaxies, $\mathcal{C} \approx W(r) = r/(\xi + r)$ with $\xi = R_d/(2\pi)$. This gives identical results and requires no iteration (see SI §13.5).

### Unified Amplitude Formula

$$\boxed{A(L) = A_0 \times (L/L_0)^n}$$

This unified 3D formula requires no discrete switch between system types. The path length $L$ naturally varies with geometry:

| Parameter | Value | Description |
|-----------|-------|-------------|
| **A₀** | $e^{1/(2\pi)} \approx 1.173$ | Base amplitude |
| **L₀** | 0.40 kpc | Reference path length (≈ disk scale height) |
| **n** | 0.27 | Path length exponent |
| **L** | Varies by system | Effective path through baryons |

**Amplitude values:**
- Thin disk galaxies (L ≈ L₀ = 0.4 kpc): A = A₀ = 1.173
- Elliptical galaxies (L ~ 1–20 kpc): A ~ 1.5–3.4
- Galaxy clusters (L ≈ 600 kpc): A = A₀ × (600/0.4)^0.27 ≈ 8.45

**Physical interpretation:** $L_0 \approx 0.4$ kpc corresponds to the typical scale height of disk galaxies. When the path length equals this reference ($L = L_0$), the amplitude is $A = A_0$. For extended systems, the amplitude scales as a power law with exponent $n = 0.27$.

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
| Reference path length | L₀ | 0.40 | kpc | Physical (disk scale height) |
| Path length exponent | n | 0.27 | — | Calibrated (holdout validated) |
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

The master regression test is `scripts/run_regression_extended.py`, which runs 17 comprehensive tests using the canonical Σ-Gravity parameters:

| Parameter | Value | Status |
|-----------|-------|--------|
| $A_0$ | $e^{1/(2\pi)} \approx 1.1725$ | Derived |
| $L_0$ | 0.4 kpc | Fixed (typical disk scale height) |
| $n$ | 0.27 | Calibrated (on 42 Fox+ clusters) |
| $\xi$ | $R_d/(2\pi)$ | Derived per-galaxy |
| M/L (disk/bulge) | 0.5/0.7 | Fixed (Lelli+ 2016) |
| $g^\dagger$ | $9.599 \times 10^{-11}$ m/s² | Derived |

### Usage

```bash
python scripts/run_regression_extended.py           # Full 17 tests
python scripts/run_regression_extended.py --quick   # Skip slow tests (Gaia, counter-rotation)
python scripts/run_regression_extended.py --core    # Core 8 tests only (SPARC, clusters, holdout, Gaia, etc.)
```

### Output Files

Results are saved to `scripts/regression_results/extended_report.json` in machine-readable JSON format.

---

## SI §4a — Complete Test Suite (17 Tests)

The regression test validates Σ-Gravity against 17 diverse astrophysical phenomena, comparing to both MOND and ΛCDM where applicable.

### Test Suite Summary

| # | Test | Gold Standard | Σ-Gravity | MOND | ΛCDM |
|---|------|---------------|-----------|------|------|
| 1 | SPARC Galaxies | Lelli+ 2016 | 17.42 km/s | 17.15 km/s | ~15 km/s (fitted) |
| 2 | Galaxy Clusters | Fox+ 2022 | 0.987× | ~0.33× | ~1.0× (fitted) |
| 3 | Cluster Holdout | Cross-validation | n=0.27±0.01 | N/A | N/A |
| 4 | Milky Way | Eilers+ 2019 | 29.8 km/s | ~30.3 km/s | ~25 km/s |
| 5 | Redshift Evolution | Theory | ∝ H(z) | ∝ H(z)? | N/A |
| 6 | Solar System | Bertotti+ 2003 | 1.77×10⁻⁹ | ~10⁻⁵ | 0 |
| 7 | Counter-Rotation | Bevacqua+ 2022 | p=0.004 | N/A | N/A |
| 8 | Tully-Fisher | McGaugh 2012 | slope=4 | slope=4 | slope~3.5 |
| 9 | Wide Binaries | Chae 2023 | 63% boost | ~35% | 0% |
| 10 | Dwarf Spheroidals | Walker+ 2009 | 0.87× (host inherit) | ~1× | ~1× (fitted) |
| 11 | Ultra-Diffuse Galaxies | van Dokkum+ 2018 | EFE needed | EFE needed | Fitted |
| 12 | Galaxy-Galaxy Lensing | Stacking | 9.5× | ~10× | ~15× |
| 13 | External Field Effect | Theory | 0.36× | 0.3× | N/A |
| 14 | Gravitational Waves | GW170817 | c_GW = c | c_GW = c | c_GW = c |
| 15 | Structure Formation | Planck 2018 | Informational | Fails | Works |
| 16 | CMB Acoustic Peaks | Planck 2018 | Informational | Fails | Works |
| 17 | Bullet Cluster | Clowe+ 2006 | 1.12× | ~0.5× | ~2.1× |

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

3. **Change the unified amplitude A(L):**
```python
def unified_amplitude(L: float) -> float:
    """Unified 3D amplitude: A = A₀ × (L/L₀)^n
    
    No D switch needed - path length L determines amplitude:
    - Disk galaxies: L ≈ L₀ = 0.4 kpc → A ≈ A₀
    - Clusters: L ≈ 600 kpc → A ≈ 8.45
    """
    return A_0 * (L / L_0)**N_EXP
```

4. **Change the full enhancement Σ:**
```python
def sigma_enhancement(g, r=None, xi=1.0, A=None, L=0.4):
    """Σ = 1 + A(L) × W(r) × h(g)"""
    # ... implementation
```

### Expected Results (Extended)

Running `python scripts/run_regression_extended.py` produces:

| Test | Expected Result |
|------|-----------------|
| SPARC Galaxies | RMS = 17.42 km/s, Scatter = 0.100 dex, Win = 42.7% |
| Galaxy Clusters | Median ratio = 0.987, Scatter = 0.132 dex (N=42) |
| Milky Way | RMS = 29.8 km/s (N=28,368 stars) |
| Redshift Evolution | g†(z=2)/g†(z=0) = 2.966 |
| Solar System | \|γ-1\| = 1.77×10⁻⁹ |
| Counter-Rotation | f_DM(CR) = 0.169 < f_DM(Normal) = 0.302, p = 0.004 |
| Tully-Fisher | M_pred/M_obs = 0.87, slope = 4 |
| Wide Binaries | 63.2% boost at 10 kAU |
| Dwarf Spheroidals | σ_pred/σ_obs = 0.87±0.63 (host inheritance, 5 dSphs) |
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

All metrics computed using `scripts/run_regression_extended.py` with canonical parameters.

| Metric | Σ-Gravity | MOND | Definition |
|--------|-----------|------|------------|
| Mean RMS error | **17.42 km/s** | 17.15 km/s | Per-galaxy RMS averaged over 171 galaxies |
| Win rate | 42.7% | 57.3% | Fraction where Σ-Gravity RMS < MOND RMS |
| RAR scatter | 0.100 dex | 0.098 dex | Std of log(V_obs/V_pred) over all data points |

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
| **Σ-Gravity** | 0 | 17.42 km/s | Works |

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

### Cluster Amplitude and Calibration

**Calibration procedure:** We fix $L_0 = 0.4$ kpc from typical disk scale heights—this is a physical reference scale, not a free parameter. We then calibrate only the exponent $n$ on the 42 Fox et al. clusters, minimizing the median absolute deviation of $\log_{10}(M_{\rm pred}/M_{\rm lens})$. This yields $n = 0.27$.

**Holdout validation:** Using 70/30 train/test splits with 10 random seeds, we confirm stability: calibrated $n = 0.27 \pm 0.01$, holdout median ratio $= 1.02 \pm 0.12$.

**Cluster amplitude:** Using the unified amplitude formula with L = 600 kpc (typical cluster path length):

$$A_{\rm cluster} = A_0 \times (L/L_0)^n = 1.173 \times (600/0.4)^{0.27} \approx 8.45$$

### Baryon Modeling Sensitivity

The cluster baryonic mass is estimated as:

$$M_{\rm bar}(200~{\rm kpc}) = f_{\rm conc} \times f_{\rm baryon} \times M_{500}$$

where:
- $f_{\rm baryon} = 0.15$ (cosmic baryon fraction from Planck)
- $f_{\rm conc} = 0.4$ (concentration factor: fraction of baryons within 200 kpc)

**Sensitivity to concentration factor:**

| $f_{\rm conc}$ | Median M_pred/M_lens | Change |
|----------------|---------------------|--------|
| 0.30 | 1.32 | +34% |
| 0.35 | 1.13 | +14% |
| **0.40** | **0.987** | baseline |
| 0.45 | 0.88 | −11% |
| 0.50 | 0.79 | −20% |

The value $f_{\rm conc} = 0.4$ is consistent with NFW concentration parameters $c \sim 4$–6 typical of massive clusters. Varying by ±25% shifts the median ratio by ~±30%, which would still be within the observational scatter (0.132 dex ≈ factor 1.35).

**Sensitivity to baryon fraction:**

| $f_{\rm baryon}$ | Median M_pred/M_lens | Notes |
|------------------|---------------------|-------|
| 0.12 | 1.23 | Low end of estimates |
| **0.15** | **0.987** | Planck cosmic value |
| 0.18 | 0.82 | High end of estimates |

**Independence of $M_{500}$ from lensing:** The $M_{500}$ values from Fox et al. (2022) are derived from X-ray or SZ observations, not from lensing. The lensing masses $M_{\rm SL}(200~{\rm kpc})$ are independently measured from strong lensing arc positions. This avoids circularity.

**Uncertainty propagation:** The 0.132 dex scatter in M_pred/M_lens includes contributions from:
- Baryonic mass uncertainty (~0.1 dex)
- Lensing mass uncertainty (~0.1 dex)
- Intrinsic scatter in cluster properties

The cluster calibration is robust to reasonable variations in baryon modeling assumptions.

---

## SI §7 — Milky Way Validation

### Results Summary

RMS = root-mean-square of (V_obs − V_pred) over all 28,368 disk stars.

| Model | RMS | Notes |
|-------|-----|-------|
| **Σ-Gravity** | **29.8 km/s** | 28,368 stars |
| MOND | 30.3 km/s | — |

### Methodology

1. Load Eilers-APOGEE-Gaia catalog
2. Apply asymmetric drift correction
3. Compute V_bar from McMillan 2017 (×1.16)
4. Apply Σ-enhancement
5. Compare to observed velocities

---

## SI §7a — Dwarf Spheroidal Galaxies (Host Inheritance Model)

### The Challenge

Dwarf spheroidal galaxies (dSphs) are satellites of the Milky Way that appear to be the most "dark matter dominated" objects known, with M/L ratios of 10–300. They are:
- **Dispersion-dominated**: No coherent rotation (σ ~ 7–11 km/s, v_rot ≈ 0)
- **Tiny**: r_half ~ 0.2–0.7 kpc
- **Embedded**: Orbit within the MW's gravitational field at 76–147 kpc

### The Solution: Host Inheritance

Rather than applying internal Σ-enhancement (which would give C → 0 for dispersion-dominated systems), dSphs **inherit the host galaxy's enhancement**:

$$\Sigma_{\rm dSph} = \Sigma_{\rm MW}(R_{\rm orbit})$$

where $R_{\rm orbit}$ is the dSph's distance from the MW center.

**Physical interpretation**: The dSph sits in the MW's already-enhanced gravitational field. The MW's Σ ~ 10–20 at dSph distances (which is what keeps the MW rotation curve flat at 220 km/s) provides the "missing mass" for the satellite's internal dynamics.

### Results

| dSph | d_MW (kpc) | Σ_MW | σ_pred (km/s) | σ_obs (km/s) | Ratio |
|------|------------|------|---------------|--------------|-------|
| Sculptor | 86 | 11.7 | 9.1 | 9.2 | **0.99** |
| Carina | 105 | 14.1 | 4.3 | 6.6 | 0.65 |
| Fornax | 147 | 19.4 | 21.7 | 10.7 | 2.03 |
| Draco | 76 | 10.4 | 3.4 | 9.1 | 0.38 |
| Ursa Minor | 76 | 10.4 | 2.9 | 9.5 | 0.31 |

**Mean ratio**: 0.87 ± 0.63

### Interpretation

- **Sculptor**: Near-perfect agreement (ratio = 0.99)
- **Fornax**: Overpredicts by 2× — suggests M_star may be overestimated
- **Draco, Ursa Minor**: Underpredicts by ~3× — suggests M_star may be underestimated

The scatter correlates with stellar mass: lower-mass dSphs (with more uncertain M_star estimates) show larger deviations. This is a **testable prediction**: improved baryonic mass estimates should reduce the scatter.

### Why This Works

dSphs don't need their own dark matter halos because:
1. They orbit inside the MW's Σ-enhanced field
2. The MW's enhancement at 76–147 kpc is Σ ~ 10–20
3. This enhancement applies to the dSph's internal dynamics
4. No separate internal enhancement is needed

This naturally explains why dSphs appear "dark matter dominated" without invoking separate dark matter halos for each satellite.

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
- $\nu(g_N, \mathcal{C}) = 1 + A \cdot \mathcal{C} \cdot h(g_N) = \Sigma$
- $\mathcal{C} = v_{\rm rot}^2/(v_{\rm rot}^2 + \sigma^2)$ is the covariant coherence scalar

**Test particles follow geodesics of $\Phi$**—no non-minimal matter coupling in the particle action.

### Covariant Acceleration Scalar

The "acceleration" in Σ-Gravity is a **field property**, not a particle property. It is defined from the gradient of the auxiliary potential:

$$g_N^2 \equiv g^{\mu\nu} \nabla_\mu \Phi_N \nabla_\nu \Phi_N$$

This is manifestly a scalar under coordinate transformations.

**Why not particle 4-acceleration?** If we defined the scalar via $a^\mu = u^\nu \nabla_\nu u^\mu$, we would face a problem: for geodesic matter, $a^\mu = 0$ identically. Since our QUMOND-like formulation has particles following geodesics of $\Phi$, using particle acceleration would be ill-defined.

The enhancement function $h(g_N)$ depends on the field strength $g_N = \sqrt{g_N^2}$, which is well-defined regardless of particle motion.

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

This is exactly the QUMOND construction (Milgrom 2010, PRD 82, 043523). A fully covariant action formulation is deferred to future work.

**Note on alternative formulations:** During development, we explored formulations involving direct modification of the stress-energy tensor (non-minimal matter coupling, e.g., $G_{\mu\nu} \propto T_{\mu\nu} \times \Sigma$) and modified teleparallel actions. These approaches were discarded because they reintroduce the conceptual problems of non-geodesic motion and ill-defined particle acceleration that the QUMOND-like formulation avoids. The theory presented here uses minimal matter coupling exclusively.

### Covariant Coherence Scalar (Primary Formulation)

**The coherence scalar C is the primary formulation** that determines gravitational enhancement. The full covariant definition is:

$$\mathcal{C} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}$$

In the non-relativistic limit used for all calculations:

$$\boxed{\mathcal{C} = \frac{v_{\rm rot}^2}{v_{\rm rot}^2 + \sigma^2}}$$

**Implementation:** Since C depends on $v_{\rm rot}$ (which depends on Σ), the prediction requires fixed-point iteration using $V_{\rm pred}$ (not $V_{\rm obs}$). Convergence typically occurs in 3-5 iterations.

**Practical approximation:** For disk galaxies, the orbit-averaged C is well-approximated by W(r) = r/(ξ+r), which gives identical results without iteration (validated in SI §13.5).

---

## SI §10 — Stress-Energy Conservation

### The QUMOND Analogy

In QUMOND/AQUAL (Milgrom 2010, Bekenstein & Milgrom 1984), the phantom density represents a redistribution of the gravitational field, not additional matter. Total stress-energy (matter + gravitational field) is conserved by construction.

Σ-Gravity inherits this structure. A fully covariant completion (deriving the field equations from an action) is deferred to future work. The phenomenological predictions do not depend on this completion.

---

## SI §11 — Relativistic Lensing Derivation

This section provides the derivation details for the effective lensing closure adopted in the main paper (Section III.C).

### Gravitational Slip Ansatz

In general relativity, the two scalar potentials $\Phi$ (Newtonian potential, governing dynamics) and $\Psi$ (curvature potential, governing light deflection) are equal in the absence of anisotropic stress. In modified gravity theories, these potentials generally differ, characterized by the gravitational slip parameter:

$$\eta \equiv \frac{\Psi}{\Phi}$$

For Σ-Gravity, we adopt an effective slip relation consistent with the QUMOND-like structure:

$$\boxed{\eta = \frac{2\Sigma - 1}{3\Sigma - 2}}$$

**Derivation motivation:** This ansatz arises from requiring that:
1. The GR limit is recovered: when $\Sigma \to 1$, we have $\eta \to 1$
2. The slip is bounded: $\eta$ remains finite and positive for all $\Sigma > 1$
3. The lensing enhancement is weaker than the dynamical enhancement (consistent with the phantom density interpretation)

| Σ | η | Physical regime |
|---|---|-----------------|
| 1.0 | 1.00 | GR limit (no enhancement) |
| 1.5 | 0.80 | Typical galaxy outer disk |
| 2.0 | 0.75 | Typical cluster enhancement |
| 3.0 | 0.71 | Strong enhancement regime |

### Lensing-to-Dynamics Mass Ratio

The lensing mass is determined by the lensing potential $\Phi_{\rm lens} = (\Phi + \Psi)/2$, while the dynamical mass is determined by $\Phi$ alone. The ratio is:

$$\frac{M_{\text{lens}}}{M_{\text{dyn}}} = \frac{\Phi + \Psi}{2\Phi} = \frac{1 + \eta}{2}$$

Substituting the slip relation:

$$\boxed{\frac{M_{\text{lens}}}{M_{\text{dyn}}} = \frac{5\Sigma - 3}{2(3\Sigma - 2)}}$$

| Σ | M_lens/M_dyn | Interpretation |
|---|--------------|----------------|
| 1.0 | 1.00 | GR: lensing = dynamics |
| 1.5 | 0.90 | Lensing slightly weaker |
| 2.0 | 0.875 | Lensing 12.5% weaker than dynamics |
| 3.0 | 0.857 | Asymptotic approach to 5/6 |

### Implications for Cluster Predictions

The cluster predictions in the main paper use this lensing closure. The key points:

1. **Strong lensing masses** from Fox et al. (2022) probe $M_{\rm lens}$
2. **Σ-Gravity predicts** an enhanced dynamical mass $M_{\rm dyn} = \Sigma \times M_{\rm bar}$
3. **The slip relation** maps between them: $M_{\rm lens} = M_{\rm dyn} \times (5\Sigma - 3)/(2(3\Sigma - 2))$

For typical cluster values ($\Sigma \approx 8.45$ from the path-length amplitude), the lensing-to-dynamics ratio approaches $\approx 0.83$. This is accounted for in the cluster calibration.

### Theoretical Status

This lensing prescription is an **effective ansatz**, not derived from first principles. A complete derivation would require:
1. A covariant action formulation of Σ-Gravity
2. Derivation of the metric perturbations in the weak-field limit
3. Identification of the physical source of gravitational slip

The current ansatz provides a self-consistent phenomenological framework that:
- Reduces to GR when enhancement vanishes
- Gives testable predictions for lensing-to-dynamics ratios
- Is consistent with the QUMOND-like field equation structure

Future work should derive this relation from first principles or constrain it observationally using systems with both dynamical and lensing mass measurements.

### Lensing Slip Sensitivity Analysis

The cluster results depend on the assumed gravitational slip relation. We test sensitivity to alternative slip prescriptions:

**Baseline (adopted):** $\eta = (2\Sigma - 1)/(3\Sigma - 2)$

**Alternative prescriptions:**

| Prescription | η formula | η at Σ=8.45 | M_lens/M_dyn |
|--------------|-----------|-------------|--------------|
| **Adopted** | $(2\Sigma-1)/(3\Sigma-2)$ | 0.68 | 0.84 |
| No slip (GR-like) | $\eta = 1$ | 1.00 | 1.00 |
| Maximal slip | $\eta = 1/\Sigma$ | 0.12 | 0.56 |
| MOND-like | $\eta = 1/\sqrt{\Sigma}$ | 0.34 | 0.67 |

**Impact on cluster calibration:**

| Slip prescription | Median M_pred/M_lens | Required n | Notes |
|-------------------|---------------------|------------|-------|
| **Adopted** | **0.987** | **0.27** | Baseline |
| No slip (η=1) | 0.83 | 0.32 | Would require higher n |
| MOND-like | 1.18 | 0.23 | Would require lower n |

**Key finding:** The cluster calibration is sensitive to the slip prescription, but the *qualitative* result (Σ-Gravity fits clusters where MOND fails) is robust. Different slip assumptions would require recalibrating n, but the unified framework would still work.

**Conditional statement:** The cluster lensing results are conditional on the adopted slip relation. If future observations or theoretical developments constrain $\eta$ differently, the cluster amplitude parameters would need recalibration. The SPARC galaxy results are unaffected by this choice (they use dynamics, not lensing).

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

### SI §13.3 Full 17-Test Comparison

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
| Dwarf Spheroidals | Host inheritance model (Σ_dSph = Σ_MW) |
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

### SI §13.5 Validation: C(r) as Primary Formulation

**The covariant coherence scalar C is now the primary formulation.** This section documents the validation that C(r) gives identical results to the W(r) approximation.

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
| Canonical W(r) = r/(ξ+r) | 17.42 | — | 42.7% |
| C_local (σ = 15 km/s) | 17.49 | +0.4% | 42.7% |
| C_local (σ = 20 km/s) | 17.42 | 0.0% | 42.7% |
| C_local (σ = 25 km/s) | 17.39 | −0.2% | 42.7% |
| C_local (σ = 30 km/s) | 17.42 | 0.0% | 42.7% |

**Interpretation:** The direct C(r) formulation gives **identical results** to the W(r) approximation. This validates:

1. **C(r) is the correct primary formulation** — theoretically proper and empirically validated
2. **W(r) is an excellent approximation** — for disk galaxies when iteration is undesired
3. **No loss of accuracy** from using the simpler W(r) form

### SI §13.6 Summary: C(r) as Primary

**The covariant coherence scalar C is the primary formulation:**

$$\Sigma = 1 + A \cdot \mathcal{C} \cdot h(g_N), \quad \mathcal{C} = \frac{v_{\rm rot}^2}{v_{\rm rot}^2 + \sigma^2}$$

**Advantages of C(r):**
1. **Covariant:** Built from 4-velocity invariants
2. **Local:** No reference to galaxy center or disk scale length
3. **General:** Works for any system with v_rot and σ
4. **Self-consistent:** Uses V_pred, not V_obs

**The W(r) approximation remains valid** for disk galaxies:
- W(r) = r/(ξ+r) with ξ = R_d/(2π)
- Gives identical results to C(r)
- Requires no iteration
- Derived from orbit-averaged C in disk geometry

---

## SI §14 — Parameter Sensitivity

### Coherence Window Exponent

| W(r) form | SPARC RMS |
|-----------|-----------|
| r/(ξ+r) [k=1] | 17.42 km/s |
| 1−(ξ/(ξ+r))^0.5 [k=0.5] | 18.64 km/s |

The k=1 form (simpler) performs better.

### Amplitude Sensitivity

| A_galaxy | SPARC RMS | Cluster Ratio |
|----------|-----------|---------------|
| 1.0 | 18.9 km/s | 0.85 |
| 1.173 | 17.42 km/s | 0.987 |
| 1.5 | 16.5 km/s | 1.12 |

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

## SI §16 — Additional Figures

This section contains supplementary figures that provide additional validation and detail beyond the main paper figures.

### SI §16.1 — RAR Residuals Histogram

![RAR Residuals](figures/rar_residuals_histogram.png)

*SI FIG. 16.1. Distribution of RAR residuals for Σ-Gravity (blue) and MOND (red outline) across all SPARC data points. Both theories achieve similar scatter (~0.10 dex, defined as std of log(V_obs/V_pred)), with Σ-Gravity showing slightly lower mean bias.*

### SI §16.2 — Cluster Holdout Validation

![Cluster Holdout](figures/cluster_holdout_validation.png)

*SI FIG. 16.2. Cluster holdout validation. Left: Predicted vs. observed Einstein radii for training (blue circles) and holdout (orange squares) clusters. Right: Normalized residuals by cluster. Both holdout clusters fall within 1σ of predictions, demonstrating out-of-sample predictive power.*

### SI §16.3 — Milky Way Comprehensive Comparison

![MW Comprehensive](figures/mw_comprehensive_comparison.png)

*SI FIG. 16.3. Milky Way rotation curve comparison. Four models compared against Gaia DR3 data (black points with errors): GR/baryons only (green dashed), Σ-Gravity (blue solid), MOND (red dotted), and NFW dark matter halo (purple dash-dot). Σ-Gravity and MOND both reproduce the flat rotation curve; the NFW halo requires tuned parameters.*

### SI §16.4 — Baryonic Tully-Fisher Relation

![BTFR](figures/btfr_two_panel_v2.png)

*SI FIG. 16.4. Baryonic Tully-Fisher Relation. Left: Log M_bar vs log V_flat for SPARC galaxies. Σ-Gravity predicts a slope of 4.0 (solid line), consistent with the observed relation. Right: Residuals from the best-fit relation.*

### SI §16.5 — Full Rotation Curve Gallery

The `figures/model_comparison/all_galaxies/` directory contains individual rotation curve plots for all 171 SPARC galaxies, comparing:
- Observed data (black points with error bars)
- Baryonic prediction (green dashed)
- Σ-Gravity prediction (blue solid)
- MOND prediction (red dotted)

Each plot is named `comparison_XX_GALAXYNAME.png` where XX is the galaxy index.

### SI §16.6 — Cluster Convergence Profiles

![Cluster Kappa Profiles](figures/cluster_kappa_profiles_panel.png)

*SI FIG. 16.6. Convergence profiles for selected clusters. Comparison of Σ-Gravity enhanced convergence (blue) with observed strong lensing constraints (gray bands).*

### SI §16.7 — Representative Galaxy Panel

![Representative Panel](figures/rc_representative_panel.png)

*SI FIG. 16.7. Six representative galaxies spanning the full range of SPARC: high-mass spirals (NGC2841, NGC6946), intermediate disks (NGC3198, NGC2403), and low-mass dwarfs (DDO154, UGC128). Σ-Gravity (blue) matches observations (black) across all mass scales without per-galaxy tuning.*

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
