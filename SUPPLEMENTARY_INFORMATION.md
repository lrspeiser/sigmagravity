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

The script `scripts/run_regression.py` reproduces all reported results using the canonical parameters:

| Parameter | Value |
|-----------|-------|
| $A_0$ | $e^{1/(2\pi)} \approx 1.1725$ |
| $L_0$ | 0.4 kpc |
| $n$ | 0.27 |
| $\xi$ | $R_d/(2\pi)$ |
| M/L (disk/bulge) | 0.5/0.7 |
| $g^\dagger$ | $9.599 \times 10^{-11}$ m/s² |

### Expected Results

Running the validation script produces:

| Test | Expected Result |
|------|-----------------|
| SPARC Galaxies | RMS = 17.75 km/s, Scatter = 0.097 dex, Win rate = 47.4% |
| Galaxy Clusters | Median ratio = 0.987, Scatter = 0.132 dex (N=42) |
| Milky Way | RMS = 29.5 km/s (N=28,368 stars) |
| Redshift Evolution | $g^\dagger(z=2)/g^\dagger(z=0) = 2.966$ |
| Solar System | $|\gamma-1| = 1.77 \times 10^{-9}$ |
| Counter-Rotation | $f_{\rm DM}$(CR) = 0.169 vs $f_{\rm DM}$(Normal) = 0.302, p = 0.004 |

### Output Files

Results are saved to `scripts/regression_results/latest_report.json` in machine-readable JSON format.

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

### Modified TEGR Action

$$S_{\Sigma} = \frac{1}{2\kappa} \int d^4x \, |e| \, \mathbf{T} + S_{\text{aux}} + \int d^4x \, |e| \, \Sigma \, \mathcal{L}_m$$

### QUMOND-Like Structure

The auxiliary field Φ_N satisfies:
$$\nabla^2 \Phi_N = 4\pi G \rho$$

The enhancement depends on g_N = |∇Φ_N|.

### Covariant Coherence Scalar

$$\mathcal{C} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}$$

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

### SI §13.1 Geometric Baseline

The canonical coherence scale is:

$$\xi_{\rm geom} = \frac{R_d}{2\pi} \approx 0.159 \times R_d$$

This corresponds to one azimuthal wavelength at the disk scale length.

### SI §13.2 Dynamical Coherence Scale

An alternative formulation uses:

$$\xi_{\rm dyn} = k \times \frac{\sigma_{\rm eff}}{\Omega_d}$$

where:
- k ≈ 0.24 (calibrated)
- σ_eff = effective velocity dispersion
- Ω_d = angular frequency at R_d

**Ablation Results:**

| Coherence Scale | SPARC RMS | Improvement |
|-----------------|-----------|-------------|
| ξ = R_d/(2π) | 17.75 km/s | Baseline |
| ξ_dyn | 16.8 km/s | +5% |

The dynamical formulation shows modest improvement but is not used for primary results due to additional complexity.

### SI §13.3 Interpretation

The dynamical coherence scale tracks the ratio of random to ordered motion:

$$\xi \propto \frac{\sigma}{\Omega} \propto T_{\rm orbit}$$

This connects to:
- Epicyclic orbit averaging
- Multi-fluid velocity dispersion
- H(z) scaling at high redshift

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
