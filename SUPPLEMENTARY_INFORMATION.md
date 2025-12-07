# Supplementary Information

## Σ-Gravity: A Coherence-Based Phenomenological Model for Galactic Dynamics

**Author:** Leonard Speiser

This Supplementary Information (SI) accompanies the main manuscript and provides complete technical details for reproducing and extending all results.

---

## Table of Contents

1. [SI §1 — Canonical Model Definition](#si-1--canonical-model-definition)
2. [SI §2 — Parameter Summary](#si-2--parameter-summary)
3. [SI §3 — Coherence Scale Derivation](#si-3--coherence-scale-derivation)
4. [SI §4 — Critical Acceleration Derivation](#si-4--critical-acceleration-derivation)
5. [SI §5 — Path Length Amplitude Scaling](#si-5--path-length-amplitude-scaling)
6. [SI §6 — Data Sources](#si-6--data-sources)
7. [SI §7 — Reproducibility: Master Regression Test](#si-7--reproducibility-master-regression-test)
8. [SI §8 — SPARC Galaxy Analysis](#si-8--sparc-galaxy-analysis)
9. [SI §9 — Galaxy Cluster Lensing](#si-9--galaxy-cluster-lensing)
10. [SI §10 — Milky Way Validation](#si-10--milky-way-validation)
11. [SI §11 — Unique Predictions](#si-11--unique-predictions)
12. [SI §12 — Theoretical Framework](#si-12--theoretical-framework)
13. [SI §13 — Stress-Energy Conservation](#si-13--stress-energy-conservation)
14. [SI §14 — Relativistic Lensing Derivation](#si-14--relativistic-lensing-derivation)
15. [SI §15 — Wide Binary Analysis](#si-15--wide-binary-analysis)
16. [SI §16 — Expected Results Reference](#si-16--expected-results-reference)

---

## SI §1 — Canonical Model Definition

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

### Unified Amplitude Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **A₀** | $e^{1/(2\pi)} \approx 1.173$ | Base amplitude |
| **L₀** | 0.40 kpc | Reference path length |
| **n** | 0.27 | Path length exponent |
| **D** | 0 (galaxy) or 1 (cluster) | Dimensionality factor |

**Amplitude values:**
- Disk galaxies (D=0): A = A₀ = 1.173
- Galaxy clusters (D=1, L≈600 kpc): A ≈ 8.45

### Rotation Curve Prediction

For disk galaxies:

$$V_{\text{pred}} = V_{\text{bar}} \times \sqrt{\Sigma}$$

where $V_{\text{bar}}^2 = V_{\text{gas}}^2 + \Upsilon_{\text{disk}} \cdot V_{\text{disk}}^2 + \Upsilon_{\text{bulge}} \cdot V_{\text{bulge}}^2$

**Mass-to-light ratios (Lelli+ 2016 standard):**
- Υ_disk = 0.5 M☉/L☉ at 3.6μm
- Υ_bulge = 0.7 M☉/L☉ at 3.6μm

---

## SI §2 — Parameter Summary

### Canonical Parameters (Current Paper)

| Parameter | Value | Status | Source |
|-----------|-------|--------|--------|
| **g†** | $cH_0/(4\sqrt{\pi}) = 9.60 \times 10^{-11}$ m/s² | **Derived** | Spherical coherence geometry (SI §4) |
| **ξ** | $R_d/(2\pi) \approx 0.159 \times R_d$ | **Derived** | One azimuthal wavelength (SI §3) |
| **A₀** | $e^{1/(2\pi)} \approx 1.173$ | **Derived** | 2D coherence geometry (SI §5) |
| **L₀** | 0.40 kpc | **Calibrated** | Reference path length |
| **n** | 0.27 | **Calibrated** | Path length exponent |
| **M/L_disk** | 0.5 M☉/L☉ | Fixed | Lelli+ 2016 |
| **M/L_bulge** | 0.7 M☉/L☉ | Fixed | Lelli+ 2016 |

### Derivation Status Key

- **Derived**: Mathematical result from stated assumptions
- **Fixed**: Standard value from literature, not fitted
- **Calibrated**: Physical motivation with final value set by data

### Key Result: No Free Parameters Per Galaxy

Unlike ΛCDM (2-3 parameters per galaxy for NFW halo fitting), Σ-Gravity uses the same formula with the same global parameters for all 171 SPARC galaxies.

---

## SI §3 — Coherence Scale Derivation

### Canonical Formula

$$\boxed{\xi = \frac{R_d}{2\pi} \approx 0.159 \times R_d}$$

### Physical Derivation

The coherence scale emerges from the condition that coherence is established over **one azimuthal wavelength** at the disk scale length:

1. **2D disk coherence**: For a thin disk with 2D structure, the superstatistical shape parameter is k = ν/2 = 1 (where ν = 2 is the effective dimensionality)

2. **Azimuthal wavelength**: At radius R_d, one complete azimuthal cycle spans $2\pi R_d$

3. **Radial coherence scale**: The radial coherence scale equals the disk scale length divided by one complete azimuthal cycle:
   $$\xi = \frac{R_d}{2\pi}$$

### Physical Interpretation

ξ is the radius where random motions (σ_eff) become comparable to ordered rotation (Ω × r):
- **Small r < ξ**: Dispersion dominates → coherence suppressed → W(r) → 0
- **Large r > ξ**: Rotation dominates → full coherence → W(r) → 1

This is an **instantaneous** property of the velocity field—purely spatial, no temporal accumulation required.

### Coherence Window

$$W(r) = \frac{r}{\xi + r} = \frac{r}{R_d/(2\pi) + r}$$

For 2D coherence (k = 1), the exponent is n_coh = k/2 = 0.5, giving the simplified form above.

### Validation

The 2D coherence framework provides:
- 16% improvement in RMS prediction error over earlier phenomenological values
- Physical interpretation (coherence suppression where dispersion dominates rotation)
- Robustness (improvement holds with baryons-only computation)

---

## SI §4 — Critical Acceleration Derivation

### Canonical Formula

$$\boxed{g^\dagger = \frac{cH_0}{4\sqrt{\pi}} \approx 9.60 \times 10^{-11} \text{ m/s}^2}$$

### Geometric Derivation

1. **Coherence radius**: The radius at which gravitational coherence develops:
   $$R_{\rm coh} = \sqrt{4\pi} \times \frac{V^2}{cH_0}$$
   
   The factor $\sqrt{4\pi}$ arises from the full solid angle (4π steradians) in spherical geometry.

2. **Critical acceleration**: At $r = 2 \times R_{\rm coh}$, the acceleration is:
   $$g^\dagger = \frac{V^2}{2 \times R_{\rm coh}} = \frac{cH_0}{4\sqrt{\pi}}$$

### Numerical Value

```python
import numpy as np
c = 2.998e8  # m/s
H0 = 70e3 / 3.086e22  # s⁻¹ (70 km/s/Mpc)
g_dagger = c * H0 / (4 * np.sqrt(np.pi))
# = 9.60e-11 m/s²
```

### Geometric Interpretation

$$4\sqrt{\pi} = 2 \times \sqrt{4\pi} \approx 7.09$$

- $\sqrt{4\pi} \approx 3.54$ arises from spherical solid angle
- Factor 2 comes from the coherence transition scale

### Comparison to MOND

| Parameter | Σ-Gravity | MOND |
|-----------|-----------|------|
| Critical acceleration | $9.60 \times 10^{-11}$ m/s² | $1.2 \times 10^{-10}$ m/s² |
| Derivation | Geometric (spherical coherence) | Empirical fit |
| Factor | $cH_0/(4\sqrt{\pi})$ | $cH_0/(2e)$ (if derived) |

---

## SI §5 — Path Length Amplitude Scaling

### Universal Scaling Law

$$\boxed{A = A_0 \times L^{1/4}}$$

where:
- $A_0 = e^{1/(2\pi)} \approx 1.173$ is the universal constant
- L is the characteristic path length through baryonic matter

### Path Length Estimates

| System | Path Length L | A = A₀ × L^(1/4) | Used A |
|--------|--------------|------------------|--------|
| Disk galaxies | 1.5 kpc | 1.30 | 1.17 |
| Ellipticals | 17 kpc | 2.38 | ~3.1 |
| Clusters | 400 kpc | 5.24 | 8.0 |

**Note**: The path length scaling provides the correct order of magnitude. The canonical A_galaxy = e^(1/2π) ≈ 1.173 is used for all disk galaxies including the Milky Way.

### Physical Interpretation

The L^(1/4) scaling suggests gravitational coherence accumulates as the field propagates through baryonic matter, analogous to a diffusion process:
- 4D spacetime random walk process
- Two nested √ processes (spatial × temporal averaging)
- Dimensional reduction from coherence integral

### Amplitude Ratio

$$\frac{A_{\text{cluster}}}{A_{\text{galaxy}}} = \left(\frac{L_{\text{cluster}}}{L_{\text{galaxy}}}\right)^{1/4} = \left(\frac{400}{1.5}\right)^{1/4} \approx 4.0$$

The observed ratio $8.0/1.17 \approx 6.8$ is explained by a combination of path length scaling and mode counting effects (see SI §9).

---

## SI §6 — Data Sources

### SI §6.1 SPARC Galaxies (N=171)

**Source:** Spitzer Photometry and Accurate Rotation Curves (SPARC)  
**Reference:** Lelli, McGaugh & Schombert (2016), AJ 152, 157  
**DOI:** 10.3847/0004-6256/152/6/157  
**URL:** http://astroweb.cwru.edu/SPARC/

**Files:**
- `data/Rotmod_LTG/*_rotmod.dat` — Individual galaxy rotation curves (175 files)
- `data/Rotmod_LTG/MasterSheet_SPARC.mrt` — Galaxy properties including disk scale lengths

**Column Format (per galaxy .dat file):**

| Column | Name | Units | Description |
|--------|------|-------|-------------|
| 1 | R | kpc | Galactocentric radius |
| 2 | V_obs | km/s | Observed rotation velocity (inclination-corrected) |
| 3 | V_err | km/s | Error on V_obs |
| 4 | V_gas | km/s | Gas velocity contribution (from HI 21cm) |
| 5 | V_disk | km/s | Disk velocity contribution (at M/L = 1) |
| 6 | V_bul | km/s | Bulge velocity contribution (at M/L = 1) |

**Critical Note:** SPARC files provide V_disk and V_bulge computed for **reference M/L = 1 M☉/L☉ at 3.6μm**. The actual baryonic velocity requires scaling:

$$V_{\rm bar}^2 = V_{\rm gas}^2 + \Upsilon_{\rm disk} \cdot V_{\rm disk}^2 + \Upsilon_{\rm bulge} \cdot V_{\rm bulge}^2$$

where Υ_disk = 0.5 and Υ_bulge = 0.7 (Lelli+ 2016 recommended values).

**Processing Steps Applied:**

```python
# 1. Apply M/L correction
V_disk_scaled = V_disk * np.sqrt(0.5)   # √0.5 ≈ 0.707
V_bulge_scaled = V_bulge * np.sqrt(0.7) # √0.7 ≈ 0.837

# 2. Compute V_bar with signed gas contribution
V_bar_sq = np.sign(V_gas) * V_gas**2 + V_disk_scaled**2 + V_bulge_scaled**2
V_bar = np.sqrt(np.abs(V_bar_sq)) * np.sign(V_bar_sq)

# 3. Estimate disk scale length R_d from rotation curve shape
# (Use R at 1/3 of rotation curve points as proxy when not in MasterSheet)
idx = len(data) // 3
R_d = R[idx] if idx > 0 else R[-1] / 2
```

**Sample Selection:**

| Criterion | N | Notes |
|-----------|---|-------|
| SPARC database | 175 | Original sample |
| Valid V_bar at all radii | 174 | Excludes UGC01281 (imaginary V_bar) |
| ≥5 rotation curve points | **171** | Quality cut for reliable RMS |

**Excluded Galaxy:** UGC01281 — excluded due to unphysical V_bar values at inner radii (negative gas velocity dominates over disk velocity, producing imaginary V_bar).

---

### SI §6.2 Fox+ 2022 Galaxy Clusters (N=42)

**Source:** Fox et al. (2022), ApJ 928, 87  
**Title:** "The Strongest Cluster Lenses: An Analysis of the Relation between Strong Lensing Strength and Physical Properties of Galaxy Clusters"  
**DOI:** 10.3847/1538-4357/ac5024

**File:** `data/clusters/fox2022_unique_clusters.csv`

**Column Format:**

| Column | Units | Description |
|--------|-------|-------------|
| `cluster` | — | Cluster name (e.g., "Abell 370") |
| `z_lens` | — | Cluster redshift |
| `M500_1e14Msun` | 10¹⁴ M☉ | Total mass within R500 (from SZ/X-ray) |
| `MSL_200kpc_1e12Msun` | 10¹² M☉ | Strong lensing mass at 200 kpc aperture |
| `spec_z_constraint` | — | "yes" if spectroscopic redshift available |

**Selection Criteria Applied:**

1. `spec_z_constraint == 'yes'` — Spectroscopic redshift confirmed
2. `M500_1e14Msun > 2.0` — Exclude low-mass clusters with large uncertainties
3. Both M500 and MSL_200kpc measurements available

**Baryonic Mass Estimate:**

$$M_{\rm bar}(200~{\rm kpc}) = 0.4 \times f_{\rm baryon} \times M_{500}$$

where:
- $f_{\rm baryon} = 0.15$ (cosmic baryon fraction)
- Factor 0.4 accounts for concentration of baryons within 200 kpc aperture

**Processing Steps:**

```python
# 1. Filter to high-quality clusters
df_valid = df[
    df['M500_1e14Msun'].notna() & 
    df['MSL_200kpc_1e12Msun'].notna() &
    (df['spec_z_constraint'] == 'yes') &
    (df['M500_1e14Msun'] > 2.0)
]

# 2. Compute baryonic mass at 200 kpc
M500 = df_valid['M500_1e14Msun'] * 1e14  # Convert to M☉
M_bar_200 = 0.4 * 0.15 * M500            # 0.4 × f_baryon × M500

# 3. Compute baryonic acceleration
r_kpc = 200
r_m = r_kpc * 3.086e19
g_bar = G * M_bar_200 * M_sun / r_m**2
```

---

### SI §6.3 Eilers-APOGEE-Gaia Milky Way (N=28,368)

**Source:** Cross-match of three catalogs:
1. **Eilers+ 2019** (ApJ 871, 120): Spectrophotometric distances
2. **APOGEE DR17**: Radial velocities and stellar parameters
3. **Gaia EDR3**: Proper motions and parallaxes

**File:** `data/gaia/eilers_apogee_6d_disk.csv`

**Column Format:**

| Column | Units | Description |
|--------|-------|-------------|
| `source_id` | — | Gaia source identifier |
| `R_gal` | kpc | Galactocentric cylindrical radius |
| `z` | kpc | Height above Galactic plane |
| `v_R` | km/s | Radial velocity component |
| `v_phi` | km/s | Azimuthal velocity (raw, needs sign correction) |
| `v_z` | km/s | Vertical velocity component |

**Sign Convention:** The raw `v_phi` column uses the opposite sign convention from standard (positive = retrograde). We apply:

```python
v_phi_obs = -df['v_phi']  # Correct to positive = prograde
```

**Selection Criteria:**
- Disk stars: 4 < R_gal < 15 kpc
- Thin disk: |z| < 0.5 kpc
- Full 6D phase space available

**Solar Motion Parameters Used:**

```python
R0_KPC = 8.122      # Distance from Sun to Galactic center (Bennett & Bovy 2019)
ZSUN_KPC = 0.0208   # Height of Sun above Galactic plane
VSUN_KMS = [11.1, 232.24, 7.25]  # Solar motion [U, V, W] (Schönrich+ 2010)
```

**Baryonic Model:** McMillan 2017, scaled by factor 1.16× to match SPARC calibration:

```python
MW_VBAR_SCALE = 1.16  # Within ~20% uncertainty of McMillan 2017 (Cautun+ 2020)
M_disk = 4.6e10 * MW_VBAR_SCALE**2   # Disk mass
M_bulge = 1.0e10 * MW_VBAR_SCALE**2  # Bulge mass
M_gas = 1.0e10 * MW_VBAR_SCALE**2    # Gas mass
```

**Asymmetric Drift Correction:**

$$V_a = \frac{\sigma_R^2}{2 V_c} \times \left(\frac{R}{R_d} - 1\right)$$

where σ_R is computed in radial bins from the data itself.

---

### SI §6.4 Counter-Rotating Galaxies (N=63)

**Sources:**

1. **MaNGA DynPop Catalog** (Zhu et al. 2023, MNRAS 522, 6326)
   - URL: https://manga-dynpop.github.io/pages/data_access/
   - File: `data/manga_dynpop/SDSSDR17_MaNGA_JAM.fits`
   - Contents: Dynamical masses and dark matter fractions for 10,296 MaNGA galaxies

2. **Bevacqua et al. 2022 Counter-Rotating Catalog** (MNRAS 511, 139)
   - VizieR: J/MNRAS/511/139
   - File: `data/stellar_corgi/bevacqua2022_counter_rotating.tsv`
   - Contents: 64 counter-rotating galaxies identified in MaNGA

**MaNGA DynPop FITS Structure:**

| HDU | Name | Contents |
|-----|------|----------|
| 1 | BASIC | Galaxy properties, `mangaid` for cross-matching |
| 4 | JAM_NFW | JAM modeling results with NFW halo, contains `fdm_Re` |

**Key Column:** `fdm_Re` in HDU 4 — Dark matter fraction within effective radius from JAM modeling

**Cross-Matching:**

```python
# MaNGA ID format: "1-113520"
dynpop_idx = {str(mid).strip(): i for i, mid in enumerate(basic['mangaid'])}
matches = [dynpop_idx[cr_id] for cr_id in cr_manga_ids if cr_id in dynpop_idx]
```

**Download Commands:**

```bash
# MaNGA DynPop catalog
mkdir -p data/manga_dynpop
curl -L -o data/manga_dynpop/SDSSDR17_MaNGA_JAM.fits \
  "https://raw.githubusercontent.com/manga-dynpop/manga-dynpop.github.io/main/catalogs/JAM/SDSSDR17_MaNGA_JAM.fits"

# Bevacqua et al. counter-rotating catalog
mkdir -p data/stellar_corgi
curl -s "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/MNRAS/511/139/table1&-out.max=200&-out.form=|" \
  > data/stellar_corgi/bevacqua2022_counter_rotating.tsv
```

---

### SI §6.5 Cosmological Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| H₀ | 70 km/s/Mpc | Planck 2018 (rounded) |
| c | 2.998×10⁸ m/s | CODATA 2018 |
| G | 6.674×10⁻¹¹ m³/kg/s² | CODATA 2018 |
| M☉ | 1.989×10³⁰ kg | IAU 2015 |
| 1 kpc | 3.086×10¹⁹ m | IAU 2012 |
| Ωₘ | 0.3 | Planck 2018 |
| ΩΛ | 0.7 | Planck 2018 |

---

## SI §7 — Reproducibility: Master Regression Test

### Single Command Validation

```bash
cd sigmagravity
python scripts/run_regression.py
```

This validates **all results** in the paper across all domains.

### Quick Mode (Skip Slow Tests)

```bash
python scripts/run_regression.py --quick
```

### Expected Output

```
================================================================================
Σ-GRAVITY MASTER REGRESSION TEST
================================================================================

UNIFIED FORMULA PARAMETERS:
  A₀ = exp(1/2π) ≈ 1.1725
  L₀ = 0.4 kpc
  n = 0.27
  ξ = R_d/(2π) ≈ 0.1592 × R_d
  M/L = 0.5/0.7 (disk/bulge)
  g† = 9.599e-11 m/s²

  For 2D disk (D=0): A = A₀ = 1.173
  For 3D cluster (D=1, L=600): A = 8.45

[✓] SPARC Galaxies: RMS=17.75 km/s, Scatter=0.097 dex, Win=47.4%
[✓] Clusters: Median ratio=0.987, Scatter=0.132 dex (42 clusters)
[✓] Gaia/MW: RMS=29.5 km/s (28368 stars)
[✓] Redshift Evolution: g†(z=2)/g†(z=0) = 2.966 (expected 2.966)
[✓] Solar System: |γ-1| = 1.77e-09 < 2.30e-05
[✓] Counter-Rotation: f_DM(CR)=0.169 < f_DM(Normal)=0.302, p=0.0039

================================================================================
SUMMARY: 6/6 tests passed
================================================================================
✓ ALL TESTS PASSED
```

### Output Files

- `regression_results/latest_report.json` — Full JSON report
- Exit code 0 = all passed, 1 = some failed

---

## SI §8 — SPARC Galaxy Analysis

### Results Summary

| Metric | Σ-Gravity | MOND | Notes |
|--------|-----------|------|-------|
| Mean RMS error | **17.75 km/s** | 17.15 km/s | 171 galaxies |
| Win rate | 47.4% | 52.6% | Fair comparison (same M/L) |
| RAR scatter | 0.097 dex | 0.098 dex | — |

### Methodology

1. **Load SPARC data** with M/L = 0.5/0.7 correction
2. **Compute V_bar** from gas + disk + bulge contributions
3. **Apply Σ-enhancement** using canonical parameters
4. **Compare to V_obs** and compute RMS residual

### Reference Implementation

```python
import numpy as np

# Constants
c, H0_SI, kpc_to_m = 2.998e8, 2.27e-18, 3.086e19
g_dagger = c * H0_SI / (4 * np.sqrt(np.pi))  # 9.60e-11 m/s²
A_GALAXY = np.exp(1/(2*np.pi))  # 1.173
XI_SCALE = 1/(2*np.pi)  # 0.159

def h_function(g_N):
    """Acceleration function h(g_N)."""
    g_N = np.maximum(g_N, 1e-15)
    return np.sqrt(g_dagger / g_N) * g_dagger / (g_dagger + g_N)

def W_coherence(r_kpc, R_d_kpc):
    """Coherence window W(r) = r/(ξ+r)."""
    xi = max(XI_SCALE * R_d_kpc, 0.01)
    return r_kpc / (xi + r_kpc)

def Sigma_enhancement(r_kpc, g_N, R_d_kpc, A=A_GALAXY):
    """Enhancement factor: Σ = 1 + A × W(r) × h(g_N)."""
    return 1 + A * W_coherence(r_kpc, R_d_kpc) * h_function(g_N)

def predict_velocity(R_kpc, V_bar_kms, R_d_kpc, A=A_GALAXY):
    """V_pred = V_bar × √Σ."""
    g_N = (V_bar_kms * 1000)**2 / (R_kpc * kpc_to_m)
    return V_bar_kms * np.sqrt(Sigma_enhancement(R_kpc, g_N, R_d_kpc, A))
```

### Reproduction Commands

```bash
# Full SPARC analysis
python scripts/run_regression.py

# Generate rotation curve gallery
python scripts/generate_representative_panel.py
# Output: figures/rc_gallery_derived.png

# Generate RAR plot
python scripts/generate_paper_figures.py
# Output: figures/rar_derived_formula.png
```

---

## SI §9 — Galaxy Cluster Lensing

### Results Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Median M_pred/M_lens | **0.955** | N=42 clusters |
| Scatter | 0.133 dex | Comparable to ΛCDM |
| Within factor 2 | 100% | No catastrophic failures |

### Comparison to Other Theories

| Theory | M_predicted/M_lensing | Notes |
|--------|----------------------|-------|
| GR + baryons only | 0.10–0.15 | The "missing mass" problem |
| MOND (standard) | ~0.33 | The "cluster problem" |
| ΛCDM (fitted halos) | 0.95–1.05 | Requires 2-3 parameters per cluster |
| **Σ-Gravity** | **0.955** | Zero free parameters per cluster |

### Cluster Amplitude Derivation

The cluster amplitude emerges from **path length scaling**:

$$A_{\rm cluster} = A_0 \times L_{\rm cluster}^{1/4} = 1.173 \times 400^{1/4} \approx 5.2$$

The empirical value A_cluster = 8.0 is higher, possibly due to:
- Mode counting (3D vs 2D geometry): factor ~2.6
- Coherence window saturation: factor ~1.9

### Methodology

1. **Estimate baryonic mass**: $M_{\rm bar} = 0.4 \times f_{\rm baryon} \times M_{500}$
2. **Compute baryonic acceleration** at 200 kpc
3. **Apply Σ-enhancement** with A_cluster = 8.0
4. **Compare** to strong lensing mass MSL(200 kpc)

### Reproduction

```bash
python scripts/run_regression.py  # Includes cluster test
```

---

## SI §10 — Milky Way Validation

### Results Summary

| Model | Mean Residual | RMS | Improvement |
|-------|---------------|-----|-------------|
| **Σ-Gravity** | **−0.7 km/s** | **29.4 km/s** | — |
| MOND | +8.1 km/s | 30.3 km/s | — |
| **Σ-Gravity vs MOND** | — | — | **+3%** |

### Methodology

1. **Load Eilers-APOGEE-Gaia catalog** (28,368 disk stars)
2. **Apply asymmetric drift correction**: $V_a = \sigma_R^2/(2V_c) \times (R/R_d - 1)$
3. **Compute V_bar** from McMillan 2017 baryonic model (scaled by 1.16×)
4. **Apply Σ-enhancement** using canonical parameters
5. **Compare** to observed circular velocities

### V_bar Scaling

The factor 1.16× brings the MW baryonic model into consistency with SPARC galaxies:
- McMillan 2017 gives V_bar(8 kpc) ≈ 172 km/s
- Scaled: V_bar(8 kpc) ≈ 200 km/s
- Observed: V_obs(8 kpc) ≈ 228 km/s
- Ratio: 228/200 = 1.14, consistent with SPARC galaxies at similar g/g†

This scaling is within the ~20% uncertainty of the McMillan 2017 model (Cautun+ 2020).

### Reproduction

```bash
python scripts/run_regression.py  # Includes MW test
```

---

## SI §11 — Unique Predictions

### 1. Counter-Rotating Disks — **CONFIRMED**

**Prediction:** Counter-rotating stellar components disrupt coherence, reducing gravitational enhancement.

**Observation (MaNGA DynPop × Bevacqua 2022):**

| Metric | Counter-Rotating (N=63) | Normal (N=10,038) | Difference |
|--------|------------------------|-------------------|------------|
| f_DM mean | **0.169** | 0.302 | **−44%** |
| f_DM median | **0.091** | 0.168 | **−46%** |

**Statistical Significance:**
- KS test: p = 0.006
- Mann-Whitney U: p = 0.004
- T-test: p = 0.001

**Uniqueness:** Neither ΛCDM nor MOND predicts any effect from rotation direction.

### 2. Velocity Dispersion Dependence

**Prediction:** High velocity dispersion reduces coherence:

$$W_{\text{eff}} = W(r) \times \exp(-(\sigma_v/v_c)^2)$$

| σ_v/v_c | W_eff | Σ | Comment |
|---------|-------|---|---------|
| 0.0 | 0.816 | 2.69 | Perfectly cold disk |
| 0.1 | 0.808 | 2.67 | Typical spiral |
| 0.2 | 0.784 | 2.61 | Thick disk |
| 0.3 | 0.743 | 2.51 | Hot disk |

**MOND has no σ_v dependence at fixed g_bar.**

### 3. Redshift Evolution

**Prediction:** Enhancement decreases at high redshift as g†(z) ∝ H(z):

$$g^\dagger(z) = \frac{cH(z)}{4\sqrt{\pi}} = g^\dagger_0 \times \sqrt{\Omega_m(1+z)^3 + \Omega_\Lambda}$$

| Redshift | g†(z)/g†(0) | Effect |
|----------|-------------|--------|
| z = 0 | 1.00 | Baseline |
| z = 1 | 1.77 | 77% higher threshold |
| z = 2 | 2.97 | 197% higher threshold |

**Consequence:** At fixed g_bar, enhancement is REDUCED at high z. This is consistent with KMOS³D observations showing reduced "dark matter fractions" at z ~ 2.

### 4. Solar System Safety

**Prediction:** Enhancement is suppressed in compact, high-acceleration systems.

| Location | g (m/s²) | g/g† | Σ−1 |
|----------|----------|------|-----|
| Earth orbit | 5.9×10⁻³ | 6×10⁷ | < 10⁻¹² |
| Saturn orbit | 6.5×10⁻⁵ | 7×10⁵ | < 10⁻⁹ |
| Neptune orbit | 6.6×10⁻⁶ | 7×10⁴ | < 10⁻⁸ |

**Cassini bound:** |γ−1| < 2.3×10⁻⁵  
**Σ-Gravity:** |γ−1| ~ 10⁻⁹ (3 orders of magnitude margin)

---

## SI §12 — Theoretical Framework

### Modified TEGR Action

$$S_{\Sigma} = \frac{1}{2\kappa} \int d^4x \, |e| \, \mathbf{T} + S_{\text{aux}} + \int d^4x \, |e| \, \Sigma[g_N, \mathcal{C}] \, \mathcal{L}_m$$

where:
- $|e|$ = tetrad determinant
- $\mathbf{T}$ = torsion scalar (standard TEGR)
- $S_{\text{aux}}$ = auxiliary field sector for Newtonian potential
- $\Sigma$ = coherent enhancement factor
- $\mathcal{L}_m = -\rho c^2$ = matter Lagrangian

### QUMOND-Like Structure

The auxiliary field $\Phi_N$ satisfies:
$$\nabla^2 \Phi_N = 4\pi G \rho$$

This is the **Poisson equation as an equation of motion**, not an external prescription.

The enhancement depends on $g_N = |\nabla\Phi_N|$, which is a well-defined functional of the matter distribution.

### Weak-Field Limit

$$\nabla^2 \Phi = 4\pi G \rho_{\text{eff}}$$

where:
$$\rho_{\text{eff}} = \Sigma \cdot \rho = [1 + A \cdot W(r) \cdot h(g_N)] \cdot \rho$$

### Covariant Coherence Scalar

$$\mathcal{C} = \frac{\omega^2}{\omega^2 + 4\pi G\rho + \theta^2 + H_0^2}$$

where:
- $\omega^2$ = vorticity scalar (from velocity field)
- $4\pi G\rho$ = Jeans rate (from local density)
- $\theta$ = expansion scalar
- $H_0^2$ = cosmic infrared cutoff

**Non-relativistic limit:**
$$\mathcal{C} = \frac{(v_{\rm rot}/\sigma_v)^2}{1 + (v_{\rm rot}/\sigma_v)^2}$$

---

## SI §13 — Stress-Energy Conservation

### The Problem

With Σ as an external functional, matter stress-energy is not conserved:
$$\nabla_\mu T^{\mu\nu}_{\text{matter}} \neq 0$$

### The Resolution

Promote Σ to a **dynamical scalar field** $\phi_C$:
$$f(\phi_C) = 1 + \frac{\phi_C^2}{M^2} = \Sigma$$

**Conservation law:**
$$\boxed{\nabla_\mu \left( T^{\mu\nu}_{\text{matter}} + T^{\mu\nu}_{\text{coherence}} \right) = 0}$$

The coherence field carries the "missing" momentum/energy, analogous to scalar fields in scalar-tensor theories.

### Validation

The dynamical field formulation exactly reproduces original Σ-Gravity predictions (0.000 km/s difference on 50 SPARC galaxies tested).

---

## SI §14 — Relativistic Lensing Derivation

### Photon Coupling

| Field | Coupling | Consequence |
|-------|----------|-------------|
| Matter (baryons) | Non-minimal ($\Sigma \cdot \mathcal{L}_m$) | Enhanced |
| Electromagnetic | Minimal | Photons follow null geodesics |

**Key choice:** EM couples minimally to preserve c_EM = c and consistency with GW170817.

### Weak-Field Metric

$$ds^2 = -(1 + 2\Phi/c^2)c^2 dt^2 + (1 - 2\Psi/c^2)d\mathbf{x}^2$$

From the field equations:
$$\nabla^2 \Phi = 4\pi G (3\Sigma - 2) \rho$$
$$\nabla^2 \Psi = 4\pi G (2\Sigma - 1) \rho$$

### Gravitational Slip

$$\eta \equiv \frac{\Psi}{\Phi} = \frac{2\Sigma - 1}{3\Sigma - 2}$$

| Σ | η | Notes |
|---|---|-------|
| 1.0 | 1.0 | GR limit |
| 1.5 | 0.80 | Transition |
| 2.0 | 0.75 | Typical cluster |

### Lensing-to-Dynamics Ratio

$$\frac{M_{\text{lens}}}{M_{\text{dyn}}} = \frac{5\Sigma - 3}{2(3\Sigma - 2)}$$

For Σ = 2: ratio = 0.875 (close to unity, within cluster systematics).

---

## SI §15 — Wide Binary Analysis

### Theoretical Status

Two possible responses to the low-g regime (wide binaries, Oort cloud):

**Option A: External Field Effect (Phenomenological)**
- The MW's gravitational field (g_MW ≈ 2.2×10⁻¹⁰ m/s²) suppresses enhancement
- Predicts 10-15% velocity boost at 10,000 AU
- **Not derived from the action**

**Option B: Coherence Requires Extended Rotation**
- Wide binaries lack disk structure → W → 0
- Predicts no enhancement (Σ = 1)
- **Limits theory scope to extended rotating systems**

### Current Observational Status

| Study | Claim |
|-------|-------|
| Chae (2023) | ~40% velocity excess |
| Banik et al. (2024) | No significant excess |

**Σ-Gravity's position:** The data is currently insufficient to distinguish between options. Resolution requires Gaia DR4 and theoretical derivation of EFE from the action.

---

## SI §16 — Expected Results Reference

### Master Regression Test Results

| Test | Expected Value | Threshold |
|------|----------------|-----------|
| SPARC mean RMS | 17.5 km/s | < 25 km/s |
| SPARC win rate vs MOND | 48% | — |
| Cluster median ratio | 0.955 | 0.5–1.5 |
| Cluster scatter | 0.133 dex | — |
| MW RMS | 29.4 km/s | < 35 km/s |
| Counter-rotation p-value | 0.004 | < 0.05 |
| Redshift g†(z=2)/g†(z=0) | 2.966 | — |
| Solar System |γ-1| | 1.8×10⁻⁹ | < 2.3×10⁻⁵ |

### Reproduction Command

```bash
git clone https://github.com/lrspeiser/SigmaGravity.git && cd SigmaGravity
pip install numpy scipy pandas matplotlib astropy
python scripts/run_regression.py
```

---

## Acknowledgments

We thank **Emmanuel N. Saridakis** (National Observatory of Athens) for detailed feedback on the theoretical framework, particularly regarding the derivation of field equations, the structure of Θ_μν, and consistency constraints in teleparallel gravity with non-minimal matter coupling.

We thank **Rafael Ferraro** (Instituto de Astronomía y Física del Espacio, CONICET – Universidad de Buenos Aires) for helpful discussions on f(T) gravity and the role of dimensional constants in modified teleparallel theories.

---

## References

- Lelli, F., McGaugh, S. S., & Schombert, J. M. 2016, AJ, 152, 157 (SPARC)
- Fox, C., Mahler, G., Sharon, K., & Remolina González, J. D. 2022, ApJ, 928, 87
- Eilers, A.-C., Hogg, D. W., Rix, H.-W., & Ness, M. K. 2019, ApJ, 871, 120
- Bevacqua, D., et al. 2022, MNRAS, 511, 139
- Zhu, L., et al. 2023, MNRAS, 522, 6326 (MaNGA DynPop)
- Milgrom, M. 1983, ApJ, 270, 365 (MOND)
- Bertotti, B., Iess, L., & Tortora, P. 2003, Nature, 425, 374 (Cassini)

---

## Legacy Documentation

For historical baseline results, exploratory analyses, and extended theoretical derivations not directly supporting the current paper's claims, see:

`archive/SUPPLEMENTARY_INFORMATION_LEGACY.md`

This includes:
- Earlier coherence scale formulations (ξ = ⅔R_d, ξ = ½R_d)
- CMB analysis (exploratory)
- Pantheon+ SNe validation
- Extended gate derivations
- Superstatistical coherence derivation details

---

*End of Supplementary Information*
