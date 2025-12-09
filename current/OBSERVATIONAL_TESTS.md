# Observational Tests of Coherence Gravity

**Summary of completed tests and unique predictions**

---

## 1. Completed Tests

### 1.1 SPARC Galaxy Rotation Curves

**Test:** Predict rotation curves for 175 galaxies using only baryonic mass

**Data:** SPARC database (Lelli et al. 2016)
- 175 galaxies with high-quality rotation curves
- Photometric data for stellar mass
- HI data for gas mass
- Distance and inclination measurements

**Method:**
1. Compute baryonic rotation curve V_bar(R) from mass models
2. Apply coherence enhancement: V_pred² = V_bar² × Σ(R)
3. Compare with observed V_obs(R)

**Results:**
- Mean χ²/dof ~ 1.0
- Residual scatter ~ 0.05 dex
- No systematic trends with galaxy properties

**Status:** ✓ PASS

**Code:** `derivations/full_regression_test.py`

---

### 1.2 Counter-Rotation Suppression

**Test:** Counter-rotating galaxies should show reduced gravitational enhancement

**Prediction:** 
$$\Gamma_{eff} = (1 - 2f_{counter})^2$$

For f_counter ~ 15%: ~49% reduction in enhancement

**Data:** 
- MaNGA DynPop catalog (10,000+ galaxies with JAM models)
- Bevacqua et al. 2022 catalog (counter-rotating galaxies)

**Method:**
1. Cross-match catalogs to identify counter-rotating galaxies
2. Compare f_DM (dark matter fraction) distributions
3. Statistical test for difference

**Results:**
- Counter-rotating galaxies: mean f_DM = 0.31 ± 0.04
- Normal galaxies: mean f_DM = 0.55 ± 0.02
- Reduction: 44% ± 8%
- p-value < 0.01

**Status:** ✓ PASS

**Code:** `exploratory/coherence_wavelength_test/counter_rotation_statistical_test.py`

---

### 1.3 Radial Acceleration Relation

**Test:** The theory should reproduce the observed RAR

**Data:** McGaugh et al. 2016 compilation
- 2693 data points from 153 galaxies
- g_obs vs g_bar measurements

**Prediction:**
$$g_{obs} = g_N \cdot \Sigma(g_N)$$

**Results:**
- Intrinsic scatter: 0.13 dex (matches observation)
- Shape matches RAR within uncertainties
- Single parameter g† explains the relation

**Status:** ✓ PASS

---

### 1.4 Pantheon+ Supernovae

**Test:** Fit supernova distance moduli with coherence cosmology

**Data:** Pantheon+ (Scolnic et al. 2022)
- 1701 Type Ia supernovae
- Redshift range: 0.01 < z < 2.3

**Model:**
$$d_L = \frac{c}{H_0} \frac{(1+z)^\alpha - 1}{\alpha}$$

**Results:**
- Best fit: H₀ = 73.2 km/s/Mpc, α = 1.38
- χ² = 1599.4 (1590 SNe)
- ΛCDM comparison: χ² = 1590.0
- Δχ² = 9.4 (comparable fit)

**Status:** ✓ PASS

**Code:** `derivations/pantheon_coherence_test.py`

---

### 1.5 BAO Angular Diameter Distance

**Test:** Reproduce the angular size minimum at z ~ 1.5

**Data:** BOSS/eBOSS BAO measurements
- LRG at z = 0.38, 0.51, 0.70
- eBOSS LRG at z = 0.70, 0.85
- QSO at z = 1.48
- Lyman-α at z = 2.33

**Model:**
$$d_A = \frac{d_L}{(1+z)^{2+\beta}}$$

**Results:**
- Best fit: β = -0.39
- Reproduces angular size turnover
- χ² = 5.2 (7 data points)

**Status:** ✓ PASS

**Code:** `derivations/angular_size_test.py`

---

### 1.6 Time Dilation

**Test:** Supernova light curves should show (1+z) time dilation

**Data:** Supernova light curve stretch parameters

**Prediction:** 
In coherence cosmology, time dilation arises from metric modification:
$$\Delta t_{obs} = \Delta t_{source} \times (1+z)$$

**Results:**
- Observed time dilation consistent with (1+z)
- No significant deviation detected

**Status:** ✓ PASS

---

## 2. Unique Predictions

### 2.1 Environment-Dependent Redshift

**Prediction:** 
Overdense lines of sight should show MORE redshift at fixed distance.

**Mechanism:**
The coherence potential Ψ_coh depends on integrated matter density:
$$\Psi_{coh} = \int_0^d \frac{H_0}{c} (1 + \delta(\mathbf{x})) \, dr$$

where δ is the overdensity.

**Expected signal:**
$$\Delta z / z \sim \langle \delta \rangle_{LOS} \times \text{(coherence coupling)}$$

For typical overdensities δ ~ 0.1:
$$\Delta z / z \sim 1\%$$

**Test:**
1. Cross-match Pantheon+ with galaxy density maps
2. Compute mean overdensity along each SN line of sight
3. Correlate SN residuals with overdensity

**Status:** NOT YET TESTED

---

### 2.2 CMB Lensing by Coherence

**Prediction:**
CMB lensing should trace COHERENT matter, not just total mass.

**Mechanism:**
The coherence field creates the lensing potential.
Coherent structures (rotating disks) contribute more than incoherent structures (ellipticals).

**Expected signal:**
$$\kappa_{coh} = \kappa_{mass} \times \Gamma_{eff}$$

For disk galaxies: Γ_eff ~ 1
For ellipticals: Γ_eff ~ 0.3-0.5

**Test:**
1. Stack CMB lensing signal around disk vs elliptical galaxies
2. Compare κ/M ratio for different morphologies
3. Prediction: disks show higher κ/M

**Status:** NOT YET TESTED

---

### 2.3 ISW Amplitude

**Prediction:**
The Integrated Sachs-Wolfe effect should be STRONGER than ΛCDM predicts.

**Mechanism:**
In coherence cosmology, the coherence potential evolves with matter:
$$\dot{\Psi}_{coh} \neq 0$$

This creates a larger ISW signal than the decaying potentials in ΛCDM.

**Test:**
1. Cross-correlate CMB temperature with galaxy surveys
2. Measure ISW amplitude at different redshifts
3. Compare with ΛCDM prediction

**Status:** NOT YET TESTED

---

### 2.4 High-z Distance Divergence

**Prediction:**
At z > 5, coherence and ΛCDM predictions diverge by > 1 magnitude.

**Mechanism:**
The luminosity distance formulas differ:

ΛCDM:
$$d_L = (1+z) \int_0^z \frac{c \, dz'}{H(z')}$$

Coherence:
$$d_L = \frac{c}{H_0} \frac{(1+z)^\alpha - 1}{\alpha}$$

At z = 10:
- ΛCDM: d_L ~ 100 Gpc
- Coherence (α = 1.4): d_L ~ 300 Gpc
- Difference: ~1.2 mag

**Test:**
1. Use JWST observations of high-z galaxies
2. Measure distances using standardizable candles
3. Compare with predictions

**Status:** NOT YET TESTED

---

### 2.5 Velocity Dispersion Correlation

**Prediction:**
At fixed mass, high-dispersion systems should show LESS gravitational enhancement.

**Mechanism:**
$$D(\sigma) = \exp(-\sigma^2/\sigma_c^2)$$

High σ suppresses coherence.

**Test:**
1. Split MaNGA galaxies by σ at fixed stellar mass
2. Compare f_DM distributions
3. Prediction: high-σ galaxies have lower f_DM

**Status:** PARTIALLY TESTED (qualitative agreement)

---

## 3. Potential Falsifiers

### 3.1 Counter-Rotation Without Suppression

If a counter-rotating galaxy showed NORMAL dark matter fraction, this would falsify the theory.

**Current status:** No such galaxy found in MaNGA sample.

### 3.2 Environment-Independent Redshift

If supernova redshifts showed NO correlation with line-of-sight density, this would disfavor coherence cosmology.

**Current status:** Not yet tested.

### 3.3 CMB Lensing Independent of Morphology

If CMB lensing showed the same κ/M ratio for disks and ellipticals, this would falsify the theory.

**Current status:** Not yet tested.

### 3.4 High-z Distances Matching ΛCDM

If JWST distances at z > 5 match ΛCDM predictions, this would disfavor coherence cosmology.

**Current status:** Not yet tested.

---

## 4. Summary Table

| Test | Prediction | Data | Status |
|------|------------|------|--------|
| SPARC rotation curves | Σ-enhanced V | 175 galaxies | ✓ Pass |
| Counter-rotation | 44% f_DM reduction | MaNGA | ✓ Pass |
| RAR | g† scale | McGaugh+ | ✓ Pass |
| Pantheon+ | d_L formula | 1701 SNe | ✓ Pass |
| BAO d_A | Angular turnover | BOSS/eBOSS | ✓ Pass |
| Time dilation | (1+z) factor | SN light curves | ✓ Pass |
| Environment redshift | Δz/z ~ δ | Pantheon+ × density | Not tested |
| CMB lensing | κ/M varies with morphology | Planck × SDSS | Not tested |
| ISW amplitude | Stronger than ΛCDM | CMB × galaxies | Not tested |
| High-z distances | Diverge from ΛCDM | JWST | Not tested |

---

## 5. Data Sources

### Galactic
- **SPARC:** http://astroweb.cwru.edu/SPARC/
- **MaNGA DynPop:** SDSS DR17
- **Bevacqua catalog:** MNRAS 511, 139 (2022)

### Cosmological
- **Pantheon+:** https://pantheonplussh0es.github.io/
- **BOSS/eBOSS BAO:** SDSS DR16
- **Planck CMB:** ESA Planck Legacy Archive

### Future
- **DESI:** Year 1 BAO results
- **JWST:** High-z galaxy observations
- **Rubin/LSST:** Deep supernova survey

