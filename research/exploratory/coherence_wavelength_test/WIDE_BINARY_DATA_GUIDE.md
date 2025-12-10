# Wide Binary Data Sources for Σ-Gravity Testing

**Date:** December 2025  
**Purpose:** Download and analyze wide binary data to test low-g Solar System predictions

---

## Executive Summary

Based on our computations (`low_g_solar_system_predictions.py`), Σ-Gravity makes different predictions depending on how the External Field Effect (EFE) is handled:

| Scenario | Prediction at 10,000 AU | Testable? |
|----------|------------------------|-----------|
| No EFE, W=1 | +54% velocity boost | YES |
| With EFE, W=1 | +12% velocity boost | YES |
| W=0 (no coherence for binaries) | No boost | Consistent with null |

**Key separation:** ~7,900 AU where g_internal = g†

---

## Data Sources

### 1. El-Badry et al. (2021) Wide Binary Catalog

**Paper:** "A million binaries from Gaia eDR3: sample selection and validation of Gaia parallax uncertainties"  
**ADS:** https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.2269E  
**arXiv:** https://arxiv.org/abs/2101.05282

**Data Download:**
```bash
# Zenodo repository (1.3 million binaries)
wget https://zenodo.org/record/4435257/files/gaia_edr3_1p3M_binaries.fits.gz

# Or from CDS VizieR
# https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/MNRAS/506/2269
```

**Key columns:**
- `separation`: Projected separation in arcsec
- `parallax`: For converting to AU
- `pmra_1`, `pmdec_1`, `pmra_2`, `pmdec_2`: Proper motions
- `phot_g_mean_mag_1`, `phot_g_mean_mag_2`: For mass estimation

**Selection for MOND regime:**
```python
# Select wide binaries with separation > 2000 AU
sep_AU = separation_arcsec * 1000 / parallax_mas
wide_binaries = catalog[sep_AU > 2000]
```

---

### 2. Chae (2023) Analysis Data

**Paper:** "Breakdown of the Newton-Einstein Standard Gravity at Low Acceleration in Internal Dynamics of Wide Binary Stars"  
**Journal:** ApJ 952, 128 (2023)  
**arXiv:** https://arxiv.org/abs/2305.04613

**Claims:** Detection of MOND-like boost at separations > 2000 AU

**Data:**
- Supplementary material available from ApJ
- Key figure: Figure 4 showing velocity anomaly vs separation
- Contact author for full data tables if needed

**Key result to compare:**
| Separation (AU) | Claimed boost |
|-----------------|---------------|
| 2000-3000 | ~5-10% |
| 5000-10000 | ~15-20% |
| >10000 | ~20-30% |

---

### 3. Banik et al. (2024) Reanalysis

**Paper:** "Strong constraints on the gravitational law from Gaia DR3 wide binaries"  
**Journal:** MNRAS 527, 4573 (2024)  
**arXiv:** https://arxiv.org/abs/2311.03436

**Claims:** No significant deviation from Newton (19σ preference for Newton over MOND)

**Data:**
- Uses same Gaia DR3 source data
- Different selection criteria and analysis methodology
- Emphasizes importance of quality cuts and contamination removal

**Key methodological differences from Chae:**
1. More stringent quality cuts
2. Different treatment of unresolved triples
3. Different statistical framework

---

### 4. Pittordis & Sutherland (2023)

**Paper:** "Testing modified gravity with wide binaries in Gaia DR3"  
**Journal:** Open Journal of Astrophysics, 6, 4 (2023)  
**arXiv:** https://arxiv.org/abs/2205.02846

**Data availability:** Methodology and selection criteria documented

---

## Download Script

```bash
#!/bin/bash
# Download wide binary data for Σ-Gravity testing

mkdir -p data/wide_binaries
cd data/wide_binaries

# El-Badry catalog (primary source)
echo "Downloading El-Badry et al. (2021) catalog..."
wget -c https://zenodo.org/record/4435257/files/gaia_edr3_1p3M_binaries.fits.gz
gunzip -k gaia_edr3_1p3M_binaries.fits.gz

# Alternative: Download from CDS if Zenodo is slow
# wget "https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/506/2269/table1.dat.gz"

echo "Download complete!"
echo "File: gaia_edr3_1p3M_binaries.fits"
```

---

## Analysis Pipeline

### Step 1: Load and Filter Data

```python
from astropy.io import fits
import numpy as np

# Load catalog
with fits.open('data/wide_binaries/gaia_edr3_1p3M_binaries.fits') as hdul:
    data = hdul[1].data

# Calculate physical separation
parallax_mas = data['parallax']
separation_arcsec = data['separation']
sep_AU = separation_arcsec * 1000 / parallax_mas

# Select quality sample
quality_mask = (
    (data['parallax_over_error'] > 20) &  # Good parallax
    (data['ruwe_1'] < 1.4) &               # Not unresolved binary
    (data['ruwe_2'] < 1.4) &
    (sep_AU > 1000) &                       # Wide enough
    (sep_AU < 50000)                        # Not too wide (contamination)
)

wide_sample = data[quality_mask]
print(f"Selected {len(wide_sample)} wide binaries")
```

### Step 2: Calculate Relative Velocities

```python
# Proper motion difference (mas/yr)
dpm_ra = data['pmra_1'] - data['pmra_2']
dpm_dec = data['pmdec_1'] - data['pmdec_2']

# Convert to velocity (km/s)
# v = 4.74 * pm (mas/yr) * d (pc)
d_pc = 1000 / parallax_mas
v_rel_tangential = 4.74 * np.sqrt(dpm_ra**2 + dpm_dec**2) * d_pc

# Expected Keplerian velocity
# v_Kep = sqrt(G * M_total / r)
G = 6.674e-11
M_sun = 1.989e30
AU_to_m = 1.496e11

# Estimate total mass from photometry (rough)
M_total = 2 * M_sun  # Assume 2 solar masses

v_Kep = np.sqrt(G * M_total / (sep_AU * AU_to_m)) / 1000  # km/s
```

### Step 3: Compare to Σ-Gravity Predictions

```python
# From low_g_solar_system_predictions.py
g_dagger = 9.60e-11  # m/s²
g_MW = 2.20e-10      # m/s² (MW field at Sun)

g_internal = G * M_total / (sep_AU * AU_to_m)**2

# Σ-Gravity with EFE
h_efe = np.sqrt(g_dagger / (g_internal + g_MW)) * g_dagger / (g_dagger + g_internal + g_MW)
Sigma_efe = 1 + np.sqrt(3) * h_efe
v_sigma_efe = v_Kep * np.sqrt(Sigma_efe)

# Compare
v_anomaly = v_rel_tangential / v_Kep
```

---

## Expected Results

### If Σ-Gravity with EFE is correct:
- ~10-15% velocity boost at 10,000 AU
- Smooth transition, not sharp threshold
- Should match Chae's direction but smaller magnitude

### If W=0 for binaries (no coherence):
- No velocity anomaly at any separation
- Consistent with Banik et al. null result

### If no EFE (pure Σ-Gravity):
- ~50% velocity boost at 10,000 AU
- Would strongly support Chae's detection

---

## Key Questions to Answer

1. **What is the observed velocity anomaly vs separation?**
   - Reproduce both Chae and Banik analyses
   - Understand methodological differences

2. **Does Σ-Gravity with EFE fit the data?**
   - Compare our predicted curve to observations
   - Test if EFE strength matches MW field

3. **Is W=0 for binaries justified?**
   - Theoretical argument: no extended coherent mass
   - Empirical test: does null result prefer W=0?

---

## References

1. El-Badry, K., et al. (2021), MNRAS, 506, 2269
2. Chae, K.-H. (2023), ApJ, 952, 128
3. Banik, I., et al. (2024), MNRAS, 527, 4573
4. Pittordis, C., & Sutherland, W. (2023), OJAp, 6, 4

---

## Status

- [x] Computed Σ-Gravity predictions
- [ ] Download El-Badry catalog
- [ ] Reproduce Chae analysis
- [ ] Reproduce Banik analysis  
- [ ] Compare to Σ-Gravity predictions
- [ ] Write up results for README


