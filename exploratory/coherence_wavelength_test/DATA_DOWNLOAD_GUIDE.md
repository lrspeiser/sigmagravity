# Data Download Guide for Σ-Gravity Testing

## 1. KMOS³D High-Redshift Survey

### Data Release Page
**URL**: https://www.mpe.mpg.de/ir/KMOS3D/data

### Direct Download Links

#### Galaxy Catalog (FITS table with all parameters)
```bash
# Main catalog with physical parameters for 739 galaxies
wget https://www.mpe.mpg.de/resources/KMOS3D/catalogs/k3d_fnlsp_table_v3.fits.tgz
tar xvfz k3d_fnlsp_table_v3.fits.tgz
```

**STATUS: ✓ DOWNLOADED** to `data/kmos3d/`

#### All Data Cubes (739 galaxies)
The cubes are available as tarballs by field:

```bash
# All KMOS3D cubes (large download ~3GB total)
wget https://www.mpe.mpg.de/resources/KMOS3D/KMOS3D_cubes.tar.gz

# Or by field:
wget https://www.mpe.mpg.de/resources/KMOS3D/KMOS3D_cubes_COSMOS.tar.gz  # 275 targets, 1.1GB
wget https://www.mpe.mpg.de/resources/KMOS3D/KMOS3D_cubes_GOODSS.tar.gz  # 224 targets, 0.96GB
wget https://www.mpe.mpg.de/resources/KMOS3D/KMOS3D_cubes_UDS.tar.gz     # 240 targets, 0.95GB
```

### Key Papers
- Wisnioski et al. 2019, ApJ, 886, 124 (Data Release Paper)
- Genzel et al. 2020, ApJ, 902, 98 (Dark matter fractions)
- Nestor Shachar et al. 2023, ApJ, 944, 78 (RC100 - 100 rotation curves)

---

## 2. NGC 4550 Counter-Rotating Galaxy

### ATLAS³D Survey Data

**Main Data Page**: https://groups.physics.ox.ac.uk/atlas3d/

NGC 4550 is included in the ATLAS³D sample of 260 early-type galaxies.

### Published NGC 4550 Data

#### Coccato et al. 2013 (A&A)
**Paper**: "Spectroscopic evidence of distinct stellar populations in the counter-rotating stellar disks of NGC 3593 and NGC 4550"
**arXiv**: https://arxiv.org/abs/1210.7807

Data from VIMOS/VLT integral-field spectroscopy:
- ESO Program: 087.B-0853A
- Contact ESO archive for raw data

#### Johnston et al. 2013 (MNRAS)  
**Paper**: "Disentangling the stellar populations in the counter-rotating disc galaxy NGC 4550"
**arXiv**: https://arxiv.org/abs/1210.0535

Data from Gemini/GMOS long-slit spectroscopy.

### ESO Archive Query for NGC 4550
```
https://archive.eso.org/wdb/wdb/eso/sched_rep_arc/query?target=NGC+4550
```

### Gemini Archive Query for NGC 4550
```
https://archive.gemini.edu/searchform/NGC4550
```

---

## 3. RC100 High-z Rotation Curve Data

### Paper
Nestor Shachar et al. 2023, ApJ, 944, 78
**arXiv**: https://arxiv.org/abs/2209.12199

### Key Data from the Paper

| Redshift Bin | f_DM(Re) | Uncertainty |
|--------------|----------|-------------|
| z ~ 1        | 0.38     | ± 0.23      |
| z ~ 2        | 0.27     | ± 0.18      |

---

## 4. Local Comparison: SPARC Database

**STATUS: ✓ HAVE DATA** in `data/Rotmod_LTG/`

### Additional SPARC Resources
**Main page**: http://astroweb.cwru.edu/SPARC/
**Paper**: Lelli, McGaugh & Schombert 2016, AJ, 152, 157

---

## 5. Other Counter-Rotating Galaxies

### NGC 3593
- Also observed by Coccato et al. 2013
- Counter-rotating disks, younger secondary disk
- VIMOS/VLT data available

### NGC 4138  
- Counter-rotating stellar disks
- Has associated star-forming ring
- Papers: Coccato et al. 2011

### NGC 7217
- Another counter-rotating candidate
- Multiple kinematic components

---

## Summary of Data Status

| Dataset | Status | Location |
|---------|--------|----------|
| SPARC (175 galaxies) | ✓ Have | `data/Rotmod_LTG/` |
| Fox+ 2022 clusters | ✓ Have | `data/clusters/` |
| Gaia DR3 MW | ✓ Have | `vendor/maxdepth_gaia/` |
| KMOS³D catalog | ✓ Downloaded | `data/kmos3d/` |
| NGC 4550 kinematics | ✗ Need | ESO/Gemini archives |
| RC100 rotation curves | ✗ Need | arXiv supplementary |

---

## Analysis Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `run_available_tests.py` | Run all tests with existing data | ✓ Done |
| `analyze_kmos3d_highz.py` | High-z test with KMOS³D | ✓ Done |
| `test_predictions_data_inventory.py` | Inventory available data | ✓ Done |

