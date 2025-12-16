# Bulge/Bar Kinematics Data Download Summary

## Download Date
December 15, 2024

## Successfully Downloaded Datasets

### 1. GIBS (GIRAFFE Inner Bulge Survey) ✓
- **Status**: Successfully downloaded
- **Files**: 
  - `data/bulge_kinematics/GIBS/gibs_catalog_0.fits` (27 stars, 20 KB)
  - `data/bulge_kinematics/GIBS/gibs_catalog_1.fits` (5,651 stars, 345 KB)
- **Total**: 5,678 stars
- **Source**: VizieR catalog J/A+A/562/A66
- **Coverage**: Fields at b = −1° and b = −2° (close to Galactic plane)
- **Paper**: Zoccali et al. 2014, A&A, 562, A66

### 2. GALACTICNUCLEUS Survey ✓
- **Status**: Successfully downloaded
- **Files**: 28 FITS catalog files in `data/bulge_kinematics/GALACTICNUCLEUS/`
- **Largest catalogs**:
  - `galacticnucleus_catalog_13.fits` (5,162 stars, 132 KB)
  - `galacticnucleus_catalog_18.fits` (735 stars, 101 KB)
  - `galacticnucleus_catalog_5.fits` (393 stars, 115 KB)
- **Source**: VizieR catalog J/A+A/620/A83
- **Coverage**: Central ~36′ × 16′ (~150 pc from Galactic Center)
- **Paper**: Nogueras-Lara et al. 2018, A&A, 620, A83

## Successfully Downloaded Datasets (continued)

### 3. BRAVA (Bulge Radial Velocity Assay) ✓
- **Status**: Successfully downloaded
- **Files**: 
  - `data/bulge_kinematics/BRAVA/brava_catalog.tbl` (8,585 stars, 3.1 MB)
- **Total**: 8,585 stars
- **Source**: IRSA Gator catalog `bravacat`
- **Coverage**: −10° < l < +10°, −4° < b < −8°
- **Paper**: Kunder et al. 2012, AJ, 143, 57
- **Download Date**: December 15, 2024
- **Columns**: l, b, ra, dec, JHK magnitudes, vhc (radial velocity), TiO, spectra URLs
- **Spectra**: Available at `https://irsa.ipac.caltech.edu/data/BRAVA/spectra/`
- **Download Script**: `scripts/download_brava_spectra.py` (for downloading FITS spectra)

## Datasets Requiring Manual Download

### 4. APOGEE (inner bulge fields)
- **Status**: Metadata created, query required
- **Location**: `data/bulge_kinematics/APOGEE/metadata.json`
- **Download Methods**:
  1. CasJobs web interface: https://skyserver.sdss.org/casjobs/
  2. astroquery (see metadata.json for SQL examples)
- **Coverage**: |l| < 10°, |b| < 10°
- **URL**: https://www.sdss.org/dr18/irspec/

### 5. VIRAC + Gaia (combined proper motions)
- **Status**: Metadata created, cross-match required
- **Location**: `data/bulge_kinematics/VIRAC_Gaia/metadata.json`
- **Description**: Requires cross-matching VIRAC and Gaia DR3 catalogs
- **Coverage**: Entire Galactic bulge region

### 6. MUSE kinematic maps (October 2025)
- **Status**: Metadata created, check A&A for publication
- **Location**: `data/bulge_kinematics/MUSE/metadata.json`
- **Description**: Maps from ~23,000 stars across 57 bulge fields
- **Source**: Check A&A October 2025 supplementary data

## Scripts Created

1. **Download Script**: `scripts/download_bulge_kinematics.py`
   - Automatically downloads available datasets
   - Creates metadata files for all datasets
   - Attempts VizieR queries for GIBS and GALACTICNUCLEUS

2. **Cross-Match Script**: `data/bulge_kinematics/crossmatch_brava_apogee_gaia.py`
   - Cross-matches BRAVA + APOGEE catalogs
   - Adds Gaia DR3 proper motions for full 6D phase space

## Recommended Next Steps

1. **BRAVA data downloaded** ✓:
   - Catalog available at `data/bulge_kinematics/BRAVA/brava_catalog.tbl`
   - To download spectra: `python scripts/download_brava_spectra.py [--all]`

2. **Query APOGEE bulge fields**:
   - Use CasJobs or astroquery
   - SQL query example in `APOGEE/metadata.json`
   - Save to `data/bulge_kinematics/APOGEE/`

3. **Cross-match datasets**:
   ```bash
   python data/bulge_kinematics/crossmatch_brava_apogee_gaia.py
   ```

4. **Add Gaia DR3 proper motions**:
   - Use astroquery.gaia to query proper motions
   - Cross-match with BRAVA/APOGEE catalogs

## Data Statistics

- **Total FITS files downloaded**: 30
- **Total stars in GIBS**: 5,678
- **Total stars in BRAVA**: 8,585
- **Total stars in GALACTICNUCLEUS**: ~10,000+ (across 28 catalogs)
- **Total data size**: ~5.6 MB (including BRAVA catalog)

## Directory Structure

```
data/bulge_kinematics/
├── BRAVA/
│   ├── metadata.json
│   ├── brava_catalog.tbl (8,585 stars, 3.1 MB)
│   └── spectra/ (for FITS spectra downloads)
├── GIBS/
│   ├── metadata.json
│   ├── gibs_catalog_0.fits (27 stars)
│   └── gibs_catalog_1.fits (5,651 stars)
├── APOGEE/
│   └── metadata.json
├── VIRAC_Gaia/
│   └── metadata.json
├── GALACTICNUCLEUS/
│   ├── metadata.json
│   └── galacticnucleus_catalog_*.fits (28 files)
├── MUSE/
│   └── metadata.json
└── crossmatch_brava_apogee_gaia.py
```

## References

- **BRAVA**: Kunder et al. 2012, AJ, 143, 57
- **GIBS**: Zoccali et al. 2014, A&A, 562, A66
- **GALACTICNUCLEUS**: Nogueras-Lara et al. 2018, A&A, 620, A83
- **APOGEE**: SDSS-IV, https://www.sdss.org/dr18/irspec/
- **VIRAC**: Smith et al. 2018, MNRAS, 480, 1460

