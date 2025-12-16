# Next Steps: Bulge/Bar Kinematics Analysis

## Current Status ✓

### Downloaded Datasets
1. **BRAVA**: 8,585 stars (catalog + spectra available)
2. **GIBS**: 5,678 stars (from VizieR)
3. **GALACTICNUCLEUS**: ~10,000+ stars (from VizieR)

### Pending Datasets
4. **APOGEE**: Need to query (metadata ready)
5. **Gaia DR3**: Need to cross-match for proper motions
6. **VIRAC + Gaia**: Need cross-match
7. **MUSE**: Check A&A October 2025

## Immediate Next Steps

### Step 1: Download APOGEE Bulge Fields

```bash
# Download APOGEE DR18 bulge fields (recommended)
python scripts/download_apogee_bulge.py --max-stars 50000 --dr 18

# Or download DR17 if DR18 has issues
python scripts/download_apogee_bulge.py --max-stars 50000 --dr 17
```

**Expected output**: `data/bulge_kinematics/APOGEE/apogee_bulge_dr18.fits`

**Alternative**: If automated query fails, use CasJobs web interface:
1. Visit: https://skyserver.sdss.org/casjobs/
2. Create account and log in
3. Run SQL query (see `APOGEE/metadata.json`)
4. Download results as FITS
5. Save to `data/bulge_kinematics/APOGEE/`

### Step 2: Cross-Match Catalogs

```bash
# Cross-match BRAVA + APOGEE (recommended starting point)
python scripts/crossmatch_bulge_catalogs.py
```

**Output**: `data/bulge_kinematics/crossmatched/brava_apogee_crossmatch.fits`

This creates a combined catalog with:
- BRAVA radial velocities
- APOGEE chemistry ([Fe/H], [α/Fe])
- APOGEE stellar parameters (Teff, log g)

### Step 3: Add Gaia DR3 Proper Motions

```bash
# Add Gaia proper motions to BRAVA
python scripts/add_gaia_proper_motions.py --catalog BRAVA

# Add Gaia proper motions to APOGEE
python scripts/add_gaia_proper_motions.py --all

# Or process all catalogs at once
python scripts/add_gaia_proper_motions.py --all
```

**Output**: `data/bulge_kinematics/crossmatched/*_with_gaia.fits`

This adds:
- Proper motions (pmra, pmdec)
- Parallaxes (for distances)
- Additional Gaia photometry and quality flags

**Note**: This may take 10-30 minutes depending on catalog size.

### Step 4: Create Final Combined Catalog

After Steps 1-3, you'll have:
- `brava_with_gaia.fits` - BRAVA + Gaia (6D phase space)
- `apogee_bulge_dr18_with_gaia.fits` - APOGEE + Gaia (6D + chemistry)
- `brava_apogee_crossmatch.fits` - Combined BRAVA + APOGEE

You can then:
1. Cross-match the Gaia-enhanced catalogs
2. Create a master catalog with all available data
3. Apply quality cuts (distance, proper motion errors, etc.)

## Analysis Workflow

### Phase 1: Data Preparation
- [x] Download BRAVA
- [x] Download GIBS
- [x] Download GALACTICNUCLEUS
- [ ] Download APOGEE ← **NEXT**
- [ ] Cross-match BRAVA + APOGEE
- [ ] Add Gaia DR3 proper motions
- [ ] Create master catalog

### Phase 2: Quality Cuts
- Distance estimates (from parallax or photometric)
- Proper motion quality (ruwe < 1.4, visibility_periods >= 8)
- Radial velocity quality (SNR, error cuts)
- Spatial cuts (R < 3 kpc from Galactic Center)

### Phase 3: Covariant Coherence Analysis
- Compute 6D phase space coordinates
- Calculate velocity dispersions
- Test covariant coherence predictions
- Compare with theoretical expectations

## Scripts Available

1. **`scripts/download_bulge_kinematics.py`**
   - Main download script (already run)
   - Can be re-run to update metadata

2. **`scripts/download_brava_spectra.py`**
   - Download BRAVA FITS spectra
   - Usage: `python scripts/download_brava_spectra.py [--all]`

3. **`scripts/download_apogee_bulge.py`** ← **NEW**
   - Download APOGEE bulge fields
   - Usage: `python scripts/download_apogee_bulge.py [--max-stars N] [--dr 18]`

4. **`scripts/crossmatch_bulge_catalogs.py`** ← **NEW**
   - Cross-match multiple catalogs
   - Usage: `python scripts/crossmatch_bulge_catalogs.py`

5. **`scripts/add_gaia_proper_motions.py`** ← **NEW**
   - Add Gaia DR3 proper motions
   - Usage: `python scripts/add_gaia_proper_motions.py --catalog BRAVA`

## Data Structure

```
data/bulge_kinematics/
├── BRAVA/
│   ├── brava_catalog.tbl (8,585 stars) ✓
│   └── spectra/ (FITS spectra)
├── GIBS/
│   └── gibs_catalog_1.fits (5,651 stars) ✓
├── APOGEE/
│   └── apogee_bulge_dr18.fits (pending) ← NEXT
├── GALACTICNUCLEUS/
│   └── galacticnucleus_catalog_*.fits (28 files) ✓
└── crossmatched/
    ├── brava_apogee_crossmatch.fits (after Step 2)
    ├── brava_with_gaia.fits (after Step 3)
    └── apogee_bulge_dr18_with_gaia.fits (after Step 3)
```

## Recommended Order

1. **Download APOGEE** (Step 1) - ~5-10 minutes
2. **Cross-match BRAVA + APOGEE** (Step 2) - ~1 minute
3. **Add Gaia proper motions** (Step 3) - ~10-30 minutes
4. **Create analysis scripts** for covariant coherence tests

## Troubleshooting

### APOGEE query fails
- Try CasJobs web interface instead
- Check network connection
- May need SDSS account authentication

### Gaia cross-match slow
- Use `--max-stars` flag to test with subset first
- Gaia queries are rate-limited, be patient
- Consider querying in smaller batches

### No matches found
- Check coordinate systems (ICRS vs Galactic)
- Increase `--max-sep` (default 2 arcsec)
- Verify catalogs have valid ra/dec columns

## References

- **BRAVA**: Kunder et al. 2012, AJ, 143, 57
- **APOGEE**: SDSS-IV, https://www.sdss.org/dr18/irspec/
- **GIBS**: Zoccali et al. 2014, A&A, 562, A66
- **Gaia DR3**: Gaia Collaboration 2023, A&A, 674, A1

