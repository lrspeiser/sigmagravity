# Bulge Kinematics Data Progress

## Completed ✓

1. **BRAVA Catalog**: 8,585 stars downloaded ✓
   - Location: `data/bulge_kinematics/BRAVA/brava_catalog.tbl`
   - Includes: positions, radial velocities, photometry, TiO band strengths

2. **GIBS Catalog**: 5,651 stars downloaded ✓
   - Location: `data/bulge_kinematics/GIBS/gibs_catalog_1.fits`
   - Includes: positions, radial velocities, metallicities

3. **GALACTICNUCLEUS**: ~10,000+ stars downloaded ✓
   - Location: `data/bulge_kinematics/GALACTICNUCLEUS/`
   - Multiple catalog files

4. **Cross-Match Script**: Created and tested ✓
   - Combined catalog: `data/bulge_kinematics/crossmatched/all_bulge_catalogs_combined.fits`
   - 14,236 total entries (BRAVA + GIBS)

5. **Gaia Cross-Match Script**: Created and tested ✓
   - Successfully cross-matched 3 test stars
   - Output: `data/bulge_kinematics/crossmatched/brava_with_gaia.fits`

## In Progress / Pending

### APOGEE Download
- **Status**: Automated query failed (known astroquery/SDSS issue)
- **Solution**: Manual download via CasJobs web interface
- **Steps**:
  1. Visit: https://skyserver.sdss.org/casjobs/
  2. Create account and log in
  3. Run SQL query (see `APOGEE/metadata.json`)
  4. Download results as FITS
  5. Save to: `data/bulge_kinematics/APOGEE/`

### Gaia Proper Motions (Full Catalog)
- **Status**: Script works but is very slow
- **Current Method**: One ADQL query per star (~1 minute per star)
- **Time Estimate**: ~140 hours for full BRAVA catalog (8,585 stars)
- **Recommendations**:
  1. **Use smaller sample** for initial analysis (e.g., 100-1000 stars)
  2. **Use pre-cross-matched catalogs** if available
  3. **Optimize query method** (bulk cross-match if possible)
  4. **Run in background** over multiple days

## Scripts Available

1. **`scripts/download_apogee_bulge.py`**
   - Downloads APOGEE bulge fields
   - Status: Needs manual CasJobs download

2. **`scripts/crossmatch_bulge_catalogs.py`**
   - Cross-matches BRAVA + APOGEE + GIBS
   - Status: ✓ Working (tested with BRAVA + GIBS)

3. **`scripts/add_gaia_proper_motions.py`**
   - Adds Gaia DR3 proper motions
   - Status: ✓ Working but slow
   - Usage: `python scripts/add_gaia_proper_motions.py --catalog BRAVA --max-stars 100`

## Recommended Next Steps

### Immediate (Quick Wins)
1. **Download APOGEE manually** via CasJobs (~30 minutes)
2. **Cross-match BRAVA + APOGEE** (~1 minute)
3. **Add Gaia to small sample** (100-500 stars, ~2-8 hours)

### Short-term (1-2 days)
4. **Create analysis script** for covariant coherence tests
5. **Apply quality cuts** (distance, proper motion errors)
6. **Compute 6D phase space** for sample stars

### Long-term (if needed)
7. **Full Gaia cross-match** (run in background, ~1 week)
8. **Combine all catalogs** into master catalog
9. **Full analysis** with complete dataset

## Current Data Summary

- **BRAVA**: 8,585 stars (radial velocities)
- **GIBS**: 5,651 stars (radial velocities + metallicities)
- **GALACTICNUCLEUS**: ~10,000+ stars (proper motions)
- **APOGEE**: Pending (chemistry + radial velocities)
- **Gaia DR3**: 3 stars tested (proper motions + parallaxes)

## Performance Notes

### Gaia Cross-Match Speed
- Current: ~1 minute per star
- For 100 stars: ~1.7 hours
- For 1,000 stars: ~17 hours
- For full BRAVA (8,585 stars): ~140 hours (~6 days)

### Optimization Options
1. **Batch queries**: Group multiple stars per query (if supported)
2. **Parallel processing**: Run multiple queries simultaneously
3. **Pre-filtered catalogs**: Use smaller, pre-filtered Gaia subsets
4. **Alternative sources**: Check for pre-cross-matched catalogs

## Files Created

```
data/bulge_kinematics/
├── BRAVA/
│   └── brava_catalog.tbl (8,585 stars) ✓
├── GIBS/
│   └── gibs_catalog_1.fits (5,651 stars) ✓
├── APOGEE/
│   └── (pending manual download)
├── crossmatched/
│   ├── all_bulge_catalogs_combined.fits (14,236 entries) ✓
│   └── brava_with_gaia.fits (3 stars, test) ✓
```

## Next Command to Run

For a practical sample size:
```bash
# Add Gaia to 100 BRAVA stars (~1.7 hours)
python scripts/add_gaia_proper_motions.py --catalog BRAVA --max-stars 100
```

Or for a quick test:
```bash
# Add Gaia to 10 BRAVA stars (~10 minutes)
python scripts/add_gaia_proper_motions.py --catalog BRAVA --max-stars 10
```

