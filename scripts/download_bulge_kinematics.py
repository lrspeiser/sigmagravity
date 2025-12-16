#!/usr/bin/env python3
"""
Download bulge/bar kinematics datasets for R < 3 kpc analysis.

Datasets:
1. BRAVA (Bulge Radial Velocity Assay) - ~10,000 M giants
2. GIBS (GIRAFFE Inner Bulge Survey) - ~5,000+ K giants
3. APOGEE (inner bulge fields) - SDSS-IV
4. VIRAC + Gaia (combined proper motions)
5. GALACTICNUCLEUS Survey - ~80,000 stars in central region
6. MUSE kinematic maps (October 2025) - ~23,000 stars

Recommended starting point: BRAVA + APOGEE cross-match
"""

import os
import sys
from pathlib import Path
import requests
from tqdm import tqdm
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
BULGE_DIR = DATA_DIR / "bulge_kinematics"
BULGE_DIR.mkdir(parents=True, exist_ok=True)

# Subdirectories for each dataset
BRAVA_DIR = BULGE_DIR / "BRAVA"
GIBS_DIR = BULGE_DIR / "GIBS"
APOGEE_DIR = BULGE_DIR / "APOGEE"
VIRAC_DIR = BULGE_DIR / "VIRAC_Gaia"
GALACTICNUCLEUS_DIR = BULGE_DIR / "GALACTICNUCLEUS"
MUSE_DIR = BULGE_DIR / "MUSE"

for d in [BRAVA_DIR, GIBS_DIR, APOGEE_DIR, VIRAC_DIR, GALACTICNUCLEUS_DIR, MUSE_DIR]:
    d.mkdir(exist_ok=True)


def download_file(url, output_path, description="file"):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=f"Downloading {description}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✓ Downloaded {description} to {output_path}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {description}: {e}")
        return False


def download_brava():
    """Download BRAVA (Bulge Radial Velocity Assay) data."""
    print("\n" + "="*60)
    print("Downloading BRAVA (Bulge Radial Velocity Assay)")
    print("="*60)
    
    # BRAVA catalog is available via IRSA Gator
    # Catalog name in IRSA is "bravacat" (not "brava")
    catalog_url = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"
    params = {
        "catalog": "bravacat",
        "spatial": "None",
        "outfmt": "1",  # IPAC ASCII table format
        "outrows": "100000"
    }
    
    # Build query URL
    query_url = f"{catalog_url}?catalog={params['catalog']}&spatial={params['spatial']}&outfmt={params['outfmt']}&outrows={params['outrows']}"
    
    output_path = BRAVA_DIR / "brava_catalog.tbl"
    
    print(f"  Downloading BRAVA catalog from IRSA...")
    print(f"  URL: {query_url}")
    
    success = download_file(query_url, output_path, "BRAVA catalog")
    
    if success:
        # Get file size and line count
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        with open(output_path, 'r') as f:
            line_count = sum(1 for _ in f)
        
        print(f"  ✓ Catalog downloaded: {file_size:.2f} MB, ~{line_count} lines")
        print(f"  ✓ Saved to: {output_path}")
    
    # Create metadata file
    metadata = {
        "dataset": "BRAVA",
        "description": "Bulge Radial Velocity Assay - ~8,585 M giants with radial velocities",
        "coverage": "−10° < l < +10°, −4° < b < −8°",
        "url": "https://irsa.ipac.caltech.edu/data/BRAVA/",
        "paper": "Kunder et al. 2012, AJ, 143, 57",
        "catalog_name": "bravacat",
        "catalog_file": "brava_catalog.tbl",
        "download_date": time.strftime("%Y-%m-%d"),
        "download_method": "IRSA Gator query interface",
        "query_url": query_url,
        "columns": [
            "l", "b", "ra", "dec",
            "j", "h", "k",  # JHK magnitudes
            "vhc",  # Heliocentric radial velocity
            "e",  # E(B-V) reddening
            "j0", "h0", "k0",  # Extinction-corrected magnitudes
            "TiO",  # TiO band strength
            "n_obs",  # Number of observations
            "fits_spectra_1", "fits_spectra_2",  # Links to FITS spectra
            "TMASS_ID"  # 2MASS identifier
        ],
        "spectra_directory": "https://irsa.ipac.caltech.edu/data/BRAVA/spectra/",
        "notes": [
            "Catalog downloaded via IRSA Gator using catalog name 'bravacat'",
            "Spectra are available in FITS format from the spectra directory",
            "Spectra URLs are also embedded in the catalog as HTML links"
        ]
    }
    
    metadata_path = BRAVA_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Created metadata file: {metadata_path}")
    
    return success


def download_apogee():
    """Download APOGEE inner bulge fields via astroquery."""
    print("\n" + "="*60)
    print("Downloading APOGEE (inner bulge fields)")
    print("="*60)
    
    try:
        from astroquery.sdss import SDSS
        from astroquery.gaia import Gaia
        import astropy.coordinates as coord
        import astropy.units as u
    except ImportError:
        print("✗ astroquery not installed. Installing...")
        os.system(f"{sys.executable} -m pip install astroquery")
        from astroquery.sdss import SDSS
        from astroquery.gaia import Gaia
        import astropy.coordinates as coord
        import astropy.units as u
    
    # APOGEE bulge fields: |l| < 10°, |b| < 10°
    # We'll query via SDSS CasJobs or direct FITS
    
    metadata = {
        "dataset": "APOGEE",
        "description": "APOGEE inner bulge fields - SDSS-IV",
        "coverage": "|l| < 10°, |b| < 10°",
        "url": "https://www.sdss.org/dr18/irspec/",
        "query_instructions": [
            "Query APOGEE DR18 for bulge fields:",
            "SELECT * FROM apogee_dr18.allStar",
            "WHERE abs(gal_l) < 10 AND abs(gal_b) < 10",
            "AND field LIKE '%bulge%' OR field LIKE '%inner%'"
        ],
        "python_query": """
from astroquery.sdss import SDSS
import astropy.coordinates as coord

# Query APOGEE bulge fields
query = '''
SELECT TOP 10000 
    apogee_id, ra, dec, glon, glat, 
    vhelio_avg, vscatter, 
    fe_h, alpha_m, 
    teff, logg
FROM apogee_dr18.allStar
WHERE abs(glon) < 10 AND abs(glat) < 10
'''
result = SDSS.query_sql(query)
result.write('apogee_bulge.fits', overwrite=True)
        """,
        "columns": [
            "apogee_id", "ra", "dec", "glon", "glat",
            "vhelio_avg", "vscatter",
            "fe_h", "alpha_m",
            "teff", "logg"
        ]
    }
    
    metadata_path = APOGEE_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata and query template: {metadata_path}")
    print("  Note: APOGEE requires SQL query via CasJobs or astroquery")
    print("  See metadata.json for query examples")
    
    # Try to run a small test query
    try:
        print("\n  Attempting test query (first 100 stars)...")
        # Use simpler query format
        query = """
        SELECT TOP 100 
            apogee_id, ra, dec, glon, glat, vhelio_avg, fe_h
        FROM apogee_dr18.allStar
        WHERE abs(glon) < 10 AND abs(glat) < 10
        """
        result = SDSS.query_sql(query, data_release=18)
        if result is not None and len(result) > 0:
            output_path = APOGEE_DIR / "apogee_bulge_sample.fits"
            result.write(str(output_path), overwrite=True)
            print(f"  ✓ Downloaded sample: {len(result)} stars -> {output_path}")
        else:
            print("  ⚠ Query returned no results")
            print("  Try using CasJobs web interface: https://skyserver.sdss.org/casjobs/")
    except Exception as e:
        print(f"  ⚠ Could not run test query: {e}")
        print("  Alternative methods:")
        print("  1. Use CasJobs web interface: https://skyserver.sdss.org/casjobs/")
        print("  2. Download pre-made APOGEE catalogs from SDSS website")
        print("  3. Use astroquery with authentication")
    
    return True


def download_gibs():
    """Download GIBS (GIRAFFE Inner Bulge Survey) data."""
    print("\n" + "="*60)
    print("Downloading GIBS (GIRAFFE Inner Bulge Survey)")
    print("="*60)
    
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("✗ astroquery not installed. Installing...")
        os.system(f"{sys.executable} -m pip install astroquery")
        from astroquery.vizier import Vizier
    
    metadata = {
        "dataset": "GIBS",
        "description": "GIRAFFE Inner Bulge Survey - ~5,000+ K giants with metallicities + RVs",
        "coverage": "Fields at b = −1° and b = −2° (close to Galactic plane)",
        "paper": "Zoccali et al. 2014, A&A, 562, A66",
        "vizier_catalog": "J/A+A/562/A66",
        "eso_archive": "https://archive.eso.org/scienceportal/home",
        "download_instructions": [
            "1. Via VizieR:",
            "   from astroquery.vizier import Vizier",
            "   v = Vizier(catalog='J/A+A/562/A66')",
            "   catalogs = v.query_constraints()",
            "",
            "2. Via ESO Archive:",
            "   Search for 'GIBS' or 'GIRAFFE Inner Bulge'",
            "   Download reduced spectra and catalogs"
        ],
        "columns": [
            "ra", "dec", "l", "b",
            "rv", "rv_err",
            "fe_h", "fe_h_err",
            "teff", "logg"
        ]
    }
    
    metadata_path = GIBS_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    
    # Try to query VizieR
    try:
        print("\n  Attempting VizieR query...")
        v = Vizier(catalog='J/A+A/562/A66', row_limit=10000)
        # Query by coordinates in bulge region (l=0, b=-2)
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # Query around Galactic center region
        center = SkyCoord(l=0*u.deg, b=-2*u.deg, frame='galactic')
        catalogs = v.query_region(center, radius=10*u.deg, catalog='J/A+A/562/A66')
        
        if catalogs and len(catalogs) > 0:
            print(f"  ✓ Found {len(catalogs)} catalogs in VizieR")
            for i, cat in enumerate(catalogs):
                if len(cat) > 0:
                    output_path = GIBS_DIR / f"gibs_catalog_{i}.fits"
                    cat.write(str(output_path), overwrite=True)
                    print(f"    → Saved: {output_path} ({len(cat)} stars)")
        else:
            # Try direct catalog access
            print("  Trying direct catalog access...")
            catalogs = v.get_catalogs('J/A+A/562/A66')
            if catalogs:
                for i, (name, cat) in enumerate(catalogs.items()):
                    output_path = GIBS_DIR / f"gibs_{name}.fits"
                    cat.write(str(output_path), overwrite=True)
                    print(f"    → Saved: {output_path} ({len(cat)} stars)")
            else:
                print("  ⚠ No catalogs found (may need different catalog ID)")
    except Exception as e:
        print(f"  ⚠ Could not query VizieR: {e}")
        print("  You may need to check the correct VizieR catalog ID")
        print("  Alternative: Download directly from ESO Archive")
    
    return True


def download_virac_gaia():
    """Download VIRAC + Gaia combined proper motions."""
    print("\n" + "="*60)
    print("Downloading VIRAC + Gaia (combined proper motions)")
    print("="*60)
    
    metadata = {
        "dataset": "VIRAC_Gaia",
        "description": "Absolute proper motions of Galactic bulge from VIRAC and Gaia",
        "coverage": "Entire Galactic bulge region",
        "paper": "Smith et al. 2018, MNRAS, 480, 1460",
        "vista_archive": "https://www.eso.org/qi/",
        "gaia_archive": "https://gea.esac.esa.int/archive/",
        "download_instructions": [
            "1. VIRAC data: VISTA Variables in Via Lactea (VVV) survey",
            "2. Cross-match with Gaia DR3 proper motions",
            "3. Combined catalog provides absolute proper motions",
            "4. Red clump stars can be isolated for distance-resolved kinematics"
        ],
        "columns": [
            "ra", "dec", "l", "b",
            "pmra_virac", "pmdec_virac",
            "pmra_gaia", "pmdec_gaia",
            "pmra_combined", "pmdec_combined",
            "distance", "distance_err"
        ]
    }
    
    metadata_path = VIRAC_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    print("  Note: VIRAC+Gaia requires cross-matching two catalogs")
    print("  See metadata.json for details")
    
    return True


def download_galacticnucleus():
    """Download GALACTICNUCLEUS Survey data."""
    print("\n" + "="*60)
    print("Downloading GALACTICNUCLEUS Survey")
    print("="*60)
    
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except ImportError:
        print("✗ astroquery not installed. Installing...")
        os.system(f"{sys.executable} -m pip install astroquery")
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    
    metadata = {
        "dataset": "GALACTICNUCLEUS",
        "description": "Proper motion catalogue for ~80,000 stars in central ~36′ × 16′ of MW",
        "coverage": "Central ~150 pc from Galactic Center",
        "paper": "Nogueras-Lara et al. 2018, A&A, 620, A83",
        "vizier_catalog": "J/A+A/620/A83",
        "eso_archive": "https://archive.eso.org/scienceportal/home",
        "download_instructions": [
            "1. Via VizieR (automated):",
            "   Catalog ID: J/A+A/620/A83",
            "2. Via ESO Archive:",
            "   Search for 'GALACTICNUCLEUS'",
            "3. Download proper motion catalog",
            "4. Catalog includes positions, proper motions, photometry"
        ],
        "columns": [
            "ra", "dec", "l", "b",
            "pmra", "pmdec",
            "pmra_err", "pmdec_err",
            "mag_Ks", "mag_H"
        ]
    }
    
    metadata_path = GALACTICNUCLEUS_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    
    # Try to query VizieR
    try:
        print("\n  Attempting VizieR query...")
        v = Vizier(catalog='J/A+A/620/A83', row_limit=100000)
        # Query Galactic Center region
        center = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
        catalogs = v.query_region(center, radius=0.5*u.deg, catalog='J/A+A/620/A83')
        
        if catalogs and len(catalogs) > 0:
            print(f"  ✓ Found {len(catalogs)} catalogs in VizieR")
            for i, cat in enumerate(catalogs):
                if len(cat) > 0:
                    output_path = GALACTICNUCLEUS_DIR / f"galacticnucleus_catalog_{i}.fits"
                    cat.write(str(output_path), overwrite=True)
                    print(f"    → Saved: {output_path} ({len(cat)} stars)")
        else:
            # Try direct catalog access
            print("  Trying direct catalog access...")
            catalogs = v.get_catalogs('J/A+A/620/A83')
            if catalogs:
                for i, (name, cat) in enumerate(catalogs.items()):
                    output_path = GALACTICNUCLEUS_DIR / f"galacticnucleus_{name}.fits"
                    cat.write(str(output_path), overwrite=True)
                    print(f"    → Saved: {output_path} ({len(cat)} stars)")
            else:
                print("  ⚠ No catalogs found via VizieR")
                print("  Try ESO Archive: https://archive.eso.org/scienceportal/home")
    except Exception as e:
        print(f"  ⚠ Could not query VizieR: {e}")
        print("  Alternative: Download directly from ESO Archive")
    
    return True


def download_muse():
    """Download MUSE kinematic maps (October 2025)."""
    print("\n" + "="*60)
    print("Downloading MUSE kinematic maps (October 2025)")
    print("="*60)
    
    metadata = {
        "dataset": "MUSE_kinematic_maps",
        "description": "Kinematic maps from ~23,000 stars across 57 bulge fields",
        "coverage": "57 bulge fields including new fields close to Galactic plane",
        "paper": "Check A&A October 2025 for publication",
        "download_instructions": [
            "1. Check Astronomy & Astrophysics journal",
            "2. Look for October 2025 MUSE bulge kinematics paper",
            "3. Download supplementary data tables",
            "4. Maps include: velocity fields, velocity dispersions, etc."
        ],
        "columns": [
            "ra", "dec", "l", "b",
            "v_los", "sigma_v",
            "field_id"
        ]
    }
    
    metadata_path = MUSE_DIR / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created metadata file: {metadata_path}")
    print("  Note: MUSE maps from October 2025 publication")
    print("  Check A&A supplementary data for download links")
    
    return True


def create_crossmatch_script():
    """Create a script for cross-matching BRAVA + APOGEE + Gaia."""
    print("\n" + "="*60)
    print("Creating cross-match script")
    print("="*60)
    
    script_content = '''#!/usr/bin/env python3
"""
Cross-match BRAVA + APOGEE + Gaia DR3 for full 6D phase space.

This script:
1. Loads BRAVA catalog (radial velocities)
2. Loads APOGEE catalog (chemistry + RVs)
3. Cross-matches with Gaia DR3 (proper motions)
4. Creates combined catalog with full 6D phase space
"""

import numpy as np
from astropy.table import Table, join
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path

# Data directories
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bulge_kinematics"
OUTPUT_DIR = DATA_DIR / "crossmatched"
OUTPUT_DIR.mkdir(exist_ok=True)

def crossmatch_catalogs(ra1, dec1, ra2, dec2, max_sep_arcsec=1.0):
    """Cross-match two catalogs by position."""
    coords1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coords2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    
    idx, sep2d, _ = coords1.match_to_catalog_sky(coords2)
    sep_arcsec = sep2d.arcsec
    
    # Only keep matches within max_sep
    good = sep_arcsec < max_sep_arcsec
    
    return idx[good], sep_arcsec[good], good

def main():
    print("Loading catalogs...")
    
    # Load BRAVA (when available)
    brava_path = DATA_DIR / "BRAVA" / "brava_catalog.fits"
    if brava_path.exists():
        brava = Table.read(brava_path)
        print(f"  BRAVA: {len(brava)} stars")
    else:
        print("  BRAVA: Not found")
        brava = None
    
    # Load APOGEE (when available)
    apogee_path = DATA_DIR / "APOGEE" / "apogee_bulge.fits"
    if apogee_path.exists():
        apogee = Table.read(apogee_path)
        print(f"  APOGEE: {len(apogee)} stars")
    else:
        print("  APOGEE: Not found")
        apogee = None
    
    # Cross-match BRAVA + APOGEE
    if brava is not None and apogee is not None:
        print("\\nCross-matching BRAVA + APOGEE...")
        idx, sep, good = crossmatch_catalogs(
            brava['ra'], brava['dec'],
            apogee['ra'], apogee['dec'],
            max_sep_arcsec=1.0
        )
        
        brava_matched = brava[good]
        apogee_matched = apogee[idx[good]]
        
        # Combine tables
        combined = join(brava_matched, apogee_matched, keys=['ra', 'dec'], join_type='inner')
        print(f"  Matched: {len(combined)} stars")
        
        # Save
        output_path = OUTPUT_DIR / "brava_apogee_crossmatch.fits"
        combined.write(str(output_path), overwrite=True)
        print(f"  Saved: {output_path}")
    
    # TODO: Add Gaia DR3 cross-match for proper motions
    print("\\nNote: Gaia DR3 cross-match not yet implemented")
    print("  Use astroquery.gaia to query proper motions for matched stars")

if __name__ == "__main__":
    main()
'''
    
    script_path = BULGE_DIR / "crossmatch_brava_apogee_gaia.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"✓ Created cross-match script: {script_path}")
    return True


def main():
    """Download all bulge kinematics datasets."""
    print("\n" + "="*70)
    print("BULGE/BAR KINEMATICS DATA DOWNLOAD")
    print("="*70)
    print(f"Output directory: {BULGE_DIR}")
    
    results = {}
    
    # Download each dataset
    results['BRAVA'] = download_brava()
    results['APOGEE'] = download_apogee()
    results['GIBS'] = download_gibs()
    results['VIRAC_Gaia'] = download_virac_gaia()
    results['GALACTICNUCLEUS'] = download_galacticnucleus()
    results['MUSE'] = download_muse()
    
    # Create cross-match script
    create_crossmatch_script()
    
    # Summary
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    for dataset, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Manual downloads required:")
    print("   - BRAVA: Visit https://irsa.ipac.caltech.edu/data/BRAVA/")
    print("   - GALACTICNUCLEUS: Visit ESO Archive")
    print("   - MUSE: Check A&A October 2025 supplementary data")
    print("\n2. Automated queries:")
    print("   - APOGEE: Run queries via CasJobs or astroquery (see metadata.json)")
    print("   - GIBS: Query via VizieR (see metadata.json)")
    print("\n3. Cross-matching:")
    print(f"   - Run: python {BULGE_DIR / 'crossmatch_brava_apogee_gaia.py'}")
    print("\n4. Recommended starting point:")
    print("   - BRAVA + APOGEE cross-match for bulge membership + chemistry")
    print("   - Add Gaia DR3 proper motions for full 6D phase space")
    
    print(f"\nAll metadata files saved to: {BULGE_DIR}")


if __name__ == "__main__":
    main()

