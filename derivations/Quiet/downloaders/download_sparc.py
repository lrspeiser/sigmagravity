"""
Download SPARC galaxy rotation curve data.

SPARC = Spitzer Photometry and Accurate Rotation Curves
Source: http://astroweb.cwru.edu/SPARC/
Reference: Lelli, McGaugh & Schombert (2016)

This is the primary dataset for testing Σ-Gravity against galaxy rotation curves.
Contains 175 galaxies with high-quality HI/Hα rotation curves and 3.6μm photometry.
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Add parent to path for config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SPARC_DIR, DATA_URLS

# =============================================================================
# DATA FILES TO DOWNLOAD
# =============================================================================

SPARC_FILES = {
    # Main galaxy table with properties
    "SPARC_Lelli2016c.mrt": "http://astroweb.cwru.edu/SPARC/SPARC_Lelli2016c.mrt",
    
    # Mass models table
    "MassModels_Lelli2016c.mrt": "http://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt",
    
    # Individual rotation curves are in separate files
    # We'll download the full archive
    "SPARC_Rotmod.tar.gz": "http://astroweb.cwru.edu/SPARC/SPARC_Rotmod.tar.gz",
}

# Individual galaxy rotation curve template
# Format: http://astroweb.cwru.edu/SPARC/NGC####_rotmod.dat
ROTCURVE_URL_TEMPLATE = "http://astroweb.cwru.edu/SPARC/{galaxy}_rotmod.dat"


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def download_sparc_main_tables():
    """Download the main SPARC data tables."""
    print("=" * 60)
    print("Downloading SPARC main data tables")
    print("=" * 60)
    
    for filename, url in SPARC_FILES.items():
        dest = SPARC_DIR / filename
        if dest.exists():
            print(f"  {filename} already exists, skipping")
            continue
        
        print(f"\nDownloading {filename}...")
        success = download_file(url, dest)
        if success:
            print(f"  Saved to {dest}")


def extract_rotation_curves():
    """Extract the rotation curve archive."""
    import tarfile
    
    archive_path = SPARC_DIR / "SPARC_Rotmod.tar.gz"
    if not archive_path.exists():
        print("Rotation curve archive not found. Run download first.")
        return
    
    extract_dir = SPARC_DIR / "rotcurves"
    if extract_dir.exists():
        print("Rotation curves already extracted")
        return
    
    print("\nExtracting rotation curves...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=SPARC_DIR)
    print(f"  Extracted to {SPARC_DIR}")


def parse_sparc_table(filepath: Path) -> list:
    """
    Parse the SPARC MRT (Machine Readable Table) format.
    
    Returns list of dicts with galaxy properties.
    """
    galaxies = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (start with # or are blank)
    data_started = False
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Parse data lines
        # Format varies by file, but main table has:
        # Galaxy  D  e_D  Inc  e_Inc  L[3.6]  e_L  Reff  SBeff  Rdisk  SBdisk  ...
        parts = line.split()
        if len(parts) < 10:
            continue
            
        try:
            galaxy = {
                'name': parts[0],
                'distance_mpc': float(parts[1]),
                'distance_err': float(parts[2]),
                'inclination': float(parts[3]),
                'inc_err': float(parts[4]),
                'luminosity_3p6': float(parts[5]),  # 10^9 Lsun
                'lum_err': float(parts[6]),
                'r_eff_kpc': float(parts[7]),
                'sb_eff': float(parts[8]),
            }
            galaxies.append(galaxy)
        except (ValueError, IndexError):
            continue
    
    return galaxies


def parse_rotation_curve(filepath: Path) -> dict:
    """
    Parse individual SPARC rotation curve file.
    
    Returns dict with:
        - r_kpc: radii in kpc
        - v_obs: observed velocities (km/s)
        - v_err: velocity errors (km/s)
        - v_gas: gas contribution (km/s)
        - v_disk: disk contribution (km/s)
        - v_bulge: bulge contribution (km/s)
    """
    data = {
        'r_kpc': [],
        'v_obs': [],
        'v_err': [],
        'v_gas': [],
        'v_disk': [],
        'v_bulge': []
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 6:
                continue
            
            try:
                data['r_kpc'].append(float(parts[0]))
                data['v_obs'].append(float(parts[1]))
                data['v_err'].append(float(parts[2]))
                data['v_gas'].append(float(parts[3]))
                data['v_disk'].append(float(parts[4]))
                data['v_bulge'].append(float(parts[5]) if len(parts) > 5 else 0.0)
            except ValueError:
                continue
    
    # Convert to numpy arrays
    import numpy as np
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def verify_download():
    """Verify that all required files are present."""
    print("\n" + "=" * 60)
    print("Verifying SPARC data")
    print("=" * 60)
    
    # Check main tables
    main_table = SPARC_DIR / "SPARC_Lelli2016c.mrt"
    if main_table.exists():
        galaxies = parse_sparc_table(main_table)
        print(f"  Main table: {len(galaxies)} galaxies")
    else:
        print("  Main table: MISSING")
    
    # Check rotation curves
    rotcurve_dir = SPARC_DIR / "SPARC_Rotmod"
    if rotcurve_dir.exists():
        rotcurve_files = list(rotcurve_dir.glob("*_rotmod.dat"))
        print(f"  Rotation curves: {len(rotcurve_files)} files")
    else:
        print("  Rotation curves: MISSING (need to extract)")
    
    print("\nSPARC data ready for analysis!")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("SPARC Data Downloader")
    print("=" * 60)
    print(f"Data directory: {SPARC_DIR}")
    print()
    
    # Download main files
    download_sparc_main_tables()
    
    # Extract rotation curves
    extract_rotation_curves()
    
    # Verify
    verify_download()
