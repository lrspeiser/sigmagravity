#!/usr/bin/env python3
"""
Download BRAVA spectra (FITS files) from IRSA.

This script:
1. Reads the BRAVA catalog (brava_catalog.tbl)
2. Extracts spectra URLs from the catalog
3. Downloads FITS spectra files to a local directory

Usage:
    python download_brava_spectra.py [--all] [--sample N] [--output-dir DIR]
"""

import os
import sys
import re
import argparse
from pathlib import Path
import requests
from tqdm import tqdm
from astropy.io import ascii

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Data directory
DATA_DIR = PROJECT_ROOT / "data" / "bulge_kinematics"
BRAVA_DIR = DATA_DIR / "BRAVA"
SPECTRA_DIR = BRAVA_DIR / "spectra"
SPECTRA_DIR.mkdir(parents=True, exist_ok=True)


def extract_spectra_urls(catalog_path):
    """Extract FITS spectra URLs from the BRAVA catalog."""
    print(f"Reading catalog: {catalog_path}")
    
    # Read the IPAC table
    try:
        table = ascii.read(catalog_path, format='ipac')
    except Exception as e:
        print(f"Error reading catalog: {e}")
        return []
    
    print(f"  Found {len(table)} stars in catalog")
    
    # Look for columns containing spectra URLs
    spectra_urls = []
    
    # Check for fits_spectra columns or HTML links
    for colname in table.colnames:
        if 'fits' in colname.lower() or 'spectra' in colname.lower():
            print(f"  Found column: {colname}")
            # Extract URLs from HTML links
            for row in table:
                value = str(row[colname])
                if value and value != 'null':
                    # Extract URL from HTML link: <a href="/data/BRAVA/spectra/Fld6_66.fits">Fits spectrum</a>
                    match = re.search(r'href="([^"]+\.fits)"', value)
                    if match:
                        url_path = match.group(1)
                        # Convert relative URL to absolute
                        if url_path.startswith('/'):
                            full_url = f"https://irsa.ipac.caltech.edu{url_path}"
                        else:
                            full_url = f"https://irsa.ipac.caltech.edu/data/BRAVA/spectra/{url_path}"
                        spectra_urls.append(full_url)
    
    # Remove duplicates
    spectra_urls = list(set(spectra_urls))
    print(f"  Found {len(spectra_urls)} unique spectra URLs")
    
    return spectra_urls


def download_spectrum(url, output_dir):
    """Download a single FITS spectrum file."""
    # Extract filename from URL
    filename = url.split('/')[-1]
    output_path = output_dir / filename
    
    # Skip if already downloaded
    if output_path.exists():
        return True, "already exists"
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return True, "downloaded"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Download BRAVA spectra from IRSA")
    parser.add_argument('--all', action='store_true',
                       help='Download all spectra (default: download sample)')
    parser.add_argument('--sample', type=int, default=10,
                       help='Number of spectra to download as sample (default: 10)')
    parser.add_argument('--output-dir', type=str, default=str(SPECTRA_DIR),
                       help=f'Output directory for spectra (default: {SPECTRA_DIR})')
    
    args = parser.parse_args()
    
    catalog_path = BRAVA_DIR / "brava_catalog.tbl"
    
    if not catalog_path.exists():
        print(f"ERROR: Catalog not found: {catalog_path}")
        print("  Please run download_bulge_kinematics.py first to download the catalog")
        return 1
    
    # Extract spectra URLs
    spectra_urls = extract_spectra_urls(catalog_path)
    
    if not spectra_urls:
        print("  No spectra URLs found in catalog")
        print("  You can browse spectra directly at: https://irsa.ipac.caltech.edu/data/BRAVA/spectra/")
        return 0
    
    # Limit to sample if not downloading all
    if not args.all:
        spectra_urls = spectra_urls[:args.sample]
        print(f"\n  Downloading sample of {args.sample} spectra (use --all to download all)")
    else:
        print(f"\n  Downloading all {len(spectra_urls)} spectra")
    
    # Download spectra
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    print()
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for url in tqdm(spectra_urls, desc="Downloading spectra"):
        success, status = download_spectrum(url, output_dir)
        if success:
            if status == "already exists":
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
            tqdm.write(f"  Error downloading {url}: {status}")
    
    print(f"\n✓ Download complete:")
    print(f"  Downloaded: {success_count}")
    print(f"  Skipped (already exists): {skip_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files in directory: {len(list(output_dir.glob('*.fits')))}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

