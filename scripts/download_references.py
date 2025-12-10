#!/usr/bin/env python3
"""
Download reference PDFs for Sigma-Gravity paper citations.

This script downloads PDFs from arXiv and other open sources for verification.
Datasets (SPARC, Eilers MW, Fox clusters) are excluded as we already have them.
"""

import os
import urllib.request
import ssl
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "references"
OUTPUT_DIR.mkdir(exist_ok=True)

# Reference papers with arXiv IDs or DOIs
# Format: (filename, arxiv_id or url, description)
PAPERS = [
    # MOND/Modified Gravity Theory
    ("Milgrom1983a_MOND_original.pdf", 
     None,  # Pre-arXiv, but available at ADS
     "Milgrom 1983 - Original MOND paper (ApJ 270, 365)"),
    
    ("Milgrom1983b_MOND_galaxies.pdf",
     None,  # Pre-arXiv
     "Milgrom 1983 - MOND implications for galaxies (ApJ 270, 371)"),
    
    ("Bekenstein2004_TeVeS.pdf",
     "astro-ph/0403694",
     "Bekenstein 2004 - TeVeS relativistic MOND (PRD 70, 083509)"),
    
    ("Milgrom2009_BiMOND.pdf",
     "0906.0571",
     "Milgrom 2009 - Bimetric MOND (PRD 80, 123536)"),
    
    ("Milgrom2010_QUMOND.pdf",
     "1001.0785",
     "Milgrom 2010 - QUMOND quasi-linear formulation (PRD 82, 043523)"),
    
    ("Sanders2002_MOND_review.pdf",
     "astro-ph/0204521",
     "Sanders & McGaugh 2002 - MOND review (ARAA 40, 263)"),
    
    # Teleparallel Gravity
    ("Ferraro2007_teleparallel.pdf",
     "gr-qc/0702125",
     "Ferraro & Fiorini 2007 - Modified teleparallel gravity (PRD 75, 084031)"),
    
    ("Bahamonde2023_teleparallel_review.pdf",
     "2106.13793",
     "Bahamonde et al 2023 - Teleparallel gravity review (RPP 86, 026901)"),
    
    # Emergent Gravity
    ("Verlinde2017_emergent_gravity.pdf",
     "1611.02269",
     "Verlinde 2017 - Emergent Gravity and Dark Universe (SciPost Phys 2, 016)"),
    
    # Cosmology
    ("Planck2020_cosmological_params.pdf",
     "1807.06209",
     "Planck 2018/2020 - Cosmological parameters (A&A 641, A6)"),
    
    # Baryonic Tully-Fisher
    ("McGaugh2000_BTFR.pdf",
     "astro-ph/0003001",
     "McGaugh et al 2000 - Baryonic Tully-Fisher (ApJL 533, L99)"),
    
    # Solar System Tests
    ("Bertotti2003_Cassini.pdf",
     None,  # Nature paper, not on arXiv
     "Bertotti et al 2003 - Cassini test of GR (Nature 425, 374)"),
    
    # Historical
    ("Zwicky1933_dark_matter.pdf",
     None,  # Historical paper, 1933
     "Zwicky 1933 - Original dark matter paper (Helv Phys Acta 6, 110)"),
]

# Data papers (we have the data, but listing for reference)
DATA_PAPERS = [
    ("Lelli2016_SPARC.pdf", "1606.09251", "SPARC database paper"),
    ("Eilers2019_MW.pdf", "1810.09466", "Milky Way rotation curve"),
    ("Fox2022_clusters.pdf", "2109.09763", "Fox et al cluster lensing"),
]

def download_arxiv_pdf(arxiv_id, output_path):
    """Download PDF from arXiv."""
    # Handle old-style arXiv IDs
    if '/' in arxiv_id:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    else:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    print(f"  Downloading from: {url}")
    
    # Create SSL context that doesn't verify (for some networks)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            with open(output_path, 'wb') as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    print("=" * 70)
    print("DOWNLOADING REFERENCE PDFs")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    downloaded = []
    failed = []
    skipped = []
    
    for filename, arxiv_id, description in PAPERS:
        output_path = OUTPUT_DIR / filename
        print(f"\n{description}")
        
        if output_path.exists():
            print(f"  Already exists: {filename}")
            downloaded.append(filename)
            continue
        
        if arxiv_id is None:
            print(f"  SKIP: No arXiv ID (pre-arXiv or paywalled)")
            skipped.append((filename, description))
            continue
        
        if download_arxiv_pdf(arxiv_id, output_path):
            print(f"  SUCCESS: {filename}")
            downloaded.append(filename)
        else:
            failed.append((filename, arxiv_id))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nDownloaded: {len(downloaded)}")
    for f in downloaded:
        print(f"  ✓ {f}")
    
    if skipped:
        print(f"\nSkipped (no arXiv): {len(skipped)}")
        for f, desc in skipped:
            print(f"  - {f}")
        print("\n  These papers are either:")
        print("    • Pre-arXiv (1983 papers)")
        print("    • Paywalled (Nature, historical journals)")
        print("  You can find them via:")
        print("    • NASA ADS: https://ui.adsabs.harvard.edu/")
        print("    • Your institution's library")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for f, aid in failed:
            print(f"  ✗ {f} (arXiv:{aid})")
    
    # Also download data papers for completeness
    print("\n" + "-" * 70)
    print("DATA PAPERS (optional - we already have the data)")
    print("-" * 70)
    
    for filename, arxiv_id, description in DATA_PAPERS:
        output_path = OUTPUT_DIR / filename
        print(f"\n{description}")
        
        if output_path.exists():
            print(f"  Already exists: {filename}")
            continue
        
        if download_arxiv_pdf(arxiv_id, output_path):
            print(f"  SUCCESS: {filename}")
        else:
            print(f"  FAILED: {filename}")

if __name__ == "__main__":
    main()

