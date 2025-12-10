#!/usr/bin/env python3
"""
Detailed citation verification - extracts more content and identifies wrong PDFs.
"""

import pdfplumber
from pathlib import Path

refs_dir = Path("/Users/leonardspeiser/Projects/sigmagravity/docs/references")

def get_pdf_info(pdf_name, expected_info):
    """Extract detailed info from PDF."""
    pdf_path = refs_dir / pdf_name
    if not pdf_path.exists():
        return f"FILE NOT FOUND: {pdf_name}"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Get first 2 pages
            text = ""
            for i, page in enumerate(pdf.pages[:2]):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Return first 2000 chars
            return text[:2000]
    except Exception as e:
        return f"ERROR: {e}"

# Check each problematic PDF
problematic = [
    ("1983ApJ-milgrom-implications-for-galaxies.pdf", 
     "Should be: M. Milgrom, Astrophys. J. 270, 371 (1983) - MOND implications for galaxies"),
    ("Milgrom2009_BiMOND.pdf", 
     "Should be: M. Milgrom, Phys. Rev. D 80, 123536 (2009) - BIMOND"),
    ("Milgrom2010_QUMOND.pdf", 
     "Should be: M. Milgrom, Phys. Rev. D 82, 043523 (2010) - QUMOND"),
    ("Ferraro2007_teleparallel.pdf", 
     "Should be: R. Ferraro and F. Fiorini, Phys. Rev. D 75, 084031 (2007) - f(T) teleparallel"),
    ("Fox2022_clusters.pdf", 
     "Should be: C. Fox et al., Astrophys. J. 928, 87 (2022) - Strong-lensing clusters"),
    ("Bekenstein2004_TeVeS.pdf",
     "Should be: J. D. Bekenstein, Phys. Rev. D 70, 083509 (2004) - TeVeS"),
]

print("=" * 80)
print("DETAILED PDF CONTENT EXTRACTION")
print("=" * 80)

for pdf_name, expected in problematic:
    print(f"\n{'='*80}")
    print(f"FILE: {pdf_name}")
    print(f"EXPECTED: {expected}")
    print("-" * 80)
    content = get_pdf_info(pdf_name, expected)
    print(content)
    print()

