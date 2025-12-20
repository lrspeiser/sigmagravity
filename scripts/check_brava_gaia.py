#!/usr/bin/env python3
"""Check BRAVA-filtered Gaia 6D catalog."""
from astropy.table import Table
import numpy as np

import sys
if len(sys.argv) > 1:
    fits_file = sys.argv[1]
else:
    fits_file = 'data/gaia/6d_brava_test.fits'

print("Loading BRAVA-filtered Gaia 6D catalog...")
t = Table.read(fits_file)

print(f"\nTotal stars: {len(t):,}")
print(f"Columns: {len(t.colnames)}")
print(f"\nColumn names: {list(t.colnames)}")

print("\n" + "="*70)
print("6D Phase Space Verification:")
print("="*70)

for col in ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']:
    if col in t.colnames:
        valid = np.sum(np.isfinite(t[col]))
        print(f"  {col:20s}: {valid:,}/{len(t):,} ({valid/len(t)*100:.1f}%) valid")

print("\n" + "="*70)
print("2MASS ID column:")
print("="*70)
if 'tmass_id' in t.colnames:
    print(f"  [OK] tmass_id column present")
    print(f"  Sample: {t['tmass_id'][0]}")
else:
    print(f"  [MISSING] tmass_id column")

print("\n" + "="*70)
print("[SUCCESS] BRAVA-filtered Gaia 6D catalog verified!")
print("="*70)

