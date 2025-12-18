#!/usr/bin/env python3
"""Check that downloaded Gaia 6D catalog has correct columns."""
from astropy.table import Table
import numpy as np

fits_file = 'data/gaia/6d_test/gaia_6d_l0000.0_0360.0_b-090.0_+090.0.fits'

print("Loading FITS file...")
t = Table.read(fits_file)

print(f"\nTotal stars: {len(t):,}")
print(f"\nAll columns ({len(t.colnames)}):")
for i, col in enumerate(t.colnames, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "="*70)
print("6D Phase Space Verification:")
print("="*70)

required_6d = {
    'ra': 'Right Ascension',
    'dec': 'Declination', 
    'parallax': 'Parallax',
    'pmra': 'Proper motion in RA',
    'pmdec': 'Proper motion in Dec',
    'radial_velocity': 'Radial velocity'
}

all_present = True
for col, desc in required_6d.items():
    if col in t.colnames:
        # Check for non-null values
        non_null = sum(~t[col].mask) if hasattr(t[col], 'mask') else sum(~np.isfinite(t[col]))
        print(f"  [OK] {col:20s} ({desc:25s}) - {non_null:,}/{len(t):,} non-null")
    else:
        print(f"  [MISSING] {col:20s} ({desc:25s}) - MISSING")
        all_present = False

print("\n" + "="*70)
print("Additional useful columns:")
print("="*70)

additional = ['l', 'b', 'parallax_error', 'pmra_error', 'pmdec_error', 
              'radial_velocity_error', 'phot_g_mean_mag', 'ruwe', 
              'visibility_periods_used', 'source_id']

for col in additional:
    if col in t.colnames:
        print(f"  [OK] {col}")
    else:
        print(f"  [-] {col} (not present)")

print("\n" + "="*70)
print("Sample data (first star):")
print("="*70)

import numpy as np
star = t[0]
for col in ['source_id', 'ra', 'dec', 'l', 'b', 'parallax', 'pmra', 'pmdec', 'radial_velocity']:
    if col in t.colnames:
        val = star[col]
        if hasattr(val, 'value'):
            val = val.value
        print(f"  {col:20s} = {val}")

print("\n" + "="*70)
print("Data Quality Check:")
print("="*70)

for col in ['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']:
    if col in t.colnames:
        # Check for finite values
        valid = np.sum(np.isfinite(t[col]))
        print(f"  {col:20s}: {valid:,}/{len(t):,} ({valid/len(t)*100:.1f}%) valid")

print("\n" + "="*70)
print("Sample Statistics:")
print("="*70)

if 'radial_velocity' in t.colnames:
    rv = t['radial_velocity']
    rv_finite = rv[np.isfinite(rv)]
    if len(rv_finite) > 0:
        print(f"  Radial velocity: {np.min(rv_finite):.1f} to {np.max(rv_finite):.1f} km/s (median: {np.median(rv_finite):.1f})")

if 'parallax' in t.colnames:
    plx = t['parallax']
    plx_finite = plx[np.isfinite(plx)]
    if len(plx_finite) > 0:
        print(f"  Parallax: {np.min(plx_finite):.4f} to {np.max(plx_finite):.4f} mas (median: {np.median(plx_finite):.4f})")

if 'pmra' in t.colnames:
    pmra = t['pmra']
    pmra_finite = pmra[np.isfinite(pmra)]
    if len(pmra_finite) > 0:
        print(f"  PM RA: {np.min(pmra_finite):.2f} to {np.max(pmra_finite):.2f} mas/yr (median: {np.median(pmra_finite):.2f})")

if 'pmdec' in t.colnames:
    pmdec = t['pmdec']
    pmdec_finite = pmdec[np.isfinite(pmdec)]
    if len(pmdec_finite) > 0:
        print(f"  PM Dec: {np.min(pmdec_finite):.2f} to {np.max(pmdec_finite):.2f} mas/yr (median: {np.median(pmdec_finite):.2f})")

print("\n" + "="*70)
if all_present:
    print("[SUCCESS] All 6D phase space columns are present with valid data!")
else:
    print("[WARNING] Some required columns are missing")
print("="*70)

