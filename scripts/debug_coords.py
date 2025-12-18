#!/usr/bin/env python3
"""Debug coordinate conversion issue."""
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
from astropy.io import ascii

# Check BRAVA star
brava = ascii.read('data/bulge_kinematics/BRAVA/brava_catalog.tbl', format='ipac')
star = brava[0]
brava_coord = SkyCoord(ra=star['ra'] * u.deg, dec=star['dec'] * u.deg, frame='icrs')
gal = brava_coord.galactic

print(f'BRAVA star:')
print(f'  ICRS: RA={brava_coord.ra.deg:.2f}, Dec={brava_coord.dec.deg:.2f}')
print(f'  Galactic: l={gal.l.deg:.2f}, b={gal.b.deg:.2f}')

# Check Gaia catalog
print(f'\nChecking Gaia catalog...')
gaia = pd.read_csv('data/gaia/gaia_processed_corrected.csv', nrows=100000)
print(f'  Gaia l range: {gaia["l"].min():.2f} to {gaia["l"].max():.2f}')
print(f'  Gaia b range: {gaia["b"].min():.2f} to {gaia["b"].max():.2f}')

# Find nearby Gaia stars in Galactic coordinates
near = (abs(gaia['l'] - gal.l.deg) < 1) & (abs(gaia['b'] - gal.b.deg) < 1)
print(f'  Gaia stars within 1 deg of BRAVA (l,b): {near.sum()}')

if near.sum() > 0:
    print(f'\n  Sample nearby Gaia stars (first 5):')
    nearby_gaia = gaia[near].head(5)
    for idx, row in nearby_gaia.iterrows():
        gaia_gal = SkyCoord(l=row['l'] * u.deg, b=row['b'] * u.deg, frame='galactic')
        gaia_icrs = gaia_gal.icrs
        sep = brava_coord.separation(gaia_icrs)
        print(f'    Gaia l={row["l"]:.2f}, b={row["b"]:.2f} -> ICRS RA={gaia_icrs.ra.deg:.2f}, Dec={gaia_icrs.dec.deg:.2f}, sep={sep.arcsec:.2f} arcsec')

