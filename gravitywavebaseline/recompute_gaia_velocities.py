"""
Recompute Galactocentric cylindrical coordinates and tangential velocities
for the full 1.8M Gaia sample using proper astropy transformations.

Output:
    data/gaia/gaia_processed_corrected.csv
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import (
    CartesianDifferential,
    Galactocentric,
    SkyCoord,
)

RAW_PATH = Path("data/gaia/gaia_large_sample_raw.csv")
OLD_PROCESSED = Path("data/gaia/gaia_processed.csv")
OUTPUT_PATH = Path("data/gaia/gaia_processed_corrected.csv")

CHUNK_SIZE = 200_000

# Galactocentric reference parameters
R_SUN = 8.2 * u.kpc
Z_SUN = 0.02 * u.kpc
V_SUN = CartesianDifferential([11.1, 238.0 + 12.24, 7.25] * u.km / u.s)


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Convert one chunk of the raw Gaia sample into Galactocentric values."""
    chunk = chunk.copy()

    # Require positive parallax
    mask = np.isfinite(chunk["parallax"].values) & (chunk["parallax"].values > 0)
    chunk = chunk.loc[mask]
    if chunk.empty:
        return pd.DataFrame()

    distance_pc = 1000.0 / chunk["parallax"].values

    coord = SkyCoord(
        ra=chunk["ra"].values * u.deg,
        dec=chunk["dec"].values * u.deg,
        distance=distance_pc * u.pc,
        pm_ra_cosdec=chunk["pmra"].fillna(0).values * u.mas / u.yr,
        pm_dec=chunk["pmdec"].fillna(0).values * u.mas / u.yr,
        radial_velocity=chunk["radial_velocity"].fillna(0).values * u.km / u.s,
        frame="icrs",
    )

    galcen_frame = Galactocentric(
        galcen_distance=R_SUN,
        z_sun=Z_SUN,
        galcen_v_sun=V_SUN,
    )

    gal = coord.transform_to(galcen_frame)
    cyl = gal.cylindrical

    R_kpc = cyl.rho.to(u.kpc).value
    z_kpc = gal.z.to(u.kpc).value
    phi = cyl.phi.to(u.rad).value

    cyl_diff = cyl.differentials["s"]
    v_phi = (cyl_diff.d_phi * cyl.rho).to(
        u.km / u.s, equivalencies=u.dimensionless_angles()
    ).value
    v_phi = np.abs(v_phi)
    v_radial = cyl_diff.d_rho.to(u.km / u.s).value

    df = pd.DataFrame(
        {
            "source_id": chunk["source_id"].values,
            "R_cyl": R_kpc,
            "z": z_kpc,
            "phi": phi,
            "v_phi": v_phi,
            "v_rad": v_radial,
            "pmra": chunk["pmra"].values,
            "pmdec": chunk["pmdec"].values,
            "distance_pc": distance_pc,
            "l": chunk["l"].values,
            "b": chunk["b"].values,
            "parallax": chunk["parallax"].values,
            "phot_g_mean_mag": chunk["phot_g_mean_mag"].values,
        }
    )

    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw Gaia file missing: {RAW_PATH}")

    start = time.time()
    frames = []
    total_rows = 0

    print(f"[INFO] Loading raw Gaia sample from {RAW_PATH}")
    for idx, chunk in enumerate(pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE)):
        t0 = time.time()
        transformed = transform_chunk(chunk)
        frames.append(transformed)
        total_rows += len(transformed)
        t1 = time.time()
        print(
            f"  Chunk {idx+1:02d}: raw {len(chunk):6d} -> kept {len(transformed):6d} "
            f"({t1 - t0:.1f}s)"
        )

    gaia_new = pd.concat(frames, ignore_index=True)
    print(f"[OK] Converted {len(gaia_new):,} stars in {time.time() - start:.1f}s")

    # Merge back stellar mass estimates and any other legacy columns
    if OLD_PROCESSED.exists():
        print(f"[INFO] Merging stellar mass columns from {OLD_PROCESSED}")
        header = pd.read_csv(OLD_PROCESSED, nrows=1)
        mass_cols = [c for c in ["source_id", "M_star", "M_star_estimated"] if c in header.columns]
        legacy = pd.read_csv(OLD_PROCESSED, usecols=mass_cols)
        gaia_new = gaia_new.merge(legacy, on="source_id", how="left")
    else:
        gaia_new["M_star"] = 1.0
        gaia_new["M_star_estimated"] = np.nan

    # Report velocity statistics
    vphi = gaia_new["v_phi"].values
    print(
        "[STATS] v_phi: min {:.1f} km/s, max {:.1f} km/s, median {:.1f} km/s".format(
            np.nanmin(vphi), np.nanmax(vphi), np.nanmedian(vphi)
        )
    )

    gaia_new.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Saved corrected catalogue to {OUTPUT_PATH} ({len(gaia_new):,} rows)")


if __name__ == "__main__":
    main()

