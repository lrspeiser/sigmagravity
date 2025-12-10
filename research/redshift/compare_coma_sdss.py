#!/usr/bin/env python3
"""
Compare observed spectroscopic redshift offsets in the Coma cluster (SDSS) to
Σ‑Gravity endpoint gravitational redshift predictions using a toy Hernquist model.

This script:
- Downloads SDSS spectra in a 6×6 deg box around Coma (if not already present)
- Selects galaxies near the Coma BCG and within |Δv| < 3000 km/s of the BCG
- Bins by projected radius and computes the mean observed Δv relative to the BCG
- Predicts Δv_grav(R) from a Hernquist + Σ kernel field via redshift/gravitational_redshift_endpoint
- Writes a CSV summary under redshift/outputs/coma_redshift_comparison.csv

Notes
- This is a research sandbox; the mass model is a simple Hernquist profile with
  tunable parameters. For rigorous work, replace with your calibrated mass model
  (and/or wire a real geff via SIGMA_GEFF and call the endpoint on the pairwise
  (x_emit, x_obs) positions).
- We use a flat H0-only angular diameter distance (low-z approximation).
"""
from __future__ import annotations
import os
import math
import csv
import json
from pathlib import Path
import numpy as np

from .redshift import c, gravitational_redshift_endpoint
from .geff_adapters import geff_hernquist_factory

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / 'data' / 'redshift'
OUTDIR = ROOT / 'redshift' / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Coma cluster (Abell 1656) reference center (approx; J2000)
RA0 = 194.953  # deg
DEC0 = 27.98   # deg
Z_CLUSTER = 0.0231

# SDSS SkyServer boxed query (6×6 deg around Coma)
SDSS_BOX = dict(ra_min=192.0, ra_max=198.0, dec_min=25.0, dec_max=31.0)
SDSS_URL = (
    "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?"
    "cmd="
    "SELECT%20TOP%20100000%20"
    "s.ra,%20s.dec,%20s.z,%20s.zErr,%20s.class,%20p.modelMag_r%20"
    "FROM%20SpecObjAll%20AS%20s%20JOIN%20PhotoObj%20AS%20p%20ON%20s.bestObjID%20=%20p.objID%20"
    "WHERE%20(s.class=%27GALAXY%27)%20AND%20s.z%20BETWEEN%200.0%20AND%200.05%20"
    f"AND%20s.ra%20BETWEEN%20{SDSS_BOX['ra_min']}%20AND%20{SDSS_BOX['ra_max']}%20"
    f"AND%20s.dec%20BETWEEN%20{SDSS_BOX['dec_min']}%20AND%20{SDSS_BOX['dec_max']}"
    "&format=csv"
)
SDSS_CSV = DATADIR / 'coma_sdss_box.csv'

# Cosmology (low-z approximation)
H0 = 70.0  # km/s/Mpc
c_kms = 299_792.458  # km/s
D_A_Mpc = (c_kms / H0) * Z_CLUSTER / (1.0 + Z_CLUSTER)  # angular diameter distance
arcsec_to_rad = (math.pi / 180.0) / 3600.0


def haversine_deg(ra1, dec1, ra2, dec2):
    ra1 = math.radians(ra1); dec1 = math.radians(dec1)
    ra2 = math.radians(ra2); dec2 = math.radians(dec2)
    d_ra = ra2 - ra1; d_dec = dec2 - dec1
    a = math.sin(d_dec/2.0)**2 + math.cos(dec1)*math.cos(dec2)*math.sin(d_ra/2.0)**2
    return 2.0 * math.asin(min(1.0, math.sqrt(a)))  # radians


def ensure_sdss_csv():
    DATADIR.mkdir(parents=True, exist_ok=True)
    if SDSS_CSV.exists() and SDSS_CSV.stat().st_size > 1000:
        return
    # Download via curl (silent); fall back message if unavailable
    os.system(f"curl -s -L \"{SDSS_URL}\" -o \"{SDSS_CSV.as_posix()}\"")


def load_sdss_csv(path: Path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(line for line in f if not line.startswith('#'))
        for r in reader:
            try:
                rows.append(dict(
                    ra=float(r['ra']), dec=float(r['dec']), z=float(r['z']), zErr=float(r['zErr'] or 0.0),
                    cls=r.get('class',''), rmag=float(r.get('modelMag_r') or 99.0)
                ))
            except Exception:
                continue
    return rows


def select_members(rows):
    # Angular selection within 2 deg; compute BCG as min rmag within 0.3 deg
    ang_deg = []
    for r in rows:
        d_rad = haversine_deg(r['ra'], r['dec'], RA0, DEC0)
        ang_deg.append(math.degrees(d_rad))
    ang_deg = np.asarray(ang_deg)
    rows = np.asarray(rows)

    mask2 = ang_deg <= 2.0
    sel = rows[mask2]
    ang2 = ang_deg[mask2]

    # Choose BCG: brightest rmag within 0.3 deg
    mask_core = ang2 <= 0.3
    if not np.any(mask_core):
        raise SystemExit("No core galaxies found within 0.3 deg; widen box or check data")
    core = sel[mask_core]
    idx_bcg = int(np.argmin([r['rmag'] for r in core]))
    bcg = core[idx_bcg]

    # LOS velocity offsets relative to BCG
    z_bcg = float(bcg['z'])
    vfac = c_kms / (1.0 + z_bcg)
    dv = vfac * (np.array([r['z'] for r in sel]) - z_bcg)

    # Membership cut: |Δv| <= 3000 km/s
    mem = sel[np.abs(dv) <= 3000.0]
    ang_mem = ang2[np.abs(dv) <= 3000.0]
    dv_mem = dv[np.abs(dv) <= 3000.0]

    return bcg, mem, ang_mem, dv_mem


def bin_stats(ang_deg, dv_kms, bins_mpc):
    # Convert angular sep to projected Mpc at cluster redshift
    theta_rad = np.radians(ang_deg)
    R_proj_Mpc = D_A_Mpc * theta_rad
    edges = np.asarray(bins_mpc, dtype=float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    means = []
    counts = []
    for i in range(len(edges) - 1):
        m = (R_proj_Mpc >= edges[i]) & (R_proj_Mpc < edges[i+1])
        if np.any(m):
            means.append(float(np.nanmean(dv_kms[m])))
            counts.append(int(np.sum(m)))
        else:
            means.append(float('nan'))
            counts.append(0)
    return mids, np.array(means), np.array(counts)


def predict_grav_dv(bins_mpc, M_Msun=1.0e15, a_kpc=300.0, ell0_kpc=200.0, p=2.0, ncoh=2.0):
    # Build Hernquist+Σ geff and compute Δv_pred(R) = c * [Ψ(BCG) - Ψ(R)]/c^2
    # Use tiny epsilon for BCG position to avoid r==0 early-exit in integrator
    Msun = 1.98840987e30
    kpc = 3.085677581e19
    Mpc = 1.0e3 * kpc

    geff = geff_hernquist_factory(M_Msun*Msun, a_kpc*kpc, ell0_kpc*kpc, p, ncoh, kernel_metric='spherical')

    eps = 1.0  # 1 meter offset
    x_obs = np.array([eps, 0.0, 0.0])
    dv_pred = []
    for R in 0.5 * (bins_mpc[:-1] + bins_mpc[1:]):
        x_emit = np.array([R*Mpc, 0.0, 0.0])
        z_rel = gravitational_redshift_endpoint(x_emit, x_obs, geff, r_max=20.0*Mpc, n_steps=4000)
        dv_pred.append(z_rel * (c / 1000.0))  # km/s
    return np.array(dv_pred)


def main():
    ensure_sdss_csv()
    rows = load_sdss_csv(SDSS_CSV)
    if not rows:
        raise SystemExit("Failed to load SDSS CSV for Coma region")
    bcg, members, ang_deg, dv = select_members(rows)

    # Bins in projected Mpc
    bins = np.array([0.0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0])
    mids, dv_mean, counts = bin_stats(ang_deg, dv, bins)

    # Toy prediction
    dv_pred = predict_grav_dv(bins)

    out_csv = OUTDIR / 'coma_redshift_comparison.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["R_mid_Mpc", "obs_mean_dv_kms", "N", "pred_dv_kms"])
        for Rm, m, n, p in zip(mids, dv_mean, counts, dv_pred):
            w.writerow([f"{Rm:.4f}", f"{m:.4f}" if math.isfinite(m) else "", int(n), f"{p:.4f}"])

    meta = dict(
        center=dict(ra=RA0, dec=DEC0, z=Z_CLUSTER),
        H0=H0, D_A_Mpc=D_A_Mpc,
        sdss_csv=str(SDSS_CSV),
        bins=list(bins.astype(float)),
        bcg=dict(ra=bcg['ra'], dec=bcg['dec'], z=float(bcg['z']), rmag=float(bcg['rmag'])),
        counts_total=int(len(members)),
    )
    (OUTDIR / 'coma_redshift_comparison_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f"Wrote {out_csv}")


if __name__ == '__main__':
    main()