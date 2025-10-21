#!/usr/bin/env python3
# Minimal SDSS-based stacked gRZ over a few low-z clusters (research demo)
# Writes redshift/outputs/stack_vs_sigma_small.csv

from __future__ import annotations
import os
import math
import csv
from pathlib import Path
import numpy as np

from .stack_grz import per_cluster_profile, stack_profiles, stack_predictions
from .geff_adapters import geff_hernquist_factory

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / 'data' / 'redshift'
OUTDIR = ROOT / 'redshift' / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Small cluster list: (id, RA, DEC, z, R200[Mpc])
CLUSTERS = [
    dict(id='Coma',  RA=194.953, DEC=27.980, z_BCG=0.0231, R200=1.8),
    dict(id='A2199', RA=247.159, DEC=39.551, z_BCG=0.0302, R200=1.5),
    dict(id='A1795', RA=207.219, DEC=26.595, z_BCG=0.0620, R200=1.5),
    dict(id='A2142', RA=239.586, DEC=27.233, z_BCG=0.0909, R200=1.7),
]

# SDSS SkyServer helper (box around each cluster)
BASE_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd={cmd}&format=csv"
BOX_DEG = 2.5

H0 = 70.0  # km/s/Mpc
c_kms = 299_792.458


def D_A_Mpc(z: float) -> float:
    return (c_kms / H0) * z / (1.0 + z)


def haversine_deg(ra1, dec1, ra2, dec2):
    ra1 = math.radians(ra1); dec1 = math.radians(dec1)
    ra2 = math.radians(ra2); dec2 = math.radians(dec2)
    d_ra = ra2 - ra1; d_dec = dec2 - dec1
    a = math.sin(d_dec/2.0)**2 + math.cos(dec1)*math.cos(dec2)*math.sin(d_ra/2.0)**2
    return 2.0 * math.asin(min(1.0, math.sqrt(a)))  # radians


def fetch_sdss_box(ra0, dec0, zmin, zmax, name):
    ra_min = ra0 - BOX_DEG; ra_max = ra0 + BOX_DEG
    dec_min = dec0 - BOX_DEG; dec_max = dec0 + BOX_DEG
    sql = (
        "SELECT%20TOP%2050000%20"
        "s.ra,%20s.dec,%20s.z,%20s.zErr,%20s.class,%20p.modelMag_r%20"
        "FROM%20SpecObjAll%20AS%20s%20JOIN%20PhotoObj%20AS%20p%20ON%20s.bestObjID%20=%20p.objID%20"
        f"WHERE%20(s.class=%27GALAXY%27)%20AND%20s.ra%20BETWEEN%20{ra_min}%20AND%20{ra_max}%20"
        f"AND%20s.dec%20BETWEEN%20{dec_min}%20AND%20{dec_max}%20"
        f"AND%20s.z%20BETWEEN%20{zmin}%20AND%20{zmax}"
    )
    url = BASE_URL.format(cmd=sql)
    out = DATADIR / f"sdss_box_{name}.csv"
    if not out.exists() or out.stat().st_size < 1000:
        os.system(f"curl -s -L \"{url}\" -o \"{out.as_posix()}\"")
    return out


def load_rows(path: Path):
    import csv as _csv
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = _csv.DictReader(line for line in f if not line.startswith('#'))
        for r in reader:
            try:
                rows.append(dict(ra=float(r['ra']), dec=float(r['dec']), z=float(r['z'])))
            except Exception:
                continue
    return rows


def build_members_for_cluster(cl):
    ra0, dec0, zc = cl['RA'], cl['DEC'], cl['z_BCG']
    D_A = D_A_Mpc(zc)
    box = fetch_sdss_box(ra0, dec0, max(0.0, zc-0.05), zc+0.05, cl['id'])
    rows = load_rows(box)
    if not rows:
        return np.zeros(0, dtype=[('Rproj_Mpc', 'f8'), ('z_spec', 'f8')])
    thetas = np.array([haversine_deg(r['ra'], r['dec'], ra0, dec0) for r in rows])  # radians
    Rproj = D_A * thetas  # Mpc
    zs = np.array([r['z'] for r in rows])
    out = np.empty(len(rows), dtype=[('Rproj_Mpc','f8'), ('z_spec','f8')])
    out['Rproj_Mpc'] = Rproj
    out['z_spec'] = zs
    return out


def main():
    DATADIR.mkdir(parents=True, exist_ok=True)

    # Members per cluster
    members_by_id = {}
    use_clusters = []
    for cl in CLUSTERS:
        mem = build_members_for_cluster(cl)
        if mem.size == 0:
            continue
        members_by_id[cl['id']] = mem
        use_clusters.append(cl)

    if not use_clusters:
        raise SystemExit("No clusters with members retrieved; aborting.")

    rbins = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0])

    # Observed robust profiles per cluster
    obs_profiles = []
    for cl in use_clusters:
        prof = per_cluster_profile(cl, members_by_id[cl['id']], rbins=rbins)
        obs_profiles.append(prof)

    rmid, dv_obs, dv_err = stack_profiles(obs_profiles, n_boot=300, seed=1)

    # Σ prediction via toy Hernquist geff (cluster-scale parameters)
    Msun = 1.98840987e30; kpc = 3.085677581e19
    geff = geff_hernquist_factory(M=1.0e15*Msun, a=300.0*kpc, ell0=200.0*kpc, p=2.0, ncoh=2.0, kernel_metric='spherical')
    _, dv_pred = stack_predictions(use_clusters, rbins, geff)

    out = OUTDIR / 'stack_vs_sigma_small.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['r_over_R200','delta_v_obs_kms','delta_v_err_kms','delta_v_pred_kms'])
        for R, o, e, p in zip(rmid, dv_obs, dv_err, dv_pred):
            w.writerow([f"{R:.3f}", f"{o:.3f}" if math.isfinite(o) else '', f"{e:.3f}" if math.isfinite(e) else '', f"{p:.3f}"])

    print(f"Wrote {out}")


if __name__ == '__main__':
    main()