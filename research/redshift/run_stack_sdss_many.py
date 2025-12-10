#!/usr/bin/env python3
"""
Stacked gravitational redshift over many SDSS clusters (hundreds) using public SQL.

Strategy (to avoid stale catalog URLs):
- Query SDSS DR18 for cluster-like groups via redshift + richness proxies in SpecObjAll/PhotoObj
- Build on-the-fly clusters from SDSS group finders or target known Abell/MaxBCG catalogs via CASJobs
- For simplicity: use SDSS DR18 CASJobs to pull pre-identified cluster centers + members

This demo pulls a list of well-known Abell clusters with SDSS coverage, fetches members per cluster via SQL,
applies robust stacking (biweight + shifting-gapper + bootstrap), and compares to Σ endpoint predictions.

Outputs: redshift/outputs/stack_vs_sigma_many.csv
"""
from __future__ import annotations
import os
import math
import csv
import json
from pathlib import Path
import numpy as np

from .stack_grz import per_cluster_profile, stack_profiles, stack_predictions
from .geff_adapters import geff_hernquist_factory

ROOT = Path(__file__).resolve().parents[1]
DATADIR = ROOT / 'data' / 'redshift'
OUTDIR = ROOT / 'redshift' / 'outputs'
OUTDIR.mkdir(parents=True, exist_ok=True)

H0 = 70.0; c_kms = 299_792.458

# Well-known Abell/ACO clusters with SDSS coverage (sample of ~50 for demonstration)
# (id, RA, DEC, z, R200_Mpc_estimate)
CLUSTERS_MANY = [
    dict(id='A119',  RA=14.070,  DEC=-1.252,  z_BCG=0.0440, R200=1.5),
    dict(id='A147',  RA=16.528,  DEC=2.180,   z_BCG=0.0440, R200=1.5),
    dict(id='A168',  RA=18.923,  DEC=0.285,   z_BCG=0.0450, R200=1.4),
    dict(id='A262',  RA=28.193,  DEC=36.153,  z_BCG=0.0163, R200=1.2),
    dict(id='A347',  RA=37.104,  DEC=41.842,  z_BCG=0.0197, R200=1.3),
    dict(id='A376',  RA=42.070,  DEC=36.877,  z_BCG=0.0484, R200=1.6),
    dict(id='A400',  RA=44.445,  DEC=6.032,   z_BCG=0.0240, R200=1.3),
    dict(id='A496',  RA=68.406,  DEC=-13.261, z_BCG=0.0330, R200=1.5),
    dict(id='A548',  RA=83.717,  DEC=-25.502, z_BCG=0.0420, R200=1.4),
    dict(id='A576',  RA=109.238, DEC=55.753,  z_BCG=0.0390, R200=1.4),
    dict(id='A634',  RA=125.187, DEC=58.157,  z_BCG=0.0312, R200=1.4),
    dict(id='A671',  RA=127.585, DEC=30.390,  z_BCG=0.0503, R200=1.5),
    dict(id='A754',  RA=137.279, DEC=-9.628,  z_BCG=0.0542, R200=1.7),
    dict(id='A779',  RA=139.406, DEC=33.754,  z_BCG=0.0223, R200=1.3),
    dict(id='A1016', RA=157.108, DEC=10.998,  z_BCG=0.0344, R200=1.4),
    dict(id='A1177', RA=165.232, DEC=21.664,  z_BCG=0.0318, R200=1.4),
    dict(id='A1185', RA=168.679, DEC=28.691,  z_BCG=0.0330, R200=1.4),
    dict(id='A1314', RA=174.429, DEC=49.041,  z_BCG=0.0340, R200=1.4),
    dict(id='A1367', RA=176.167, DEC=19.834,  z_BCG=0.0216, R200=1.3),
    dict(id='A1656', RA=194.953, DEC=27.980,  z_BCG=0.0231, R200=1.8),  # Coma
    dict(id='A1750', RA=203.042, DEC=-1.734,  z_BCG=0.0860, R200=1.9),
    dict(id='A1795', RA=207.219, DEC=26.595,  z_BCG=0.0620, R200=1.7),
    dict(id='A2029', RA=227.733, DEC=5.743,   z_BCG=0.0773, R200=1.8),
    dict(id='A2052', RA=229.187, DEC=7.021,   z_BCG=0.0348, R200=1.4),
    dict(id='A2063', RA=230.770, DEC=8.607,   z_BCG=0.0354, R200=1.4),
    dict(id='A2065', RA=230.618, DEC=27.703,  z_BCG=0.0726, R200=1.8),
    dict(id='A2142', RA=239.586, DEC=27.233,  z_BCG=0.0909, R200=1.9),
    dict(id='A2147', RA=240.569, DEC=15.970,  z_BCG=0.0351, R200=1.4),
    dict(id='A2152', RA=241.050, DEC=16.432,  z_BCG=0.0410, R200=1.5),
    dict(id='A2197', RA=246.992, DEC=40.927,  z_BCG=0.0308, R200=1.4),
    dict(id='A2199', RA=247.159, DEC=39.551,  z_BCG=0.0302, R200=1.5),
    dict(id='A2255', RA=258.127, DEC=64.054,  z_BCG=0.0809, R200=1.8),
    dict(id='A2256', RA=255.944, DEC=78.649,  z_BCG=0.0581, R200=1.7),
    dict(id='A2589', RA=351.250, DEC=16.777,  z_BCG=0.0416, R200=1.5),
    dict(id='A2597', RA=351.335, DEC=-12.119, z_BCG=0.0852, R200=1.8),
    dict(id='A2634', RA=354.611, DEC=27.021,  z_BCG=0.0312, R200=1.4),
    dict(id='A2657', RA=356.353, DEC=9.196,   z_BCG=0.0404, R200=1.5),
    dict(id='A2670', RA=358.560, DEC=-10.413, z_BCG=0.0763, R200=1.8),
    dict(id='A3112', RA=49.167,  DEC=-44.233, z_BCG=0.0750, R200=1.8),
    dict(id='A3158', RA=55.727,  DEC=-53.631, z_BCG=0.0590, R200=1.7),
    dict(id='A3266', RA=67.863,  DEC=-61.454, z_BCG=0.0590, R200=1.7),
    dict(id='A3376', RA=90.488,  DEC=-39.973, z_BCG=0.0456, R200=1.5),
    dict(id='A3391', RA=96.591,  DEC=-53.693, z_BCG=0.0531, R200=1.6),
    dict(id='A3526', RA=192.199, DEC=-41.311, z_BCG=0.0103, R200=1.1),
    dict(id='A3558', RA=202.016, DEC=-31.500, z_BCG=0.0480, R200=1.6),
    dict(id='A3571', RA=206.854, DEC=-32.862, z_BCG=0.0391, R200=1.5),
    dict(id='A3667', RA=303.134, DEC=-56.827, z_BCG=0.0556, R200=1.7),
    dict(id='A4038', RA=354.563, DEC=-28.144, z_BCG=0.0283, R200=1.3),
]


def D_A_Mpc(z: float) -> float:
    return (c_kms / H0) * z / (1.0 + z)


def haversine_deg(ra1, dec1, ra2, dec2):
    ra1 = math.radians(ra1); dec1 = math.radians(dec1)
    ra2 = math.radians(ra2); dec2 = math.radians(dec2)
    d_ra = ra2 - ra1; d_dec = dec2 - dec1
    a = math.sin(d_dec/2.0)**2 + math.cos(dec1)*math.cos(dec2)*math.sin(d_ra/2.0)**2
    return 2.0 * math.asin(min(1.0, math.sqrt(a)))


def fetch_sdss_box(ra0, dec0, zmin, zmax, name, box_deg=3.0):
    ra_min = ra0 - box_deg; ra_max = ra0 + box_deg
    dec_min = dec0 - box_deg; dec_max = dec0 + box_deg
    sql = (
        "SELECT%20TOP%2050000%20"
        "s.ra,%20s.dec,%20s.z,%20s.zErr,%20s.class,%20p.modelMag_r%20"
        "FROM%20SpecObjAll%20AS%20s%20JOIN%20PhotoObj%20AS%20p%20ON%20s.bestObjID%20=%20p.objID%20"
        f"WHERE%20(s.class=%27GALAXY%27)%20AND%20s.ra%20BETWEEN%20{ra_min}%20AND%20{ra_max}%20"
        f"AND%20s.dec%20BETWEEN%20{dec_min}%20AND%20{dec_max}%20"
        f"AND%20s.z%20BETWEEN%20{zmin}%20AND%20{zmax}"
    )
    url = f"https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd={sql}&format=csv"
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
    box = fetch_sdss_box(ra0, dec0, max(0.0, zc-0.06), zc+0.06, cl['id'], box_deg=2.5)
    rows = load_rows(box)
    if not rows:
        return np.zeros(0, dtype=[('Rproj_Mpc','f8'), ('z_spec','f8')])
    thetas = np.array([haversine_deg(r['ra'], r['dec'], ra0, dec0) for r in rows])
    Rproj = D_A * thetas
    zs = np.array([r['z'] for r in rows])
    out = np.empty(len(rows), dtype=[('Rproj_Mpc','f8'), ('z_spec','f8')])
    out['Rproj_Mpc'] = Rproj
    out['z_spec'] = zs
    return out


def main():
    DATADIR.mkdir(parents=True, exist_ok=True)

    # Limit to first 40 for demo (full run: use all ~48)
    use_list = CLUSTERS_MANY[:40]

    members_by_id = {}
    use_clusters = []
    print(f"Fetching members for {len(use_list)} clusters...")
    for i, cl in enumerate(use_list):
        if i % 10 == 0:
            print(f"  {i}/{len(use_list)}")
        mem = build_members_for_cluster(cl)
        if mem.size == 0:
            continue
        members_by_id[cl['id']] = mem
        use_clusters.append(cl)

    if len(use_clusters) < 10:
        raise SystemExit(f"Only {len(use_clusters)} clusters with members; need more for robust stack.")

    print(f"Stacking {len(use_clusters)} clusters...")
    rbins = np.arange(0.0, 2.01, 0.20)

    obs_profiles = []
    for cl in use_clusters:
        prof = per_cluster_profile(cl, members_by_id[cl['id']], rbins=rbins)
        obs_profiles.append(prof)

    rmid, dv_obs, dv_err = stack_profiles(obs_profiles, n_boot=500, seed=2)

    # Σ endpoint predictions (toy Hernquist)
    Msun = 1.98840987e30; kpc = 3.085677581e19
    geff = geff_hernquist_factory(M=1.0e15*Msun, a=300.0*kpc, ell0=200.0*kpc, p=2.0, ncoh=2.0, A=4.6, kernel_metric='spherical')
    _, dv_pred = stack_predictions(use_clusters, rbins, geff)

    out = OUTDIR / 'stack_vs_sigma_many.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['r_over_R200','delta_v_obs_kms','delta_v_err_kms','delta_v_pred_kms','N_clusters'])
        for R, o, e, p in zip(rmid, dv_obs, dv_err, dv_pred):
            w.writerow([f"{R:.3f}", f"{o:.3f}" if math.isfinite(o) else '', f"{e:.3f}" if math.isfinite(e) else '', f"{p:.3f}", len(use_clusters)])

    meta = dict(N_clusters=len(use_clusters), cluster_ids=[c['id'] for c in use_clusters], bins=list(rbins.astype(float)))
    (OUTDIR / 'stack_vs_sigma_many_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f"Wrote {out}")
    print(f"Stacked {len(use_clusters)} clusters; check stack_vs_sigma_many.csv for comparison.")


if __name__ == '__main__':
    main()
