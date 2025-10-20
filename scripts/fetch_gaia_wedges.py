#!/usr/bin/env python3
"""
Fetch targeted Gaia DR3 wedges (inner disk and anticenter) via astroquery
and write CSVs under data/gaia/new/.

Regions supported (via --region):
- inner_rvs       : |l|<30 deg, |b|<10 deg, radial_velocity not null
- inner_pm        : same sky cuts, no RV requirement
- anticenter_rvs  : |l-180|<30 deg, |b|<10 deg, RV not null
- anticenter_pm   : same sky cuts, no RV requirement

Quality cuts (modifiable with flags): ruwe<1.6, visibility_periods_used>=8.
We select TOP N rows (default 200000) with ORDER BY random_index to sample.

Dependencies: astroquery>=0.4, astropy; network access to Gaia TAP.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from astroquery.gaia import Gaia

REGIONS = {
    # Prefer native galactic columns l,b in gaiadr3.gaia_source to avoid heavy TRANSFORM calls
    "inner": "(g.l > 330 OR g.l < 30) AND ABS(g.b) < 10",
    "anticenter": "ABS(g.l - 180) < 30 AND ABS(g.b) < 10",
}

BASE_SELECT = """
SELECT TOP {top_n}
  g.source_id, g.ra, g.dec,
  g.l AS l,
  g.b AS b,
  g.parallax, g.parallax_error, g.pmra, g.pmdec,
  g.radial_velocity, g.radial_velocity_error,
  g.phot_g_mean_mag, g.bp_rp,
  g.ruwe, g.visibility_periods_used,
  ap.teff_gspphot, ap.ag_gspphot, ap.ebpminrp_gspphot
FROM gaiadr3.gaia_source AS g
LEFT JOIN gaiadr3.astrophysical_parameters AS ap USING (source_id)
WHERE
  ({sky_where})
  AND g.ruwe < {ruwe_max}
  AND g.visibility_periods_used >= {vis_min}
  {rv_clause}
ORDER BY g.random_index
""".strip()


def build_query(region: str, kind: str, top_n: int, ruwe_max: float, vis_min: int) -> str:
    sky = REGIONS['inner'] if region == 'inner' else REGIONS['anticenter']
    rv_clause = "AND g.radial_velocity IS NOT NULL" if kind == 'rvs' else ""
    return BASE_SELECT.format(top_n=top_n, sky_where=sky, ruwe_max=ruwe_max, vis_min=vis_min, rv_clause=rv_clause)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', choices=['inner','anticenter'], required=True)
    ap.add_argument('--kind', choices=['rvs','pm'], required=True)
    ap.add_argument('--top-n', type=int, default=200000)
    ap.add_argument('--ruwe-max', type=float, default=1.6)
    ap.add_argument('--vis-min', type=int, default=8)
    ap.add_argument('--outdir', default=str(Path('data')/'gaia'/'new'))
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    adql = build_query(args.region, args.kind, args.top_n, args.ruwe_max, args.vis_min)
    print('Submitting ADQL to Gaia TAP...')
    print(adql)
    job = Gaia.launch_job_async(adql, dump_to_file=False, output_format='csv')
    tbl = job.get_results()

    tag = f"{args.region}_{args.kind}_dr3_top{args.top_n}"
    out = Path(args.outdir) / f"gaia_{tag}.csv"
    tbl.write(out.as_posix(), format='csv', overwrite=True)
    print(f'Wrote {out} (rows={len(tbl)})')


if __name__ == '__main__':
    main()
