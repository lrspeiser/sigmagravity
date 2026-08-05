# Sigma V19CA SkyMapper/Gaia foreground diagnostics

## Purpose

V19BZ showed that H I overlap cannot safely choose among the 17,094 SkyMapper
candidate occurrences. V19CA adds independent foreground evidence using the
exact Gaia DR3 links already published in the SkyMapper DR4 database. This is
smaller and more policy-compliant than bulk-harvesting optical image cutouts.

The SkyMapper DR4 table browser exposes nearest and second-nearest Gaia DR3
identifiers and distances, and the public `ext.gaia_dr3` table supplies the
associated astrometry. The query projects no Gaia radial velocity. Official
access documentation is at <https://skymapper.anu.edu.au/how-to-access/> and
the table browser is at
<https://skymapper.anu.edu.au/table-browser-standalone/>.

## Frozen diagnostic

Every one of the 17,034 unique SkyMapper object IDs is queried exactly once by
an ID-batched left join. A candidate receives foreground-astrometry evidence
only when the nearest Gaia source lies within one arcsecond and has at least
one of:

- positive parallax at five sigma;
- absolute RA proper motion at five sigma; or
- absolute declination proper motion at five sigma.

A stricter quality-controlled flag additionally requires `RUWE <= 1.4` and a
five- or six-parameter Gaia solution. Both flags remain diagnostics. A moving
star can overlap a background galaxy, so V19CA applies no hard mask, weight,
candidate removal or counterpart selection.

## Honesty and target boundary

The TAP schema and three actual candidate rows were inspected before this
contract was written; that pilot is disclosed. The complete candidate
population has not been queried. The thresholds were fixed from conservative
astrometric significance and match-quality rules, not from the pilot outcome.

No WALLABY kinematic row, galaxy velocity, lensing result, halo map, gravity
residual, action, constant, evidence split or Solar-System calculation enters
this acquisition.

## Reproduction after the contract commit

```powershell
python scripts/acquire_sigma_v19ca_skymapper_gaia_foreground_diagnostics.py
python -m pytest tests/test_sigma_v19ca_skymapper_gaia_foreground_diagnostics.py -q
```
