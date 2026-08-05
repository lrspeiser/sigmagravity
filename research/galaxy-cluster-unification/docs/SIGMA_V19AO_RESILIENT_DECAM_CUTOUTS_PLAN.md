# Sigma V19AO resilient DECam cutout plan

## Purpose

V19AN failed because 37 frozen groups referenced 22 stale NSC `*_d2`
filenames.  V19AO preserves every group and every measurement membership while
replacing only those transport identities with authoritative NOIRLab Archive
file/HDU subsets.

## Frozen source-routing rule

Source routing is determined by the already-frozen exposure identifier, never
by pixel quality:

- 102 groups whose exposure does not end in `_d2` retain the exact V19AN NSC
  SIA grouped-cutout URL.
- All 37 `_d2` groups use the public NOIRLab Archive `vohdu` metadata service.
  The query uses the lexicographically smallest frozen NSC object ID in the
  group and the stale filename prefix with only the terminal `d2` token
  removed.

Each archive query must return exactly one DECam `InstCal` image with the same
filter.  The retrieval URL must request exactly the primary header and the
spatially matched HDU as `?hdus=0,<hdu_idx>`.

The 37 raw identity responses are preserved and hashed.  The complete hybrid
plan is frozen at SHA-256
`52c254325ea1af47b7265f418cd878feb82cb9a50e95c79771060faa04e18f89`
before any V19AO science pixel is downloaded.

## Identity result before retrieval

| Quantity | Frozen result |
|---|---:|
| Total groups | 139 |
| Measurement memberships | 1,032 |
| NSC SIA grouped cutouts | 102 |
| Archive selected-HDU subsets | 37 |
| Unique current archive files | 22 |
| Unique archive file/HDU pairs | 37 |
| Missing or ambiguous identities | 0 |

The observed old-to-current indices are consistently offset by one in this
sample.  That is an outcome, not a rule: V19AO records the sky-matched HDU from
every archive response and performs no arithmetic extension conversion.

## Retrieval gates

All 139 frozen URLs must return HTTP 200, open as valid FITS, contain a
two-dimensional image and celestial WCS, contain every assigned frozen anchor,
and have at least one finite pixel.  Every file and the completed manifest are
hashed.

V19AO may not inspect fluxes, morphology, backgrounds, PSFs, deblending or
validation colors.  Passing authorizes only a later, separately frozen
development-only photometry protocol.
