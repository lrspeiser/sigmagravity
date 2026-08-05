# Sigma V19Y HSC member-photometry acquisition plan

## Outcome

V19Y determines whether the 141 published spectroscopic members in the
Bullet Cluster and Abell 2146 have enough common HST photometry to support a
later physical member mass-current reconstruction.  It is an acquisition
gate, not a counterpart-matching or gravity test.

The need is concrete.  V19H and V19I measured member number-current geometry,

\[
j_{\rm los}^{(N)}(x)=n(x)u_{\rm los}(x),
\]

but frame dragging and any baryon-sourced directional long-wave metric mode
require a mass-weighted current.  The next physical proxy would be closer to

\[
j_{\rm los}^{(M)}(x)=\rho_\star(x)u_{\rm los}(x),
\]

with uncertainty from the photometry-to-mass conversion and the unobserved
transverse velocities.  V19Y only acquires the source-side photometry needed
to decide whether that next step is defensible.

## Frozen acquisition

- Query the documented HSC v3 `summary/magauto` CSV endpoint.
- Query all 78 Bullet and all 63 Abell 2146 member coordinates.
- Use a 6-arcsec Bullet cone because the published right ascensions are
  quantized to a whole time-second; use a 1-arcsec Abell 2146 cone because its
  coordinates are reported to 0.0001 degree.
- Request common ACS F435W, F606W and F814W photometry and its repeat-count and
  dispersion metadata, plus catalog astrometry and quality metadata.
- Preserve one raw CSV and one exact URL file per member and hash both.
- Retain all candidates, including ambiguous cones, and record zero-candidate
  cones without substitution.

No HSC candidate is selected in V19Y.  No quality threshold, stellar
population model, mass-to-light ratio, k-correction, current map, gravity
formula, or lensing target is opened.

## Why selection is deferred

The Bullet coordinate precision permits several HSC sources inside one frozen
cone.  Choosing the nearest entry after inspecting colors or downstream field
geometry would create an unreported degree of freedom.  V19Y first measures
the candidate landscape.  A separate V19Z protocol must then freeze a
positional-plus-photometric association likelihood, null-match handling,
quality requirements, stellar-mass model, and uncertainty propagation before
constructing any `rho v` map.

## Decision after acquisition

V19Y authorizes a V19Z design only when all 141 frozen queries return HTTP 200,
all nonempty responses have the exact requested schema, every candidate has a
finite sky position, and every raw payload and URL is hashed.  Candidate
coverage is reported, not used as a pass/fail threshold at this acquisition
stage.

If the common three-band coverage is too sparse or ambiguity is too high, the
honest conclusion is that HSC summary photometry alone cannot support the
joint physical current map.  The next source would then need to be frozen
before acquisition (for example, calibrated HST mosaics/catalogs processed
homogeneously), rather than loosening V19Y after seeing the data.

## Relationship to the long-wavelength gravity idea

The newly retained hypothesis is not that gravity receives an arbitrary boost
outside a star system.  It is that a baryon-sourced metric mode may have a
universal wavelength `lambda_Sigma` much longer than a stellar system.  A
local system of diameter `D` then samples nearly one phase, leaving the first
intrinsic tidal anomaly suppressed approximately as
`(D/lambda_Sigma)^2`; galaxies and clusters can sample meaningful phase and
tensor-polarization structure.

The already failed linear isotropic wavelength filter remains retired.  A new
version advances only if its phase, direction, amplitude and polarization are
predicted from baryonic stress-energy—especially mass current, stress,
overlap or tidal orientation—and both matter and photons follow the same
metric.  V19Y supplies one missing source input for such a test; it does not
itself validate the wave hypothesis.

## Reproduction after the freeze commit

```powershell
python scripts/download_sigma_v19y_hsc_member_photometry.py
python -m pytest -q tests/test_sigma_v19y_hsc_member_photometry.py
```
