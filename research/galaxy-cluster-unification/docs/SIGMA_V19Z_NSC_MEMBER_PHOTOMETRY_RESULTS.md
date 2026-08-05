# Sigma V19Z NSC member-photometry results

## Result

V19Z passed its acquisition gates.  All 141 frozen NSC DR2 cone queries
returned HTTP 200 with the exact requested schema.  The runner retained and
hashed every raw CSV, exact ADQL statement and exact URL.  It also recovered
all 78 Bullet paper rows and the preregistered 72 complete Bessel `B/R/I`
measurements.  No counterpart, quality cut, filter transformation, stellar
mass, current map, lensing target or gravity setting was opened.

Unlike the HSC-only result, the combined source inventory is now adequate to
design a probabilistic association protocol for both clusters.

## Candidate coverage

| Diagnostic | Bullet Cluster | Abell 2146 |
|---|---:|---:|
| Published members | 78 | 63 |
| NSC cones with candidates | 77 | 59 |
| NSC cones without candidates | 1 | 4 |
| NSC candidate rows | 185 | 59 |
| Exactly one candidate | 18 | 59 |
| Multiple candidates | 59 | 0 |
| Median candidates/member | 2 | 1 |
| Maximum candidates/cone | 6 | 1 |
| Candidates with all `g/r/i/z` bands | 136 (73.5%) | 0 |
| Exactly-one cones with all `g/r/i/z` | 18 | 0 |
| Members with published Bessel `B/R/I` and at least one NSC candidate | 72 | not applicable |
| Published-`B/R/I` members with exactly one all-`g/r/i/z` candidate | 15 | not applicable |

For Bullet, NSC reduces the raw ambiguity from 793 HSC candidates to 185 and
extends coverage from 58 to 77 of 78 members.  All 72 members having published
Bessel `B/R/I` also have at least one NSC candidate.  The remaining task is a
real, constrained association problem rather than a missing-data search.

For Abell 2146, every nonempty NSC cone is unique.  NSC and HSC both cover 58
members; NSC covers one additional member.  Four members (61, 76, 109 and 116)
are absent from both summaries.  NSC's Abell imaging lacks `i` and has only
`z`, or `g/r/z`, so HST's three ACS bands remain the better mass input; NSC is
valuable as independent precise astrometry.  Bullet member 03 is the only
member absent from both HSC and NSC, and it also lacks the published Bessel
photometry.

## Why no match has been made yet

An NSC candidate is not automatically the spectroscopic galaxy.  The Bullet
paper coordinate may be displaced by several arcseconds solely because its
right ascension was rounded to a whole time-second.  A nearest-neighbor rule
would ignore local background density, measurement errors, source morphology,
proper-motion evidence, the published colors and overlapping cones.

The later association should be a global posterior over assignments and null
states.  Its evidence can include:

- paper-coordinate quantization convolved with NSC astrometric uncertainty;
- NSC source density estimated from preregistered annuli;
- galaxy/star morphology and proper-motion likelihoods, not hard retrospective
  cuts;
- a forward model from a redshifted galaxy SED through Bessel and NSC
  passbands, with calibration/model uncertainty;
- one-to-one constraints when neighboring member cones share an NSC object;
  and
- an explicit null-match probability and posterior ambiguity propagation.

The 15 Bullet cones having one candidate, published Bessel `B/R/I`, and full
NSC `g/r/i/z` are a useful validation subset, but they must not be assumed
correct merely because they are singletons or used to tune a transformation
until it agrees.

## Physical consequence

The project can now proceed toward the projected stellar mass current

\[
j_{\rm los}^{(M)}(x)=\rho_\star(x)u_{\rm los}(x),
\]

with association and mass uncertainty, instead of the number-current proxy
`n(x)u_los(x)`.  This is necessary for three distinct calculations:

1. the ordinary GR gravitomagnetic/frame-dragging control;
2. a universal extra susceptibility to baryonic current, if preregistered; and
3. the directional long-wave metric mode whose wavelength is longer than a
   stellar system but whose galaxy/cluster phase and tensor polarization are
   sourced by baryonic stress-energy.

V19Z supports the source reconstruction only.  It is not evidence that any of
those gravitational effects has the required amplitude or topology.

## Next frozen stage

V19AA should freeze the association and mass-inference model before producing
member matches.  Its concrete gates should include posterior predictive tests
on coordinate offsets and colors, leave-one-out behavior on the 15 Bullet
singleton/full-color cases, catalog-to-catalog consistency for the 58 Abell
HSC/NSC overlaps, explicit null states for the five members absent from both
sources, and sensitivity to the unreported Bullet photometric uncertainties.

Only after those gates pass should the project build mass-current map draws and
score the GR/current/long-wave responses.  Lensing targets remain sealed.

Machine-readable evidence is in
`results/sigma_v19z_nsc_member_photometry/provenance.json` and
`coverage_analysis.json`; the extracted paper values are in
`data/derived/sigma_v19z_member_photometry/bullet_published_bri.csv`.
