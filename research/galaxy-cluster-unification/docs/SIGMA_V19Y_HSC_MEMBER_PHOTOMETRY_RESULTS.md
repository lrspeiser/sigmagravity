# Sigma V19Y HSC member-photometry results

## Result

The acquisition gate passed: all 141 frozen member cones returned HTTP 200,
all raw CSV responses and exact query URLs were retained and hashed, and no
counterpart, quality cut, stellar mass, current map, lensing target or gravity
parameter was opened.

The scientific source-sufficiency result is mixed.  HSC summary photometry is
close to sufficient for Abell 2146 but is not sufficient by itself for the
Bullet Cluster.  We therefore must not construct a nominally homogeneous
two-cluster mass-current map by silently treating nearest HSC objects as the
Bullet members.

## Candidate landscape

| Frozen diagnostic | Bullet Cluster | Abell 2146 |
|---|---:|---:|
| Published spectroscopic members | 78 | 63 |
| Cones with at least one HSC candidate | 58 (74.4%) | 58 (92.1%) |
| Cones with no candidate | 20 | 5 |
| Total candidates | 793 | 62 |
| Cones with exactly one candidate | 0 | 54 |
| Cones with multiple candidates | 58 | 4 |
| Median candidates per member, including empty cones | 8.5 | 1.0 |
| Maximum candidates in one cone | 53 | 2 |
| Candidates with all F435W/F606W/F814W measurements | 46 (5.8%) | 60 (96.8%) |
| Members with exactly one candidate having all three bands | 0 | 54 |
| Median nearest-candidate separation in a nonempty cone | 1.14 arcsec | 0.72 arcsec |

The two catalogs behave differently because their input astrometry differs.
Abell 2146 positions were published to 0.0001 degree and could use a 1-arcsec
cone.  Bullet right ascensions were published only to a whole time-second;
at its declination, half of that quantization interval is about 4.2 arcsec, so
the preregistered 6-arcsec cone is necessary.  Its high source density makes a
position-only association underdetermined.

The Bullet paper itself contains `B`, `B-R` and `B-I` for 72 of the 78 members.
Those measurements were not carried into V19D because that stage was a
velocity-catalog extraction.  They can become an independent association
constraint, but only after a matching likelihood and filter transformation
are frozen.  They cannot be used by eye to choose among the 793 candidates.

## What the data now support

For Abell 2146, a later probabilistic association can represent the four
two-candidate cones and five null cones explicitly.  Fifty-four members have
one candidate with all three requested bands; 60 of 62 candidate rows have
all three bands, and 60 of 62 have `NumImages > 1`.  This is a credible basis for
stellar-mass inference with a frozen stellar-population model and propagated
association uncertainty.

For the Bullet Cluster, the HSC-only route fails as a full member-mass source:
no nonempty cone is unique, 20 members have no HSC candidate, and only 46 of
793 candidate rows have all three requested ACS bands.  The correct next move
is a source-side acquisition with full-field, precise ground-based astrometry
and multiband photometry, not a wider HSC cone or a nearest-neighbor rule.
DES DR2 and the NOIRLab Source Catalog are candidate public sources; coverage
and schema must be checked without opening member matches and then frozen
before acquisition.

## Consequence for the directional and long-wave physics lanes

The current relativistic source proxy remains

\[
j_{\rm los}^{(N)}(x)=n(x)u_{\rm los}(x),
\]

not the desired mass current

\[
j_{\rm los}^{(M)}(x)=\rho_\star(x)u_{\rm los}(x).
\]

V19Y makes the latter feasible for most Abell 2146 members, but not yet for
the joint cluster pair.  It therefore does not authorize a physical
Lense--Thirring amplitude or a source-generated long-wave metric test.

The retained wave hypothesis has characteristic wavelength
`lambda_Sigma` longer than a stellar system.  If a baryon-sourced tensor mode
varies as `cos(2 pi s/lambda_Sigma + phase)`, a stellar system samples nearly
one phase.  Its constant offset and uniform free-fall gradient are not local
intrinsic observables; the anomalous tidal response can be suppressed roughly
as `(D_star/lambda_Sigma)^2`.  A galaxy or cluster can sample enough of the
mode for its phase, direction and polarization to matter.

This does not rescue the already failed scalar wavelength-only filter.  The
new lane must predict wave amplitude, phase and tensor orientation from the
baryonic density, mass current, stress and overlap, and must place massive
objects and photons on the same metric.  The mass-current acquisition is
therefore a prerequisite rather than evidence for the mode.

## Frozen next-step boundary

Before any member is matched or mass is inferred:

1. extract the 72 available Bullet `B`, `B-R`, `B-I` measurements losslessly;
2. preflight full-field DES DR2 and NOIRLab Source Catalog coverage and exact
   public query schemas without querying member coordinates;
3. freeze one all-candidate acquisition protocol if coverage is adequate;
4. freeze a global probabilistic association with catalog-precision error
   models, a null-match state, one-to-one constraints where appropriate,
   morphology information, and preregistered color transformations;
5. freeze one stellar-population/mass model and propagate photometric,
   association and line-of-sight velocity uncertainty; and
6. score ordinary GR frame dragging before any extra current susceptibility or
   long-wave coupling is fitted.

The acquisition report and descriptive coverage analysis are
`results/sigma_v19y_hsc_member_photometry/provenance.json` and
`coverage_analysis.json`.  The latter reports aggregate separations and band
coverage but records no selected candidate identifiers.
