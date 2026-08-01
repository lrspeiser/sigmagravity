# R1 execution targets

Status: R1A observable audit complete; R1A.1 failed the structural three-radial-
mode gate; R1A.2 has met its 2-system structural-promotion threshold while its
30-host inventory is complete at 45 hosts. RX J2129 strict-readiness preparation
is active. A full Jacobian, R1 freeze, and R2 fitting are not authorized.

## Outcome first

The first immediate outcome—a machine-readable audit of at least 30 unique BCG
hosts or a source-by-source hard public-data shortfall—is complete at 45 hosts.
The second, turning RX J2129 into the first strict R1-ready system, remains active.
The later outcome remains a
residual-blind freeze of at least ten clusters on which the dynamical potential
and Weyl potential are separately identifiable over the same radii. The current
baseline is 45 source-screened hosts, 2 non-disturbed structural promotions
(MACS J1206 and RX J2129), and 0 strict R1-ready systems. Host-count expansion is
closed; observable likelihood and covariance acquisition remains due under R1A.2.

The exact machine-readable milestones are in
`configs/r1_execution_targets.json`.

## R1A: pilot integrity

Before bulk acquisition, verify the BCG coordinate for each of the three RELICS
systems and recenter its MCMC maps. Locate the multiple-image/source-redshift
inputs and Lenstool configuration needed to rerun each model, or explicitly mark
the map ensemble as a standard-GR derived product. Attach a stated uncertainty or
posterior to the stellar M/L normalization.

Advance only when at least one non-disturbed system has a complete per-system
schema, three dynamics bins, and three lensing bins on verified common support.
A2537 remains useful for engineering, but the Newman source calls it the sample's
most likely disturbed cluster, so it cannot be the sole R1A pass.

Completed R1A observable audit: the primary article sources yield 175 published
image-table entries across all ten candidates. After excluding source-marked or
text-identified model predictions, a source-table error, and A383's explicitly
unused image, 166 observable-level positions remain; 106 also have a spectroscopic
source redshift and the declared per-coordinate image-plane error. Newman publishes
BCG-relative positions and a 0.5-arcsec likelihood error (1.0 arcsec for A2390).
Kaleidoscope publishes a 0.5-arcsec error model for its six cluster models. None
publishes a full systematic covariance.

The strict family-count proxy fails 0/10. A383 has two inner images from one source
family; A2537 and MS2137 each have one inner image from one family; all other
systems have none or unauditable centering. The earlier A2537 three-annulus map
count remains a derived-field diagnostic, not three independent observations.

## R1A.1: structural likelihood-rank audit

The family-count failure triggered a structural audit before any full nuisance
model was built. Each strict image inside the dynamics aperture samples at most one
scalar radial response value. Outer counterimages help anchor its unknown source
position but do not add another in-support radius. Therefore the number of strict
inner images is an upper bound on radial rank, and nuisance marginalization can
only reduce it.

The result is 0/10 passes. A383 has a structural radial-rank upper bound of 2;
MS2137 and A2537 each have an upper bound of 1; the other seven have 0. A full
three-node Jacobian cannot overturn those ceilings, so none of the current systems
advances. This is a completed failed gate, not evidence for or against a gravity
theory.

## R1A.2: residual-blind replacement qualification

Screen at least 30 unique BCG hosts, or exhaust the located primary-source samples
and produce a hard-shortfall ledger. Every row must record resolved BCG dynamics,
kinematic radial support, baryonic-profile availability, image positions and source
redshifts, strict inner-image count, structural radial-rank bound, nuisance/covariance
products, provenance, and a precise exclusion reason.

Promote only systems with at least three selected dynamics bins, three strict inner
image positions, radial-rank upper bound at least 3, enough family-wide constraints
to anchor source coordinates, a verified BCG center/common radial support, and
observable-level lens inputs. At least two non-disturbed systems must pass before
strict-readiness preparation begins. A full Jacobian still requires covariance,
complete baryonic inputs, and a rerunnable lens nuisance likelihood. One acquisition
cycle screens ten new hosts or exhausts one named resolved-BCG sample. Three
consecutive unsuccessful cycles or seven active research days triggers a premise/
data-availability rethink; the gate may not be weakened. Exact fields and thresholds are frozen in
`configs/r1_replacement_search_targets.json`.

First-cycle progress: MACS J1206 is the first non-disturbed structural promotion.
Its six published BCG-dynamics annuli reach about 50 kpc; the independent strong-
lens catalog supplies 82 retained spectroscopic image positions, including 11
inside 50 kpc from nine source families. Its radial-rank upper bound is therefore
11, with 46 family-wide positional degrees of freedom after source coordinates.
Abell S1063 has BCG dynamics to 40 kpc but only one strict inner image and fails
with rank bound 1; its source also reports evidence for a recent off-axis merger.
This branch produced the first of the required two promotions. MACS J1206 is not
R1-ready: numerical dispersion values/covariance, numerical baryonic arrays, and
a complete rerunnable lens nuisance posterior remain to be acquired. See
`results/r1_replacement_search_cycle1/report.json`.

The official-data audit resolves the first of those questions without pretending
the problem is solved. Neither the MACS J1206 nor Abell S1063 publication package
contains a numerical BCG dispersion table, measurement covariance, or likelihood.
Both do have public, calibrated level-3 MUSE cubes in the ESO Science Archive. For
MACS J1206 the coadded cube is `ADP.2017-06-19T11:32:26.411` (17,477.713 s); Abell
S1063 has separate SW and NE products. Thus digitizing Figure 2 is limited to an
engineering check. The evidence path is an independent pPXF reconstruction from
the cube, with the object mask, PSF, templates, spectral masks, and bin covariance
frozen before inspecting a gravity residual. Machine-readable targets and status
are in `configs/r1_dynamics_public_data_targets.json` and
`results/r1_dynamics_public_data_audit/report.json`.

The MACS J1206 level-3 cutout is local and FITS-verified: 150 x 150 spatial pixels,
1841 wavelength planes spanning 4859.66-7159.66 Angstrom, with `DATA` and `STAT`
extensions. Its size is 331,548,480 bytes and SHA-256 is
`B2250F1DEDDD7E452697C26497422EA832D94B220183797EE552C7BA3868DF7F`.
The frozen level-3 reconstruction passes the inner five-bin engineering check but
fails the outer half-field gate: its two 8-12 arcsec halves return velocities near
-617 and -52 km/s and dispersions near 639 and 140 km/s.

The follow-up archive-footprint audit selected all six homogeneous central 095.A-
0181(A) level-2 products before looking at their kinematics. Their combined
16,855.219 s exposure was extracted per product on independently registered
elliptical annuli and inverse-variance coadded spectrally; their sub-pixel sky and
wavelength offsets were not image-averaged. This removes the catastrophic velocity
split: the outer halves differ by 16.4 km/s. It does not pass the profile gate.
The 5-8 and 8-12 arcsec opposite-half dispersion differences are 0.428 and 0.327,
above the frozen 0.20 limit; the outer dispersion is 120.8 +/- 82.7 km/s; and a
leave-one-product-out half fit shifts by as much as 0.994 in fractional dispersion.
The outer coadd S/N is only 9.30 per Angstrom. Therefore the public-cube route is
paused, covariance production and gravity fitting are not authorized, and the
failure is recorded rather than averaged away. See
`results/r1_m1206_level2_ppxf/report.json` and
`data/derived/r1_m1206_level2_ppxf_profile.csv`.

Cycle 1 is now complete under its predeclared alternative-success rule. The full
Sand et al. six-cluster resolved-BCG sample has been exhausted: RXJ 1133 has three
numerical BCG bins but only two visually inferred one-dimensional critical radii,
and Abell 1201 has eight slit measurements but only one critical radius. Neither
has an image-position likelihood. A separate audit of MACS J0416 parses all 237
spectroscopic image positions but finds no resolved BCG radial dynamics: the 64
quoted dispersions are aperture measurements of distinct member galaxies, and the
lens paper identifies merger-like geometry. These three exact exclusions are in
`results/r1_replacement_search_cycle1_extensions/report.json`. The first Cycle 2
bridge, SDSS J0100+1818, has six BGG kinematic bins and 18 spectroscopic image
positions, but only one image lies within the 3-arcsec kinematic support. Its rank
ceiling is therefore 1; its candidate-fossil label is not counted as evidence for
an undisturbed halo because the source says deeper X-ray data are needed.

RX J2129 is the second non-disturbed structural promotion. It was selected from
the published image geometry, relaxed-state evidence, and public MUSE availability
before any BCG dispersion was extracted. Three spectroscopic images from three
families lie within 5 arcsec, giving a structural radial-rank ceiling of 3 and 12
family-wide positional degrees of freedom after source coordinates. A frozen
four-annulus pPXF reconstruction from ESO product
`ADP.2017-12-14T12:30:03.217` passes the predeclared S/N, formal-error, and
opposite-half checks. A subsequent deterministic resolution audit invalidated
E-MILES as the resolution-sensitivity baseline: its 2.51-A rest-frame templates
are broader than the 2.024-2.186-A MUSE rest-frame range, causing pPXF to clip the
nominal resolution differences. That provisional covariance result is retained
but cannot advance the gate.

The correction was frozen before the XSL bootstrap result, without changing the
center, annuli, spatial mask, or numerical thresholds. XSL has 0.402-0.592-A
resolution over the fitted rest-frame range. Its profile gives dispersions of
293.1, 306.4, 322.0, and 343.6 km/s and again passes both half-field checks. All
100 anchored 2x2-spaxel bootstrap replicates and all nine non-baseline sensitivity
protocols complete. The resulting 4x4 covariance is positive definite, with total
errors of 6.74, 1.91, 2.07, and 3.32 km/s; the largest fractional total uncertainty
is 0.0231 and the largest protocol shift is 0.0391, below the frozen 0.30 and 0.10
limits. See `results/r1_rxj2129_covariance_xsl/report.json`.

The first baryonic gate has also been executed under
`configs/r1_rxj2129_baryonic_protocol.json`, without reading a gravity or lens
residual. The primary Tian text resolves the apparent CDS-label ambiguity: the
Cooke value of (5.81\times10^{11}\,M_\odot) is the total BCG stellar mass used in
a spherical Hernquist profile with (r_h=0.551R_e) and
(R_e=41.4\pm0.54) kpc. Propagating the shared 10% mass error and radius error
gives a four-bin BCG acceleration baseline from (1.414\times10^{-10}) to
(5.674\times10^{-11}) m s\(^{-2}\) with a full 4x4 covariance.

That is a partial pass, not a completed baryonic likelihood. The same paper
reports a fitted Sersic index of 2.70 while using the n=4-like Hernquist shape,
and it does not publish the PSF/GALFIT configuration, Sersic-index error, or a
BCG/ICL decomposition. The public Chandra lineage provides only one cumulative
gas mass, (2.18\pm0.07\times10^{11}\,M_\odot) at 14.3 kpc. The Donahue source
package contains the gas model equations and an RX J2129 plot but no numerical
radial samples or covariance, so no gas law was inferred from the single point.

The predeclared conservative Molino membership audit finds one low-mass
photometric candidate inside 5 arcsec (nominal (8.5\times10^6\,M_\odot)). It
finds 66 candidates inside 30 arcsec with a nominal summed catalog mass of
(2.09\times10^{11}\,M_\odot); because most are interval-overlap candidates
rather than normalized membership probabilities, neither number is a complete
off-center satellite likelihood or a negligibility proof. The exact ledger,
component blockers, and diagnostic are in
`results/r1_rxj2129_baryons/report.json`. Strict R1 readiness remains false.

The next prerequisite, an empirical HST PSF audit, passes. Three predeclared
unflagged point sources with catalog S/N above 30 produce consistent PSFs in both
bands. F125W has a 0.0165 spread in the fraction of light inside 3 pixels, a
maximum pairwise normalized-profile difference of 0.146, and a maximum
leave-one-out shift of 0.055. F814W gives 0.0354, 0.272, and 0.111. These all pass
the frozen 0.15, 0.35, and 0.30 gates. The empirical PSF product therefore
authorizes the BCG+ICL decomposition, but no subsequent baryonic or gravity gate.
See `results/r1_rxj2129_hst_psf/report.json`.

The Cooke source adds an aperture caveat that must be carried into that fit. Its
MAGPHYS mass uses SDSS ugriz Petrosian magnitudes and WISE profile-fit photometry,
not an HST aperture explicitly matched to Tian's one-component light profile or
to a new BCG/ICL split. The decomposition must therefore include a mass-aperture
nuisance; assigning all 5.81e11 solar masses to whichever fitted component looks
like the BCG would not be source-traceable.

The frozen nonparametric HST extraction also passes. After catalog-ellipse masks,
12-sector profiling, 500 shared sector bootstraps, and the CLASH correlated-noise
allowance, 49 of 60 radial bins are usable in each band. Their joint 98x98
covariance is positive definite, and the fitted HST center is only 0.0464 arcsec
from the adopted dynamics center, below the 0.30-arcsec gate.

The preregistered PSF-convolved model comparison is now complete. Relative to one
Sersic term, two terms improve held-out chi2 by 0.6977 in F125W and 0.7710 in
F814W, both above the frozen 0.20 threshold, and their effective radii differ by a
factor of 10.97. But the putative outer term contains 0.9486 of F125W light inside
30 arcsec, violating the frozen 0.05-0.80 component-fraction range. This is a
decisive conjunctive-gate failure: the total-light profile requires curvature,
but the two terms are not identifiable as physical BCG and ICL components. The
protocol therefore retains the nonparametric total-light profile, explicitly
records BCG/ICL non-identifiability, skips a sensitivity grid that cannot rescue
the already-failed baseline gate, and does not authorize component mass mapping.
See `results/r1_rxj2129_bcg_icl/report.json`.

The satellite workstream has advanced beyond its original obstruction. The active
Jauzac appendix contains 156 machine-readable redshift rows despite a prose count
of 158; this discrepancy is retained explicitly. A 0.5-arcsec unique crossmatch
produces 112 extended Molino labels, including 34 members, and the exact
interval-overlap candidate domain contains 43 training objects. The frozen
spatially grouped classifier improves held-out Brier score by 0.5708 and log loss
by 0.3057, reaches AUC 0.9178, and has five-bin calibration error 0.0770. Four of
five spatial folds beat prevalence, although one held-out fold contains only one
object and is retained as a limitation.

The resulting 500 correlated membership-bootstrap vectors feed 2,000 off-center
stellar-force draws. The four-bin acceleration covariance is positive definite;
Plummer-size sensitivity peaks at 0.0118 of the published BCG acceleration, and a
worst-case point-mass tidal bound that assigns every one of 199 candidates beyond
30 arcsec membership one and +0.30 dex mass peaks at 0.00871. Both pass their
frozen gates. This is a numeric satellite *stellar* likelihood, not a lens member
dark-subhalo scaling law. See
`results/r1_rxj2129_satellite_membership/force_report.json`.

The observable lens-input gate passes independently. All 25 Jauzac table rows are
retained in a ledger; the likelihood excludes four photometric System 2 images and
uses 21 spectroscopic images in seven families with a 42x42 diagonal coordinate
covariance. Images 5.2, 6.3, and 8.2 from three families lie inside 5 arcsec. No
published model RMS or GR convergence map enters this likelihood. See
`results/r1_rxj2129_lens_observables/report.json`.

The independent lens implementation has now completed, with an important limit.
The smooth model reproduces all 21 images with exact radial RMS 0.3833 arcsec,
reduced coordinate chi2 0.6856, and a finite 24x24 local covariance. The blind
seven-image holdout instead has 1.4299-arcsec RMS. Adding the 66 member candidates
reduces its training RMS but worsens heldout RMS to 2.7265 arcsec, so that layer is
rejected. The frozen protocol did not contain a numerical heldout-adequacy
threshold; therefore these values cannot be used retroactively to authorize a
predictive or Weyl-response claim. The all-image fit remains an engineering
control, and a corrected adequacy gate must use a fresh system or unspent split.

The hot-gas route is now closed under its frozen gate. ObsID 552 retains only
81.809% of its exposure versus the 90% floor; the blank-sky `BKGSCAL` values fall
outside 0.5-2.0; and the event headers do not meet the frozen CALDB lineage.
Although the response, counts, centering, and compatibility checks pass, no gas
likelihood is authorized and the thresholds are unchanged.

Cycle 3 exhausts the complete 32-BCG Loubser et al. Gemini long-slit sample. Five
systems overlap the prior ledger and 27 are new, taking the inventory to 45/30.
All 32 have spatially resolved profiles (nine CCCP bins; generally 11, 13, or 15
MENeaCS bins, typically to 15 kpc per side), and the arXiv source contains an
individual profile plot for every host. Only central dispersion and a power-law
slope/error are tabulated; radial values, measurement covariance, promised r-band
light profiles, hot-gas profiles, and satellite baryons are absent. None of the 27
new hosts has an image-position/source-redshift likelihood in the source package or
current normalized lens ledger. The host-count gate passes, but no new system is
strict-ready. The inventory is therefore at 45/30 hosts, 2/2 structural
promotions, and 0 strict R1-ready systems. See
`results/r1_replacement_search_cycle3/report.json`.

RX J2129 and Abell 1689 are now explicit frozen-gate failures. A1689 completes
200/200 bootstrap replicates but fails its 27-run pPXF systematic grid with a
36.6% maximum signed-bin dispersion shift versus the 10% ceiling; no final
dynamics covariance is assembled.

The raw-dynamics replacement cycle is closed. A2261 fails its continuum-center
gate, A383 fails its arc-RMS gate, MS2137 fails its pre-fit registration/geometry
gate, and the predeclared disturbed A2537 control fails its arc-RMS gate before
science processing. The cycle produces zero new structural promotions, below the
predeclared progress threshold.

R1B3 is frozen in `configs/r1_rxj2129_strict_observable_next_stage.json`. Public
XMM ObsID 0093030201 and the two local HST bands pass the metadata/header gate,
and all 2,824 XMM archive objects are checksum-provenanced. XMM X1-X2 now pass:
MOS2 and pn survive the calibration, flare, immutable-mask, sector-level FWC/corner,
and local 650-900 kpc transfer gates; MOS1 is excluded by its CCD5 corner scale.
X3 also passes all six immutable annuli with 76,279.260 conservative net counts
and minimum annular S/N 84.091. The active targets are now coverage-complete
MOS2+pn direct responses, both 6x6 PSF cross-region ARF matrices, central-source
response vectors, and independent HST H1-H3 measurement covariance. The X3
wide-annulus responses are barred from fitting because their default detector map
did not cover the full DSS extent. Details and failure rules are in
[`R1_NEXT_STAGE_2026-07-26.md`](R1_NEXT_STAGE_2026-07-26.md).

The MOS2+pn X4 response interface passes, but the first resolution baseline does
not: SAS realized the nominal 650 request as 642x642 with 81-detector-unit pixels,
exceeding the frozen 80-unit ceiling. That comparison is retained only as an
invalid engineering record. The active numerical gate is the predeclared
sqrt(2)-promoted 920-versus-1302 a04-to-a04 comparison for both detectors, with
unchanged limits of 2% integrated response, 2% median fit-band shape, and 5% p95
fit-band shape. Full X4 production cannot start until this gate passes.

The promoted gate passes. MOS2 changes by 1.803%, 1.841%, and 2.034% for the
integrated, median, and p95 metrics; pn changes by 0.464%, 0.463%, and 0.523%.
The active XMM action is therefore the complete 12-RMF, 12-direct-ARF,
72-cross-region-ARF, and 12-central-source-ARF production and immutable manifest
audit. Temperature and density inference remain locked until that audit passes.

## New rank-three source search: SDSS J0946+1006

The first candidate outside the original 15-system universe was selected from
observable geometry only. The Jackpot lens has spectroscopic source planes at
redshifts 0.609, 2.035, and 5.975; published ring scales of about 1.4, 2.1, and
2.5 arcsec; 53 published MUSE kinematic bins measured to about 2.7 arcsec; and
public HST and ESO archive products. Four exact arXiv source packages are now
checksum-provenanced locally.

It does not pass the frozen same-support gate. Turner et al. exclude the nine
outermost bins and restrict their model-valid dynamical predictions to 1.95
arcsec because the outer envelope is asymmetric and possibly tidal. Therefore
only the 1.4-arcsec ring scale is inside accepted dynamics support. The second
and third ring scales remain outside, and the full image-level marginalized
Weyl rank has not been established. Public metadata confirm eleven MUSE cube
products, including one level-3 cube, and five expected HST observations. Public
analysis code exists, but normalized theory-neutral likelihood products and
chains were not identified as public downloads.

The concrete outcome is `rank-one repair candidate, not promotion`. It leaves
the structural ceiling at 3/10 and the minimum new rank-three requirement at
seven. No Jackpot science pixels may be downloaded under the current protocol.
The object is revisited only if a separate pre-pixel protocol can validate
dynamics through at least 2.5 arcsec with asymmetry and tidal systematics; if
that cannot be made testable in one feasibility cycle, replace the object rather
than relaxing the support threshold. See
`results/r1_j0946_jackpot_feasibility/report.json`.

## New rank-three source search: ESO 325-G004

E325 was selected from its data geometry, without using the published value of
the GR slip parameter or a gravity residual. The exact source archive is now
hash-locked. It establishes a lens redshift of 0.035, one extended source at
redshift 2.1, a 2.95-arcsec Einstein ring, 0.6-arcsec MUSE kinematic sampling,
and a deliberately accepted central 4-arcsec dynamics support. The source also
states that the extended arc widths and curvature constrain radial magnification
and exclude profiles that share the same Einstein radius.

The public-data audit passes. ESO exposes three level-2 MUSE products: one
7.38-GB science cube and two sky cubes. MAST exposes two F814W visit mosaics
totalling 18,882 seconds and two F475W visits totalling 4,800 seconds. General
`pylens` code is public, but no E325-specific normalized theory-neutral
likelihood, source reconstruction, covariance, or chains were identified. The
paper's joint 20-parameter posterior is not an observable product and is barred
from this test.

This is an acquisition authorization, not a structural promotion. The frozen J1
protocol names four exact HST DRC files and a 19.7-MB, six-arcsecond-radius ESO
SODA cutout over 4750-5600 Angstrom. It defines four radial-deflection basis
directions at 0.7, 0.9, 1.1, and 1.3 times the published Einstein radius. After
whitening and nuisance projection, E325 advances only if at least three
directions have >=3-sigma response to a one-percent-Einstein-radius deflection,
relative singular value >=0.001, and condition number <=1000. Rank three must
survive half-step finite differences, 60x60/80x80/100x100 source grids, gradient
and curvature regularization, mask erosion/dilation, all PSF variants, synthetic
coverage, a blank-sky null, and held-out-visit prediction. At least three
numerical MUSE bins with covariance must overlap the retained response support.

Passing those outcomes would raise only the structural ceiling from 3/10 to
4/10; strict readiness and R2 would remain locked. Rank one or two is a useful
lower-rank control but cannot be relabelled a success. Receipt failure, control
failure, or no result after one receipt cycle plus one implementation cycle ends
the E325 branch without changing thresholds. See
`results/r1_e325_feasibility/report.json` and
`configs/r1_e325_acquisition_jacobian_protocol.json`.

## R1B: five-system checkpoint

Add hot gas, extended ICL, satellite-galaxy components, their nuisance parameters,
and covariance to a residual-blind subset. Acquire observable-level or rerunnable
lens inputs rather than treating a GR/Lenstool kappa map as theory-independent.

R1B acquisition is active only for the RX J2129 strict-observable package. Passing
the XMM and HST measurement gates would produce the first complete candidate data
package; it would authorize only a separate dynamical-versus-Weyl identifiability
audit, not R2 or a new field equation.

Advance at five strict-ready systems. A completed acquisition cycle must add at
least two strict-ready systems (20% of the current ten-system gap) or identify and
test a specific public-data obstruction. Otherwise search a replacement sample
before doing another cycle.

## R1C: ten-system freeze

At ten strict-ready systems, freeze identities, centers, radial masks, covariance
treatment, baryonic nuisance priors, and grouped validation splits without viewing
gravity residuals. Only that frozen artifact authorizes R2.

If three acquisition cycles or seven active research days fail to produce ten
systems, stop the fit and report that the proposed dynamics-lensing response is
not identifiable with the assembled public data. Do not lower the sample gate
after inspecting residuals.

## R2: theory-free response test

Reconstruct the dynamical potential and Weyl potential separately. A one-field
description survives only if it closes at least 50% of both held-out benchmark
gaps. A two-potential description survives only if the second response is
independently required on the same systems. If neither is identifiable, stop
before proposing another force-law variant.
