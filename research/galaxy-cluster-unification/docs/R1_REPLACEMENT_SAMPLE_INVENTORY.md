# R1 replacement-sample inventory

Status: published-count screen passed; strict R1 gate not passed.

The current CLASH and MaNGA/SPIDERS bridge samples have no same-object systems
with resolved dynamics and lensing. A residual-blind replacement search found two
published cluster samples with complementary coverage:

- Newman et al. (2013): seven clusters with resolved BCG kinematics, strong and
  weak lensing, and HST surface photometry.
- Cerny et al. (2025): six radial-arc clusters with MUSE BCG dispersion profiles,
  strong-lensing constraints, and HST photometric BCG fits.

A383 and MS2137 occur in both samples. Selecting the newer resolved kinematics for
those duplicates leaves 11 unique systems. A963 has four dynamics bins but only
two strong-lensing degrees of freedom in the Newman fit, so it fails the preliminary
3+3 count screen. The remaining ten systems are the acquisition queue:

1. A2390
2. A2537
3. A2667
4. A383
5. A611
6. MACS J0326
7. MACS J0417
8. MACS J0949
9. MACS J1427
10. MS2137

This is not yet a frozen R1 science sample. Strong-lens image positions are not
automatically three independent radial mass constraints, and published parametric
BCG light fits are not the same as numerical surface-brightness likelihoods. The
strict gate remains at 0/10 because none of the ten currently has all of the
following local and verified:

- numerical lens-model likelihood or posterior chain;
- covariance that preserves shared strong/weak-lensing systematics;
- numerical baryonic surface-brightness/profile data for forward modelling;
- at least three dynamics and three lensing constraints over demonstrably
  overlapping radial support.

## RELICS acquisition update

MAST's RELICS high-level archive supplies an independent Lenstool reconstruction
for three queue systems: A2537, MACS J0417, and MACS J0949. We downloaded one best
convergence map and 100 MCMC uncertainty-range maps for each system. These 300
realizations support a full sample covariance for radial projected convergence,
although they remain standard-lens-equation model products rather than raw image
likelihoods.

Using one-pixel-wide annuli and requiring the entire annulus to lie inside the
published BCG kinematic support gives:

- A2537: 3 fully overlapping lensing annuli inside 0-3.65 arcsec;
- MACS J0417: 1 fully overlapping annulus inside 0-3.40 arcsec;
- MACS J0949: 1 fully overlapping annulus inside 0-3.18 arcsec.

Thus only A2537 passes the provisional radial-count part of the gate among these
three. The count is based on FITS-reference-centered annuli, whereas the dynamics
are BCG-centered, so exact overlap is still unverified. It is also not R1-ready
because the complete numerical baryonic profile and joint observable covariance
are missing. The two MACS maps have 2.01-arcsec range-map pixels, too coarse to
place three full annuli inside their BCG kinematic support.

A primary-source centering audit verifies the BCG centers for MACS J0417 and
MACS J0949; each is about 0.10 arcsec from its archived map reference. Neither
passes the radial count. The Cerny et al. A2537 model-table caption mistakenly
prints the Abell 2163 BCG coordinate (and repeats it in the Abell 2163 table), so
A2537's exact map-to-BCG center cannot be certified from that publication. The
verified BCG-centered 3+3 count is therefore 0/3, not 1/3.

## Observable-level strong-lens update

The exact article sources now supply published image tables for all ten candidates:
175 entries in total. We retain 166 as observable-level positions after excluding
model-predicted entries, a source-table coordinate error, and A383's explicitly
unused constraint. Of these, 106 also have a spectroscopic source redshift and a
declared Gaussian image-plane error. Newman supplies BCG-relative coordinates and
a 0.5-arcsec error per coordinate (1.0 arcsec for A2390); Kaleidoscope supplies a
0.5-arcsec likelihood error for its six models. These inputs can be forward-modeled
under a declared alternative lens equation. The kappa maps cannot be substituted
for them, and no publication supplies a full systematic covariance.

Using the residual-blind preferred dynamics source for duplicate systems, only
three clusters have strict image positions inside the BCG dynamics aperture:
A383 has two images from one source family, while A2537 and MS2137 each have one.
No system has three distinct inner source families. The BCG-relative Newman table
also resolves A2537's absolute-coordinate problem for this image-level audit.

The failed family-count proxy now feeds a stricter identifiability test rather than
a gravity fit. That structural test rejects all ten current systems before a full
Jacobian: A383 has only two strict inner images and hence radial-rank upper bound 2;
A2537 and MS2137 each have upper bound 1; the remaining systems have 0. Outer
counterimages can constrain a source position, but they do not add in-support
radial samples. Nuisance marginalization cannot raise these bounds. See
`configs/r1_identifiability_targets.json`.

The next stage is therefore the residual-blind replacement-host audit in
`configs/r1_replacement_search_targets.json`. It targets at least 30 unique BCG
hosts, or a source-by-source hard-shortfall certificate, and requires at least two
non-disturbed systems to pass the same structural gate before strict-readiness
preparation begins. A full likelihood Jacobian still requires the post-promotion
covariance, baryonic, and lens-nuisance gates.

The first detailed replacement audit adds MACS J1206 and Abell S1063. MACS J1206
passes the structural gate with six dynamics annuli to about 50 kpc and 11 strict
inner image positions from nine source families (radial-rank upper bound 11).
Abell S1063 has only one strict image within its 40-kpc kinematic support (rank
bound 1) and is independently flagged as a recent off-axis merger. Thus the stage
has one of the required two non-disturbed structural promotions, initially 13 of 30 hosts
source-screened, and still 0 R1-ready systems. The exact image-level audit is in
`data/derived/r1_replacement_cycle1_image_support.csv`.

An official archive audit then separated publication-level and raw-data
availability. Neither detailed dynamics paper supplies its BCG velocity-dispersion
values, errors/covariance, or likelihood as a numerical table; both supply those
measurements only in Figure 2. The ESO ObsCore/DataLink records do, however, expose
public level-3 MUSE cubes for both systems. MACS J1206 is therefore
raw-reconstructible but not publication-likelihood-ready. Its 11.45-GB full cube
has been reduced to a local 331,548,480-byte, BCG-centered 15-arcsec,
4860-7160-Angstrom SODA cutout for the six-annulus pPXF reconstruction. This
raw-data route does not change the strict
R1-ready count until the mask choices and propagated covariance have been frozen.
See `data/derived/r1_dynamics_public_data_availability.csv` and
`data/derived/r1_dynamics_archive_products.csv`.

The first-cycle extension then exhausts the complete Sand six-cluster sample and
audits MACS J0416. RXJ 1133 (three BCG bins, two critical radii) and Abell 1201
(eight slit measurements, one critical radius) fail the three-mode and
image-position requirements. MACS J0416 supplies 237 spectroscopic multiple-image
positions but no resolved BCG profile; its 64 stellar dispersions are measurements
of separate member galaxies, and it is independently merger-like. The cumulative
inventory after Cycle 1 is therefore 16/30 with the promotion count unchanged at
1/2. The parsed ledger is
`data/derived/r1_replacement_cycle1_extension_ledger.csv`.

Cycle 2 begins with the group-scale bridge SDSS J0100+1818. It has six resolved
BGG kinematic bins to 3 arcsec (about 20 kpc) and 18 spectroscopic multiple-image
positions, but only image C2 lies inside that kinematic support. Its radial-rank
ceiling is therefore 1, and the source's candidate-fossil classification is not
enough to count the halo as non-disturbed without the deeper X-ray test it calls
for. The system is retained as a low-overlap control, not promoted.

RX J2129 then supplies the second non-disturbed structural promotion. Three
spectroscopic images from three source families fall inside the 5-arcsec BCG
aperture, and their nine family-wide image positions leave 12 positional degrees
of freedom after source coordinates. Its independently frozen four-bin public-
MUSE reconstruction passes every baseline S/N, formal-error, and opposite-half
consistency check. The first E-MILES covariance run was then invalidated because
its templates are broader than MUSE in the fitted rest-frame range, making the
instrumental-resolution variants non-informative. Under the frozen correction,
the resolution-valid XSL profile spans 293.1-343.6 km/s; all 100 spatial block
bootstraps and all nine non-baseline protocols complete; the positive-definite
4x4 covariance has fractional total errors below 0.024 and a maximum protocol
shift of 0.0391. The kinematic covariance workstream therefore passes.

Complete BCG/ICL/gas/satellite inputs and a rerunnable lens nuisance likelihood
are still missing, so RX J2129 is not strict R1-ready. The source states that the
underlying lens-model data are available on request rather than publishing the
configuration or chain.

Cycle 3 exhausts the complete 32-BCG Loubser et al. 2018 Gemini long-slit sample.
After five overlaps it adds 27 unique hosts, bringing the cumulative inventory to
45/30. The source archive supplies a per-object radial-kinematics plot for all 32
hosts and tabulates central dispersion plus a power-law slope/error, but not the
radial profile values or covariance. The 27 new hosts have no local observable
image-position/source-redshift likelihood in the current project ledger, and the
source omits complete baryonic profiles. Thus the count boundary is passed while
strict readiness remains at zero; future acquisition targets missing likelihoods
and covariance rather than more host names. See
`results/r1_replacement_search_cycle3/report.json`.

The structural promotion threshold is 2/2, and the strict-ready count remains 0. See
`results/r1_replacement_search_cycle2/report.json`,
`results/r1_rxj2129_covariance_xsl/report.json`, and
`configs/r1_rxj2129_readiness_targets.json`.

The next work is therefore split but still residual-blind: acquire the missing
lens likelihoods and numerical covariance for the frozen inventory while completing
the baryonic readiness workstreams. A gravity-law or full
Jacobian fit is not authorized by the covariance pass alone.

## Newman BCG stellar-profile reconstruction

The Newman source also publishes Chabrier-IMF stellar population-synthesis
mass-to-light ratios for all seven BCGs. We combine these with the rest-frame
V-band luminosities and circularized dPIE light fits to reconstruct annular BCG
stellar surface-density profiles on every Newman dynamics bin and, for A2537, on
the RELICS reference-annulus grid. The reconstruction includes a conditional
covariance from the stated 0.1 mag luminosity uncertainty and the reported
`r_cut` uncertainty.

This is a partial baryonic input, not a full baryonic profile. It omits SPS M/L
uncertainty, intracluster gas, extended intracluster light not captured by the
single BCG component, satellite galaxies, and cross-probe covariance. It therefore
does not change the strict R1 count of 0/10.

The source parser generates 70 published velocity-dispersion bins, validates the
Newman strong-lensing degrees of freedom and the four Kaleidoscope image-position
tables, deduplicates systems without looking at gravity residuals, and emits the
acquisition queue. Continue that residual-blind acquisition queue while preparing
RX J2129 for strict readiness; do not build an A383/MS2137 Jacobian and do not add
a new force law.

Primary sources:

- Newman et al. (2013), https://arxiv.org/abs/1209.1391
- Cerny et al. (2025), https://arxiv.org/abs/2506.21531
- Jauzac et al. (2019), https://arxiv.org/abs/1811.02505
- Allingham et al. (2023), https://arxiv.org/abs/2207.10520
- MAST RELICS lens models, https://archive.stsci.edu/hlsp/relics
