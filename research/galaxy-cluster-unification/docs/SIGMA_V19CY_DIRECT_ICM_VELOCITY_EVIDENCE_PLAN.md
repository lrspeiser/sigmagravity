# Sigma V19CY direct ICM velocity evidence plan

## Why this is the next admissible direction

V19CX combined every registered spectrum correctly but failed because the
Bullet Cluster's complete spectrum was not adequately described by the frozen
one-temperature plasma model. That blocks the planned 494-region thermodynamic
source reconstruction and prevents us from deriving an action from I4/I5.

The most useful new evidence is not another temperature interpolation. It is a
direct measurement of gas motion. XRISM/Resolve measures iron-line centroids
and widths well enough to provide sign-resolved line-of-sight bulk velocity and
velocity dispersion. This gives us one observed component of baryonic current,
which is precisely the time-odd information that the P2 causal/memory route was
missing.

## Frozen three-system split

| Role | System | Public observations | Outcome status at freeze |
|---|---|---|---|
| Development | Abell 2319 | 000101000, 000102000, 000103000 | published and known |
| Validation | Abell 3667 | 201051010, 201050010 | not inspected |
| Holdout | Abell 754 | 201015010, 201016010 | not inspected |

The validation and holdout were selected from official public-archive metadata
only. Abell 3667 has two Resolve pointings totaling 415.478 ks and targets a
prominent cold front. Abell 754 has two pointings totaling 320.097 ks and was
proposed specifically to measure hydrodynamic motion relative to its merger
axis. Their scientific velocity outcomes remain sealed.

## Archive inventory result

The metadata-only inventory found 568 files totaling 30,602,430,184 bytes:

| Role | Files | Remote bytes | Approx. GiB |
|---|---:|---:|---:|
| Development | 209 | 11,323,323,935 | 10.545 |
| Validation | 191 | 9,416,570,852 | 8.770 |
| Holdout | 168 | 9,862,535,397 | 9.185 |

No file body or scientific outcome was read. The exact 568-row manifest is
`results/sigma_v19cy_direct_icm_velocity_evidence/archive_manifest.csv`.

For development, cleaned Resolve events, auxiliary files, Resolve housekeeping,
pipeline products, and processing logs total 4,145,263,140 bytes (3.861 GiB).
The Resolve unfiltered events add 5,191,117,778 bytes (4.835 GiB), and cleaned
Xtend events/products add 325,847,859 bytes (0.303 GiB). The A2319 commissioning
gain treatment must be checked against the published method and current XRISM
calibration guidance before freezing which of those additional files are
scientifically required. This prevents both an incomplete reduction and an
unnecessary 11.3-GB download.

Abell 2319 is development-only because its result is already known. The
published analysis measured five sky regions and found a roughly 300 km/s
velocity range across the core, including a region blueshifted by about
230 km/s and a region with roughly 400 km/s velocity dispersion. We first have
to reproduce those values and their spatial-spectral-mixing treatment.

## Frozen A2319 acquisition closure

The published Appendix and NASA's official Resolve energy-scale report show
that the three science ObsIDs are not calibration-self-contained. The starting
gain fiducial for 000101000 lies in predecessor ObsID 000100000, and the later
fiducials cross the boundaries between 000101000, 000102000, and 000103000.
The 000103000 history also crosses an ADR recycle and may use only the linear
post-recycle interval beginning six hours later. Standard cleaned events alone
therefore cannot reproduce the commissioning energy scale.

The exact development acquisition was frozen before payload download. It has
197 files totaling 12,742,865,194 bytes (11.868 GiB):

| Asset group | Files | Remote bytes | Purpose |
|---|---:|---:|---|
| Three XRISM science ObsIDs | 116 | 9,124,771,397 | Unfiltered/cleaned events, auxiliary data, HK, and logs needed for gain reconstruction and screening |
| 000100000 calibration predecessor | 15 | 1,567,139,641 | Only the Fe-55/cal-pixel gain dependency and required time/HK files; no predecessor science exposure |
| Current CALDB archives | 3 | 1,780,998,985 | General, XRISM GEN 20241115, and Resolve 20260315 calibration |
| Chandra ObsIDs 3231 and 15187 | 62 | 265,771,249 | Reprocessed 0.5-7.0 keV SSM surface-brightness input |
| Official gain-quality report | 1 | 4,183,922 | Independent energy-scale acceptance reference |

NASA's combined 000100000 gain report records 2,294 Fe-55 solutions and 471
calibration-pixel solutions with zero failed fits. That does not itself prove
our reduction. We must still verify complete open-filter time coverage, exclude
every ADR/SAA interval, reproduce the observation-specific calibration-pixel
centroid and width, and propagate the published 0.51, 0.33, and 0.30 eV
systematic uncertainties for 000101000, 000102000, and 000103000.

The manifest is
`results/sigma_v19cy_direct_icm_velocity_evidence/development_acquisition_manifest.csv`
and its SHA-256 is
`3ef7816ccfa069a49f34cf18d22d1cd22da1c4fd1dc0ac2ed9151c2e66f16cac`.
The predecessor cannot contribute a science region or exposure. Xtend,
pipeline quick-look products, validation, holdout, lensing, halo, and gravity
targets remain excluded.

### Terminal development download result

The frozen acquisition completed without changing its scope or concurrency.
All 197 files were downloaded, independently size-checked, and SHA-256 hashed.
The local tree contains exactly 12,742,865,194 bytes and no partial files. The
terminal provenance report has SHA-256
`4df72dea94daf51bd0c0d6d1fbb1567651561357ebd06de93aa5f2edb82f386f` and is
stored at
`results/sigma_v19cy_direct_icm_velocity_evidence/development_download_provenance.json`.

The exact terminal totals are 116 A2319 science files (9,124,771,397 bytes),
15 predecessor calibration files (1,567,139,641 bytes), three CALDB archives
(1,780,998,985 bytes), 62 Chandra files (265,771,249 bytes), and one official
gain report (4,183,922 bytes). No A3667 or A754 payload was accessed, no
lensing/halo/gravity target was opened, and no velocity fit was performed.
This closes acquisition only; it does not yet pass the gain-reconstruction,
spectral-reproduction, or spatial-spectral-mixing gates.

### Frozen reduction-environment result

The isolated A2319 calibration environment passed after one recorded,
fail-closed path-conversion correction. The first execution downloaded and
hash-froze NASA's current `caldb.config` and `alias_config.fits`, but stopped
before listing or extracting any archive because `wslpath` did not preserve the
native Windows path. Its empty staging directory was removed, and the
correction replaced that external conversion with a deterministic local-drive
mapping before a second frozen execution.

The terminal environment contains 139 files totaling 2,361,956,732 bytes.
HEASoft 6.36, XSPEC 12.15.1, the six pinned Resolve/CALDB executables, both
official setup files, and the GEN/Resolve calibration indexes all matched their
frozen hashes. Live `caldbinfo` queries report both XRISM/GEN and
XRISM/RESOLVE as configured and accessible. The terminal report SHA-256 is
`b9ded07723d38b7444f5bfb1abbd54337533b8c8c11fe0da043d9583d1cbd394`.

This environment pass authorizes freezing the development gain-reconstruction
protocol. It does not authorize inspecting gain/event arrays, fitting a
spectrum or velocity, opening A3667/A754, or using lensing, halo, gravity, or
action targets.

### FITS metadata-only closure

Before authorizing row-level gain work, a frozen structural inventory opened
only FITS headers and column schemas. It covered 87 development files totaling
7,661,862,987 compressed bytes and 452 HDUs. Astropy reported that every HDU
data object remained unloaded; the runner never accessed an HDU `.data`
property. The terminal inventory SHA-256 is
`90f9e075cc80536bc0c104c25533cd6dc9a3570312f3cd7c58abdaa4c6524c1e`.

The metadata establishes that each Fe-55 and calibration-pixel gain history
has a `Drift_energy` table with the required `TIME`, `PIXEL`, fitted/average
correction, temperature, width, exposure, event-count, and fit-quality fields.
The raw table sizes exceed the 2,294 Fe-55 and 471 calibration-pixel solutions
summarized in the official combined gain report, so raw row count cannot be
used as an acceptance result; the later protocol must apply the exact time,
pixel, ADR, SAA, fit-quality, and science-interval rules.

Each science cleaned event file contains roughly 1.25 million event rows and
33 general GTIs. Each science open-filter file has zero `GTIOPEN1` intervals
and two `GTIOPEN2` intervals. The ADR files also expose separate on/off GTI
extensions. These are schemas and row counts only: no time boundary, gain
value, event energy, spectrum, or velocity has yet been read.

### Scalar gain-timeline audit result

The first frozen row-level audit read only the declared scalar gain-history
columns and GTI `START`/`STOP` values. Across the four required calibration
histories it found 11,405 raw Fe-55 rows and 4,219 raw calibration-pixel rows.
None of the nine preregistered pooled count variants reproduced the official
2,294/471 comparison, so the runner stopped before gain application exactly as
specified. This failed comparison is retained rather than silently replacing
it with a post-hoc filter.

The timeline topology nevertheless closes the required science support. All
six open-filter intervals have a preceding Fe-55 anchor for all 34 science
pixels. The first five have a following anchor for all 34 pixels; only the
final interval of 000103000 lacks one, matching the published need for a final
one-sided forward extrapolation. The audit also recovered all ADR-on intervals
and the six-hour post-recycle candidate times without reading an event energy
or fitting a spectrum.

Visual and text inspection of the official `000100000` Resolve energy-scale
report then supplied the missing accounting scope. The report-specific table
contains 2,294 Fe-55 and 471 calibration-pixel solutions with zero failures,
and its methodology requires gain-history solutions to exclude ADR-recycle
and SAA intervals. The earlier pooled comparison combined four ObsIDs and
treated the continuous calibration-pixel history as part of the same
denominator. The separately frozen closure instead tested the report's exact
intermittent `000100000` history and passed: its 2,765 rows split into 2,294
non-pixel-12 Fe-55 solutions and 471 pixel-12 solutions, every per-pixel count
matched Table 1, and zero solution times overlapped any of the 45 SAA or four
ADR intervals. The terminal report SHA-256 is
`22b3022320d99f8b2617b0a2228da2976e62756aaeeb6cc8edb88f0b858ddb14`.

This closes the official solution accounting and authorizes freezing a
separate gain-reconstruction protocol. It still does not authorize gain
interpolation, event correction, spectral fitting, or a velocity claim.

### Gain-reconstruction scalar topology result

The paper-specific reconstruction topology passed before any gain-history
array or event row was opened. A frozen 10,800-second segmentation recovered
four complete Fe-55 fiducial blocks, one from each of 000100000 through
000103000. The same four blocks, row membership, and boundaries were recovered
with independent 7,200- and 14,400-second thresholds, so the result is not an
artifact of choosing a convenient gap size after inspecting the residuals.

All seven preregistered branches have the required preceding, following, or
two-sided anchors for every main-array pixel and pixel 12. These branches are:

1. two cross-ObsID intervals for 000101000;
2. two cross-ObsID intervals for 000102000;
3. forward extrapolation before the 000103000 ADR recycle;
4. back-extrapolation beginning six hours after that recycle starts; and
5. final forward extrapolation where no ending fiducial exists.

The continuous calibration-pixel histories supplied 1,111 usable rows across
the seven science branches. Their residual relative to the intermittent
pixel-12 prediction is finite in every branch. The median residual is small in
the long first 000101000 interval (`4.20e-6` in equivalent-temperature units)
and substantially larger around the 000103000 ADR branches (`1.85e-5` before
and `-1.09e-5` after), showing why a single per-ObsID pipeline interpolation is
not an adequate reproduction of the published analysis.

The terminal topology report SHA-256 is
`f5fcdfe793681d228fa78ab84ecb29d5e823bfdc7de48f98a340594ccbe4624e`.
This authorizes freezing calibration-application candidates only. It does not
select the common-mode correction, calculate an event energy, fit a line, or
measure a cluster velocity.

### Calibration-application candidate result

Three candidates were frozen before recalculating any calibration energy:

1. the branch-specific Fe-55 fit alone;
2. that fit plus the branch-median calibration-pixel residual; and
3. that fit plus a linear residual passing through the frozen branch median.

Each candidate was applied to high-resolution primary pixel-12 events in all
seven branches with the audited HEASoft `rslpha2pi` executable and the archived
pipeline settings (`method=FIT`, `secphacol=PHA2`, random seed 7). All 28
selection/application commands exited zero. The 21 outputs contain 2,153,106
candidate-event rows in total, with identical selected rows across candidates.
Every row has finite `EPI2` and `TEMP`.

Two fail-closed corrections are retained. Version 1.0.0 stopped after one
application because a copied FITS extension name was normalized to uppercase;
the correction made each drift build copy an immutable template and compare
extension names case-insensitively. Version 1.0.1 completed all applications
but stopped at a zero-null-PI gate. The official `rslpha2pi` rule sets PI to
NULL for finite negative `EPI2`; a count-only audit confirmed a one-to-one
match in every output, with no unexplained null PI and no negative `EPI2`
retaining a PI. Version 1.0.2 therefore passed the documented gate without
adding a performance tolerance.

The 49 final scratch files occupy 498,323,520 bytes and are retained under
`tmp/sigma_v19cy_a2319_calibration_application` for the next frozen line-shape
test. The terminal report SHA-256 is
`2fe0daffdf6f6eac722d375f9420ba7f14ba6a28d04bf40a974eceb3f6b173e7`.
No centroid, width, or energy distribution has yet been inspected, so no
candidate has been preferred. No cluster sky event has been opened.

### Calibration line-shape gate result

The preregistered eight-component CALDB Mn K-alpha Voigt fit rejected all
three candidates. The required terminal action is therefore
`stop_before_cluster_event_application`: no candidate was selected, no
cluster sky event was opened, and no cluster velocity was fit.

| Candidate | 000101 shift / FWHM (eV) | 000102 shift / FWHM (eV) | 000103 shift / FWHM (eV) | Maximum absolute z | Result |
|---|---:|---:|---:|---:|---|
| Fe-55 branch only | -0.600 / 5.489 | +0.820 / 5.346 | -0.514 / 8.000 | 177.00 | Fail |
| Branch-median common mode | +0.277 / 5.260 | +0.091 / 5.281 | -0.100 / 5.167 | 68.70 | Fail |
| Branch-linear common mode | +0.321 / 4.779 | +0.157 / 4.545 | -0.054 / 4.598 | 73.08 | Fail |

The branch-linear correction had the lowest total score and substantially
narrowed every line. It nearly reproduced the 000102000 commissioning values,
but the frozen rule required all six observables to lie within five quoted
statistical errors. Its 000101000 centroid was +0.731 eV above the published
target, and its 000101000 FWHM was +0.329 eV broader. It also left smaller but
still disqualifying residuals in 000103000.

This result falsifies the three frozen long-branch approximations; it does not
yet falsify reconstruction of the published calibration. A single linear fit
over each full Fe-55 anchor segment can discard local boundary behavior and
within-segment curvature. The next admissible step is calibration-only
diagnosis, frozen before inspecting its outcomes: time-resolved Mn K-alpha
fits can distinguish a constant reference offset from a slope error or
curvature, while the continuous calibration-pixel gain history can provide a
mission-pipeline baseline for the line fitter. Cluster-event access remains
sealed until a separately frozen reconstruction passes its calibration gate.

The terminal line-shape report SHA-256 is
`868ed6bbe284b936312470b1bae7474c32881ade1bf94f9c935f6913269a1307`.

### Calibration failure diagnosis result

The frozen calibration-only diagnosis separates an analysis failure from a
gain-reconstruction failure. Reapplying each observation's original
continuous pixel-12 gain history produced the following controls:

| ObsID | Control shift (eV) | Control FWHM (eV) | Published FWHM (eV) | Result |
|---|---:|---:|---:|---|
| 000101000 | -0.0028 | 4.5468 | 4.45 | Pass |
| 000102000 | +0.0028 | 4.5047 | 4.45 | Pass |
| 000103000 | +0.0137 | 4.5343 | 4.46 | Pass |

All controls meet the preregistered absolute-centroid and FWHM tolerances.
All fourteen `ftcopy` and `rslpha2pi` commands exited zero. This validates the
line template, fitter, event selection, mission executable, and continuous
gain application at the precision needed for the diagnosis.

The four equal-time fits show that the branch-linear candidate is not failing
through one removable global offset. Every one of the seven branches has at
least 0.1 eV of within-branch curvature relative to its endpoint line. Every
ObsID also has at least 0.1 eV separation among its branch-total centroids.
The largest curvature is 0.759 eV in the long first 000101000 interval;
000101000 and 000102000 additionally contain material endpoint slopes.

This authorizes freezing one replacement calibration-only reconstruction
protocol that preserves local time structure instead of compressing a full
anchor or science branch to one straight line. It does not authorize choosing
that reconstruction after seeing its calibration performance, applying gain
to a cluster sky event, or fitting a cluster velocity. The seven continuous
controls are a validation reference, not a proposed science calibration,
because the continuously illuminated pixel behaves differently from the main
array that observes the cluster.

The diagnosis retained 17 scratch files totaling 260,815,680 bytes. Its
terminal report SHA-256 is
`a418ad26fe170408cd19a1483aa1b543bd13076409484ac3bf2046b7bd4c2a1a`.

### Common/differential held-out closure result

The parameter-free decomposition

\[
T_i(t)=C_{12}(t)+D_i(t)
\]

was tested by omitting each complete filter-wheel Fe-55 interval in turn.
The exact continuously illuminated pixel supplied common mode `C12`; a linear
pixel-specific differential `Di` was learned from only the other three
intervals. Candidate and native-control histories were then applied to the
same held-out main-array Fe-55 events.

| Held-out segment | Role | Whole shift delta (eV) | Whole FWHM increase (eV) | Pixel shift p90 (eV) | Pixel FWHM p90 (eV) | Result |
|---:|---|---:|---:|---:|---:|---|
| 0 / 000100000 | Endpoint extrapolation | +0.204 | +0.416 | 0.691 | 0.169 | Fail |
| 1 / 000101000 | Interior interpolation | -0.008 | +0.203 | 0.396 | 0.142 | Pass |
| 2 / 000102000 | Interior interpolation | -0.088 | +0.256 | 0.356 | 0.135 | Pass |
| 3 / 000103000 | Endpoint extrapolation | +0.279 | +0.396 | 0.778 | 0.167 | Fail |

All twelve mission-tool commands exited zero. The global gate is false, so no
science-branch application is authorized. The scenario pattern is nevertheless
specific and useful: both bracketed interpolation folds pass every frozen
whole-array and per-pixel criterion, while both one-sided extrapolation folds
fail. The common/differential mechanism therefore remains active for bracketed
regimes. Endpoint extrapolation is parked for a separately frozen local model;
it is not treated as a falsification of the successful interpolation regime.

The closure retained 20 scratch files totaling 4,604,342,400 bytes. Its
terminal report SHA-256 is
`fa3bd67b4e81c84e11812d6724fe36a56744eda5da6a82e37b5ecdd40885c13e`.

### Endpoint extrapolation falsification and reduced-scope decision

Two additional endpoint rules were frozen together and tested on both failed
outer folds. Neither was allowed to switch by endpoint:

| Candidate | Fold | Whole shift delta (eV) | Whole FWHM increase (eV) | Pixel shift p90 (eV) | Pixel FWHM p90 (eV) | Result |
|---|---:|---:|---:|---:|---:|---|
| Nearest-anchor constant | 0 | +0.164 | +0.194 | 0.583 | 0.146 | Fail |
| Nearest-anchor constant | 3 | +0.197 | +0.332 | 0.566 | 0.160 | Fail |
| Nearest-anchor linear | 0 | -0.651 | +0.528 | 1.351 | 0.169 | Fail |
| Nearest-anchor linear | 3 | -0.487 | +0.746 | 1.695 | 0.153 | Fail |

All six mission-tool commands exited zero. The constant rule came close at
whole-array level but failed the frozen per-pixel 90th-percentile criterion at
both ends. The local linear rule was materially worse. Together with the
earlier global-linear endpoint failure, three physically distinct closures
now fail the same endpoint gate. We therefore stop modifying one-sided
differential extrapolation and park those branches rather than adding another
calibration degree of freedom.

This is not a falsification of common/differential interpolation: both
bracketed held-out folds passed. The authorized next step is a reduced-scope
A2319 development protocol using only the three genuinely bracketed science
fragments. A fourth topology branch was also rejected because its following
anchor lies across the ADR, so it is not an undisturbed bracket. The three
one-sided intervals around the ADR and final tail remain excluded and
explicitly unresolved.

The endpoint experiment retained 10 scratch files totaling 2,636,691,840
bytes. Its terminal report SHA-256 is
`9616b6774e38f4a253dede22b02783afec6e9e30d8fa186d0f4701c00aecb2ed`.

### Reduced A2319 bracketed science calibration

The science calibration was frozen before any cluster-event row was opened.
Two fail-closed setup stops were recorded and preserved: version 1.0.0 found
that the second pointing's official `OBJECT` value is `Abell2319_Cor1`, and
version 1.0.1 found that a topology branch extended beyond valid native
calibration-pixel time support. Neither stop read a sky-event row. Version
1.0.2 therefore froze the application interval as the intersection of the
topology branch, valid calibration-pixel support, and cleaned-event
`TSTART`/`TSTOP`; it never extrapolates common mode.

| Science fragment | Supported interval (mission seconds) | Selected/output rows | Null EPI2 | Null TEMP | Result |
|---|---:|---:|---:|---:|---|
| 000101 open 0 | 150943122.000–151036810.979 | 18,301 / 18,301 | 0 | 0 | Pass |
| 000101 open 1 | 151201971.014–151204864.023 | 857 / 857 | 0 | 0 | Pass |
| 000102 open 0 | 151208096.000–151289711.018 | 19,038 / 19,038 | 0 | 0 | Pass |

All six `ftcopy`/`rslpha2pi` mission-tool commands exited zero. The calibration
gate passed for all 38,196 retained events. No cluster energy distribution was
summarized or fit, no velocity was estimated, and the A3667 validation and
A754 holdout assets remained sealed. The short 857-event fragment is valid
but may limit the number of independently useful spatial/spectral regions;
the next protocol must freeze that handling before inspecting Fe-K energies.

The retained calibration scratch tree contains nine files totaling
280,800,000 bytes. The terminal report SHA-256 is
`6f474938d674e353d2339cf150d5eef1f41b0231b5e8e2f31688c7bec548136e`.

### Detector-region count readiness

Before reading any recalibrated energy value, the published detector geometry
was frozen from figure 1 and section 3.3 of the A2319 paper. P1 uses detector
regions `a`, `b`, and `d`; P2 uses `b'`, `c'`, `d'`, and `e'`. Together each
pointing partitions all 34 science pixels exactly once after excluding the
off-field calibration pixel 12 and the aperiodically noisy pixel 27.

The count-only audit then applied the official high-resolution primary,
rise-time, frame-event, and noisy-pixel screen. It copied rows without using an
energy column and read only output `NAXIS2` headers.

| Pointing | Detector region | Aggregate screened events | Frozen floor | Result |
|---|---|---:|---:|---|
| P1 | a | 5,557 | 1,000 | Pass |
| P1 | b | 4,500 | 1,000 | Pass |
| P1 | d | 3,774 | 1,000 | Pass |
| P2 | b' | 3,979 | 1,000 | Pass |
| P2 | c' | 3,643 | 1,000 | Pass |
| P2 | d' | 3,022 | 1,000 | Pass |
| P2 | e' | 3,594 | 1,000 | Pass |

All 13 mission-tool commands exited zero, and each branch's detector regions
sum exactly to its screened full-array count. The geometry therefore remains
usable without merging or redrawing a weak region. No energy distribution,
spectrum, response, velocity, validation object, or holdout object was opened.

The readiness scratch tree contains 13 files totaling 10,154,880 bytes. Its
terminal report SHA-256 is
`84452380c22ba22f415464875cd0278bf61c67382c43eca0f5cbc12a31087fc1`.

### Raw Fe-K relative-shift diagnostic

A response-independent diagnostic was frozen before opening any science
energy. It built each region's line template from only the other six regions,
fit a positive shift plus smooth continuum with Poisson deviance, and compared
the centered seven-region result with the paper's detector-region table only
after fitting. This is a calibration/topology check, not a plasma or absolute
velocity measurement.

| Frozen gate | Result | Pass? |
|---|---:|---|
| Centered RMSE versus published detector velocities | 108.19 km/s (limit 100) | No |
| Spearman ordering | 0.464 (minimum 0.6) | No |
| Centered sign agreement | 3/7 (minimum 5/7) | No |
| Constant-velocity rejection | 2.47 sigma (minimum 3) | No |
| Most blueshifted region | b' (required b') | Yes |
| Most redshifted region | e' (required e') | Yes |
| Bootstrap completion | 150/150 | Yes |
| Maximum regional uncertainty | 154.54 km/s (limit 200) | Yes |

The result is mixed rather than empty: it recovers both extrema and a
441 km/s full range, but not the intermediate ordering. The He-like and
H-like sub-bands also diverge, and three low-count H-like fits report optimizer
non-convergence. That pattern is compatible with the empirical template
confusing temperature-dependent Fe line ratios with energy shifts. The paper
independently reports detector temperatures spanning about 7.7–9.8 keV and a
different He-like/H-like velocity in `e'`. This specific failure must be
diagnosed before authorizing response production; no template smoothing,
window, region, bound, or threshold may be retuned to the observed values.

The terminal report SHA-256 is
`df73f2bdb944c2d0349b0e64c997ffc64163110468f4a094f438e7baf2b9b941`;
the diagnostic plot SHA-256 is
`cbe7b28f848674129f1bfe2818d9f2bf10fc7bf465ad287eb508d7eab8d0fe4d`.

The subsequent frozen-report-only diagnosis did not reopen or refit an event.
It found four independent warnings: three non-converged sparse H-like fits, a
maximum 400 km/s He/H disagreement, a published 2.18 keV detector-temperature
span, and 77–155 km/s primary uncertainties. Calibration execution and exact
region partitions remain intact, and both velocity extrema were recovered.
Calibration failure is therefore disfavored but explicitly not ruled out. The
failed empirical template remains failed; the authorized next closure is a
materially different response-aware absorbed `bapec` analysis with NXB, broad,
narrow, and shared-velocity two-temperature checks. Validation and holdout
data remain sealed. The diagnosis report SHA-256 is
`b41524c93b591b2fa7deddbd79ce33914113350d9573c113795aa362b5df1caf`.

## The new observable source terms

The signed projected gas current is

\[
J_{\parallel}(\mathbf x)
=
\Sigma_g(\mathbf x)
\left[v_{\rm los}(\mathbf x)-v_{\rm sys}\right].
\]

Unlike density, temperature, or pressure, this changes sign if the motion is
reversed. It is therefore genuine time-odd evidence. We also construct the
time-even kinetic stress

\[
\Pi_{\parallel}(\mathbf x)
=
\Sigma_g(\mathbf x)
\left(
[v_{\rm los}-v_{\rm sys}]^2+\sigma_v^2
\right).
\]

We then test, without lensing, whether the frozen I4 thermodynamic-gradient
axis follows the observed velocity-gradient axis and whether I5 baroclinicity
tracks kinetic-stress activation. I5 remains scalar and can never substitute
for I4's direction.

## What must pass

Each validation or holdout cluster needs at least eight independent usable sky
regions, with at least 75% having velocity uncertainty no larger than
200 km/s. Broad one-temperature, broad two-temperature/shared-velocity, and
narrow Fe-K fits must retain the same sign topology. Their shifts must be no
larger than both 100 km/s and one combined standard deviation, and the velocity
gradient axis may move by at most 15 degrees.

A time-odd source is admitted only if all three systems independently:

- reject a spatially constant velocity field at at least 3 sigma;
- detect signed projected current at at least 3 sigma;
- retain at least 20% velocity variance unexplained by gas density,
  temperature, and member light;
- preserve the sign under leave-one-region-out tests at least 90% of the time.

No cluster or spectral branch is averaged away. The holdout is opened only
after development and validation pass with frozen code and thresholds.

## Possible decisions

| Result | Consequence |
|---|---|
| Direct velocity field fails robustness | Do not admit any dynamic source |
| Signed current passes but I4/I5 fail | Admit P2 current/memory to mathematical comparison; retire I4/I5 for this route |
| Signed current and I4/I5 pass | Compare P2 with the supported thermodynamic placement using constraints and degrees of freedom |
| Only kinetic stress passes | Evidence is time-even; P2 remains unauthorized |

Even a complete pass does not demonstrate modified gravity. It only shows that
ordinary baryons contain a stable source structure worth placing in a
covariant theory. Lensing, halo maps, gravity fitting, Solar-System tuning, and
action derivation remain closed during this protocol.

The complete frozen specification is
[`sigma_v19cy_direct_icm_velocity_evidence.json`](../configs/sigma_v19cy_direct_icm_velocity_evidence.json).

## Public sources

- [NASA HEASARC XRISM archive](https://heasarc.gsfc.nasa.gov/docs/xrism/archive/index.html)
- [XRISM data organization and public archive documentation](https://heasarc.gsfc.nasa.gov/docs/xrism/analysis/abc_guide/XRISM_Data_Specifics.html)
- [Published Abell 2319 Resolve velocity analysis](https://arxiv.org/abs/2508.05067)
- [Independent XMM-Newton velocity-map demonstration in merging Abell 3266](https://arxiv.org/abs/2408.00837)
