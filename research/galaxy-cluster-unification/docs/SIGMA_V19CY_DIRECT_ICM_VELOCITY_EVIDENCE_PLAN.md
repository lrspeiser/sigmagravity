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
