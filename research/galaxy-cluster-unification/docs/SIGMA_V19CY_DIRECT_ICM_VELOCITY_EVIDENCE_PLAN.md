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

Abell 2319 is development-only because its result is already known. The
published analysis measured five sky regions and found a roughly 300 km/s
velocity range across the core, including a region blueshifted by about
230 km/s and a region with roughly 400 km/s velocity dispersion. We first have
to reproduce those values and their spatial-spectral-mixing treatment.

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
