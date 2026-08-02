# P0691 multipole-gated source-routing topology

Frozen before scores: 2026-08-02
Verdict: numerical field **passes**; the uniquely calculated global shape gate **fails raw topology**

## Tested operator

P0691 asks whether the amount of source routing can be calculated from the
registered baryonic shape instead of fitted to the lens. For the mass-weighted
baryonic covariance tensor `C`, it defines

\[
q_b=\sqrt{3\over2}{\|C-\mathrm{tr}(C)I/3\|_F\over\mathrm{tr}(C)}.
\]

This has fixed geometric limits: `q_b=0` for a spherical source and `q_b=1`
for a line-like source. The registered RX J2129 baryonic map gives
`q_b=0.1188629`. The source is then fixed without an adjustable blend:

\[
S_{\rm mix}=(1-q_b)S_{\rm local}+q_bS_{\rm route}.
\]

No gravity amplitude, photon amplitude, routing fraction, or object-specific
constant is fitted. The spherical limit is exactly the P0684 local generator;
the line-like limit is exactly the P0690 fully routed source.

## Frozen result

| Domain | Metric | Result | Gate | Verdict |
|---|---|---:|---:|---|
| geometry | calculated routing fraction | `0.118863` | `[0,1]` | pass |
| 3D field | normalized equation residual | `3.20e-14` | `<=1e-10` | pass |
| 3D field | mixed-source identity error | `0` | `<=1e-14` | pass |
| 3D field | median / RMS physical deflection | `10.39 / 10.22 arcsec` | median `1-20` | pass |
| 3D field | normalized deflection curl | `3.92e-16` | `<=1e-8` | pass |
| raw lens | training / heldout exact roots | `13/15`, `5/7` | `15/15`, `7/7` | fail |
| topology | missing / exact families | `3 / 4` | missing `0` | fail |
| topology | parity-diverse families | `5/7` | `7/7` | fail |
| topology | critical-curve families | `7/7` | `7/7` | pass |
| nuisance fit | parameters near a bound | `1` | `0` | fail |

The positional RMS values are infinite because the frozen scoring rule assigns
an infinite error whenever a required image root is absent. The optimizer cost
is `217.25`, but that cannot substitute for the failed root and topology gates.

## What this teaches us

The geometric blend is mathematically clean and produces a well-resolved,
curl-free field of plausible overall amplitude. It does not produce the
observed lens mapping. Families 2, 3, and 6 are missing images, and families 2
and 6 also fail parity diversity. One external-shear component reaches its
frozen bound.

The failure is informative because the candidate is not simply too weak or
too strong. Fully local routing and the `q_b` blend both miss three families,
whereas fully routed gravity misses only one family but creates two
potentially observable surplus families and grossly over-bends the radial
cluster tests. A single global shape number cannot encode where within the map
the effective source must move.

P0691 therefore rejects `q_b` as a sufficient global routing controller. It
does not reject all spatially varying or multipole-resolved routing laws.

## Next diagnostic

Before another function of `q_b` is invented, run a preregistered continuum
atlas on the fully spent RX J2129 system:

\[
S_f=(1-f)S_{\rm local}+fS_{\rm route}.
\]

The fractions and all topology gates must be frozen before their fields are
calculated. This atlas is diagnostic only: no row may be promoted or called a
validation result. If no interval produces the required topology, retire the
linear-routing family. If an interval does, the next hypothesis must predict
that spatial behavior from a baryonic field quantity and then face a new
frozen test; the best post-hoc fraction cannot itself become the theory.

## Reproduction

```powershell
python scripts/run_p0691_multipole_gated_source_routing_topology.py
python -m pytest tests/test_source_routing_qumond.py tests/test_source_routing_spherical.py tests/test_spatial_qumond_3d.py -q
```

Artifacts are in
`results/p0691_multipole_gated_source_routing_topology/`.

## Hosted researcher model

The eventual public simulator will expose this same distinction between a
frozen theory test and a diagnostic parameter atlas. A researcher can select a
catalog object or a seeded synthetic galaxy/cluster, submit a unit-checked
formula through a safe expression language, and receive immutable solver,
dataset, seed, comparator, and parameter-accounting hashes. The planned web
and API gateway runs on Vercel; isolated Cloud Run Jobs or Modal workers run
the field solves and lens-root searches asynchronously. See
[`PUBLIC_SIMULATOR_API_PLAN.md`](PUBLIC_SIMULATOR_API_PLAN.md).

## Claim boundary

RX J2129 and the inherited spherical controls are spent development evidence.
This result is not a relativistic metric theory and does not validate a
different light law. P0633 galaxy kinematics and P0640 raw cluster constraints
remain sealed.
