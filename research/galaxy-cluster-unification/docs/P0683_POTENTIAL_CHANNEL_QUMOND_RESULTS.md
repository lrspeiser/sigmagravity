# P0683 potential-channel QUMOND reconnaissance

Frozen before scores: 2026-08-02  
Verdict: numerical integrity **pass**; dimension-fixed primary **does not advance**

## Equation tested

P0683 starts from a QUMOND-style source boost rather than adding separate
galaxy and cluster formulas:

\[
\nabla^2\Phi=\nabla\!\cdot\left[
\nu_0\!\left({|\nabla\Phi_N|\over a_0}\right)^{p(\chi_b)}
\nabla\Phi_N\right],
\]

\[
\nu_0(y)={1\over1-e^{-\sqrt y}},\qquad
\chi_b={|\Phi_b|\over c^2},
\]

\[
S={\chi_b^n\over\chi_b^n+\chi_t^n},\qquad
p(\chi_b)=1+(p_\infty-1)S.
\]

The primary hypothesis fixes `p_infinity=4` and `n=2`, motivated by a smooth
transition from one local channel toward four spacetime-channel factors. Only
`chi_t` is a fitted universal setting. This interpretation is motivation, not
a microscopic derivation.

The fixed base boost is exactly the spent RAR comparator in the one-channel
spherical limit and approaches one exponentially at Solar accelerations. An
earlier pre-score draft used the slow-tailed simple AQUAL interpolation; it was
replaced and recommitted before any P0683 grid score because it necessarily
failed the frozen Earth-force proxy.

## Frozen coverage

- 264 combinations of endpoint exponent, transition power, and `chi_t`;
- primary subgrid: 11 `chi_t` values at `p_infinity=4`, `n=2`;
- 968 already-spent outer points in 131 SPARC galaxies;
- 72 radial deflection targets across six spent clusters;
- no per-object gravity setting;
- no raw lens roots, 3D field solve, or sealed P0633/P0640 outcome.

## Primary result

The selected primary value is

\[
\chi_t=3\times10^{-6}.
\]

| Metric | Selected | Frozen gate | Result |
|---|---:|---:|---|
| Galaxy equal-system RMSE | `11.308 km/s` | no more than `1.05x` fixed RAR | pass (`1.035x`) |
| All-five cluster log RMS | `0.285 dex` | `<=0.200 dex` | fail |
| Reliable-three cluster log RMS | `0.309 dex` | `<=0.200 dex` | fail |
| Fixed-RAR cluster gap closed | `51.2%` | `>=75%` | fail |
| Solar force proxies | zero at reported precision | frozen limits | pass |

Comparators on the identical spent data are:

| Comparator | Galaxy RMSE (km/s) | all-five cluster log RMS (dex) |
|---|---:|---:|
| Newton/baryons | 67.298 | 0.988 |
| fixed RAR | 10.925 | 0.583 |
| simple algebraic AQUAL | 11.032 | 0.579 |

The primary is therefore meaningful progress relative to fixed RAR on the
cluster radial field, but it remains about a factor `10^0.285 = 1.93` from the
compact-halo radial target and cannot advance to topology.

## Failure anatomy

The failure is not simply too little or too much universal amplitude. The
selected law underpredicts the required deflection in MACS0329, MACS0429, and
MACS1115, but overpredicts it in MACS1931 and RXJ2129. Median
prediction/target ratios are approximately `0.60`, `0.69`, `0.62`, `2.06`,
and `2.44`, respectively.

Potential depth cleanly separates spent galaxies from clusters, but orders
the cluster response in the wrong direction: deeper cluster potentials receive
more channel activation, while the P0682 target tends to require less radial
amplification in those systems.

No grid row passes the galaxy, all-five-cluster, and reliable-three-cluster
gates together. The nearest galaxy-safe diagnostic uses `p_infinity=3`,
`n=4`, and `chi_t=1.5e-6`: it scores `1.003x` fixed RAR on galaxies and
`0.191 dex` on all five clusters, but misses the reliable-three gate at
`0.227 dex` and closes only about `67%` of the fixed-RAR cluster gap. It also
does not have the frozen dimension-fixed form and cannot be promoted post hoc.

## New development clue

A post-result baryonic audit—not a preregistered P0683 result—finds that the
required P0682 radial ratio is inversely ordered by

\[
\eta={|\Phi_b|\over r g_b},
\]

the baryonic potential path ratio. Across the five non-boundary clusters,
`Spearman rho=-0.90` (`p=0.037`, only five systems). The ratios range from
about `1.4` to `14.4` in their lens annuli, while most spent galaxy points lie
near one. This is hypothesis-generating evidence for *path dilution inside the
cluster branch*: extended profiles may distribute the available response over
more radial acceleration lengths.

The next allowed equation should keep the successful potential-depth onset but
reduce its channel exponent with a dimensionless function of `eta`, frozen
before another score. It must not use cluster identity, target halo amplitude,
or lens-image radii as formula inputs.

## Reproduction

```powershell
python scripts/run_p0683_potential_channel_qumond_reconnaissance.py
python -m pytest tests/test_potential_channel_qumond.py -q
```

Artifacts are in `results/p0683_potential_channel_qumond_reconnaissance/`.

