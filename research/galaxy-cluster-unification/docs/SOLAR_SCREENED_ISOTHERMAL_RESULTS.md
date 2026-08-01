# Solar-screened one-parameter isothermal result

## Result

The tested law is

\[
g(r)=g_{\rm bar}(r)+
\lambda\,g_{\rm bar}(r_*)\frac{r_*}{r}
\frac{a_0}{a_0+g_{\rm bar}(r)},
\]

with fixed `a0 = 1.2e-10 m/s^2`, fixed `r* = 200 kpc`, and one fitted
universal parameter. The two development clusters select

\[
\boxed{\lambda=10.5}.
\]

No cluster-specific gravity amplitude, transition radius, or photon multiplier
was fitted. The same value was then used for Mercury, the Cassini force proxy,
the two cluster replay holdouts, and the two stress clusters.

The formula passes every gate in the frozen screened-law protocol. It predicts
a Mercury supplementary precession of only `-2.36e-7 mas/century`, well inside
the frozen `+/-3.1 mas/century` margin, and obtains `5.261 arcsec` on the
two-cluster replay holdout. It nevertheless remains a phenomenological survivor,
not a validated replacement for GR plus dark matter.

## What the screen does

The final factor is

\[
S(g_{\rm bar})=\frac{a_0}{a_0+g_{\rm bar}}.
\]

- In the Solar System, `g_bar` is many orders of magnitude above `a0`, so the
  extra force is strongly suppressed.
- At cluster accelerations near or below `a0`, the screen is substantially open,
  preserving most of the long-range `1/r` tail.
- The screen adds no fitted number because `a0` was fixed before the run. Its
  *functional form* is still theoretical flexibility and was proposed after the
  unscreened Mercury conflict was known.

## Frozen selection and holdout

| Role | Clusters | Purpose |
|---|---|---|
| Development | MACS0329, MACS0429 | Select one shared `lambda` from a frozen grid and refine it |
| Validation replay | MACS1115, MACS1931 | Score the locked value on six withheld images |
| Stress replay | RXJ1347, RXJ2129 | Apply the locked value after validation |

The coarse grid selected the neighborhood around `lambda=12`. The nine-point
refinement selected `lambda=10.5`; the lower value `10.0` had a better numeric
RMS but was ineligible because it lost a development-image root. The selected
value is not a refinement boundary and retains every development held-out root.

## Solar-System calculation

For the first Solar diagnostic, the Sun is treated as an isolated point source:

\[
g_{\rm bar}=\frac{GM_\odot}{r^2},\qquad
g_{\rm bar}(r_*)=\frac{GM_\odot}{r_*^2}.
\]

The additional secular precession is calculated by time-averaging the planar
Gauss planetary equation around each unperturbed Kepler orbit. The unit tests
verify that the numerical quadrature reproduces the closed unscreened `1/r`
result.

The frozen Mercury limit is the conservative INPOP15a criterion-1 result,
`0.0 +/- 3.1 mas/century`. The screen was not adjusted to center the prediction;
the cluster-selected `lambda=10.5` was inserted directly. See
[Fienga et al. (2016)](https://arxiv.org/abs/1601.00947).

| Planet | Unscreened precession | Screened precession | Screened extra force at semimajor axis |
|---|---:|---:|---:|
| Mercury | -26.223 mas/cy | **-2.36e-7 mas/cy** | 2.99e-19 |
| Venus | -19.393 | -6.16e-7 | 1.95e-18 |
| Earth | -16.492 | -1.00e-6 | 5.15e-18 |
| Mars | -13.333 | -1.87e-6 | 1.82e-17 |
| Jupiter | -7.222 | -1.19e-5 | 7.26e-16 |
| Saturn | -5.338 | -2.95e-5 | 4.47e-15 |
| Uranus | -3.760 | -8.40e-5 | 3.64e-14 |
| Neptune | -3.006 | -1.65e-4 | 1.40e-13 |

The Mercury prediction is about **13.1 million times smaller** than the allowed
one-sigma absolute margin. The unscreened control fails that margin by a factor
of 8.46, confirming that the screen—not a relaxed error bar—causes the pass.

The maximum screened force fraction between 1.6 solar radii and Saturn is
`4.47e-15`, versus the frozen `2.3e-5` Cassini proxy. Cassini directly measured
the PPN light-time parameter `gamma`, however, not this force fraction. The
measured result was `gamma = 1 + (2.1 +/- 2.3)e-5`; our zero-slip closure remains
an assumption until a covariant metric is derived. See
[Bertotti, Iess and Tortora (2003)](https://doi.org/10.1038/nature01997).

## Cluster replay result

| Model | Equal-cluster held-out RMS | Pooled coordinate chi-square | Fitted gravity parameters |
|---|---:|---:|---:|
| Screened isothermal tail | **5.261 arcsec** | **942.9** | 1 universal |
| Prior unscreened tail | 9.423 | 2202.4 | 1 universal |
| Compact cluster halo | 9.989 | 1615.4 | object-specific halo parameters |
| Baryons-only GR | 25.199 | 13751.6 | 0 |
| Simple MOND | 25.636 | 14808.2 | 0 fixed |
| Fixed RAR | 25.673 | not recorded here | 0 fixed |

The screened law reduces equal-cluster RMS by 44.2% relative to the unscreened
tail and 47.3% relative to the compact-halo aggregate. It also has 41.6% lower
pooled chi-square than that compact-halo comparator.

The per-cluster result is still uneven:

| Validation cluster | Screened law | Baryons GR | Compact halo |
|---|---:|---:|---:|
| MACS1115 | **1.800** | 29.931 | 14.057 |
| MACS1931 | 7.218 | 19.343 | **1.401** |

Thus the screened law wins the aggregate because of a very large improvement on
MACS1115; the compact halo remains much better on MACS1931. All six validation
roots and all nineteen validation-training roots converge.

## Gates and stress test

| Frozen screened-law gate | Result |
|---|---|
| Mercury within `+/-3.1 mas/cy` | Pass |
| Cassini fractional-force proxy | Pass |
| Every validation held-out root | Pass |
| Both validation clusters beat baryons | Pass |
| Validation RMS no worse than 1.25 times compact halo | Pass |
| Selected parameter interior | Pass |

For continuity, the earlier program's more aggressive absolute target of
`2 arcsec` is also reported: **it fails** at 5.261 arcsec. This was a post-result
continuity diagnostic, not a changed frozen gate.

The locked RXJ2129 stress replay scores `15.563 arcsec`, compared with `17.908`
for baryons but only `2.521` for the compact halo. Its error is 6.17 times the
halo error. RXJ1347 has no eligible within-family holdout; its training RMS is
5.154 versus 0.805 for the halo. The formula therefore has not demonstrated
uniform dark-halo-quality prediction across clusters.

## What this proves—and what it does not

This run shows that one fixed acceleration screen and one universal fitted
amplitude can simultaneously:

1. remove the direct Mercury conflict of the unscreened `1/r` tail;
2. preserve every root in the two-cluster validation replay; and
3. improve the limited compact-halo aggregate under both equal-cluster RMS and
   pooled coordinate chi-square.

It does not yet prove Solar-System compatibility or a fundamental theory:

- The Solar calculation is a first-order published-precession test, not a fit to
  raw MESSENGER, Cassini, Mars-ranging, or lunar-laser-ranging observations.
- The point-source normalization does not specify how the Sun, Milky Way, and
  external masses combine in one translation-invariant field equation.
- Zero gravitational slip was imposed rather than derived, so Cassini light
  propagation is not yet predicted from the same action.
- Every cluster used here appeared in earlier project analyses. A genuinely new,
  frozen cluster sample is still required.
- The screen form was designed after seeing the unscreened Mercury failure, so
  Mercury is a development constraint, not independent confirmation.

## Reproduce

```powershell
python -m pytest tests/test_one_parameter_lens.py tests/test_solar_system_tail.py -q
python scripts/run_solar_screened_isothermal.py
python -m pytest tests/test_solar_screened_isothermal_results.py -q
```

The complete machine-readable report, parameter grids, image predictions,
geometry, radial profiles, and Solar-System table are in
`results/solar_screened_isothermal/`.
