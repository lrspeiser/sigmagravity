# P0554 route/softness interaction results

## Outcome

The first direct combination of the two previously separate useful effects is
complete:

- a slightly softer radial photon-addition law improves continuous lensing
  residuals but does not recover an exact root from the observed MACS1931
  image seed; and
- the conservative A0279 angular route recovers that observed-seed root but
  does not improve the already-complete clusters.

Together they recover **18/18 observed-seed raw image roots in all five clusters**,
retain the exact P0554 galaxy and Solar predictions, and improve the CLASH
summary error from 0.19908 to 0.19641 dex. This is a real structural advance
over either component alone.

A later frozen global search sharpens this statement. Every formula already
has at least the three roots required by the observed MACS1931 family; the
17/18 versus 18/18 label instead tracks an exact **3-to-5-root bifurcation**.
The two extra roots lie near image 2c and create a companion-image prediction,
not an unqualified success. See
[`P0554_CAUSTIC_MARGIN_RESULTS.md`](P0554_CAUSTIC_MARGIN_RESULTS.md).

It is not an accuracy solution. On the four systems where scalar P0554 and the
combined formula are both complete, the improvement is only **0.310%**. The
combined historical-validation RMS is 18.144 arcsec, 1.816 times the limited
compact-halo comparator's 9.989 arcsec. Seventy-two of 90 geometry fits touch a
nuisance bound. No formula is promoted.

## Equation tested

The radial parent remains P0554. Its lensing enhancement is generalized from
ordinary addition to the parent-preserving mean

$$
 A_{\rm lens}=\mathcal A_{k_\gamma}
 \left[\mu_\gamma(A_{\rm dyn}-1)\right],
 $$

where $\mathcal A_1(x)=1+x$. The interaction parent uses
$k_\gamma=0.98$; galaxies still respond to $A_{\rm dyn}$ and are therefore
unchanged.

Only the excess above baryonic GR is angularly redistributed:

$$
 \boldsymbol\alpha=\boldsymbol\alpha_{\rm radial}
 + f_{\rm route}^{,p}\,\delta\boldsymbol\alpha_{\rm A0279},
 \qquad p=2.5.
$$

The A0279 field routes member-weighted excess toward the baryonic light center.
Its annular convergence mean and independently sampled circular radial
deflection are removed. Consequently it changes two-dimensional image geometry
without changing the radial SPARC, CLASH, or point-mass Solar controls.

The interaction grid changed one route coordinate at a time by approximately
4--5%, and changed $k_\gamma$ from 0.98 to 0.97 or 0.99. Six ordinary lens
geometry parameters were refit with eight starts in every formula/system pair;
no gravity or route parameter was fitted to a cluster.

## Coverage

| Domain | Coverage |
|---|---:|
| SPARC | 131 galaxies, 968 outer held-out points |
| CLASH radial acceleration | 20 systems, 84 points |
| Raw strong lensing | 5 clusters, 18 held-out images |
| Exact lens geometry fits | 90 |
| Conservative angular fields audited | 80 |
| Solar controls | Cassini, Earth, Saturn, Mercury proxies |

## What each component does

The raw comparison is restricted to the four systems complete for both scalar
P0554 and the candidate. The full five-system RMS is shown only when the
candidate has all roots; it cannot be compared directly with an undefined
five-system scalar RMS.

| Formula | Matched four-system RMS | Change vs P0554 | Observed-seed roots | Candidate-only full RMS |
|---|---:|---:|---:|---:|
| Scalar P0554 | 18.536 arcsec | reference | 17/18 | undefined |
| Photon softness 0.98 only | 18.068 | **+2.525%** | 17/18 | undefined |
| A0279 route only | 18.563 | -0.149% | **18/18** | 17.207 |
| Combined parent | 18.478 | +0.310% | **18/18** | 16.839 |
| Combined, route power 2.4 | 18.474 | +0.332% | **18/18** | **16.808** |

The two effects therefore solve different problems, but they are not additive.
Across the four mutually complete systems:

| Formula | Equal-system RMS |
|---|---:|
| P0554 | 18.5357 arcsec |
| Photon softness alone | 18.0676 |
| Route alone | 18.5633 |
| Combined | 18.4782 |

The descriptive aggregate interaction is +0.3829 arcsec. If the effects added
independently it would be near zero. Instead, adding the route cancels most of
the radial softness gain after lens geometry is refit.

## Small-change impact ranking

Log elasticity measures the fractional raw-RMS response per fractional
parameter change. It is evaluated only on clusters with all roots in both
directions. A changed root count is reported separately because an undefined
image solution is not an RMS value.

| Parameter | Low--high test | Common complete clusters | Raw RMS span | Log elasticity | Root-count change |
|---|---:|---:|---:|---:|---:|
| Photon-addition softness | 0.97--0.99 | 4 | **0.3121 arcsec** | **0.8354** | 1 |
| Return width | 57--63 kpc | 5 | 0.0489 | 0.02895 | 0 |
| Base routed fraction | 0.4275--0.4725 | 4 | 0.0461 | 0.02491 | 1 |
| Route-amplitude power | 2.4--2.6 | 4 | 0.0364 | 0.02458 | 1 |
| Source-light exponent | 0.95--1.05 | 4 | 0.0425 | 0.02301 | 1 |
| Return length | 237.5--262.5 kpc | 4 | 0.0168 | 0.00907 | 1 |
| Extent slope | 0.95--1.05 | 5 | 0.0063 | 0.00373 | 0 |

Photon softness is about 29 times as elastic as return width, the strongest
smooth route coordinate. Route coordinates are weak continuous RMS levers once
ordinary geometry can adjust. Their important effect is image topology. The
later global diagnostic shows that the toggled topology is a three-to-five-root
transition near image 2c.

## The topology boundary is not a single amplitude threshold

Small changes around the combined parent toggle the same observed-seed
MACS1931 solution and, in a global search, the same nearby extra root pair:

- $k_\gamma=0.97$ loses it, while 0.99 retains it;
- route power 2.4 retains it, while 2.6 loses it;
- lower routed fraction retains it, while higher routed fraction loses it;
- shorter return length retains it, while longer return length loses it; and
- a shallower source-light exponent loses it, while a steeper exponent retains
  it.

Both tested return widths and both extent slopes retain all roots. The mixed
directions are decisive: root recovery is not controlled by one monotonic
"more routing" amplitude. It depends on where the two-dimensional caustic is
placed by the shape of the redistributed field.

## Cross-domain effects

| Formula | SPARC outer RMSE | CLASH RMSE | Mercury proxy | Solar pass |
|---|---:|---:|---:|---|
| P0554 | 12.5709 km/s | 0.19908 dex | -1.730 mas/century | yes |
| Photon softness 0.98 | 12.5709 | 0.19641 | -1.730 | yes |
| A0279 route | 12.5709 | 0.19908 | -1.730 | yes |
| Combined parent | 12.5709 | 0.19641 | -1.730 | yes |

This preservation is built into the zero-monopole experiment and must not be
misread as an independent galaxy or Solar success. Its value is that it
isolates angular topology from radial amplitude.

## Universal lessons

1. **Continuous accuracy and image existence are separate observables.** The
   photon response controls the former; the angular route controls the latter.
2. **A radial and an angular improvement cannot be assumed to add.** Their
   +0.3829-arcsecond interaction removes most of the radial gain.
3. **Route geometry is weak in ordinary RMS but strong near a caustic.** Tiny
   changes often leave four clusters nearly unchanged while creating or
   deleting an extra MACS1931 image pair near 2c.
4. **The root boundary is multidimensional and nonmonotonic.** A single routed
   fraction or strength parameter cannot characterize it.
5. **Width is the strongest smooth route-shape lever that keeps every root in
   both directions, but its effect is small.** A 10% full span changes the
   five-system RMS by only 0.049 arcsec.
6. **Extent dependence is locally almost inert after exact refitting.** Its
   full span changes RMS by only 0.006 arcsec despite its earlier map-level
   usefulness.
7. **The next field law must predict caustic topology from complete baryonic
   geometry, not tune a radial coefficient.** Gas and diffuse ICL are the most
   important absent source maps.

## Why this still does not challenge a dark-matter fit

The best complete descriptive formula is the combined route with $p=2.4$.
It has 16.808-arcsecond full five-system RMS, but the scalar baseline has no
defined five-system comparator because it misses an image. On the historical
validation pair it gives 18.081 arcsec, 1.810 times the limited compact-halo
control. Its matched MACS1115 change is a 0.0041% worsening.

Moreover, 72/90 fitted geometries touch at least one declared nuisance bound.
This simplified radial-plus-shear lens is not a precision alternative to a
modern multi-component cluster model. The supported claim is narrower:

> A conservative baryon-derived angular route can robustly alter an extra-pair
> caustic topology while an independent photon law controls continuous lensing
> strength, but their present combination does not predict accurate cluster
> image positions.

## Numerical audits and limits

The maximum normalized curl RMS is $3.54\times10^{-17}$ and the maximum
annular convergence-mean error is $4.21\times10^{-16}$. The correction is
therefore a scalar lens-potential field to numerical precision, not an
arbitrary curl-bearing vector patch.

All components, formulas, and clusters are spent exploratory evidence. Member
catalogs omit a registered hot-gas map and separately measured diffuse ICL;
CLASH acceleration targets were reconstructed with conventional assumptions;
and the Solar checks remain analytic proxies. No covariant action has yet been
derived for the route operator.

## Reproduction

```powershell
python scripts/run_p0554_route_softness_interaction.py
python -m pytest tests/test_p0554_route_softness_interaction.py -q
```

Machine-readable outputs and the diagnostic figure are in
`results/p0554_route_softness_interaction/`.
