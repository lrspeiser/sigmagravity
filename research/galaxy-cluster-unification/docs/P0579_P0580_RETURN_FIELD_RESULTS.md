# P0579-P0580 baryonic return-field results

## Result in plain language

The inverse-derived return rule produced the first encouraging raw transfer of
this route family: without using either new cluster's inferred dark-halo map to
place endpoints, its locked geometry improved held-out image consistency in
both SMACS0723 and SPT0615. The equal-cluster gain over a 100-kpc baryon-light
control was 12.20%.

That result is not yet a pass. Its SMACS deflection has mass-sheet similarity
`R2=0.960`, just above the frozen 0.95 limit, and only two clusters are
available. A much larger 432-setting calibration search overfit badly.

Applying the same kernel as a strictly conservative redistribution on SPARC
moved 110 of 131 galaxies in the correct direction but barely changed the
overall error: 72.40 to 70.93 km/s, compared with 10.35 km/s for fixed RAR.
This establishes a useful boundary: **route geometry can help determine where
cluster lensing appears, but redistribution of a fixed baryonic gravity budget
cannot supply the missing galaxy-scale amplitude.**

## Field representation

The conservative two-dimensional route law is

\[
\Sigma_{\rm eff}=(1-f_{\rm eff})B_w+f_{\rm eff}R_{\ell,w,m}[B],
\qquad
f_{\rm eff}=q_{\rm route}s(C),
\]

\[
C={R_{50}\over R_{80}},\qquad
s(C)={1\over1+\exp[-\beta(C-0.648526)]},
\]

where `B_w` is member light deposited at its observed locations, `R` moves the
same normalized source weights along a return route, `ell=lambda*R80`, and
`w=eta*R80`. Both components integrate to one, so the directional kernel does
not create map-integrated source strength.

Four route-residence interpretations were frozen:

1. endpoint only;
2. uniform residence along the chord;
3. an outward radial bow of height `0.5 R80`; and
4. equal left/right transverse bows of height `0.5 R80`, which insert no
   handed direction.

The previously inferred values were locked at `lambda=0.36`, `eta=0.23`, the
standard concentration gate, endpoint deposition, and full routed fraction.

## P0579 raw cluster test

The inputs were 29 raw image positions in nine source subfamilies. Four
subfamilies were used for normalization/calibration and five remained held
out. Each candidate received one positive amplitude per cluster; route
geometry was shared.

### Locked inverse replay

| Cluster | B100 held-out RMS | Locked return RMS | Improvement |
|---|---:|---:|---:|
| SMACS J0723.3-7327 | 6.385 arcsec | 5.852 arcsec | 8.34% |
| SPT-CL J0615-5746 | 7.569 arcsec | 6.399 arcsec | 15.46% |
| Equal-cluster mean | 6.977 arcsec | 6.126 arcsec | 12.20% |

The locked rule improved three of five held-out subfamilies, exactly the frozen
60% gate. Its maximum mass-sheet `R2` was 0.960: much lower than the failed
fractional propagators and lower than the B100 control in SMACS, but still over
the predeclared 0.95 cutoff. Consequently every performance gate passed except
the mass-sheet gate.

The prior normalized-map-selected route did not transfer: its raw held-out RMS
was 10.897 arcsec. This matters because it shows that normalized convergence
shape and raw lens-equation response select different route geometries.

### What the 432-setting selection did

Calibration selected a no-gate, `0.5 R80`, `0.28 R80` symmetric transverse
arc at full fraction. It achieved 2.000 arcsec on the small calibration set but
14.724 arcsec held out, 111% worse than B100, and `R2=0.9963`. This is a clear
multiple-testing and degeneracy failure, not a candidate theory.

The held-out parameter spans, which are descriptive after opening the holdout,
were:

| Coordinate | Held-out RMS span |
|---|---:|
| Route-residence mode | 8.527 arcsec |
| Endpoint width | 1.796 arcsec |
| Return distance | 1.342 arcsec |
| Concentration gate | 0.631 arcsec |
| Routed fraction | 0.387 arcsec |

Endpoint deposition was best on held-out images and transverse residence was
worst. This is the largest robust qualitative result of the sweep: **whether
the field contributes only where it returns or all along its route matters far
more than small changes to the gate or routed fraction.**

A post-hoc low-degeneracy setting reached 5.413 arcsec with `R2=0.859`, but it
was not selected by calibration and is not validation evidence.

## P0580 conservative SPARC translation

For each galaxy the measured baryonic acceleration was translated into a
spherical force-equivalent cumulative source,

\[
M_b(<r)={g_b(r)r^2\over G}.
\]

Positive shell increments were moved through the same frozen route geometries,
and

\[
M_{\rm eff}(<r)=(1-f_{\rm eff})M_b(<r)+f_{\rm eff}M_{\rm route}(<r),
\qquad
V^2(r)={GM_{\rm eff}(<r)\over r}.
\]

All routed profiles were renormalized to the original total force-equivalent
mass. No multiplicative extra-gravity amplitude was fitted.

| Formula | Outer RMS |
|---|---:|
| Newtonian baryons | 72.399 km/s |
| Locked inverse return | 70.926 km/s |
| Best of 432, post hoc | 69.321 km/s |
| Prior scalar arc-apogee R1322 | 12.966 km/s |
| Fixed RAR | 10.348 km/s |

The locked return improved 84.0% of galaxies and the post-hoc best improved
90.8%, showing that inward relocation has the right sign. Its magnitude is
fundamentally limited: outside a surface enclosing the routed source, the
enclosed budget is unchanged.

The conservative SPARC impact ranking was:

| Coordinate | Median outer-RMS span |
|---|---:|
| Route mode | 1.823 km/s |
| Return distance | 0.476 km/s |
| Width | 0.224 km/s |
| Concentration gate | 0.047 km/s |
| Routed fraction | 0.026 km/s |

Endpoint-only return and longer travel were best. Once again, route residence
dominates the environmental gate.

## Solar and conservation audits

The map-integrated conservation error was `4.44e-16` in P0579 and the radial
mass-conservation error was below the stored `1e-10` gate in P0580. For an
isolated point source, `R50=R80=0`; launch and return coincide, so this
directional redistribution gives exactly zero Solar force or perihelion
change. This statement applies to the conservative directional channel, not
to an independently added scalar enhancement.

## What this changes in the research direction

1. A target-blind, baryon-derived endpoint kernel has a real but preliminary
   raw-lensing transfer signal.
2. Path residence is the most impactful new formula choice. Treating every
   point along an arc as an additional source is disfavored; arrival-only
   response is more promising.
3. Pure conservative redirection is far too weak for galaxy rotation even
   though it helps most galaxies in sign.
4. The next field equation therefore needs two derived pieces: a scalar
   response that determines total low-acceleration strength and a normalized
   endpoint kernel that determines where that response appears.
5. Those pieces cannot simply be fitted independently by object. A defensible
   model must derive both from the same baryonic invariants and retain one
   universal strength law.

The next informative formula change is an **arrival-weighted nonlocal response**:
keep ordinary baryonic gravity local, generate one universal low-acceleration
excess, and route only that excess through the endpoint kernel. This preserves
the successful Solar screen and avoids depleting the ordinary local field.
It should be tested on a third untouched raw cluster before any refinement of
the `0.36` and `0.23` coefficients.

## Reproduction

```powershell
python scripts/run_p0579_extent_gated_return_raw.py
python scripts/run_p0580_conservative_return_sparc.py
pytest -q tests/test_p0579_p0580_return_field_results.py
```

Machine-readable outputs are in
`results/p0579_extent_gated_return_raw/` and
`results/p0580_conservative_return_sparc/`.
