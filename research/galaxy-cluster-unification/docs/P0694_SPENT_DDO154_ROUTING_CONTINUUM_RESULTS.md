# P0694 spent DDO154 routing continuum

Frozen before intermediate-fraction scores: 2026-08-02

Verdict: no viable row; the shared linear source endpoint pair is **retired**
as a galaxy-cluster unifier

## Diagnostic

P0694 evaluates 13 preregistered fractions of

\[
S_f=(1-f)S_{\rm local}+fS_{\rm route},\qquad 0\le f\le1,
\]

on the spent real gas-plus-stellar DDO154 map. The P0693 value
`f=0.0835241` is an exact reproduction marker. No row may be selected,
promoted, or described as validation.

## Result

| Fraction | RMSE | Weighted RMSE | Mean bias | RMSE / algebraic MOND | Weighted / algebraic MOND | All gates |
|---:|---:|---:|---:|---:|---:|---|
| `0.0` | `3.94308728` | `3.27400669` | `-1.40952` | `1.3521870` | `2.6713901` | fail |
| `0.083524` | `3.94308728` | `3.27400670` | `-1.40952` | `1.3521870` | `2.6713901` | fail |
| `0.5` | `3.94308729` | `3.27400672` | `-1.40952` | `1.3521870` | `2.6713901` | fail |
| `1.0` | `3.94308731` | `3.27400674` | `-1.40952` | `1.3521870` | `2.6713902` | fail |

Zero of 13 rows pass. Both ordinary and weighted RMSE are minimized at
`f=0`; every positive routing fraction worsens them, although only in the
eighth decimal place. The P0693 marker reproduces its two errors exactly.

## Physical meaning

The continuum is almost perfectly flat because DDO154 never activates a
material difference between the two source endpoints. Its dimensionless
baryonic potential depth is far below the cluster transition, so the positive
extra generator available for relocation is negligible. Changing the routing
fraction cannot change the galaxy curve if there is essentially nothing to
route.

This closes two loopholes:

1. P0693 did not fail because `e_2D=0.0835` happened to choose a bad fraction.
   The entire allowed interval gives the same noncompetitive galaxy field.
2. A new transform, cap, or exponent applied to the global covariance scalar
   cannot repair DDO154 within these endpoints.

The shared `S_local`/`S_route` linear pair is therefore retired as a universal
galaxy-cluster equation. The useful RX J2129 source-relocation topology from
P0693 remains a mechanism clue, not a surviving joint theory.

## Next operator generator

The galaxy endpoint must change while the cluster topology correction is
preserved. A parameter-free path-potential construction is:

\[
\Phi_{\rm path}(\mathbf{x})=\Phi_N(\mathbf{x}_c)+
\int_0^1 -\nu\!\left({|\mathbf g_N(\mathbf{x}_c+t\mathbf d)|\over a_0}\right)
\mathbf g_N(\mathbf{x}_c+t\mathbf d)\cdot\mathbf d\,dt,
\]

where `d=x-x_c` and `x_c` is the baryonic centroid. Its radial derivative is
the algebraic MOND radial force along each baryon-centered ray, while the
gradient of the completed scalar potential remains curl-free. It is a genuine
path-integrated geometric object rather than another scalar amplitude gate.

The cluster topology correction can then be added as a zero-boundary
potential difference:

\[
\Phi_{\rm joint}=\Phi_{\rm path}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

This has no fitted physical constant. For galaxies with negligible routing
activation it uses the path endpoint; for morphologically extended clusters
it retains the measured relocation correction. This is only a formula
generator. It requires a no-observation numerical audit of path integration,
spherical limits, rotation covariance, boundary behavior, curl, and grid
convergence before any DDO154 or lens score is calculated.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0694_spent_ddo154_routing_continuum.py
python -m pytest tests/test_source_routing_qumond.py tests/test_p0635_ddo154_map_commissioning.py -q
```

Artifacts are in
`results/p0694_spent_ddo154_routing_continuum/`.

## Claim boundary

P0694 is spent mechanism evidence and finite sampling cannot exclude an
arbitrarily narrow numerical interval. Here, however, the monotonic changes
over the whole interval are many orders of magnitude smaller than the missing
galaxy performance. P0633 and P0640 remain sealed.
