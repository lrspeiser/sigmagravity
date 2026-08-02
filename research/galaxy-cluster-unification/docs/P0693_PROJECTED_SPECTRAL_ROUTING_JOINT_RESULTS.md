# P0693 projected spectral routing joint screen

Frozen before candidate scores: 2026-08-02

Verdict: **fails jointly**; parameter-free cluster topology succeeds, but the
real-map galaxy remains noncompetitive with algebraic MOND

## Candidate

For the mass-centered projected baryonic covariance `C`, P0693 freezes

\[
e_{2D}=1-\frac{\lambda_{\min}(C)}{\lambda_{\max}(C)},
\qquad
S_{\rm spec}=(1-e_{2D})S_{\rm local}+e_{2D}S_{\rm route}.
\]

Galaxies use the registered face-on/deprojected gas-plus-stellar plane.
Clusters use the observed sky plane. All accepted map cells are included; no
threshold, fitted aperture, exponent, cap, multiplier, routing parameter,
gravity amplitude, or photon amplitude is allowed.

## Joint result

| Domain | Metric | Candidate | Comparator / gate | Verdict |
|---|---|---:|---:|---|
| DDO154 | calculated `e_2D` | `0.083524` | baryons only | pass |
| DDO154 | rotation RMSE | `3.943 km/s` | algebraic MOND `2.916` | fail (`1.352x`) |
| DDO154 | weighted RMSE | `3.274 km/s` | algebraic MOND `1.226` | fail (`2.671x`) |
| DDO154 | rotation RMSE | `3.943 km/s` | 3D QUMOND `3.936` | fail (`1.0018x`) |
| DDO154 | mean bias | `-1.410 km/s` | absolute `<=3` | pass |
| RX J2129 | calculated `e_2D` | `0.272023` | baryons only | pass |
| RX J2129 | physical median deflection | `12.64 arcsec` | `1-20` | pass |
| RX J2129 | training / heldout roots | `15/15`, `7/7` | exact coverage | pass |
| RX J2129 | training / heldout RMS | `0.601 / 2.670 arcsec` | each `<=3` | pass |
| RX J2129 | heldout / compact halo | `1.053x` | `<=1.25x` | pass |
| RX J2129 | missing / observable-surplus families | `0 / 2` | `0 / <=2` | pass |
| RX J2129 | parity-diverse / critical families | `7/7 / 7/7` | `7/7 / 7/7` | pass |
| RX J2129 | near-bound nuisances | `0` | `0` | pass |

Every numerical, identity, boundary, parameter-accounting, cluster-field,
raw-root, topology, and nuisance gate passes. Only the three preregistered
galaxy-comparison gates fail, so the candidate does not advance.

## Strong cluster result, limited claim

P0693 is the cleanest cluster result in the source-routing branch. The
parameter-free baryonic geometry lands inside the P0692 topology transition
without selecting its fraction from lens residuals. Five families have exact
global multiplicity; two contain potentially observable surplus roots. All
seven recover both parities and critical curves. The heldout positional error
is only 5.3% above the object-specific compact-halo comparator.

This still is not independent evidence. The equation was generated after the
spent P0692 atlas exposed the useful interval, RX J2129 is fully spent, and the
zero-slip photon closure is not a relativistic metric theory.

## Galaxy failure anatomy

The DDO154 total baryons are nearly circular because gas supplies 94.4% of the
map mass. Gas alone gives `e_2D=0.0595`, stars give `0.7498`, and the nominal
total gives `0.0835`. The resulting field barely moves away from the local
endpoint:

- P0693 spectral mixture: `3.943 km/s` RMSE;
- ordinary 3D QUMOND: `3.936 km/s`; and
- axisymmetric P0693 control (`e_2D=0`): `4.023 km/s`.

The spectral controller is therefore not damaging an otherwise competitive
galaxy endpoint. The underlying non-spherical full-field endpoint itself is
about 35% worse than the algebraic circular-orbit MOND comparator. The
weighted discrepancy is larger because the field curve underpredicts the
inner points that carry small observational uncertainties.

The 13 sealed baryon maps, inspected without kinematics, span total
`e_2D=0.016-0.776` with median `0.517`. A controller that is harmless for
gas-dominated DDO154 could create large changes in other dwarfs. Those outcomes
remain sealed, so this is a falsifiability warning rather than a score.

## Next diagnostic

Before changing the controller, freeze a DDO154-only continuum of

\[
S_f=(1-f)S_{\rm local}+fS_{\rm route},\qquad 0\le f\le1.
\]

This diagnostic cannot select a fraction or advance a candidate. It asks
whether any permitted mixture of the current endpoints can close the
full-field/algebraic galaxy gap at all.

- If no row becomes competitive with algebraic MOND, retire this shared linear
  endpoint pair as a galaxy-cluster unifier. A new operator must change the
  radial galaxy endpoint while preserving the cluster topology mechanism.
- If some row is competitive, identify a baryon-derived *spatial* quantity
  that predicts the response without adopting a spent fraction, and freeze it
  jointly before another score.

The P0693 failure already rejects the global covariance scalar as a universal
controller. No alternative transform of `e_2D` may be fitted to DDO154 or RX
J2129.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0693_projected_spectral_routing_joint_screen.py
python -m pytest tests/test_source_routing_qumond.py tests/test_source_routing_spherical.py tests/test_spatial_qumond_3d.py tests/test_potential_channel_qumond.py tests/test_p0635_ddo154_map_commissioning.py -q
```

Artifacts are in
`results/p0693_projected_spectral_routing_joint_screen/`.

## Public simulator implication

This is exactly the distinction the hosted researcher API must preserve: a
model may pass a cluster suite while failing its conjunctive universal score.
API responses must expose domain-level gates, comparator ratios, surplus-image
counts, and sealed/open evidence status rather than returning one aggregate
leaderboard number. See
[`PUBLIC_SIMULATOR_API_PLAN.md`](PUBLIC_SIMULATOR_API_PLAN.md).

## Claim boundary

P0693 uses spent DDO154 and RX J2129 outcomes. The 13 P0639 maps contribute
baryonic geometry only; no kinematics or candidate residual was opened. P0633
and P0640 outcomes remain sealed.
