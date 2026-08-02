# P0692 spent linear source-routing continuum atlas

Frozen before intermediate-fraction scores: 2026-08-02

Verdict: one diagnostic row is viable; the sampled linear family is **not
retired**, but no fraction is selected or advanced

## Question and evidence class

P0692 does not fit a new theory. It maps the fully spent RX J2129 behavior of

\[
S_f=(1-f)S_{\rm local}+fS_{\rm route}
\]

at 17 preregistered fractions from zero to one. Every row uses the same
baryonic endpoints, boundary, zero-slip photon closure, nuisance bounds, 12
optimizer starts, and global-root search. The P0691 quadrupole value
`f=0.1188629` is a registered reproduction marker. No row may be promoted,
selected as a constant, or called validation evidence.

## Main result

| Routing fraction | Median deflection | Train roots / RMS | Heldout roots / RMS | Missing families | Observable surplus | Both parities | Near-bound nuisances | All gates |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0.118863` | `10.39 arcsec` | `13/15 / inf` | `5/7 / inf` | 3 | 0 | 5/7 | 1 | fail |
| `0.15` | `10.86 arcsec` | `15/15 / 0.953` | `7/7 / 2.912` | 1 | 0 | 7/7 | 1 | fail |
| `0.20` | `11.61 arcsec` | `15/15 / 0.735` | `7/7 / 2.574` | 1 | 0 | 7/7 | 0 | fail |
| **`0.30`** | **`13.05 arcsec`** | **`15/15 / 0.495`** | **`7/7 / 2.692`** | **0** | **2** | **7/7** | **0** | **diagnostic pass** |
| `0.40` | `14.53 arcsec` | `15/15 / 0.722` | `7/7 / 5.947` | 1 | 2 | 7/7 | 0 | fail |
| `0.50` | `16.08 arcsec` | `15/15 / 1.089` | `7/7 / 7.682` | 0 | 5 | 7/7 | 1 | fail |
| `1.00` | `23.81 arcsec` | `15/15 / 3.486` | `6/7 / inf` | 1 | 3 | 7/7 | 3 | fail |

The `f=0.30` row is the only row that passes every frozen diagnostic gate. Its
heldout RMS is `2.692 arcsec`, or `1.0615x` the object-specific compact-halo
comparator at `2.536 arcsec`. Five families have exact global multiplicity and
two have potentially observable surplus images. All seven contain both
parities and a critical curve.

The registered P0691 marker reproduces its field amplitude, training and
heldout root counts, missing-family count, and parity count exactly. This
confirms that the transition is not a code-path discrepancy.

## What changes across the continuum

The atlas separates three regimes:

1. Below about `f=0.15`, the field is too locally hollow to create all of the
   observed image branches. Missing roots and missing parities dominate.
2. Around `f=0.15-0.30`, all observed-image root searches converge and the
   global topology fills in. At `f=0.20`, only one global family root is still
   missing; at `f=0.30`, no family is missing.
3. Above about `f=0.40`, extra critical structure grows faster than the useful
   roots. Heldout positions worsen, surplus images proliferate, and nuisance
   parameters return to their bounds.

This is not a simple amplitude optimum. The useful interval is a topology
bifurcation: too little relocation lacks image branches; too much relocation
creates branches that should have been seen.

## The next formula generator

The value `0.30` cannot be adopted as a universal routing fraction because it
was exposed by a spent atlas. A new law must calculate comparable behavior
from baryons before another score is opened.

One clean post-hoc generator is the spectral anisotropy of the projected
baryonic covariance:

\[
e_{2D}=1-{\lambda_{\min}(C_{xy})\over\lambda_{\max}(C_{xy})}.
\]

It has exact limits zero for a circular projected source and one for a line.
RX J2129 gives `e_2D=0.272023`, near the only viable sampled row without a
fitted coefficient. Unlike the P0691 three-dimensional Frobenius quadrupole,
this statistic measures the strongest projected extent contrast and avoids
dependence on the artificial line-of-sight lift.

This is a hypothesis generator, not evidence. It is still a global scalar and
may fail for barred, edge-on, lopsided, or intrinsically elongated galaxies.
The next protocol must therefore freeze:

- the exact projected/deprojected covariance convention;
- a real two-dimensional spent-galaxy morphology screen, including bulges,
  disks, bars, inclination, surface density, and lopsidedness;
- the RX J2129 field and raw-topology gates;
- map-resolution, field-of-view, and baryonic mass-to-light perturbations; and
- a rule preventing any adjustment of `e_2D` after those scores.

Only if that parameter-free law survives spent mechanism tests should it face
the still-sealed P0633 galaxy kinematics and P0640 cluster constraints.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0692_spent_linear_routing_continuum_atlas.py
python -m pytest tests/test_source_routing_qumond.py tests/test_source_routing_spherical.py tests/test_spatial_qumond_3d.py tests/test_potential_channel_qumond.py -q
```

Artifacts are in
`results/p0692_spent_linear_routing_continuum_atlas/`.

## Researcher API implications

The seven-minute local runtime validates the planned split architecture:
Vercel serves the browser interface, typed API, authentication, job records,
and cached summaries; isolated Cloud Run Jobs or Modal workers execute 3D
fields and global root searches. Each API job must label whether it is a frozen
test or a diagnostic sweep, return all attempted rows rather than only the
best one, and hash the dataset, object, seed, formula AST, solver, nuisance
bounds, and comparator versions. See
[`PUBLIC_SIMULATOR_API_PLAN.md`](PUBLIC_SIMULATOR_API_PLAN.md).

## Claim boundary

P0692 is fully spent, post-hoc mechanism evidence. Its single viable row is
not a universal setting and does not beat a dark-matter model on independent
data. Finite sampling cannot exclude a narrow unsampled interval, and the
zero-slip light rule is not a relativistic metric theory. P0633 and P0640
remain sealed.
