# P0573-P0574 fresh replication and symmetry-gated arrival results

## Outcome

The cluster-side pattern replicated on a genuinely unused sample, but the
original formula was not safe for an extended circular galaxy. A single
baryon-only symmetry factor repairs that separation while keeping most of the
cluster improvement.

This is evidence for a useful empirical structure. It is not evidence that
dark matter has been eliminated, because the targets are normalized standard-
GR lens reconstructions and the formula does not yet predict absolute lens
strength or galaxy rotation speeds.

## P0573: genuinely fresh replication

The sample was chosen from the official RELICS archive using filenames and
coverage only. No candidate convergence pixels or scores were inspected before
the freeze.

| System | HST strict members | Lenstool maps | GLAFIC map |
|---|---:|---:|---:|
| RXC J2211.7-0350 | 82 | 100 | yes |
| SMACS J0723.3-7327 | 41 | 100 | yes |
| SPT-CL J0615-5746 | 74 | 100 | yes |

The locked formula was

\[
A=\sqrt{1-C}\,B_T,\qquad D=A\|T\|_F,
\]

\[
\Sigma_{\rm pred}=0.2B_{100}+0.8\,\widehat{G_{50}*D}.
\]

It has no parameter fitted to any of the three fresh systems.

| Fresh metric | Local light | Locked arrival | Change |
|---|---:|---:|---:|
| Equal-system Lenstool JS | 0.03890 | 0.03175 | 18.37% better |
| Mean Pearson | 0.83135 | 0.83838 | +0.00704 |
| Systems improved | — | 3/3 | pass |
| Lenstool realizations improved | — | 266/300 | 88.7% |
| Equal-system GLAFIC JS | 0.04296 | 0.03079 | 28.32% better |
| GLAFIC systems improved | — | 3/3 | pass |

This passes every predeclared cluster and independent-method gate.

## The cross-domain correction

The earlier work had tested only one centered source and called it an
"axisymmetric null." That was too broad. A resolved circular exponential disk
gives activation RMS 0.439 and maximum 0.510. Multiple source vectors cancel
partly even in a perfectly circular distribution, so `1-C` is not zero.

The Solar point-source null remains exact. The error concerned resolved
axisymmetric matter, not the one-source limit.

## P0574: one universal symmetry factor

The smallest tested repair measures how different the baryon map is after a
90-degree rotation about the system center:

\[
Q_{90}={\sum |B_{50}(x)-B_{50}(R_{90}x)|\over 2\sum B_{50}(x)}.
\]

It then changes only the fraction assigned to the angular destination:

\[
H(Q_{90})={Q_{90}^{n}\over Q_{90}^{n}+Q_0^n},\qquad
f_{\rm eff}=fH.
\]

For a centered circular field, `Q90=0` and the angular response is exactly
off. For clumpy cluster member maps, the measured `Q90` values span 0.179 to
0.539 in the P0573 sample.

Fourteen variants were frozen before any symmetry-gated lens score. Selection
used 13 older, globally spent clusters. The selected universal setting was

\[
\alpha=0.5,\quad \beta=1,\quad w=60\ {\rm kpc},\quad f=0.8,
\quad Q_0=0.05,\quad n=4.
\]

| Variant-validation metric | Local light | Symmetry-gated | Change |
|---|---:|---:|---:|
| Equal-system Lenstool JS | 0.03890 | 0.03277 | 15.77% better |
| Systems improved | — | 3/3 | pass |
| Lenstool realizations improved | — | 253/300 | 84.3% |
| GLAFIC change | — | — | 24.97% better |
| GLAFIC systems improved | — | 3/3 | pass |
| Retained no-gate P0573 gain | — | 85.8% | pass |

SPT-CL J0615 improves by only 0.46%, so the all-system count should not be read
as three equally strong wins. RXC J2211 and SMACS J0723 improve by 31.8% and
21.9%, respectively.

## Galaxy and Solar meaning

All 175 local SPARC files are one-dimensional deprojected axisymmetric mass
models. They therefore have `Q90=0` by construction, and this angular layer
changes every tabulated rotation speed by exactly 0 km/s. The same is true for
the centered circular disk audit and Solar point-source audit.

This is compatibility, not a galaxy solution. The formula supplies no radial
extra acceleration and cannot by itself reproduce any SPARC rotation curve.
A future unified equation must generate a radial trace for galaxies and this
symmetry-gated angular response for clusters from the same field or action.

## Parameter lesson

Spatial reach is the only large local coordinate. Moving the arrival width
from 40 to 60 kpc changes the 13-system mean score by 5.21%. The route fraction
changes it by 1.11%; tidal and cancellation powers by 0.61% and 0.20%.
Reasonable changes to `Q0` and `n` move it by less than 0.08%.

That insensitivity is encouraging: the symmetry gate acts as an on/off domain
separator, while the map morphology is governed mainly by where the response
is spread.

## What the test still cannot say

- The lens maps are inferred under GR and standard parametric lens modeling.
- Only normalized morphology was tested; absolute convergence is absent.
- Hot gas, intracluster light, and stellar mass-to-light variations are absent
  from the baryon source proxy.
- `Q90` is a projected statistic, not a covariant four-dimensional scalar.
- P0573 is fresh to the no-gate formula; it is only prospective-to-variant for
  P0574 because the target pixels had already been opened.

The next test should freeze the P0574 potential and predict raw multiple-image
positions and, separately, weak-shear observables without using a reconstructed
convergence map as the target.

## Reproduce

```powershell
python scripts/download_gravity_arc_fresh_sample.py --config configs/p0573_tidal_arrival_fresh_replication_protocol.json
python scripts/audit_gravity_arc_fresh_sample.py --config configs/p0573_tidal_arrival_fresh_replication_protocol.json
python scripts/run_p0573_tidal_arrival_fresh_replication.py
python scripts/run_p0574_symmetry_gated_arrival_microvariation.py
python -m pytest -q tests/test_p0573_p0574_fresh_symmetry_results.py
```
