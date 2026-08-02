# P0661 quadratic coherence tensor audit

## Hypothesis

P0660 showed that domain separation is carried mostly by persistence length,
not by a different local star-versus-gas angle. P0661 tested the fixed kernel

\[
F(\ell)=1-\exp[-(\ell/L_c)^2]
\]

with the same `L_c=10 kpc`, `a0`, transverse tensor, and all other settings.
The motivation is coherent-vector accumulation: an amplitude first grows
linearly with path length, while its power or variance begins quadratically.
The integer exponent adds no fitted constant.

## Frozen result

P0661 solves the scale-separation problem but fails its resolution gate:

- short-path log slope: `1.99855`;
- survival at `ell/Lc=2`: `0.981684`;
- galaxy nominal median weighted `sigma`: `0.00121364`;
- cluster nominal median weighted `sigma`: `0.0746497`;
- nominal cluster/galaxy ratio: `61.5088x`;
- weakest ratio under mass-map sensitivity: `55.6835x`;
- cluster median resolution change: `11.20%`;
- galaxy median resolution change: `50.96%`; and
- frozen maximum permitted resolution change: `35%`.

All other gates pass. The candidate does not advance, and no target outcome was
opened.

## Diagnosis

The quadratic kernel magnifies errors in small galaxy coherence lengths. The
current inherited estimator clips its trace length in **pixels**, so halving the
map resolution doubles the physical minimum trace step. For affected galaxies,
the estimated path length grows strongly when a 65-cell map is reduced to 33
cells; the quadratic survival then magnifies that change.

This is not evidence that the quadratic physics is correct. It does establish
that the present pixel-bounded estimator cannot support it. The next legitimate
test is a physical tidal-length estimator `|g|/||grad g||` whose numerical bounds
are based on the physical map extent rather than a number of pixels. That test
must again be frozen and outcome-blind.

## Claim boundary

P0661 tests no velocities and no lensing topology. The coherent-amplitude story
is a motivation, not a microscopic derivation, and the 10 kpc length remains
phenomenological.

## Reproduction

```powershell
python scripts/run_p0661_quadratic_coherence_tensor_audit.py
python -m pytest tests/test_p0661_quadratic_coherence_tensor_audit.py -q
```
