# Sigma v5A cosmological-branch result

## Decision

The exact Sigma v5A action is **retired before any observational fit**. Its
causal polarization sector remains mathematically viable, but the inherited
Sigma-v2 galaxy base is not a globally real differentiable action around the
FLRW background.

No galaxy or cluster score was calculated, and no holdout was opened.

## FLRW contraction

In coincident gauge,

\[
ds^2=-dt^2+a(t)^2\delta_{ij}dx^i dx^j,
\]

the nonmetricity traces satisfy

\[
Q_0=6H,
\qquad
\widetilde Q_0=0,
\qquad
\widetilde Q_a\widetilde Q^a=0.
\]

The Sigma-v2 primitive is a function of

\[
Y={\widetilde Q_a\widetilde Q^a\over4q_\Sigma^2}.
\]

The background is therefore exactly at `Y=0`. Generic perturbations can make
the Lorentzian contraction either positive or negative, so a covariant action
must be real and differentiable on an open signed neighborhood of zero.

## Failure

The inherited primitive was defined for `Y>=0` and has

\[
\mathcal H_Y=1-\nu_s(\sqrt Y).
\]

It is not real on the negative side. The executable implementation correctly
rejects `Y=-1e-12`. On the positive side its derivative is also singular:

| `Y` | `H_Y` |
|---:|---:|
| `1e-4` | `-9.51249` |
| `1e-8` | `-99.5012` |
| `1e-12` | `-999.500` |
| `1e-16` | `-9999.50` |

The derivative grows by a factor `1051.2` across this already small probe,
exceeding the frozen factor-100 gate. There is no regular perturbative action
neighborhood around FLRW.

Replacing `Y` with `|Y|` would be nondifferentiable at the background.
Replacing it with `Y^2` would define a materially different galaxy law. Neither
is accepted as a post-failure repair of v5A.

## What survives

The polarization source itself was already built from

\[
Z=Y^2,
\qquad
J={Z\over(1+Z)^2}.
\]

It is real, even, and smooth for signed `Y`; its measured evenness error and
derivative at zero are both exactly zero. The bounded disformal polarization
kinetic tensor also remains Lorentzian and causal for either timelike or
spacelike \(\mathcal W_a\).

The physically clean successor is therefore Sigma v5B: retain the causal
polarization action and place it directly on the symmetric-teleparallel
equivalent of GR, removing the ill-defined Sigma-v2 primitive. This does not
add a parameter. It also removes the concern that the new theory is MOND plus
a separate cluster correction: the polarization must generate any galaxy and
cluster departures from one GR base.

## Reproduction

```powershell
python scripts/check_sigma_v5a_cosmological_branch.py
python -m pytest tests/test_sigma_causal_polarization.py -q
```
