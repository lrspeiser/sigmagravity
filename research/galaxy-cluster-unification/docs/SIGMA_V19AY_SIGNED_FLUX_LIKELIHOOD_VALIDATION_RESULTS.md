# Sigma V19AY signed-flux likelihood validation results

## Decision

V19AY **failed closed**. The signed-flux, color-integrated, brightness-profiled
likelihood retrieved only 1/5 validation identities at rank one and achieved a
mean reciprocal rank of 0.50. The unchanged gates were 3/5 and 0.65.

No ambiguous candidate was scored, no positional posterior was changed, and no
mass, current, lensing, halo or gravity quantity was inferred.

## Frozen result

| Validation member | True-pair rank |
|---|---:|
| 26 | 2 |
| 57 | 1 |
| 66 | 4 |
| 71 | 4 |
| 21 | 2 |

All 25 flux-pair scores were finite. The failure is discrimination, not a
numerical problem.

## Why the apparently principled likelihood did worse

The model did two reasonable things: it integrated the already-measured color
scatter and profiled out total brightness. But its profile likelihood used the
per-source flux uncertainty. A noisy flux vector is easy to fit because many
color templates lie within its broad error bars. Without a separately
validated background-flux model or a properly normalized population prior,
that compatibility is not positive evidence that the source is the member.

An analytic marginalization of the same Gaussian amplitude model was checked
after the frozen failure and produced the same five ranks. The missing
uncertainty-volume factor was therefore not a simple rescue. Adding a tuned
temperature, SNR switch or empirical background density after seeing these
five ranks would only move the fitting freedom into the association layer.

This closes the tested signed-flux identity likelihood. It does not invalidate
the signed measurements themselves.

## Better route to the baryonic map

The association problem and the mass problem do not have to be solved by the
same photometry. Each spectroscopic Bullet row already has published Bessel
`B/R/I` photometry. That photometry can supply a stellar-mass distribution,
while the V19AA positional posterior supplies a distribution over precise
candidate locations plus an explicit null/location-kernel state.

The next map should therefore:

1. infer each member's stellar-mass probability distribution directly from its
   published BRI and spectroscopic cluster membership;
2. place that mass at every candidate location with the unchanged V19AA
   positional posterior;
3. place the null probability over the original coordinate-quantization kernel
   rather than deleting the member;
4. sample the joint one-to-one association constraints rather than selecting a
   nearest object; and
5. propagate the resulting ensemble through any long-wavelength source term.

This makes the uncertainty visible. It avoids pretending that an ambiguous
counterpart has become secure merely because a flexible photometric likelihood
can be written down.

## Consequence for the wave hypothesis

The long-wave proposal depends on baryonic direction, overlap, current and
stress. The correct input is therefore an ensemble
`{T_mu_nu^(s)}` of allowed baryonic maps, not one best-guess image. A candidate
field term is credible only if its predicted shear/topology survives this map
ensemble. V19AY tested the construction of that ensemble; it did not test the
gravity equation itself.

Reproducibility:

- `configs/sigma_v19ay_signed_flux_likelihood_validation.json`
- `scripts/run_sigma_v19ay_signed_flux_likelihood_validation.py`
- `results/sigma_v19ay_signed_flux_likelihood_validation/report.json`
- `data/derived/sigma_v19ay_signed_flux_likelihood_validation/validation_scores.csv`
