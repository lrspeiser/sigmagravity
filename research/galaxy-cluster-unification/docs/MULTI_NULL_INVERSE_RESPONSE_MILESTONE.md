# Multi-null inverse-response milestone

Date: 2026-08-02

## Outcome

The inverse baryon-to-response workbench can now ask a harder question than
"does one compact kernel reconstruct the supplied target?" It asks whether the
observed source/target relationship is better than every declared wrong-answer
family.

The request remains backward compatible with the original radial-angle form,
but a suite may declare up to eight families with independent deterministic
counts and seeds. The global `signal_against_null` value is true only when all
families have Monte Carlo p-values at or below 0.05 and the observed aggregate
R-squared is at least 0.25. No family can be silently dropped from this rule.

## Implemented controls

| Control | What changes | What is preserved | Question it probes |
|---|---|---|---|
| `source_radial_angle_shuffle` | Baryonic values move around each radius shell | Every shell's source values | Does angular baryonic structure matter beyond the radial profile? |
| `source_phase_scramble` | Fourier phases of the baryonic map are randomized | Full Fourier power spectrum and mean | Does the particular spatial arrangement matter beyond scale power? |
| `target_system_permutation` | Whole target, uncertainty, and mask packages move between systems | The set of submitted target packages | Is each baryonic system paired with its own target better than a wrong pairing? |
| `target_radial_angle_shuffle` | Target values move around each radius shell | Every shell's target values | Does target morphology matter beyond its radial profile? |
| `source_missing_baryon_dropout` | Random source cells are removed | Total source integral after rescaling | Is the inferred response fragile to a declared incomplete-source perturbation? |

Fourier-phase scrambling can produce a map that is not a physically valid
non-negative baryonic density. It is therefore a structural null, not a rival
physical model. Missing-baryon dropout is also a sensitivity control, not a
claim about the location of real unobserved gas or stars.

## Request example

```json
{
  "nullControls": {
    "combinationRule": "all_declared_families",
    "families": [
      {"kind": "source_radial_angle_shuffle", "count": 19, "seed": 23},
      {"kind": "source_phase_scramble", "count": 19, "seed": 24},
      {"kind": "target_system_permutation", "count": 19, "seed": 25},
      {"kind": "target_radial_angle_shuffle", "count": 19, "seed": 26},
      {
        "kind": "source_missing_baryon_dropout",
        "count": 19,
        "seed": 27,
        "dropoutFraction": 0.2
      }
    ]
  }
}
```

The original object remains accepted:

```json
{
  "nullControls": {
    "kind": "source_radial_angle_shuffle",
    "count": 19,
    "seed": 23
  }
}
```

## Deterministic output

`scientific_result.json` now reports per-family counts, seeds, median null
error, Monte Carlo p-value, preserved quantity, family decision, the maximum
family p-value, and the all-family decision. `null_controls.csv` contains every
replicate. The generated HTML report and optional-LLM briefing show every
family rather than compressing the suite into one unexplained number.

The resource preflight counts all requested null fits. The null declaration is
part of the canonical job identity, so a changed family, count, seed, or
dropout fraction produces a different job hash.

## Verification boundary

The known-answer suite injects one shared kernel into two synthetic systems.
The workbench recovers that kernel and, with 19 trials per family, beats all
five controls at p=0.05. Separate tests execute every family on 3D maps,
confirm exact Fourier-power preservation, confirm total-source preservation,
check deterministic repetition, and retain the original single-control API.

This proves the controls and reporting machinery behave as specified. It does
not show that a recovered response is a path taken by gravity, that a
dark-matter reconstruction is a direct observation, or that the fitted kernel
predicts unseen galaxies or clusters.

## What is required for a useful scientific discovery test

1. Register complete baryonic posterior ensembles for a multi-cluster
   development set, including gas, stars, intracluster light, line-of-sight
   structure, masks, calibration, and provenance.
2. Register effective-response posteriors from at least two independent lens
   modeling pipelines rather than one preferred dark-matter map.
3. Add physically motivated central-halo, local-light, baryon-catalog, and
   conservation controls; the present shuffle/dropout controls do not replace
   them.
4. Fit inverse response families only on the development set and expose their
   non-identifiability and dependence on lens method and baryonic uncertainty.
5. Compress any stable many-cell response into a small analytic forward law
   with explicit units, boundaries, photon/matter coupling, and universal
   parameter accounting.
6. Freeze that law and remove all target halo maps.
7. Predict raw held-out cluster image positions, shear or magnification and
   held-out galaxy velocity fields from baryons alone.
8. Run the same constants through Solar-System, numerical convergence,
   resolution, boundary, conservation, and stability gates.
9. Compare against baryons-only GR, fixed MOND/RAR, and published dark-matter
   models while displaying nuisance and effective parameter counts.
10. Connect durable storage, a queue, isolated workers, authentication,
    quotas, retries, cancellation, signed artifacts, and audit logs so external
    researchers can run these jobs safely through the hosted API.
