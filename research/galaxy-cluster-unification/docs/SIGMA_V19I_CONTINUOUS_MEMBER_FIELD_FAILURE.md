# Sigma V19I continuous member-field result

## Outcome

The frozen continuous source-only successor to the failed discrete mixture did
not pass both clusters.  **Abell 2146 passed every gate; the Bullet Cluster did
not identify an admissible merger pair.**  This is a source-identifiability
failure, not a gravity-formula failure: no lensing target was opened, no
gravity equation was selected, and no gravity parameter was changed.

The machine-readable report is
`results/sigma_v19i_continuous_member_field/report.json`, with SHA-256
`6bbd82f51103b2d877324605c9b3d799800df338a85a985cdb8c1a05bcc28b55`.

## What was tested

The previous V19H test forced neither two nor three components, but its
heteroscedastic Gaussian-mixture selection chose one component in both
clusters.  V19I therefore changed the measurement representation rather than
relaxing that failed threshold:

- select one physical Gaussian-kernel bandwidth from `50, 75, 100, 150, 200,
  300 kpc` by leave-one-out positional likelihood;
- construct continuous member number density, mean line-of-sight velocity,
  intrinsic velocity variance, number current and random stress on a fixed
  `5 kpc` grid;
- identify density modes with a descending-superlevel-set merge tree;
- require peak-to-saddle ratio at least `1.5`, peak density at least `10%` of
  the global peak, and pair separation from `100` to `1200 kpc`; and
- repeat catalog resampling, quoted velocity perturbation and bandwidth
  selection in `2,000` bootstraps per cluster.

Published Abell 2146 A/B labels were used only after selection as validation.
They did not enter the bandwidth, modes, pair or bootstrap recovery.

## Results

| Quantity | Bullet | Abell 2146 |
|---|---:|---:|
| Primary bandwidth | 150 kpc | 75 kpc |
| Qualifying density modes | 2 | 2 |
| Mode coordinates (kpc) | `(55,-35)`, `(-1455,650)` | `(-170,135)`, `(185,-245)` |
| Candidate separation | about 1658 kpc | 520.0 kpc |
| Pair inside 100--1200 kpc | no | yes |
| Mode-1 recovery | not applicable | 89.90% |
| Mode-2 recovery | not applicable | 71.25% |
| Finite bootstrap draws | 2000/2000 | 2000/2000 |
| Published-label ARI, validation only | unavailable | 0.6498 |
| Frozen cluster gate | **fail** | **pass** |

The Bullet result is not a near-threshold rejection.  Its second accepted
peak is a low-density concentration far to the northwest; the separation from
the central peak is roughly `1.66 Mpc`, well beyond the frozen `1.2 Mpc`
maximum.  The algorithm did not resolve two persistent central merger modes.

Abell 2146 is the positive control.  Its two peaks are comparable in density,
have peak-to-saddle ratio `1.888` for the lower-persistence mode, and remain
recoverable above the frozen `68%` threshold.  The validation-only agreement
with published A/B labels is substantial but not perfect.

## Physical meaning and claim boundary

The current proxy is

\[
j_{\rm los}(x)=n(x)u(x),
\]

not the physical mass current `rho v`.  The two source catalogs do not contain
a homogeneous common photometric or stellar-mass observable.  Therefore this
stage can measure projected flow geometry but cannot calculate a
Lense--Thirring amplitude.  A physical frame-dragging test requires member
stellar masses, their uncertainties, and a projection model for unobserved
transverse velocities.  Gas bulk-current constraints would also be needed for
a complete baryonic `T_0i` source.

The result also does not test the nonlocal photon-phase proposal.  That lane
requires a separately frozen, gauge-preserving propagation kernel and
frequency, polarization and time-delay controls.  It cannot be inferred from
member density modes.

## Decision

Do not lower the persistence or recovery threshold, expand the pair bound,
force two Bullet modes, or use published mode coordinates.  The two current
member catalogs are retired as a jointly sufficient assembly-current
coordinate.  A future member-current gate requires richer homogeneous
source-side data, preferably common multiband photometry/stellar masses and
integral-field or denser spectroscopic velocities.

The independent automated X-ray shock and thermodynamic measurements may
continue, but they cannot be combined into a final causal source map until the
missing member mass-current information is supplied.

## Reproduction

```powershell
python scripts/run_sigma_v19i_continuous_member_field.py
python -m pytest -q tests/test_sigma_v19i_continuous_member_field.py
python -m ruff check scripts/run_sigma_v19i_continuous_member_field.py tests/test_sigma_v19i_continuous_member_field.py
```
