# Initial engineering baseline

Date: 2026-07-25

This report records pipeline state, not a scientific conclusion.

## Data and split

- Imported 175 SPARC rotation curves containing 3,391 raw radial rows.
- Source checkout and all 178 copied files are recorded in the provenance
  manifest with SHA-256 hashes.
- Prespecified quality, inclination, and minimum-length cuts retained 131
  galaxies and 3,034 points.
- The deterministic radial split contains 2,066 inner training points and 968
  untouched outer holdout points.

## Machine state

- Physical GPU: NVIDIA GeForce RTX 5090, 32,607 MiB.
- NVIDIA driver: 580.88; driver-reported CUDA compatibility: 13.0.
- Python used for the engineering baseline: 3.13.5.
- Installed global PyTorch: 2.7.1+cpu; CUDA unavailable in that environment.
- Isolated project environment: Python 3.12.10, PyTorch 2.12.1+cu130; CUDA is
  available and identifies the RTX 5090 at compute capability 12.0.
- `scripts/setup_cuda.ps1` created that environment without altering the global
  Python installation.

## Preliminary 300-step CPU comparison

| Model | Outer chi² / point | Outer RMSE (km/s) |
|---|---:|---:|
| Fixed RAR | 5.103 | 10.682 |
| Smooth void, p=0.5 | 5.886 | 11.107 |
| Smooth void, free p | 5.923 | 11.108 |
| NFW | 14.019 | 17.089 |
| Newtonian baryons | 108.384 | 41.307 |

The CUDA free smooth model moved to $p=0.516$, $a_t=2.36\times10^{-10}$ m/s²,
$A_0=0.600$, and $w=0.500$ dex. This is an optimization smoke baseline from one
seed and only 300 steps. It is not a converged posterior, uncertainty interval,
or evidence for a void mechanism. At the time of this baseline, no independent
void score had been supplied, so $\beta$ was not tested in the numbers above.

CPU and CUDA predictions agreed to a maximum $1.7\times10^{-9}$ km/s through
100 identical steps. By step 300, small numerical differences sent the
nonconvex optimizer to nearby paths: CPU gave $p=0.509$ and outer chi²/point
5.845, while CUDA gave $p=0.516$ and 5.923. Production claims therefore require
multi-seed convergence and distributional comparisons, not one optimizer path.

The poor NFW extrapolation at this stage is also not a general verdict on NFW:
its two halo parameters per galaxy were trained only on inner radii under a weak
generic prior, and the short optimizer run may be inadequate. The result serves
as a warning that the comparator needs convergence and prior-sensitivity checks.

## Next locked work

1. Add synthetic recovery tests for all global and nuisance parameters.
2. Run multiple initializations and convergence diagnostics.
3. Add whole-galaxy cross-validation and bootstrap orchestration.
4. Estimate environmental coupling using the now-frozen CF4 table, with
   whole-galaxy validation and the two prespecified reconstruction sensitivities.

The inspected SigmaGravity checkout contains SPARC coordinates but no independent
Cosmicflows, DESI, 2MRS, or other 3-D void/density catalog. Its files named
“environment estimator” derive disk-state quantities from SPARC observables and
do not qualify for the causal $\beta$ test.

## Independent environment data added 2026-07-26

The separate repository now contains the official grouped and ungrouped
Cosmicflows-4 density reconstructions, the CF4 group catalog, and the official
2026 128^3 ungrouped grid. A deterministic build assigns all 175 SPARC galaxies
an external score using only sky position, distance, and reconstructed density.

The primary grouped score spans -0.586 to 1.197 across SPARC and has standard
deviation 0.318. Its rank correlation with the ungrouped 64^3 score is 0.810;
the 128^3 release gives a more demanding catalog-version sensitivity at 0.564.
These are input diagnostics, not evidence for or against the gravity model. No
environment-enabled scientific fit is reported in this status document.
