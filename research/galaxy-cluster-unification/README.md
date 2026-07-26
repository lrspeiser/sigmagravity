# Void Screening Lab

An independent, falsification-first project for testing whether a smoothly
activated outer-galaxy acceleration can predict SPARC rotation curves, and
whether any surviving amplitude is independently associated with cosmic void
environment.

This is deliberately isolated from SigmaGravity's main theory and application
code even though it now lives under `research/galaxy-cluster-unification` in the
same repository. SigmaGravity is used only as a read-only source for a copied
SPARC snapshot; this subproject has its own package, configurations, data,
tests, and results. Every imported SPARC file is recorded with its source path,
source Git commit, size, and SHA-256 hash in
`data/raw/sparc/provenance.json`. See
[`docs/SIGMAGRAVITY_PLACEMENT.md`](docs/SIGMAGRAVITY_PLACEMENT.md) for the
placement and origin record.

## What the first milestone tests

The primary phenomenological model is

$$
g_{\rm pred}=g_{\rm bar}+
A_0 e^{\beta \mathcal V},a_t
\left(\frac{g_{\rm bar}}{a_t}\right)^p
S(g_{\rm bar}),
$$

with a gradual unscreening function

$$
S(g)=\left[1+\exp\left(\frac{\log_{10}g-\log_{10}a_t}{w}\right)\right]^{-1}.
$$

The data are allowed to determine the exponent $p$. A value near $p=1/2$
produces a flat added rotation contribution in the point-mass outer limit. The
same nuisance treatment and radial holdout are used for Newtonian, empirical
RAR, and NFW comparators.

SPARC rotation curves are mostly H I and H-alpha gas kinematics, not tracks of
individual outer stars. “Negative gravity” is therefore only an operational
label here for an additional inward effective acceleration. A SPARC fit alone
cannot establish a void origin.

## Current scope

- Parse all 175 SPARC mass-model files and the published metadata table.
- Preserve signed gas contributions when constructing $V_{\rm bar}^2$.
- Apply explicit quality, inclination, and minimum-point cuts.
- Fit common physical parameters plus per-galaxy mass-to-light, distance, and
  inclination nuisance parameters with stated priors.
- Train on the inner 70% of each retained curve and score the untouched outer
  30%.
- Compare Newtonian, RAR, NFW, free-$p$ void, and fixed-$p=1/2$ void models.
- Include a frozen, independent Cosmicflows-4 density score for every SPARC
  galaxy. The code never derives an environment score from rotation-curve
  velocities or residuals.
- Run on CPU or CUDA and save machine-readable summaries, predictions, model
  state, optimization history, and diagnostic plots.

The preregistered decision rules and project phases are in
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md).
The first tested CPU/CUDA engineering baseline is recorded in
[`docs/INITIAL_STATUS.md`](docs/INITIAL_STATUS.md).

## Quick start on this machine

The SPARC snapshot is already imported. To rebuild it from the local source:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/import_sigmagravity_data.ps1
```

Create an isolated environment and install the package:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For the RTX 5090, the included setup script creates a Python 3.12 environment
and installs the official PyTorch 2.12.1 CUDA 13.0 wheel. The version and index
are explicit so the environment is reproducible:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_cuda.ps1
.venv\Scripts\Activate.ps1
```

Confirm the environment rather than trusting the presence of `nvidia-smi`:

```powershell
python check_device.py
```

If PyTorch has released a newer stable build, verify the current Windows/Pip/CUDA
command at <https://pytorch.org/get-started/locally/> before changing the pinned
version.

Run a single model:

```powershell
python fit.py --model void --device auto --steps 5000 --output results/void_free_p
python fit.py --model void --fixed-flat-power --device auto --steps 5000 --output results/void_p05
```

Run the initial comparator suite:

```powershell
python compare_models.py --device auto --steps 5000 --output results/comparison
```

For a fast execution check, add `--steps 25`. That verifies the pipeline, not
scientific convergence.

## Independent environment input

Download the official Cosmicflows-4 grids and catalog, verify their byte sizes
and hashes, and rebuild the frozen SPARC cross-match with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_cosmicflows4.ps1
python scripts/build_cf4_environment.py
```

The generated `data/derived/void_scores_cf4.csv` contains one row per galaxy.
Its required model-input columns begin:

```text
galaxy,void_score
CamB,...
DDO154,...
...
```

The primary score is the negative grouped-grid density contrast, so larger
values mean a more underdense reconstructed environment. The ungrouped 64^3
grid and official 128^3 grid are retained as prespecified sensitivity columns.
The construction uses only sky position, SPARC distance, and the external CF4
density field. Run the environment-enabled model with:

```powershell
python fit.py --model void --environment-csv data/derived/void_scores_cf4.csv `
  --device cuda --steps 10000 --output results/void_environment
```

The full coordinate convention, grid hashes, score distributions, cross-grid
correlations, and catalog validation are recorded in
`data/derived/cf4_environment_report.json`.

## Current CF4 theory-test result

The completed 5,000-step radial and five-fold galaxy tests do **not** support
the specific prediction that a stronger CF4 void environment increases the
additional galactic acceleration. The smooth low-acceleration law remains much
better than Newtonian baryons alone, but the environment effect harms strict
held-out-galaxy prediction for both 64^3 reconstructions, and the 128^3 result
has the opposite beta sign. See `docs/CF4_THEORY_TEST.md` for the full results,
bootstrap intervals, limitations, and decision-rule audit.

The prior-art audit and frozen next-model sequence are in
[`docs/PRIOR_ART_AND_NEXT_TESTS.md`](docs/PRIOR_ART_AND_NEXT_TESTS.md). It
separates known ideas from this project's narrower test architecture and defines
potential-screened, environment-shifted, boundary-layer, void-wall, and physical
tidal checks before any of those variants are fit.

Those staged tests are now complete. The self-potential screen is competitive
with, but does not outperform, fixed RAR; CF4 threshold shifting, a boundary
layer, and independently cataloged void-wall depth all fail strict held-out-
galaxy prediction. Ordinary CF4 tides are about five orders of magnitude too
small at the median. See [`docs/NEXT_MODEL_RESULTS.md`](docs/NEXT_MODEL_RESULTS.md).

## Joint galaxy-dynamics and cluster-lensing result

The project now tests a stricter target: one baryon-linked acceleration field,
with zero gravitational slip and no lensing-only multiplier, must predict SPARC
circular speeds and CLASH lensing accelerations. The strongest tested bridge is
an explicitly **EMOND-like prior-art control** in which the RAR acceleration
scale rises with baryonic potential depth. In five-fold whole-system validation
it reduces the equal-domain score from 25.98 to 7.16 while making the SPARC
score 3.94% worse. It therefore clears the frozen relative advancement gate,
but its cluster chi-square remains too large for it to be called a completed
theory.

On 50 MaNGA BCG dynamical points that were not used in fitting, the frozen law
improves chi-square per point from 9.96 to 7.15, but a separately labeled
cluster-scale RAR reaches 2.19. The bridge is therefore partial. A post-hoc
inverse check finds that the missing transition corresponds to a plausible host
baryonic potential scale, but that quantity must be independently measured
before it can be used predictively. See
[`docs/UNIFIED_GALAXY_CLUSTER_RESULTS.md`](docs/UNIFIED_GALAXY_CLUSTER_RESULTS.md).

Reproduce the joint and external tests with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_clash_rar.ps1
python scripts/cross_validate_unified.py
powershell -ExecutionPolicy Bypass -File scripts/download_manga_bcg.ps1
python scripts/build_manga_bcg_table.py
python scripts/test_external_bcg.py
python scripts/diagnose_bcg_host_potential.py
```

Reproduce the strict validation and report with:

```powershell
python scripts/cross_validate_cf4.py --device cuda --steps 5000 `
  --folds 5 --bootstrap-draws 100000 --output results/cf4_galaxy_cv_5000
python scripts/summarize_cf4_test.py
```

## Reproducibility

All CLI defaults are stored in [`configs/baseline.json`](configs/baseline.json).
Each result summary records the seed, data hash, device, cuts, parameter count,
and train/holdout metrics. The source data are never edited in place.
