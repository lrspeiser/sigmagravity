# Periodic FFT Poisson worker milestone

Date: 2026-08-03

## Outcome

The formula-neutral worker now executes the previously declared
`fft_poisson` solver family on uniform periodic Cartesian 2-D and 3-D grids.
It solves independent scalar equations of the form

\[
\nabla^2 u = s
\]

without dispatching on a theory name. The same typed manifest, exact hash
confirmation, content-addressed input bundle, asynchronous field-job path and
immutable artifact report used by the finite-volume solver are retained.

This closes one explicit generic-worker requirement. It does not turn a
periodic simulation cell into an isolated galaxy or cluster.

## Confirmed numerical contract

An FFT Poisson manifest must declare all of the following:

- Cartesian 2-D or Cartesian 3-D geometry;
- one scalar solved field and one exact `laplacian(field)` equation per solved
  field;
- `boundary.type=periodic` on every solved field;
- `solver.family=fft_poisson` and `maxIterations=1`;
- `periodicZeroMode=require_zero_mean` or `subtract_mean`;
- `potentialGauge=zero_mean`; and
- a dimensionless `zeroModeTolerance` in `(0,1)`.

Version one requires independent right-hand sides. A right-hand side that
references a solved field is rejected rather than being treated as a coupled
solve. Boundary-value overrides, variable-coefficient left-hand sides and
iterative/nonlocal controls that the direct solver would ignore are also
rejected.

The published example is
[`periodic-poisson.json`](../hosted-simulator/examples/models/periodic-poisson.json),
whose exact confirmed computational hash is
`cc3bc73ccb40db010cce358c84babe9f3455cfb48d5c441dd94e8c928a9b9cbe`.

## Zero-mode behavior

Periodic Poisson has no solution when the source integral is nonzero. The
worker therefore never silently drops the Fourier zero mode:

- `require_zero_mean` rejects the job when
  `abs(mean(source))/rms(source)` exceeds the confirmed tolerance; and
- `subtract_mean` explicitly solves the density-contrast source
  `source-mean(source)`.

Both policies set the potential zero mode to zero. The report records the raw
source mean and integral, removed mean, effective integral, potential mean and
the exact policy. Mean subtraction is consequently visible in the model hash
and result artifacts.

## Diagnostics

The worker publishes, per equation:

- continuum-Fourier spectral residual;
- raw and effective source conservation values;
- potential gauge residual;
- maximum absolute and relative imaginary leakage;
- gradient-energy and minus-potential-source integrals;
- their integration-by-parts balance error;
- physical cell/domain volume and axis lengths; and
- minimum, maximum and total nonzero Fourier-mode support.

On even grids, first derivatives set the unpaired Nyquist component to zero so
a real nodal field produces a real nodal derivative. That convention is
reported explicitly. The energy diagnostic uses the full spectral Parseval
sum, including the Nyquist contribution, rather than the sampled first-
derivative array.

Gradient, divergence and Laplacian operators used inside FFT-job right-hand
sides or observables use the same periodic spectral derivative convention.
The returned acceleration therefore does not switch to a nonperiodic edge
stencil after the field solve.

## Measured acceptance

The anisotropic 3-D manufactured case used shape `[12,10,8]`, spacing
`[0.25,0.4,0.8]` and a resolved product Fourier mode. Its measured result was:

| Diagnostic | Value |
|---|---:|
| Relative field L2 error | `4.516614129620976e-16` |
| Relative spectral residual | `8.779858498704359e-16` |
| Energy-balance relative error | `1.2911135769671397e-16` |
| Relative imaginary leakage | `8.550883806990097e-17` |
| Potential mean | `-9.295842113745288e-20` |
| Nonzero modes | `959` |

Additional deterministic tests cover:

- anisotropic Cartesian 2-D and 3-D grids;
- spectral acceleration derivatives;
- explicit mean subtraction and its conservation ledger;
- rejection of a nonzero mean under `require_zero_mean`;
- rejection of nonperiodic boundaries, boundary overrides, coupled right-hand
  sides and ignored iterative controls;
- the same resolved physical mode at 12, 24 and 48 cells per axis; and
- hash-identical field-job replay with rehashed NPZ/JSON/CSV artifacts.

The real local asynchronous HTTP acceptance now contains six jobs. The two new
FFT cases passed upload registration, immutable array-byte upload,
queued/running/succeeded events, worker/gateway source-hash agreement, artifact
download/rehash and zero per-object gravity parameters:

| HTTP case | Relative field L2 error |
|---|---:|
| Periodic Cartesian 2-D, `24 x 24` | `3.464425937954617e-16` |
| Periodic Cartesian 3-D, `12 x 12 x 12` | `4.1083258285824224e-16` |

The Linux container runs this same six-case script in GitHub Actions.

GitHub Actions run
<https://github.com/lrspeiser/sigmagravity/actions/runs/30803981203> passed for
implementation commit `3d564c2d0fece41985b72332608d645ebafc5cc4`. It
built the non-root scientific-worker and advanced plug-in sandbox images,
passed all 160 hosted/control-plane tests, ran all six real container jobs and
re-ran the plug-in isolation acceptance. The Linux periodic 2-D and 3-D field
errors were `3.6301665671303565e-16` and `6.064489774045424e-16`.

Production deployment `dpl_84y1NAYwFEgdsEcmfF2GUyeFRcr4` is ready at
<https://sigma-gravity-research-simulator-gnpwerllt-horizon3.vercel.app> and
aliased to the stable site. The live HTTP smoke validated version
`0.35.0-preview`, guide example 16, the published periodic manifest and its
confirmed hash. The deployment queue canary recorded identity hash
`e8566836011c69b34a294bedcea8d93d7016f82af966b5d21f2bd4daaa974b6f`
and private acknowledgement hash
`55cf62199cb07ffe42a264be3785e37fbf9264f463dc88344ea2962622f2b5e4`.

## Scientific boundary

A periodic FFT solve represents a torus: mass, density contrast and field are
repeated beyond every face. It is useful for periodic simulation boxes,
manufactured tests and theories whose boundary contract is genuinely
periodic. It is not an isolated far-field boundary and must not be used to
claim a galaxy or cluster result merely because it is faster or spectrally
exact.

The existing isolated/Dirichlet finite-volume route remains the appropriate
reference for a finite isolated domain, together with box-size and resolution
convergence tests. The next generic numerical gaps are vector/tensor solves,
mixed and Neumann boundary execution, refinement/multiresolution and
production scaling.
