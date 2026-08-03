# Generic coupled-field and photon/matter execution milestone

Date: 2026-08-02  
Release: hosted simulator v0.19

## Outcome

The local reference backend now has direct acceptance evidence that one generic
field manifest can solve multiple potentials and expose different predictions
for massive tracers and photons. No branch checks the name "two-potential" or
selects a theory-specific solver.

The published fixture declares

```text
laplacian(Psi) = 4 pi G rho_b
laplacian(Phi) = 4 pi G eta rho_b
a_matter       = -gradient(Psi)
a_photon       = -gradient((Psi + Phi) / 2)
eta            = 1.5, universal
```

With identical zero far-field boundaries, linearity gives the known answer
`Phi = 1.5 Psi`, and therefore `a_photon = 1.25 a_matter`. The value 1.5 was
chosen to make a non-degenerate numerical acceptance test. It was not inferred
from observations.

## Evidence

The exact confirmed model hash is

```text
bcc7c218ec4d11ee77c85837530daa342e98748c3eb04e460b35f93a7e17accc
```

The real local HTTP workflow performed an immutable array upload, queued the
confirmed model, executed the generic 3D worker, downloaded every artifact, and
verified each content hash. The measured result was:

| Check | Result |
|---|---:|
| State | succeeded |
| Grid | 9 x 9 x 9 |
| Solved fields | 2 |
| Equations | 2 |
| `Phi / Psi` target | 1.5 |
| relative field-ratio error | 4.16e-16 |
| photon/matter target | 1.25 |
| largest relative observable-ratio error | 4.19e-16 |
| verified artifacts | 8 / 8 |
| per-object gravity parameters | 0 |

A separate two-dimensional manufactured solution uses

```text
laplacian(u) = forcing_u + lambda v
laplacian(v) = forcing_v + lambda u
```

so each solved field changes the other field's right-hand side. It converges by
sequential Gauss-Seidel/Picard updates and recovers both analytic fields within
the finite-difference error threshold. This closes the loophole where two
independent equations might have been presented as evidence for coupling.

## What this establishes

- Multiple solved fields can be read from equation trees rather than a theory
  label.
- Fields can feed back into one another.
- Matter and photon observables remain separately typed and separately stored.
- The exact confirmed manifest survives upload, preflight, execution, artifact
  generation, hashing, and download.
- The report discloses solver family, equation count, solved-field count,
  multi-field update scheme, and parameter accounting.

## What this does not establish

- The fixture does not fit any galaxy or cluster observation.
- A different photon response is not by itself a covariant relativistic theory.
- The test does not establish conservation laws, stability, causality, or the
  Solar-System limit.
- A converged discretization can still solve the wrong physical equation.
- Public Vercel endpoints still do not run heavy Python jobs; they publish the
  contract and return an explicit 503 until durable workers are connected.

## Reproduction

```powershell
python -m pytest -q tests/test_generic_field_worker.py -k "two_potential or two_fields_can_feed"
python -m pytest -q tests/test_field_job.py -k two_potential
cd hosted-simulator
npm run dev
npm run smoke:two-potential
```

## Next scientific gate

Use this capability to express a small, motivated Sigma Gravity candidate,
freeze its universal constants on development systems, and request both
massive-tracer and photon observables. The held-out test must start from
baryonic observations and predict raw galaxy velocities and raw lensing. An
inferred dark-matter or convergence map may help discover a candidate response
law, but it cannot be supplied to the held-out forward run.
