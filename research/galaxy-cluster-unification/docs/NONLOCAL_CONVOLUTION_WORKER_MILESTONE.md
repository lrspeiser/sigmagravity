# Nonlocal convolution worker milestone

Status: implemented and numerically verified for the local reference worker
on 2026-08-02.

## Capability added

The generic field worker can now execute a nonlocal source term of the form

```text
response(y) = integral K(y - x) source(x) dV
```

inside an otherwise generic stationary elliptic manifest. The published
example uses

```text
laplacian(Phi) = 4 pi G [rho_b + q convolution(rho_b, K)]
```

This is a neutral acceptance fixture for baryon-to-effective-response ideas.
The worker dispatches on the typed expression tree, not the model name.

## Frozen numerical meaning

Every manifest using `convolution` must declare all of these conventions:

- `solver.family = nonlocal_elliptic`;
- `nonlocalBoundary = zero_padded`;
- `convolutionMode = linear_same`;
- `kernelOrigin = centered_sample`; and
- `convolutionMeasure = physical_volume`.

The submitted field and kernel share one odd Cartesian grid. The central
kernel sample means zero separation. The worker computes a linear convolution,
returns an array with the source shape, treats the field outside the submitted
box as zero, and multiplies the discrete sum by the physical cell volume. It
does not normalize the kernel. These choices are copied into result metadata.

## Verification

The acceptance tests establish that:

- a centered discrete unit impulse reproduces the submitted kernel after the
  physical-volume factor;
- a corner impulse does not reappear at the opposite corner, excluding
  periodic wraparound;
- a missing convention is rejected before solving; and
- the published 3D nonlocal response model produces finite potential and
  acceleration fields through the same generic worker used by other models.

## What this does not establish

This implementation does not infer a kernel from dark-matter maps, prove that
gravity propagates along a reconstructed route, or show that the example fits
galaxy dynamics, cluster lensing, or Solar-System constraints. Zero padding is
also only a finite-domain approximation to an isolated infinite domain.
Kernel resolution and box-size convergence must be checked for each study.

## Next scientific build

The inverse halo-response workbench should ingest uncertainty ensembles of
registered baryonic maps and independently derived effective-mass or lensing
maps on development systems. It should recover families of compatible kernels,
compare them with radius-preserving and shuffled nulls, and disclose
non-identifiability. A candidate kernel may advance only after it is frozen and
used to predict raw held-out lensing and galaxy observations without access to
the held-out halo reconstruction.
