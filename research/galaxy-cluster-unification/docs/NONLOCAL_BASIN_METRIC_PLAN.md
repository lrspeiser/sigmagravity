# NBM0: nonlocal two-potential gravity-basin plan

## Why this branch

The mass-scaled galaxy formula is retired as a theory candidate because it
reproduces the baryonic Tully-Fisher/MOND regularity and does not predict cluster
lensing. Direct void tides, environmental generalized-Aether, and the minimal
environmental MOG realization have also failed their declared gates.

NBM0 instead develops the earlier *gravity valley* idea as a covariant basin
field and treats dynamics and lensing as two outputs of one physical metric. It
does not start with an acceleration interpolation.

## Zero-order field and metric

Let the dimensionless basin field `X` be sourced by density contrast relative to
the cosmological background:

```
(Box - L_X^-2) X = -kappa_X (T - T_background)/M_Pl^2,
```

with retarded boundary conditions. This makes an extended underdense or
overdense basin a boundary-value problem rather than a collection of invented
point voids.

Matter and light follow one physical metric:

```
g_tilde_mn = exp(2 alpha X) [g_mn + 2 beta X U_m U_n],
U_m U^m = -1.
```

`U_m` supplies the cosmological time direction. A completed action must make it
dynamical and prove its stability; NBM0 first tests the unavoidable weak-field
metric mapping.

For the Einstein-metric Newtonian potential `U_N`, linearization gives

```
Psi = U_N + c^2 (alpha-beta) X,
Phi = U_N - c^2 alpha X,
(Phi+Psi)/2 = U_N - c^2 beta X/2.
```

Consequently massive bodies and light cannot receive independently fitted
responses:

```
q_X = lensing extra / dynamics extra
    = -beta/[2(alpha-beta)].
```

Pure conformal coupling gives no additional lensing. Pure disformal coupling
gives `q_X=1/2`. The no-slip choice `beta=2 alpha` gives `Phi=Psi` and `q_X=1`.
These are theory statements, not lensing calibration options.

## Staged outcomes

### N0 — algebraic metric gate

Implement and unit-test the weak-field coefficients, special limits, GR limit,
and singular cases. Reject any version that needs a lensing-only factor.

### N1 — empirical identifiability gate

Require at least ten same objects with three or more overlapping radial points
for dynamics, raw/rerunnable lensing likelihoods, and baryonic profiles. The
ratio `q_X` cannot be inferred by dividing CLASH lensing enhancements by SPARC
dynamical enhancements because they are different systems in different
compactness regimes.

### N2 — full action

Only after N1, freeze an action with at most four new global parameters:
`kappa_X`, `L_X`, `alpha`, and `beta`. Derive all reciprocal field equations,
stress-energy conservation, metric slip, mode speeds, and screening. Do not add
an empirical `mu(g/a0)` or mass-velocity exponent.

### N3 — one-potential prediction run

Fit no lensing parameter. Solve `X`, `Psi`, and `Phi` from baryons plus boundary
data, freeze constants on training systems, and predict both resolved dynamics
and raw lensing coordinates/shear in held-out systems. Whole systems, not radial
points, are held out.

### N4 — external constraints

Require Solar-System PPN recovery, luminal tensor propagation, positive kinetic
and gradient matrices, and cosmological stability. A good astrophysical fit does
not override a failed health gate.

## Current stopping boundary

The current public package has zero strict-ready same systems. CLASH contains 84
GR/NFW-deprojected acceleration summaries rather than raw alternative-metric
likelihoods; the BCG bridge contains dynamics summaries without same-object
lensing. NBM0 can therefore pass or fail its algebraic gate now, but no empirical
coupling or unification claim is authorized yet.
