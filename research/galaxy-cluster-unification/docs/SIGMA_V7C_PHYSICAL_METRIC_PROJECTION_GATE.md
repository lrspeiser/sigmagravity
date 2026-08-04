# Sigma v7C physical-metric projection gate

## Decision

The frozen scalar-only v7C equation is retired as a lensing carrier and retained
only as a nonlinear scalar-dynamics and solver control.  Its construction result
is real: the scalar response is healthy on the tested branch and is `7.223%`
nonadditive for separated sources.  But that scalar response is not itself the
physical lensing potential.

No observational map or raw holdout was opened.  The failure is an analytic
metric-projection result.

## Why the projection is mandatory

Use the weak-field convention

$$
ds^2=-(1+2\Psi)dt^2+(1-2\Phi)\delta_{ij}dx^i dx^j.
$$

Slow massive bodies respond primarily to `Psi`, whereas photons respond to the
Weyl combination

$$
W=\frac{\Psi+\Phi}{2}.
$$

The decoupling-limit field redefinition for ghost-free massive gravity contains

$$
\widehat h_{\mu\nu}
=\widetilde h_{\mu\nu}
-\eta_{\mu\nu}\pi
-\frac{\widetilde\alpha}{\Lambda_3^3}
\partial_\mu\pi\,\partial_\nu\pi.
$$

This is Eq. 63 in the
[Babichev--Deffayet Vainshtein review](https://arxiv.org/abs/1304.7240); the
complete decoupling-limit interaction structure is derived by
[Ondo and Tolley](https://arxiv.org/abs/1307.4769).

## Leading conformal response

First retain only `delta h_mn=-eta_mn pi`.  With signature `(-,+,+,+)`,

$$
\delta h_{00}=+\pi=-2\,\delta\Psi,
$$

and

$$
\delta h_{ij}=-\pi\delta_{ij}=-2\,\delta\Phi\delta_{ij}.
$$

Therefore

$$
\boxed{
\delta\Psi=-\frac{\pi}{2},\qquad
\delta\Phi=+\frac{\pi}{2},\qquad
\delta W=0.
}
$$

The executable audit obtains a maximum absolute Weyl response of exactly zero
over 1,001 signed field samples.  This is not a small-parameter constraint; it
is an algebraic cancellation.  Equivalently, the perturbation is conformal and
does not change unparameterized null trajectories at this order.

The same identity explains the linear vDVZ decomposition.  In units where the
helicity-2 response has `Psi=Phi=1`, the scalar supplies

$$
(\delta\Psi,\delta\Phi,\delta W)
=\left(\frac13,-\frac13,0\right).
$$

The complete massive response is thus

$$
(\Psi,\Phi,W)=\left(\frac43,\frac23,1\right),
$$

so `gamma=Phi/Psi=1/2`.  After calibrating Newton's constant to the enhanced
massive force, the light response is `3/4` of GR.  These are the standard vDVZ
ratios summarized in the same review.

## What could produce lensing, but is missing from frozen v7C

There are two legitimate non-conformal routes in the parent theory.

### Static disformal metric

For a static scalar,

$$
\delta h_{00}^{\rm dis}=0,
\qquad
\delta h_{ij}^{\rm dis}
=-D\,\partial_i\pi\,\partial_j\pi.
$$

It is anisotropic.  For a null ray with spatial direction `n`,

$$
\delta h_{\mu\nu}k^\mu k^\nu
=-D(\mathbf n\cdot\nabla\pi)^2,
$$

which can be nonzero.  The audit obtains `-1` for a unit ray aligned with a
unit gradient and zero for an orthogonal ray.  Therefore we do **not** claim
that every disformal Galileon has zero lensing.

However, v7C froze only

$$
3\nabla^2\pi+kappa
\left[(\nabla^2\pi)^2-pi_{,ij}\pi_{,ij}\right]=J_b.
$$

It did not freeze the disformal physical-metric coefficient together with the
complete action-linked scalar equation and matter coupling.  Adding that term
after observing a failed lensing result would be a new theory, not a projection
of the frozen one.

### Residual helicity-0/helicity-2 mixing

For the general massive-gravity parameter branch, the helicity-2 equation
retains an `X^(3)` source built from the scalar Hessian.  The scalar and tensor
must then be solved together.  The frozen v7C scalar PDE contains no tensor
equation, so it cannot use this route either.

## Gate result

| Check | Result |
|---|---:|
| Conformal projection identity verified | pass |
| Standard vDVZ decomposition verified | pass |
| Nonzero action-derived lensing response in frozen v7C | **fail** |
| Complete scalar-to-metric mapping frozen | **fail** |
| Coupled tensor closure available | **fail** |
| Object-specific or lens-only multiplier used | no |
| Observational data opened | no |

The construction solver remains useful for studying nonlinear scalar dynamics,
but its field cannot be inserted into a lens model as though it were `W`.

## Scope and next decision

This result rejects the **frozen scalar-only v7C lensing interpretation**.  It
does not reject all Galileons, all disformal scalar-tensor theories, or full
ghost-free bimetric gravity.

It is nevertheless the third positive-spin-2 carrier formulation in this
sequence to miss the combined requirement of a healthy, Solar-safe, useful
universal lensing response:

1. v7A: the unscreened carrier is Solar-limited to a `1.0000075` lensing factor;
2. v7B: spherical screening cannot distinguish equal-density systems and the
   healthy exterior caps lensing at `1.5`;
3. v7C: the geometry-sensitive scalar response has no closed nonzero lensing
   projection in the frozen equations.

The project must now issue the planned mechanism-level falsification synthesis
instead of adding another term to v7C.  A future return to bimetric gravity is
allowed only as a separately preregistered, complete coupled-metric theory.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/audit_sigma_v7c_metric_projection.py
python -m pytest -q tests/test_sigma_v7_metric_projection.py
```

Machine-readable evidence is stored in
`results/sigma_v7c_metric_projection_gate/report.json`.
