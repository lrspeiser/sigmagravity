# Sigma v14 local covariant gauge-carrier falsification

## Decision

The frozen local covariant gauge-reduced tidal carrier is rejected before any
observational data. Three materially different completions fail the common
requirement of a positive, gauge-reduced orientation carrier on an arbitrary
one-metric galaxy or cluster spacetime:

1. **v14A minimal covariantization:** the flat scalar gauge symmetry is broken
   by the Riemann tensor;
2. **v14B partially-massless correction:** the constant-curvature part can be
   cancelled, but the Weyl curvature needed to describe real tidal fields
   remains in the gauge variation; and
3. **v14C conformal/Bach completion:** the conserved neutral source exists,
   but the local fourth-order propagator has two spin-two poles with opposite
   residues, so one has negative energy.

The project's three-formulation stopping rule therefore resets this mechanism
instead of adding another curvature counterterm. The v13B reduced convexity
theorem is unaffected, but no covariant gravity theory has been obtained.
`theory_viable=false`; galaxies, clusters, Solar data, and holdouts remained
closed.

## v14A: exact curvature obstruction

In flat space the proposed symmetric field uses

$$
\delta A_{\mu\nu}=\partial_\mu\partial_\nu\alpha,
$$

and the first-curl field strength

$$
F_{\mu\nu|\rho}=2\partial_{[\mu}A_{\nu]\rho}
$$

is gauge invariant because partial derivatives commute. The minimal covariant
replacement is

$$
F_{\mu\nu|\rho}=2\nabla_{[\mu}A_{\nu]\rho},
\qquad
\delta A_{\mu\nu}=\nabla_\mu\nabla_\nu\alpha.
$$

Its exact variation is

$$
\boxed{
\delta F_{\mu\nu|\rho}
=R_{\mu\nu\rho}{}^\sigma\nabla_\sigma\alpha .
}
$$

This is not a small correction to an otherwise exact gauge theory. The
symmetry that was supposed to remove unphysical tensor components no longer
exists on a curved background. Setting the curvature to zero would also remove
the gravitational environment whose tidal geometry the new carrier was meant
to retain.

The frozen audit gives zero residual in flat space and a nonzero residual on a
constant-curvature sentinel. V14A therefore fails before a source or
Hamiltonian is considered.

## v14B: partially-massless correction and the Weyl remainder

On a constant-curvature background,

$$
R_{\mu\nu\rho\sigma}
=H^2(g_{\mu\rho}g_{\nu\sigma}-g_{\mu\sigma}g_{\nu\rho}),
$$

the corrected transformation

$$
\delta A_{\mu\nu}
=(\nabla_\mu\nabla_\nu+H^2g_{\mu\nu})\alpha
$$

cancels the curvature term exactly. This is the linear partially-massless
escape, and the executable constant-curvature residual is zero to machine
precision.

A galaxy or cluster is not a constant-curvature spacetime. Decompose its
Riemann tensor into the constant-curvature part and Weyl tensor. After the
correction cancels the former, the exact remainder is

$$
\boxed{
\delta F_{\mu\nu|\rho}
=C_{\mu\nu\rho}{}^\sigma\nabla_\sigma\alpha .
}
$$

The Weyl tensor is precisely the trace-free tidal curvature responsible for
shear and lensing topology. Requiring it to vanish would delete the signal the
carrier was introduced to model. A synthetic electric Weyl tensor in the
audit satisfies all pair symmetries and has zero trace, yet leaves a gauge
residual of `0.6` in normalized units. The computed remainder agrees with the
direct Weyl contraction to machine precision.

This matches the established boundary of partially-massless spin two:
consistent free symmetry is tied to special Einstein/constant-curvature
backgrounds, and perturbatively local unitary coupling to gravity has strong
no-go results. See [Deser, Joung, and Waldron](https://arxiv.org/abs/1301.4181),
[Joung, Li, and Taronna](https://arxiv.org/abs/1406.2335), and
[Garcia-Saenz and Rosen](https://arxiv.org/abs/1410.8734). The project result
does not rely only on those papers: the exact Weyl residual is reproduced in
the local audit.

## Source compatibility does not rescue the symmetry

A coupling

$$
S_{\rm int}=\int\sqrt{-g}\,A_{\mu\nu}J^{\mu\nu}
$$

requires

$$
\nabla_\mu\nabla_\nu J^{\mu\nu}=0
$$

under the scalar gauge transformation.

The obvious source choices separate cleanly:

| Source | Result |
|---|---|
| Full conserved stress tensor `T_mn` | Gauge-compatible, but restores the forbidden direct mass monopole and its wrong fourth-order point-source law |
| Trace-free stress `T_mn-g_mn T/4` | Not conserved for a generic varying trace; its double divergence contains `-box(T)/4` |
| Flat improvement `(nabla_mn-g_mn box)S` | Neutral and conserved in flat space, but its curved divergence is `R_ns nabla^s S` |
| Bach tensor `B_mn` | Symmetric, trace-free, and covariantly conserved; advances only to v14C's energy gate |

Thus a neutral current can be written, but it does not repair the kinetic gauge
symmetry on a Weyl-curved background.

## v14C: Bach/conformal completion and the energy sign

The Bach tensor is the metric Euler tensor of the local Weyl-squared action.
It provides the desired covariant, trace-free, conserved geometric current.
Combined with the required Einstein--Hilbert/GR base, it gives a fourth-order
spin-two operator. Its representative nondegenerate quadratic propagator
factor has the exact partial fraction

$$
\boxed{
{1\over k^2(k^2+m^2)}
={1\over m^2}
\left({1\over k^2}-{1\over k^2+m^2}\right).
}
$$

The two residues have equal magnitude and opposite sign. Changing the overall
sign only exchanges which pole is negative; it cannot make both positive. In
the frozen row the residues are `+1/3` and `-1/3`.

This is the local negative-energy spin-two obstruction of the conformal/Bach
lane. It is materially different from v14A's broken symmetry and v14B's
background restriction, but it fails the same combined carrier requirement.
Accepting the negative pole would violate the goal's bounded-Hamiltonian gate.

## Failure accounting

- v14A minimal curved field strength: rejected;
- v14B partially-massless curvature completion: rejected;
- v14C conformal/Bach completion: rejected;
- common failed gate: healthy local covariant gauge-reduced tidal carrier on
  arbitrary one-metric backgrounds;
- materially distinct failures: `3`;
- mechanism reset: **triggered**;
- observational data opened: no;
- theory viable: no.

This closes the frozen local covariant scalar-gauge rank-two carrier. It is not
a theorem against every nonlocal, explicitly Lorentz-breaking, or arbitrary
higher-spin construction. Under the current project history, however, those
routes are not free loopholes: nonlocal localization, preferred-clock/aether
placement, and positive massive spin two have already triggered their own
mechanism resets. A successor must identify a genuinely different constraint
structure rather than combining their names.

## Reproduction

```powershell
python scripts/audit_sigma_v14_gauge_carrier.py
python -m pytest -q tests/test_sigma_v14_gauge_carrier.py
```

Machine-readable evidence is in
`results/sigma_v14_gauge_carrier_gate/report.json`.
