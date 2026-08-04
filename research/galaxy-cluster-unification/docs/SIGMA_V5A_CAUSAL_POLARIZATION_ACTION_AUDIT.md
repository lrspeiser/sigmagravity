# Sigma v5A causal-polarization action audit

## Outcome

Sigma v5A is the first local causal completion of the post-v4 orientation
idea. It passes its current theory-only source, local kinetic, gradient,
causality, static-uniqueness, weak-invariant, high-field, and parameter-count
screens. No observational data were accessed.

It is **not yet frozen as an empirical theory**. The complete metric and
connection equations, cosmological branch, nonlinear constraint count,
background tensor speed, full weak two-potential equations, PPN limit, and
prior-art audit remain mandatory. No galaxy or cluster score is authorized.

## Geometric definitions

Use one physical metric and a flat, torsion-free symmetric-teleparallel
connection. Retain

\[
Q_a=Q_{a\ m}{}^m,
\qquad
\widetilde Q_a=Q^m{}_{am},
\qquad
\mathcal W_a=Q_a-4\widetilde Q_a.
\]

Let

\[
q_\Sigma={a_\Sigma\over c^2},
\qquad
Y={\widetilde Q_a\widetilde Q^a\over4q_\Sigma^2},
\qquad
Z=Y^2.
\]

The dimensionless source is the fixed band-pass

\[
\boxed{J(Z)={Z\over(1+Z)^2}.}
\]

On the static weak branch, \(Y=(g_\Phi/a_\Sigma)^2\), so

\[
J={x^4\over(1+x^4)^2},
\qquad x={g_\Phi\over a_\Sigma}.
\]

It vanishes in flat space, reaches `1/4` at the universal transition
`x=1`, and falls as `x^-4` at high field. At `x=10^5`, its value is
`1e-20`; Solar suppression is structural rather than fitted.

## Causal orientation transport

Introduce one real dimensionless gravitational polarization field
\(\sigma\). Its effective inverse metric is

\[
\boxed{
\mathcal G_\sigma^{ab}=g^{ab}
-{\alpha_\Sigma\over1+\alpha_\Sigma}
{\mathcal W^a\mathcal W^b
\over\sqrt{(\mathcal W_c\mathcal W^c)^2+(4q_\Sigma)^4}}.
}
\]

For finite \(\alpha_\Sigma\ge0\), the rank-one correction has magnitude
strictly below \(\alpha_\Sigma/(1+\alpha_\Sigma)<1\).

- If \(\mathcal W^a\) is spacelike, one spatial kinetic eigenvalue is reduced
  but remains positive.
- If it is timelike, the magnitude of the time kinetic eigenvalue increases.
- In both cases the scalar cone is Lorentzian and lies on or inside the metric
  light cone.

The theory-only admissible range is frozen to

\[
0\le\alpha_\Sigma\le10,
\]

which keeps the minimum spatial kinetic eigenvalue at least `1/11`. An
`alpha=1e6` singular control remains positive but falls to `9.99999e-7`,
showing why unbounded anisotropy would approach strong coupling.

## Local covariant action candidate

Let the Sigma-v2 base action be

\[
S_{\Sigma2}=-{c^4\over16\pi G}\int d^4x\sqrt{-g}
\left[\mathbb Q+2q_\Sigma^2\mathcal H(Y)\right]+S_b[g,\psi_b].
\]

The v5A local candidate is

\[
\boxed{
S_{\Sigma5A}=-{c^4\over16\pi G}\int d^4x\sqrt{-g}
\left\{
\mathbb Q+2q_\Sigma^2\mathcal H(Y)
+{\eta_\Sigma\over L_\Sigma^2}
\left[
L_\Sigma^2\mathcal G_\sigma^{ab}
\nabla_a\sigma\nabla_b\sigma
+\sigma^2-2\sigma J(Z)
\right]
\right\}
+S_b[g,\psi_b].
}
\]

The four provisional universal constants are

\[
\{a_\Sigma,L_\Sigma,\alpha_\Sigma,\eta_\Sigma\},
\qquad \eta_\Sigma>0.
\]

Matter remains minimally coupled to one metric. The polarization is in the
gravitational action and has no object-specific center, scale, orientation, or
initial profile. Its derived stress must be included in every lensing and
dynamics calculation; it is not declared to be ordinary baryons.

## Scalar equation and source uniqueness

Variation with respect to \(\sigma\) gives

\[
\boxed{
\sigma-L_\Sigma^2\nabla_a
\left(\mathcal G_\sigma^{ab}\nabla_b\sigma\right)=J(Z).
}
\]

In a local constant background, the homogeneous dispersion relation is

\[
|\mathcal G^{00}|\omega^2
=\mathcal G^{ij}k_i k_j+L_\Sigma^{-2}.
\]

The time kinetic coefficient, spatial gradient eigenvalues, and mass squared
are positive. The maximum local characteristic speed is at most `c`.

For a stationary isolated system, subtract any two regular solutions and call
their difference \(s\). Multiplying the homogeneous equation by \(s\) gives

\[
\int d^3x\left[
s^2+L_\Sigma^2K^{ij}\partial_i s\partial_j s
\right]=0.
\]

Positive \(K^{ij}\) forces \(s=0\). Thus fixed geometry and decaying boundary
data have one static polarization profile. In exact vacuum, `J=0`, so there is
no regular static source-free halo solution.

The dynamical theory still contains one real scalar mode. Its cosmological
initial state must be a single universal retarded prescription, not fitted per
object. Because the action is quadratic in \(\sigma\), it has no self-supported
static soliton in this candidate. Whether cosmological waves or backgrounds
remain acceptably small is an unresolved calculation, not an assumption.

## Current executable screen

| Check | Result | Gate |
|---|---:|---:|
| Weyl-trace weak identity max error | `2.68e-13` | at most `1e-12` |
| Plain `F(sigma)R` lensing-null error | `2.22e-16` | at most `1e-14` |
| Source at `g/a_sigma=1` | `0.25` | exactly `0.25` |
| Source at `g/a_sigma=1e-5` | `1.00e-20` | at most `1.0001e-20` |
| Source at `g/a_sigma=1e5` | `1.00e-20` | at most `1.0001e-20` |
| Minimum admissible kinetic eigenvalue | `0.0909091` | at least `0.0909091` |
| Maximum characteristic speed/c | `1.0` | at most `1.0` |
| Universal constants | `4` | at most `5` |

All current theory-screen gates pass. The machine-readable report is
`results/sigma_v5a_causal_polarization_action_audit/report.json`.

## Why this is not simply dark matter with a new name

At this stage the distinction is structural, not semantic:

1. \(\sigma\) is part of a universal gravitational action.
2. Its source is a fixed local invariant of the one metric, not an inferred
   mass density or per-object halo profile.
3. Its static profile is unique for fixed geometry and boundary data.
4. It has no regular static source-free lump in the quadratic candidate.
5. The same derived stress affects matter and photons through one metric.

Nevertheless, it is a new gravitational field that carries energy. A critic
can reasonably call any extra gravitating field “dark-sector-like.” The
scientifically meaningful distinction will be whether four universal
constants predict diverse systems without halo initial data or object fits.
That remains to be tested only after the theory is complete.

## Unresolved hard gates

Before any observational score:

1. derive the complete metric and connection Euler equations, including every
   dependence of \(\mathcal G_\sigma\) and \(J\) on nonmetricity;
2. derive the full weak equations for \(\Psi\), \(\Phi\), and
   \(W=(\Psi+\Phi)/2\), including polarization backreaction;
3. count all nonlinear modes of the combined nonmetricity-scalar action and
   exclude hidden ghosts/strong coupling from the base gravity sector;
4. prove `c_T=c` on cosmological and quasistatic backgrounds;
5. define a globally real cosmological branch for the Sigma-v2 primitive;
6. calculate PPN parameters and the Solar response including the universal
   cosmological polarization state; and
7. complete prior-art and field-redefinition audits.

Failure of any item retires v5A before a map fit.

## Reproduction

```powershell
python scripts/check_sigma_v5a_causal_polarization_action.py
python -m pytest tests/test_sigma_causal_polarization.py tests/test_sigma_nonmetricity.py -q
python -m ruff check src/voidscreen/sigma_causal_polarization.py scripts/check_sigma_v5a_causal_polarization_action.py tests/test_sigma_causal_polarization.py
```
