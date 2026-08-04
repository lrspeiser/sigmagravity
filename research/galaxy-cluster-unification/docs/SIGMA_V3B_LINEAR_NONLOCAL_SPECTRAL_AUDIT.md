# Sigma v3B linear nonlocal spectral audit

## Outcome

Sigma v3B made the earlier "universal gravity is stronger, but its local
summation is incomplete" proposal mathematically explicit.  A scale-dependent
metric propagator can be essentially Newtonian in the Solar System and inside
galaxies, become stronger across a cluster, and spread baryonic structure into
a broad convergence and shear pattern.  The manufactured two-component map
confirms that this is not merely a radial multiplier: the same filtered metric
potential changes both shear components, with a relative shear-map L2 change of
`1.054` in the frozen fixture.

The simplest healthy linear explanation does not survive.  Once the local
high-momentum coupling is normalized to measured Newtonian gravity, every
ordinary positive Kallen--Lehmann spectrum is at least as strong at high
momentum as at low momentum.  It can make gravity weaken with distance, not
grow toward cluster scales.  A rational filter that gives the desired growth
has a negative-residue massive pole.  An entire nested-exponential filter
avoids extra finite poles and retains the luminal massless tensor pole, but it
reverses the standard positive-spectral monotonicity and does not yet have a
proved causal Lorentzian prescription for this infrared use.

Accordingly, no linear Sigma v3 equation is frozen.  Ordinary positive-spectrum
and rational linear infrared enhancement are retired.  The entire filter is
retained as a precise mathematical clue, not as an accepted theory.  The next
candidate must be nonlinear and leave the quadratic Sigma-v1/GR propagator
unchanged.

This is a pre-fit theory audit.  It does not count as a third raw cluster
topology failure.

## Quadratic metric question

Around a flat background, write a gauge-invariant one-metric quadratic action
schematically as

\[
S^{(2)}={M_*^2\over8}\int d^4x\,
h\,\mathcal P_{\rm GR}\,a(\Box)\Box h
+{1\over2}\int d^4x\,h_{\mu\nu}T_b^{\mu\nu}.
\]

`P_GR` denotes the usual constrained massless spin-2 projector.  If `a(z)` has
no finite zero, the propagator adds no finite pole.  In a static problem,

\[
\widetilde \Phi(\mathbf k)
=-{4\pi G_N\over k^2}\,T(k^2)\widetilde\rho_b(\mathbf k),
\qquad
T(k^2)={1\over a(-k^2)}.
\]

A covariant quadratic-curvature envelope can generate such a kinetic operator
by combining `R F_1(Box) R`, `R_mn F_2(Box) R^mn`, and Weyl terms so that the
spin-0 and spin-2 pole conditions are satisfied.  Sigma v3B audits the
propagator that any such completion must have; it does not claim that the
nested-exponential choice already has a complete nonlinear covariant action.

## The positive-spectrum obstruction

For Euclidean momentum `s=k^2`, an ordinary positive spectral propagator has

\[
D_E(s)={Z_0\over s}
+\int_0^\infty {\rho(m^2)\over s+m^2}\,dm^2,
\qquad Z_0>0,\quad\rho(m^2)\ge0.
\]

Normalize the force carrier to its high-momentum value:

\[
T_{\rm KL}(s)=
{sD_E(s)\over Z_0+\int\rho(m^2)dm^2}.
\]

Then

\[
{dT_{\rm KL}\over ds}
={1\over Z_0+\int\rho}
\int_0^\infty
{\rho(m^2)m^2\over(s+m^2)^2}\,dm^2\ge0.
\]

Therefore

\[
T_{\rm KL}(0)\le T_{\rm KL}(\infty)=1.
\]

This is the key result.  A healthy linear collection of massless and massive
exchanges cannot make the infrared gravitational coupling exceed the locally
calibrated ultraviolet coupling.  The proof is independent of how many Yukawa
masses are used and applies separately to every polarization having a standard
positive spectral density.  Adding a tidal projector supplies orientation but
does not change the residue sign argument.

## Rational causal control

The simplest transfer with the desired direction is

\[
T_{\rm rat}(k^2)=1+{A_{\rm rat}\over1+k^2L_\Sigma^2}.
\]

It gives the point-force law

\[
{g(r)\over g_{\rm local}}
=1+A_{\rm rat}\left[1-(1+r/L_\Sigma)e^{-r/L_\Sigma}\right].
\]

It is locally screened and approaches `1+A_rat` at large radius.  But its
propagator decomposes as

\[
{T_{\rm rat}\over k^2}
={1+A_{\rm rat}\over k^2}
-{A_{\rm rat}\over k^2+L_\Sigma^{-2}}.
\]

The massive residue is negative.  The spent cluster amplitude anchor requires
`A_rat=5.7268`, giving residues `+6.7268` and `-5.7268`; the negative residue is
85.1% of the massless residue.  Interpreted as a fundamental linear mode, that
is precisely the negative-norm degree of freedom prohibited by the project.

## Entire no-pole escape

The more interesting transfer is

\[
\boxed{
T_{\rm ent}(k^2)
=\exp\!\left[A\exp(-k^2L_\Sigma^2)\right]
},
\]

with inverse kinetic form factor

\[
a(-k^2)=\exp\!\left[-A\exp(-k^2L_\Sigma^2)\right].
\]

It has no finite zeros.  The only finite propagator pole is still `k^2=0`, so
the tensor dispersion remains luminal at this quadratic level.  Its limiting
couplings are

\[
T_{\rm ent}(\infty)=1,
\qquad
T_{\rm ent}(0)=e^A.
\]

Expanding the nested exponential gives a useful physical picture:

\[
T_{\rm ent}(k^2)
=1+\sum_{n=1}^\infty {A^n\over n!}e^{-nk^2L_\Sigma^2}.
\]

Each term is a positive Gaussian image of the baryonic source, with width
`sqrt(n) L_sigma`.  No image center, ellipticity, orientation, or amplitude is
chosen per object.  For a point source, the force ratio is

\[
{g(r)\over g_{\rm local}}=1+
\sum_{n=1}^{\infty}{A^n\over n!}
\left[
\operatorname{erf}\!\left({x\over2\sqrt n}\right)
-{x\over\sqrt{\pi n}}e^{-x^2/(4n)}
\right],
\quad x={r\over L_\Sigma}.
\]

The correction begins as `x^3` locally and tends to `e^A-1` far away.  This is
the cleanest mathematical realization so far of local gravity vectors being
"lost" while the universal large-scale coupling is stronger.

Using only the spent median convergence ratio,

\[
e^A={\kappa_{\rm halo}\over\kappa_{\rm AQUAL}}
={0.68868\over0.10238}=6.72679,
\qquad A=1.90610.
\]

For an illustrative, unfitted `L_sigma=100 kpc`:

| Radius | Predicted force ratio |
|---|---:|
| 1 AU | fractional addition `3.07e-32` |
| 10 kpc | 1.000269 |
| 500 kpc | 5.89565 |
| asymptotically large | 6.72679 |

This demonstrates scale separation, not empirical validation.  The amplitude
was read from a spent halo diagnostic and the length was illustrative.

The unresolved issue is foundational.  The transfer decreases with Euclidean
momentum,

\[
{dT_{\rm ent}\over d(k^2L_\Sigma^2)}
=-A e^{-k^2L_\Sigma^2}T_{\rm ent}<0,
\]

opposite to the standard positive Kallen--Lehmann condition.  Since it is an
entire function, absence of extra poles is not by itself a proof of reflection
positivity, a causal Lorentzian initial-value problem, or nonlinear stability.
Generalized nonlocal spectral representations may evade the standard theorem,
so the audit does not label this form factor a proved ghost.  It labels the
required health proof **missing**, which is enough to prevent freezing it under
the project gates.

## Manufactured shear result

The fixture contains two offset Gaussian baryonic components and a faint
bridge.  The same transfer multiplies the common metric potential before its
Hessian is calculated:

\[
\kappa={1\over2}(W_{,xx}+W_{,yy}),\quad
\gamma_1={1\over2}(W_{,xx}-W_{,yy}),\quad
\gamma_2=W_{,xy}.
\]

The filtered-minus-local shear norm is `1.054` times the local shear norm.  The
linear nonlocal idea therefore passes the orientation test that Sigma v1,
Sigma v2, and the local spherical edge bound could not address.  Its failure is
the spectral/causal completion, not its ability to generate a two-dimensional
tidal pattern.

![Sigma v3B spectral and manufactured-shear audit](../results/sigma_v3b_linear_nonlocal_spectral_audit/linear_nonlocal_spectral_audit.png)

## Prior art and novelty boundary

Covariant quadratic-curvature actions with entire form factors and no added
flat-background propagator poles are established prior art; see
[Biswas et al.](https://arxiv.org/abs/1110.5249).  Generalized spectral
representations for nonlocal quantum gravity are an active technical subject;
see [Briscese et al.](https://arxiv.org/abs/2405.14056).  A traditional
variation of actions containing explicitly retarded inverse operators can
reintroduce advanced kernels, as demonstrated by
[Zhang et al.](https://arxiv.org/abs/1601.03808).  These facts are why
"entire," "retarded," and "ghost-free" are treated as separate gates.

The exact nested-exponential infrared transfer was not found in the audited
formula set.  That does not establish originality.  The project contribution
at this stage is the explicit transfer, its analytic point-force series, the
spent amplitude translation, the manufactured shear test, and the transparent
statement of the missing health proof.

## Decision and next derivation

The next candidate must turn on **nonlinearly**, so its expansion about the
high-field/Minkowski background starts beyond quadratic order.  Then the
linear propagator, its positive residue, and `c_T=c` can remain those of
Sigma-v1/GR while a retarded tidal memory changes the cluster solution.

The nonlinear candidate must still provide:

1. a causal closed-time-path or equivalent retarded variational definition;
2. a unique baryon-forced state with no homogeneous halo profile;
3. a bounded interaction and a positive principal kinetic matrix on the
   backgrounds actually used;
4. automatic high-acceleration and short-distance suppression;
5. both metric potentials and a nonzero trace-free shear prediction; and
6. at most two additional universal constants beyond `a_sigma`.

## Reproduction

```powershell
python scripts/check_sigma_v3b_linear_nonlocal.py
python -m pytest tests/test_sigma_v3b_linear_nonlocal.py -q
python -m ruff check src/voidscreen/sigma_nonlocal_spectral.py scripts/check_sigma_v3b_linear_nonlocal.py tests/test_sigma_v3b_linear_nonlocal.py
```
