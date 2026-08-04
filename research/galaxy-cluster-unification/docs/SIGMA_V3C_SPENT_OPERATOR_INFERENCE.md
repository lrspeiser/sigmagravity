# Sigma v3C spent baryon-to-Weyl operator inference

## Outcome

Before inventing another nonlinear action, Sigma v3C asked what operator the
already-spent cluster maps actually require.  The input was the complete
two-dimensional AQUAL convergence and shear generated from registered baryons;
the target was the same-catalog compact-halo reconstruction.  A single real
translation-invariant transfer was required to act on convergence and both
shear components.  The analysis used no new holdout and makes no validation
claim.

The result rejects wavelength alone as the missing variable.  The joint
two-parameter entire filter reaches normalized Fourier RMSE `0.800`, above the
frozen `0.500` plausibility threshold.  More importantly, an arbitrary radial
transfer fitted independently in each wavelength bin still leaves RMSE `0.773`
in AS295 and `0.708` in PLCKG287.  Moving that nonparametric operator to the
other cluster worsens the errors to `0.956` and `0.800`.  Median radial phase
coherence is only `0.276` and `0.291`, versus the declared `0.800` requirement.

Thus a universal real isotropic convolution cannot turn the measured baryonic
metric into the halo-like Hessian, even if its radial response is allowed to be
an arbitrary table.  The next action must respond to local tensor orientation,
component overlap, or a wider baryonic environment in addition to wavelength.

This is the first full-map result that distinguishes two separate issues:

1. Sigma v3B showed that a linear infrared enhancement has a spectral/causal
   health problem.
2. Sigma v3C shows that even ignoring that problem, scale-only filtering does
   not reproduce the required two-dimensional field.

## Frozen measurement

For each of AS295 and PLCKG287, the calculation sampled a square with 350 kpc
half-width on a `193 x 193` grid.  The source redshift was fixed to `z_s=2`, and
the common coordinate convention was north rows and east columns.  The three
channels were

\[
\kappa,qquad\gamma_1,qquad\gamma_2.
\]

Each map was mean-subtracted and multiplied by a two-dimensional Tukey window.
The scored Fourier band corresponds to wavelengths from 18 to 500 kpc.  These
choices were frozen before the full-map metrics were calculated.

Let `S_c(k)` be an AQUAL source channel and `H_c(k)` the halo target channel,
where `c` runs over convergence and both shear components.  In each radial
wavenumber bin, the best possible shared real transfer is

\[
T_b={\operatorname{Re}\sum_{c,\mathbf k\in b}
H_c(\mathbf k)S_c^*(\mathbf k)
\over
\sum_{c,\mathbf k\in b}|S_c(\mathbf k)|^2}.
\]

Its phase coherence is

\[
\mathcal C_b={
\left|\sum_{c,\mathbf k\in b}H_cS_c^*\right|^2
\over
\left(\sum|S_c|^2\right)\left(\sum|H_c|^2\right)}.
\]

The normalized error gives equal weight to the three physical channels:

\[
\mathrm{NRMSE}=
\left[{1\over3}\sum_c
{\sum_{\mathbf k\in B}|T(\mathbf k)S_c-H_c|^2
\over\sum_{\mathbf k\in B}|H_c|^2}
\right]^{1/2}.
\]

An error near zero would mean that a common wavelength response predicts the
full Hessian.  An error near one means it leaves most target power unexplained.

## Numerical results

| Full-map test | AS295 | PLCKG287 | Frozen gate |
|---|---:|---:|---:|
| Unmodified AQUAL NRMSE | 0.899 | 0.916 | diagnostic |
| Unmodified Newtonian NRMSE | 0.952 | 0.969 | diagnostic |
| Same-cluster arbitrary radial transfer | 0.773 | 0.708 | lower bound only |
| Same-cluster nonnegative radial transfer | 0.773 | 0.708 | lower bound only |
| Other-cluster radial transfer | 0.956 | 0.800 | at most 0.500 |
| Joint entire filter | 0.830 | 0.770 | joint at most 0.500 |
| Median radial coherence | 0.276 | 0.291 | at least 0.800 |
| Negative real-transfer bins | 9.52% | 9.52% | 0% |

The joint entire form

\[
T(k)=\exp\left[Ae^{-(kL_\Sigma)^2}\right]
\]

selects

\[
A=1.3753,qquad L_\Sigma=10.0\ {\rm kpc},
\]

and reaches combined NRMSE `0.8003`.  The length hits its frozen lower bound.
A declared post-failure resolution sensitivity extended the bound to one
quarter pixel (`0.91 kpc`).  It selected `L_sigma=0.95 kpc` and improved the
score only to `0.7867`, still far above the gate.  The primary result is not an
artifact of stopping the length at 10 kpc.

The independently optimized amplitudes are also different:

\[
A_{\rm AS295}=1.155,qquad
A_{\rm PLCKG287}=1.735.
\]

Those values are diagnostics, not proposed object parameters.  Their mismatch
reinforces that a single scale/amplitude response is not identifying the same
operator in both systems.

![Spent full-map Sigma v3 operator inference](../results/sigma_v3c_spent_operator_inference/spent_operator_inference.png)

## What the failure means physically

A real isotropic convolution can broaden every baryonic peak and change how
different spatial wavelengths are weighted.  It cannot independently rotate a
mode or decide that one overlap of gas, BCG light, and member galaxies should
respond differently from another overlap at the same wavelength.

The low coherence shows that the halo comparator is not simply a blurred and
amplified version of the AQUAL field.  Its convergence and shear power occupy
different phases and orientations.  This agrees with the earlier arc-local
result: the candidate-to-halo shear correlation was `-0.067`, and the median
shear-axis error was 17.7 degrees.

The map contours also show why componentwise nonlinearity improved root
multiplicity in P0718.  Applying a nonlinear operation before summing separated
baryonic components can retain information that is destroyed when all baryons
are first compressed into one smooth scalar field:

\[
\mathcal N\!\left(\sum_i\rho_i\right)
\ne\sum_i\mathcal N(\rho_i).
\]

That ordering alone did not transfer accurately between the two clusters, but
it identifies a legitimate variable for a field theory: local overlap and
tidal eigenstructure, not an object label.

## Constraint on the next action

The next theory should use a retarded symmetric trace-free memory rather than a
scalar wavelength filter.  A concrete derivation target is

\[
\mathcal M_{ab}
=L_\Sigma^2
\left(1-L_\Sigma^2\Box_{\rm ret}\right)^{-1}
\left[\mathcal S(X)\,\mathcal E_{ab}\right],
\]

where

- `E_ab` is the electric, symmetric trace-free part of the metric Weyl tensor;
- `S(X)` suppresses the source automatically at high Sigma/nonmetricity
  acceleration;
- the retarded rule and universal past boundary condition uniquely fix
  `M_ab`; and
- invariants such as `M_ab M^ab` and `M_a^b M_b^c M_c^a` distinguish tidal
  eigenvalue patterns without saying "galaxy" or "cluster."

This equation is a response definition, not yet an accepted action.  A viable
interaction must begin beyond quadratic order so that the GR/Sigma-v1 linear
propagator and luminal tensor pole are unchanged.  It must also be derived from
a causal closed-time-path effective action or a local constrained completion;
inserting a retarded inverse into an ordinary single-copy action is not enough.

The next derivation therefore has five pre-fit gates:

1. `M_ab` is transverse/spatial and trace-free under evolution;
2. no homogeneous solution is adjustable per object;
3. the interaction begins at cubic or higher order around the high-field/flat
   background;
4. the principal kinetic matrix remains positive on Solar, galaxy, and cluster
   background ranges; and
5. its weak variation changes both shear orientation and convergence with one
   physical metric.

Only after those gates pass should a 2D nonlinear solver or another raw
holdout be run.

## Claim boundary

The compact-halo map is a useful target decomposition, not direct truth.  It
was constrained by the same lensing images and includes fitted external shear.
The Fourier window omits baryons outside 350 kpc and mixes nearby wavelengths.
Consequently, this result rejects translation-invariant real linear filtering
of the registered source maps; it does not reject every nonlocal or nonlinear
modified-gravity theory.

## Reproduction

```powershell
python scripts/infer_sigma_v3c_spent_operator.py
python -m pytest tests/test_sigma_v3c_operator_inference.py -q
python -m ruff check src/voidscreen/sigma_operator_inference.py scripts/infer_sigma_v3c_spent_operator.py tests/test_sigma_v3c_operator_inference.py
```
