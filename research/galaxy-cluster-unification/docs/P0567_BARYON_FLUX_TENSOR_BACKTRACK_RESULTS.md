# P0567: baryon-flux tensor backtracking

## The field idea

Treat baryonic matter as the only material source. Instead of adding a dark
mass density, allow the surrounding field geometry to redirect the relation
between a conserved baryon-sourced flux and the acceleration reconstructed
from lensing:

\[
\nabla\!\cdot\mathbf J_b=-C\,\Sigma_b,
\qquad
\mathbf J_b=\mathbf K(\mathbf x)\,\mathbf g_L.
\]

Here `J_b` is the conserved projected gravity flux, `Sigma_b` is the baryonic
surface-density proxy, and `g_L=-grad(Psi_L)` is the field implied by the
standard lensing convergence map. `K` is a symmetric positive-definite routing
tensor:

\[
\mathbf K=\mathbf R(\theta)
\begin{pmatrix}e^{u+v}&0\\0&e^{u-v}\end{pmatrix}
\mathbf R(\theta)^T.
\]

The exponential eigenvalues keep the response positive. The tensor may rotate
and rescale the field, but it cannot reverse its sign locally.

An observer who assumes `K=I` would assign the difference to an apparent
source term. In this interpretation, the apparent dark distribution is a
divergence pattern caused by spatial variation of `K`, not additional material:

\[
\rho_{\rm app}-\rho_b\ \propto
\nabla\!\cdot[(\mathbf I-\mathbf K)\mathbf g_L].
\]

This is a weak-field, two-dimensional ansatz. It is not yet a relativistic
field equation.

## A hard local test

At each map pixel, a positive-definite `K` can map `g_L` into `J_b` only when

\[
\mathbf g_L\!\cdot\mathbf J_b>0.
\]

This makes the hypothesis falsifiable without first choosing the power of the
effect. If the two fields point more than 90 degrees apart, no local positive
routing tensor can connect them. Where the condition passes, the minimum
required eigenvalue ratio is

\[
\chi_{\min}=\frac{1+|\sin\theta|}{1-|\sin\theta|},
\]

where `theta` is the angle between the two fields. `chi=1` is no distortion;
large `chi` means strongly directional spacetime response.

## Data and computation

The run used 13 RELICS clusters, 1,300 Lenstool convergence realizations, ten
GLAFIC best maps, and strict photometric-member positions weighted by F160W
light. Both the baryonic and lensing maps were smoothed at 20 kpc. A padded FFT
Poisson solve converted each map into a field. Absolute normalization was
removed, so this stage tests geometry rather than gravity strength.

The total-convergence maxima initially overlapped bright member galaxies and
produced nearly zero-length backtracks. That valid but uninformative diagnostic
was amended before interpretation: the backtrack starts were changed to peaks
of the positive residual after subtracting the best non-negative local
baryon-light projection. The original full-field gates were preserved.

For each residual peak, the code integrates the baryonic flux direction in
5 kpc steps until it reaches within 20 kpc of a catalogued baryonic member or
travels 1,000 kpc. The resulting line is the projected source attribution under
this ansatz. It is not an observed trajectory and cannot recover line-of-sight
curvature.

## Results

Across the ten fresh systems:

- Full-field, convergence-weighted local feasibility: **95.82%**.
- Apparent-dark-residual-weighted feasibility: **95.83%**.
- Weakest full-field systems: **MACS J0257.1-2325 (77.33%)** and
  **ACT-CL J0102-4915 (82.90%)**.
- Pooled minimum tensor anisotropy: median **1.56:1**, 90th percentile
  **5.59:1**.
- New-analysis holdout feasibility: **99.94%**, versus **94.05%** in the seven
  development systems. The holdout is only internal to P0567; all ten systems
  had been inspected by earlier project analyses.
- Median absolute Lenstool-versus-GLAFIC feasibility difference: **0.16
  percentage points**.
- Residual-peak backtracks: **24 of 25** reached a catalogued member, with
  median projected path length **58.7 kpc** and median tortuosity **1.024**.
- The one unresolved path is the second residual peak in A3192. It approaches
  to 25.7 kpc of a member but does not cross the frozen 20 kpc arrival radius
  before the 1,000 kpc integration cap, consistent with a field equilibrium or
  an incomplete baryon catalog rather than a successful attribution.

All three predeclared representation gates pass. This means the proposed local
tensor is geometrically possible over most of these maps and does not require
extreme anisotropy over most of the weighted area. It does **not** mean that the
tensor predicted the lensing maps: the target lens field was used to calculate
the pointwise lower bound.

## What the observation tells us

The useful empirical clue is not simply that baryons and apparent halos are
near one another. It is that their reconstructed field directions are usually
compatible with a positive local mapping, while two clusters contain sizable,
repeatable opposition regions. A universal routing theory should predict both
the broad alignment and those failures from baryonic geometry alone.

ACT-CL J0102 is the strongest stress case. Its residual-weighted feasibility
ranges from 77.1% to 86.3% across the sampled Lenstool realizations and falls to
74.0% under GLAFIC. MACS J0257 stays near 78% under both methods. Tuning a rule
to hide these regions would erase the most discriminating evidence.

## What is still missing

1. The baryon map omits hot gas, diffuse intracluster light, and mass-to-light
   variation. Those omissions matter most in clusters.
2. The convergence maps are standard-GR parametric reconstructions, not raw
   image, shear, flexion, or time-delay data.
3. A pointwise tensor has too much freedom. Many smooth tensor fields can share
   the same local lower bound.
4. Projection prevents recovery of the true three-dimensional arc or any path
   through an additional field dimension.
5. Normalized morphology cannot determine the absolute multiplier needed to
   explain lensing strength or galaxy rotation.

## Next predictive experiment

Fit a low-parameter tensor from baryonic information only:

\[
\mathbf K_\vartheta(E_b)=
\exp\!\left[
 a_0\mathbf I+
 a_1\,\widehat{\nabla\Sigma_b\nabla\Sigma_b}+
 a_2\,\widehat{\mathbf T_b}+
 a_3\,s(\Phi_b)\widehat{\mathbf T_b}
\right],
\]

where `T_b` is the traceless baryonic tidal tensor, hats indicate normalized
tensor shapes, and `s(Phi_b)` is a universal environment transition. The matrix
exponential guarantees positive eigenvalues. Fit the four universal
coefficients on the seven P0567 development systems, then predict the complete
convergence morphology of the three P0567 holdouts without consulting their
target pixels during fitting.

That is the point where the idea changes from an inverse explanation to a
scientific prediction. A survivor would then need complete gas/stellar maps
and tests against raw strong- and weak-lensing observables.

## Forward follow-up completed

P0568 performed that baryon-only compression with nine tensor families. A
small low-density tidal term transferred directionally, improving the locked
normalized holdout maps by 8.01% versus the development-selected local-light
null, but it missed the frozen 10% gate and failed the SPARC transfer badly.
Width/coupling refinements showed that broad baryonic extent dominates the
result and that the stronger development optimum does not improve prior
holdout transfer. See `docs/P0568_BARYON_ONLY_TENSOR_FORWARD_RESULTS.md`.
