# Quantitative stage gates for the void-field theory program

Status: prospective plan recorded 2026-07-26 after the U0 results and before
fitting an H7 weak-field closure. Exact machine-readable thresholds are in
`configs/theory_stage_gates.json`.

## Outcome first

The project will not advance because a formula looks promising. Each stage has
a measurable output, a continue threshold, a stronger scientific-success
threshold, and a bounded rethink trigger. A completed research cycle means one
specified derivation, an implementation with tests, and one frozen evaluation;
changing a parameter bound does not count as a new cycle.

The primary target is one field theory with at most five global physical
parameters, no per-object force parameters, no class label, and no lensing-only
multiplier. Both matter and light must follow one physical metric.

## Stage 0: controls and scale

**Deliverables**

- Reproduce 131 SPARC galaxies/3,034 points, 20 CLASH clusters/84 points, and 50
  external MaNGA BCG points.
- Preserve H0 as the zero-parameter density-contrast/void-tide control.
- Express any enhanced direct-void candidate as the required multiplier over
  H0, not as a newly normalized arbitrary force.

**Decision**

- A direct void force can remain a primary mechanism only if the median required
  amplification is at most 100 and its directional prediction agrees with
  resolved velocity fields.
- At 1,000 or more, direct amplification is considered structurally implausible
  unless independent void-flow data demand it.
- The existing result is of order $10^5$, so H1 is already demoted to a null
  test. It cannot be rescued by fitting SPARC alone.

## Stage 1: reconstruct the response the theory must produce

For spherical/quasistatic nonlinear gravity,

$$
g_{\rm bar}=\mu(g/a_X,X)g.
$$

Every observed point therefore supplies the necessary target

$$
\mu_{\rm req}=\frac{g_{\rm bar}}{g_{\rm obs}},
\qquad
\nu_{\rm req}=\frac{1}{\mu_{\rm req}}.
$$

For points with $g_{\rm obs}>g_{\rm bar}$, invert three declared constitutive
families: the existing RAR closure, simple
$\mu(x)=x/(1+x)$, and standard $\mu(x)=x/\sqrt{1+x^2}$.

**Deliverables**

- `results/constitutive_target/targets.csv` with the required response and
  inferred local $a_X$ for each point.
- A report containing domain quantiles, transition coverage, and numerical
  round-trip errors.
- A plot showing whether SPARC and CLASH constrain a continuous transition or
  are separated enough that a logistic law can merely classify the samples.

**Decision**

- At least 70% of points in each domain must permit the pointwise inverse. If
  not, abandon inverse fitting and use the full forward likelihood without
  clipping or deleting $g_{\rm obs}\le g_{\rm bar}$ points.
- All analytic inverse/forward round trips must agree to relative error
  $10^{-10}$.
- The transition region should contain at least ten systems from each domain.
  If it does not, U0 cannot establish a continuous unified law; the independent
  BCG/host-potential stage becomes mandatory before choosing the transition.

## Stage 2: derive the minimal EV-SVT weak-field closure

Start with H3+H4+H6+H7. Derive the field equation from an action before fitting
its free functions. The first closure may use no more than five global physical
parameters: one response scale, one environment coupling, one environment
range, one transition shape, and at most one metric/vector coupling.

**Required mathematical outcomes**

- $\mu>0$ and $d(\mu g)/dg>0$ throughout the observed domain.
- No singularity over the observed acceleration, potential, or radius ranges.
- For $g_{\rm bar}/a_X\le10^{-3}$, the spherical solution approaches
  $g=\sqrt{a_Xg_{\rm bar}}$ within 5%.
- For $g_{\rm bar}/a_X\ge10^5$, the fractional fifth force is below $10^{-5}$.
- Before cross-validation, the development median absolute acceleration
  residual is at most 0.10 dex separately in SPARC and CLASH.
- The same physical metric yields $\Psi$ for dynamics and $\Phi+\Psi$ for
  lensing. A second fitted normalization is prohibited.

Failure of positivity, monotonicity, or the high-field limit is a theory failure,
not an optimizer problem.

## Stage 3: whole-system predictive validation

Use the existing five whole-system folds and grouped bootstrap. The development
sample is no longer blind, so clearing this stage only licenses the independent
environment test.

### Continue gate

All of the following must hold:

- SPARC $\chi^2/N\le9.41784$, the existing 5% allowance over fixed RAR.
- Raw CLASH $\chi^2/N\le5.00$.
- Equal-domain macro $\chi^2/N\le7.20$.
- The predicted environment sign is the same in all five folds.
- No physical parameter sits on a hard bound.

### Scientific-success gate

- SPARC still satisfies $\chi^2/N\le9.41784$.
- CLASH $\chi^2/N\le2.5$ after the declared 0.063-dex intrinsic-scatter
  sensitivity.
- CLASH RMS is at most 0.16 dex.
- The grouped-bootstrap 95% interval supports the macro improvement.

For comparison, U0 currently gives 9.323 on SPARC, 4.994 raw/2.757 with
intrinsic scatter on CLASH, 0.169-dex CLASH RMS, and a 7.159 macro score. The
domain-labeled oracle gives raw CLASH $\chi^2/N=1.648$ but is not a unified
model.

## Stage 4: independent host and void environment

Construct $X$ without using any scored rotation speed, BCG acceleration, or
lensing mass. Host profiles must come from independently measured gas, stars,
and satellite components. Require at least 30 usable BCG systems and 80%
coverage of the selected sample.

### Continue gate

- BCG $\chi^2/N\le5.0$.
- Absolute mean BCG residual at most 0.15 dex.

### Scientific-success gate

- BCG $\chi^2/N\le3.0$.
- BCG RMS at most 0.17 dex.
- Absolute mean residual at most 0.10 dex.

U0 currently gives 7.149, 0.299 dex, and -0.258 dex respectively. The
cluster-scale RAR reference gives 2.188 but uses a domain-specific scale. The
new theory must approach that accuracy without being told that an object is a
cluster galaxy.

Resolved two-dimensional H I fields provide the separate H2 directional test.
A direct vector force predicts side-to-side and orientation effects; a scalar
environment modulation primarily moves the radial transition. Those signatures
must not be averaged into one folded curve.

## Stage 5: relativistic consistency

Only a candidate that passes Stages 2--4 proceeds. Required outcomes include:

- $|\gamma-1|\le2.3\times10^{-5}$ in its declared Solar-System limit.
- $|c_T/c-1|\le10^{-15}$ for tensor waves.
- Positive kinetic eigenvalues and positive squared gradient speeds for every
  propagating mode on galaxy, Solar-System, and cosmological backgrounds.
- A well-posed initial-value problem and one universal matter metric.

Passing rotation curves cannot compensate for a ghost, gradient instability,
excluded wave speed, or nonconserved matter source.

## Stage 6: cosmology and structure

The final candidate must be evolved on an FLRW background and through linear
perturbations. It must predict expansion, structure growth, void velocity
profiles, and lensing consistently. No claim of a replacement for dark matter
or GR is permitted before this stage. Galaxy and cluster success only establish
an effective low-redshift field law.

## Rethink clock

Use both an effort clock and a calendar checkpoint:

- Maximum three completed research cycles at any stage.
- Checkpoint after seven active research days even if a cycle is unfinished.
- Each cycle must close at least 20% of the gap between the current metric and
  the stage threshold, or identify and test a specific structural obstruction.
- If two materially different closures fail for the same mathematical reason,
  revisit the action rather than add another interpolation parameter.
- Candidate order is minimal EV-SVT, environmental generalized Aether, then
  environmental MOG.
- If all three fail the same gate, stop adding fields and reconsider either the
  void-coupling premise or the requirement that one baryon-only field account
  for all of the inferred mass discrepancy.

This clock counts active work, not periods when the project is unattended.
