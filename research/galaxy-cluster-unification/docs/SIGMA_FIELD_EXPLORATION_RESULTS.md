# Screened Sigma-field exploration

## Outcome

The proposed Newtonian `+ Sigma` model has now been implemented and explored on
a frozen 125-point universal parameter grid.  It produces a real finite-size
vacuum response, a strong galaxy/cluster gravity enhancement, and a modest
disk-geometry effect without inserting RAR, MOND, a cluster label, or a fitted
lensing amplitude.  Its present bounded coupling does not yet give the same
radial behavior to a galaxy and RX J2129: the boost saturates, after which a
finite baryonic mass returns to a Keplerian decline.

This is an exploratory result, not an advancement or rejection verdict.

## Equation tested

The matter density first determines a dimensionless environmental field:

$$
L_\Sigma^2\nabla^2\Sigma =
\left(\frac{\rho_b}{\rho_s}-1\right)\Sigma+\Sigma^3,
\qquad 0\leq\Sigma\leq1.
$$

That field changes the gravitational permittivity:

$$
\epsilon(\Sigma)=1-\eta\Sigma^2,
\qquad
\nabla\!\cdot\!\left[\epsilon(\Sigma)\nabla\Phi\right]=4\pi G\rho_b.
$$

The three universal quantities have simple roles:

- $\eta$ sets the maximum possible gravity enhancement,
  $g/g_N=1/(1-\eta)$.
- $\rho_s$ is the density at which the low-density Sigma phase begins to be
  favored locally.
- $L_\Sigma$ is the distance over which surrounding high- and low-density
  regions influence one another.

In spherical symmetry the prediction follows directly from Gauss's law:

$$
g(r)=\frac{G M_b(<r)}{\epsilon[\Sigma(r)]r^2}.
$$

## Frozen exploration

The grid contained five values of each of $\eta$, $\rho_s$, and $L_\Sigma$,
for 125 universal settings.  The same setting was applied to a Hernquist galaxy
and the RX J2129 cluster profile.  Both radial field solves converged in 120 of
125 rows; the five unresolved rows all had the smallest density threshold and
shortest length.  None of those rows was a leading result.

The descriptive joint setting was

$$
\eta=0.80,\qquad \rho_s=10^{-23.5}\ {\rm g\,cm^{-3}},\qquad
L_\Sigma=3\ {\rm kpc}.
$$

It has a maximum spherical enhancement of $1/(1-\eta)=5$.

## Results for items 1--6

| Item | Concrete result | What it establishes so far |
|---|---:|---|
| 1. Galaxy scaling | 0.2187 dex RMSE from the comparison RAR over 5--50 kpc; velocity slope $d\log v/d\log r=-0.344$ over 10--50 kpc and $-0.481$ over 100--250 kpc | Sigma supplies a large outer boost, but the joint setting does not maintain a flat curve. The far field is almost Keplerian ($-1/2$). |
| 2. Cluster transfer | 0.2076 dex RMSE from the RX J2129 derived target, with a mean residual of $-0.2055$ dex | One universal setting brings the baryonic cluster field to within a factor of about 1.6 of the derived target, but tends to underpredict it. |
| 3. Dense/local environment | At 7.81 kpc in the disk model, $\Sigma=0.853$, $\epsilon=0.418$, and the galaxy-scale midplane force is 2.84 times its Newtonian value | This measures the Milky-Way-scale background, not AU-scale Solar-System screening. A nested stellar/laboratory calculation is still required. |
| 4. Finite void buildup | An empty spherical cavity remains at $\Sigma=0$ through $R/L_\Sigma=3$, while $\Sigma(0)=0.99995$ at $R/L_\Sigma=10$ | The equation produces genuine accumulation with void size rather than responding only to local density. |
| 5. Disk geometry | At $(R,z)=(12.19,2.97)$ kpc, the vertical/radial force ratio changes from 0.4495 to 0.4785, a focusing ratio of 1.0645 | Disk plus bulge geometry changes the direction of the field by about 6.4% in this example; it is not merely a spherical multiplier. |
| 6. Zero-slip lensing diagnostic | Training radial RMS 0.880 arcsec; seven spent-heldout images all recover roots with 2.982 arcsec radial RMS | The same potential can be propagated to photons, but this diagnostic closure is not yet a relativistic theory and the raw image agreement is not yet precise. |

The RX J2129 radial target is NFW-deprojected and helped rank the parameters, so
it is not independent evidence against dark matter.  The raw image coordinates
are closer to the desired observable, but the holdout has already been examined
in earlier work and is labeled spent.

## The useful parameter tension

The grid did find good settings for either scale separately:

- The galaxy-only optimum is $(\eta,\log_{10}\rho_s,L_\Sigma)=(0.9,-25.5,30)$,
  with 0.0510 dex RAR RMSE and a slightly rising 10--50 kpc velocity slope of
  0.0897.  On the cluster it gives 0.5085 dex RMSE and only 1.63 times Newtonian
  gravity at 100 kpc.
- The cluster-only optimum is $(0.9,-23.5,10)$, with 0.0928 dex cluster RMSE and
  9.49 times Newtonian gravity at 100 kpc.  It over-amplifies the galaxy, whose
  RAR RMSE becomes 0.4647 dex.
- The flattest sampled galaxy curve has slope $-0.0015$, demonstrating that the
  transition can make a flat-looking interval.  Its joint score is poor,
  however, so flatness alone is not the missing universal rule.

This is informative rather than terminal: $\rho_s$ and $L_\Sigma$ can place a
transition at either the galaxy or cluster scale, but the present local-density
trigger does not scale that transition appropriately for both objects.

## A first-principles void threshold

For an empty spherical cavity with a dense wall imposing $\Sigma(R)=0$, the
quadratic field energy near $\Sigma=0$ changes stability when

$$
L_\Sigma^2\left(\frac{\pi}{R}\right)^2=1.
$$

Therefore the analytic onset is

$$
R_{\rm critical}=\pi L_\Sigma.
$$

The numerical solver gives $\Sigma(0)=0$ at $R/L_\Sigma=3.14$ and
$\Sigma(0)=0.288$ at 3.20, then 0.843 at 4 and 0.963 at 5.  This is the cleanest
new behavior in the current model: the Laplacian makes the amount of empty
space matter even when its local density is identically zero everywhere.

## What to explore next

The next equation should preserve the finite-size instability while changing
the saturated radial response.  The present difficulty follows algebraically:
once $\Sigma\to1$, $\epsilon\to1-\eta$ is constant, so
$g\propto r^{-2}$ and $v\propto r^{-1/2}$.  A durable flat point-mass curve
instead requires an effective $\epsilon\propto r^{-1}$ over the relevant
region.

Three controlled follow-ups are therefore more informative than a denser scan
of the same formula:

1. Derive a covariant scalar action whose weak-field limit is the tested
   equation, so stellar motion and lensing are fixed by one metric rather than
   an assumed zero-slip rule.
2. Test one scale-aware coupling at a time, such as a dependence on the Sigma
   correlation length or its gradient, and require it to retain the analytic
   $R_{\rm critical}=\pi L_\Sigma$ behavior.
3. Replace the single idealized galaxy with a small mass-and-size ladder before
   refitting RX J2129.  That will reveal whether a new term actually generates
   mass-dependent transition radii rather than moving one preferred radius.

## Reproduction and artifacts

Run:

```powershell
$env:PYTHONPATH='src'
python scripts/run_sigma_field_exploration.py
python -m pytest tests/test_sigma_field.py tests/test_sigma_field_exploration_results.py -q
```

The machine-readable report is
`results/sigma_field_exploration/report.json`; the complete grid is
`parameter_grid.csv`; radial, void, disk, and raw-lensing predictions are stored
beside it.  The overview figure is `sigma_field_exploration.png`.
