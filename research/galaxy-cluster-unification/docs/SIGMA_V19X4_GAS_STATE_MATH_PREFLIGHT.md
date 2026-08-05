# Sigma V19X4 gas-state mathematics preflight

## Outcome

The cluster source pipeline now has a target-blind, hash-frozen executable
conversion from future regional X-ray fits to gas density, surface density,
pressure, entropy, sound speed and the thermodynamic inputs needed by a later
shock-state reconstruction. It admits all 494
already-frozen adaptive regions without opening a regional spectrum, lensing
target, inferred halo map or gravity parameter.

The audit also found and corrected a material algebra error in the historical
V19H prose. That parent is hash-bound and remains untouched. The correction is
frozen prospectively here, before the full regional fits exist.

## Correct APEC conversion

The official XSPEC APEC normalization is

$$
N={10^{-14}\over4\pi[D_A(1+z)]^2}\int n_e n_H\,dV.
$$

Let $R=n_e/n_H=1.2$, let $E_A=(\int n_e n_H dV)/A$, and approximate one
region as a uniform slab of line-of-sight depth $L$. Then

$$
E_A={n_e^2L\over R},\qquad
n_e=\sqrt{{R E_A\over L}},
$$

and therefore

$$
\boxed{\Sigma_{\rm gas}=\mu_e m_p\sqrt{R E_A L}}.
$$

V19H had $R$ in the denominator. The corrected surface density is exactly
$R=1.2$ times the historical value: a 20% upward correction, equivalent to a
16.7% understatement relative to the corrected result. This is a baryon-map
correction, not a new force parameter.

## Frozen geometry and uncertainty

For a region containing $N_{\rm pix}$ pixels,

$$
A=N_{\rm pix}(0.984\ {\rm arcsec}\;s)^2,
$$

where $s$ is the frozen Planck18 kpc-per-arcsecond scale for that cluster. The
reference depth is $L_{\rm ref}=\sqrt A$. Each of 4,096 scrambled-Sobol draws
uses a universal log-uniform depth factor from 0.5 to 2.0. This implements the
already-frozen V19H geometric-mean depth prior identically in both clusters.

Ordered temperature and normalization profiles drive bounded asymmetric
uncertainty draws. A finite fit with a failed profile is not discarded: the
complete frozen fit-bound log-uniform prior is used and the region is flagged.
Dependence between temperature and normalization is tested at rank
correlations -0.9, 0 and +0.9 when a joint likelihood surface is unavailable.

The scrambled-Sobol construction has $2N+1$ dimensions for a cluster with $N$
regions. Each region receives distinct temperature and normalization draws;
one shared depth-factor draw represents the cluster-wide deprojection scale.
The same underlying Sobol points are reused across all three dependence
branches, so differences between them are caused by the declared dependence
rather than Monte Carlo noise.

No hydrostatic-equilibrium assumption is used. Shock Mach number and shock
speed follow from the measured density/temperature jumps and ordinary
Rankine-Hugoniot relations.

## Common physical grid

Both clusters are placed on the same east-positive, north-positive physical
grid spanning $-1200$ to $+1200$ kpc at 10-kpc spacing: 241 cells on each axis.
Adaptive-region identities are transferred by nearest-neighbor sampling before
any field value is assigned. Invalid V19M bins are masked rather than filled.

Posterior summaries are stored at the 5th, 16th, 50th, 84th and 95th
percentiles. Every summary is exposed at predeclared 50-kpc and 100-kpc FWHM
resolutions. Mask-normalized smoothing prevents missing pixels from acting as
zero density, and surface-density maps are rescaled within the admitted mask
to conserve their discrete gas mass to a required relative error of $10^{-6}$.

The real frozen geometry passes the manufactured admission test: all 366
Bullet and all 128 Abell 2146 region IDs occur on the 241-by-241 grids. The
measured regional fit values remain unavailable, so this proves geometry and
algorithm coverage only.

## What this enables

After V19W4, V19X2 and V19X3 pass, these maps can distinguish several cases
that a radial acceleration-only test cannot:

- a field tied simply to baryonic density;
- a field tied to overlap between separated baryonic components;
- a field tied to relative currents or merger direction;
- a field tied to anisotropic pressure/stress;
- a field tied to thermodynamic gradients or baroclinicity; and
- a density-only negative control.

Those are precisely the geometrical differences needed to test cluster
offsets and lensing topology without inserting a “cluster mode.” They remain
source hypotheses until a universal invariant passes the frozen gates and is
derived from one covariant action.

## Claim boundary

This preflight verifies algebra, provenance, posterior execution and region
admission. It is not evidence for Sigma Gravity. The synthetic representative
normalizations used by the checker are unit tests only. The executable refuses
to run until the hash-bound V19X3 configuration and terminal passing report
exist. Observed gas maps, source-invariant scores, lensing predictions and
gravity claims all remain unopened.
