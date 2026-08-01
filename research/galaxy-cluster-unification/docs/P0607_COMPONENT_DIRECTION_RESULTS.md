# P0607 baryonic-component route directions

## Outcome

RX J2129 does not support a substantial component-directed gravity arc in this
raw-image test. The training-only screen chooses the registered Chandra gas
morphology for the positive route, but at the smallest tested strength,
`s_theta=0.0025`. After an eight-start geometry refit, its training RMS changes
from 0.495243 to 0.495153 arcsec (0.018% better), while its spent held-out RMS
changes from 1.808663 to 1.810617 arcsec (0.108% worse).

The signed falsification control is more revealing. Training prefers a small
route *opposite* the HST starlight direction, `s_theta=-0.005`, but that also
worsens spent held-out RMS to 1.825896 arcsec. The sub-percent training changes
are therefore consistent with the six ordinary geometry parameters absorbing
a weak perturbation; they are not evidence for a measured flow direction.

## Formula tested

The already frozen P0599 radial field remains the scalar parent. Only its
excess above baryons-only GR is allowed to acquire angular structure:

$$
\boldsymbol\alpha(\boldsymbol\theta,z_s)=
\boldsymbol\alpha_{0599}(\boldsymbol\theta,z_s)+
s_\theta\,\delta\boldsymbol\alpha_P(\boldsymbol\theta,z_s).
$$

For each hard photometric member at $\mathbf x_i$, an observed baryonic
component $P$ supplies a unit attraction direction $\hat{\mathbf d}_{P,i}$.
The unit route template lands at

$$
\mathbf y_i=\mathbf x_i+0.25R_{80}\hat{\mathbf d}_{P,i}
$$

and is smoothed by $0.50R_{80}$. Its convergence perturbation is proportional
to the P0599-minus-baryon carrier, but the mean in every one-arcsecond annulus
is removed. Solving the two-dimensional Poisson equation gives a curl-free
deflection, and its independently sampled circular radial mean is also
subtracted. Thus this stage tests where the excess field points without
silently adding a radial halo.

## Observed directions

The test used 51 hard photometric members, 12,435 positive F160W cells, 15,401
soft X-ray events on the registered grid, and 39.6 ks of Chandra exposure.
The luminosity-weighted direction cosines are:

| Pair | Mean cosine |
|---|---:|
| Continuous starlight vs discrete members | 0.983 |
| Gas vs discrete members | 0.832 |
| Gas vs continuous starlight | 0.775 |

The gas map contains some genuinely different angular information, but the raw
images prefer effectively zero use of it. All 40 signed component variants
retain all 15 training and seven held-out roots.

## What this says about backtracked arcs

The ten-cluster inverse transport result still supplies a valid *way to draw
candidate projected routes*: match baryonic launch weights to the positive
lensing-excess arrival map with balanced optimal transport. This P0607 forward
test says that simply turning those routes into a component-attraction field is
not enough. In RX J2129, component identity changes the unit field by several
arcseconds, but raw images tolerate only a few thousandths of that correction.

That separates three questions which must not be conflated:

1. **Origin attribution:** which baryonic source can be paired with an apparent
   excess location? Existing inverse maps answer this conditionally.
2. **Projected direction:** does a baryonic component predict the angular
   correction? P0607 finds no transferable training/held-out improvement.
3. **Hidden path:** how high or how long the line arcs outside the lens plane?
   A single two-dimensional lens map cannot answer this at all.

The next falsifiable observable is source-redshift tomography. A normal
single-plane correction scales with $D_{ls}/D_s$. A field that leaves and
re-enters the lens plane, or occupies more than one effective plane, can instead
produce a different redshift dependence, plus time-delay or weak-shear
residuals. Those observables can constrain an arc model in a way that a 2-D
endpoint map cannot.

## Cross-domain boundary

The angular layer is exactly absent for the one-dimensional axisymmetric SPARC
profiles and the one-source Solar control. P0599 therefore retains its galaxy
equal-RMSE value of 10.883 km/s, its 20-cluster radial value of 0.1196 dex, and
its screened Solar behavior. This is preservation, not a new success.

RX J2129 is fully spent, F160W is not stellar mass, and X-ray brightness is not
gas mass. A nonzero route must be fixed here and improve a different raw
cluster before it can become evidence.

## Reproduction

```powershell
python scripts/run_p0607_component_direction_raw_lensing.py
python -m pytest tests/test_p0607_component_direction_raw_lensing_results.py -q
```

Machine-readable outputs are in
`results/p0607_component_direction_raw_lensing/`.
