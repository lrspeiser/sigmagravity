# Arc-apogee cross-domain results

## Result in one sentence

The data favor putting baryonic concentration in the **directional routing** of
the proposed field, not in its scalar strength; after making that separation,
the best Solar-safe universal galaxy law is 25.3% worse than fixed RAR in
held-out rotation speed and the best cluster geometry is competitive with a
local-light baseline but not with the strongest previously tested arc controls.

This is an exploratory formula-discovery result. It is not evidence that
gravity physically travels along the reconstructed paths, and it does not beat
dark-matter or MOND analyses.

## Hypothesis being tested

The inverse cluster reconstruction suggested that influence sourced by
baryonic galaxies might spend part of its path away from the source and return
over a characteristic distance. We represented the scalar acceleration by

$$
g(r)=g_b(r)\left[1+q\,
\frac{x^\alpha}{1+(x/\zeta)^\alpha}
\frac{1}{1+(g_b/a_0)^n}\right],
\qquad x=\frac{r}{R_b},
$$

where

- $g_b$ is ordinary Newtonian acceleration from the observed baryons;
- $q$ is one universal dimensionless residence/return strength;
- $x^\alpha$ lets returned influence accumulate with distance;
- $\zeta R_b$ is the apogee or saturation scale;
- $[1+(g_b/a_0)^n]^{-1}$ turns the modification off in high-acceleration
  environments such as the Solar System; and
- $a_0=1.2\times10^{-10}\ \mathrm{m\,s^{-2}}$ is held fixed.

The cluster map uses a separate normalized routing kernel,

$$
K=(1-s(C))K_{\rm local}+s(C)K_{\rm arc},
$$

with $C=R_{50}/R_{80}$. The soft function $s(C)$ controls whether influence is
left near the observed galaxy light or deposited along a center-directed arc.
It does **not** change the total routed amount. In plain language: concentration
chooses the route, while $q$ sets the amount.

## What was actually run

The initial frozen sweep contained:

| Test | Coverage | Quantity scored |
|---|---:|---|
| SPARC galaxy rotation | 540 variants, 131 galaxies, 968 outer points | Five-fold held-out-galaxy outer velocity RMSE |
| RELICS cluster morphology | 180 kernels, 10 clusters, 2 reconstructions | 3,600 normalized-map scores using JS divergence and Pearson correlation |
| Solar proxies | All 540 galaxy variants | Fractional force, Earth orbit, and Mercury perihelion checks |

Every galaxy variant fitted one global $q$ to the inner portions of all
training galaxies. No galaxy received its own strength. The baryonic,
distance, and inclination nuisance values were inherited unchanged from the
fixed-RAR fit so that this stage measures the response of the formula rather
than a new full likelihood fit.

The boundary-refinement sweep then tested 1,440 variants with

$$
R_b=R_{80}^{1-\mu}R_M^\mu,
\qquad R_M=\sqrt{GM_b/a_0},
$$

where $\mu=0$ uses measured baryonic extent and $\mu=1$ uses the square-root
mass radius. It also refined $\alpha$, $\zeta$, and $n$.

## Galaxy result

Using the same RAR-derived nuisances, the reference outer-curve errors are:

| Formula | Outer RMSE (km/s) |
|---|---:|
| Newtonian baryons | 72.399 |
| Fixed RAR | 10.348 |
| Simple MOND | 10.440 |
| Best Solar-safe arc-apogee law | 12.966 |

The selected exploratory law is candidate `R1322`:

$$
\mu=1,\quad \alpha=0.75,\quad \zeta=100,\quad n=1,
\quad q=1.455674.
$$

Its five training-fold values of $q$ range only from 1.446807 to 1.468545,
which is good evidence that the fitted universal amplitude is numerically
stable across these folds. Its held-out error is 25.3% above fixed RAR. It is a
large improvement over Newtonian baryons but does not match RAR or MOND.

The radius interpolation was the dominant control and improved monotonically:

| $\mu$ | Meaning | Best Solar-safe held-out RMSE (km/s) |
|---:|---|---:|
| 0.00 | measured $R_{80}$ only | 23.524 |
| 0.25 | mostly measured extent | 21.354 |
| 0.50 | equal geometric mixture | 18.655 |
| 0.75 | mostly square-root mass radius | 15.421 |
| 1.00 | $\sqrt{GM_b/a_0}$ only | 12.966 |

This trend is scientifically important: most of the galaxy success arrives
when the model imports the same square-root baryonic-mass scale that underlies
deep-MOND/RAR behavior. The proposed route interpretation is different, but the
successful radial scaling is not yet an independent alternative to MOND.

The best score without Solar restrictions was 11.890 km/s. It used $n=0.5$
and failed badly at Earth and Mercury. Solar screening therefore matters, but
it costs only about 1.08 km/s; it is not the main reason the surviving model
misses RAR.

### Galaxy-type residuals

| Subsample | Arc RMSE | RAR RMSE | Arc/RAR |
|---|---:|---:|---:|
| Late type | 9.037 | 8.132 | 1.111 |
| Intermediate mass | 11.552 | 9.828 | 1.175 |
| Disk dominated | 11.918 | 9.942 | 1.199 |
| Gas rich | 10.639 | 8.625 | 1.234 |
| Bulge dominated | 15.587 | 12.306 | 1.267 |
| Dwarf mass | 8.116 | 6.315 | 1.285 |
| Giant mass | 17.489 | 12.808 | 1.365 |
| Gas poor | 13.424 | 9.628 | 1.394 |

The clearest remaining failure is not a random set of galaxies. The law does
best on late-type and disk-dominated systems and overpredicts increasingly in
giant and gas-poor systems. A future change should target that residual without
adding a per-galaxy knob.

## Cluster-map result

The inverse-derived primary kernel obtained median JS divergence 0.06678 and
median Pearson correlation 0.708. The best same-data replay, `K0077`, used a
soft extent gate, a return length of $0.5R_{80}$, width $0.18R_{80}$, and a
half-weight outward-arc deposition. It obtained median JS 0.05772 and median
Pearson 0.783.

For JS divergence, lower is better:

| Normalized cluster map | Median JS | Mean JS | Median Pearson |
|---|---:|---:|---:|
| Local light, 75-kpc smoothing | 0.06210 | 0.08910 | 0.739 |
| Central smooth halo | 0.07891 | 0.08817 | 0.736 |
| Previous C0351 arc | 0.05391 | 0.07249 | 0.819 |
| Previous W060 arc | 0.05344 | 0.06773 | 0.834 |
| New K0077 routed arc | 0.05772 | 0.10404 | 0.783 |

K0077 improves the median over local light, but its worse mean exposes severe
failures in some clusters. It also trails both C0351 and W060. Because K0077 was
selected on these same ten maps, it requires entirely untouched clusters before
being treated as predictive.

The parameter ranking explains the placement decision:

| Domain | Largest control | Best setting | Best-to-worst span |
|---|---|---|---:|
| Galaxy scalar strength | concentration gate | no gate | 18.586 km/s |
| Galaxy radial scale | $R_b$ choice | square-root mass | 9.009 km/s |
| Cluster spatial shape | concentration gate | soft gate | 0.0300 JS |
| Cluster deposition | path geometry | chord-like | 0.0121 JS |

The concentration gate is simultaneously the largest harm to galaxy amplitude
and the largest help to cluster morphology. This is why the separated placement
is more than an arbitrary domain switch: the scalar trace and the directional
distribution are different mathematical observables of one conserved kernel.

## Solar-System screen

For `R1322`, the diagnostic values are:

| Check | Predicted modification | Applied bound | Result |
|---|---:|---:|---|
| Earth fractional force | $3.84\times10^{-11}$ | $1.0\times10^{-10}$ | pass |
| Maximum fractional force, solar limb to Saturn | $1.35\times10^{-8}$ | Cassini proxy $2.3\times10^{-5}$ | pass |
| Mercury supplementary precession | $-2.04$ mas/century | $|\Delta\dot\varpi|<3.1$ mas/century | pass |

These are first-order, zero-slip diagnostics. They are not a derived PPN limit
or a multi-planet ephemeris fit, so “Solar safe” here means only that the
candidate passes this stage's specified proxies.

## What we learned and what we did not

The most defensible new observation is a placement rule:

1. The total low-acceleration enhancement wants a universal square-root mass
   scale and little or no concentration dependence.
2. The *location* of reconstructed cluster convergence benefits from
   concentration-dependent nonlocal routing.
3. A high-acceleration screen with $n\geq1$ is required by Solar proxies.
4. The scalar galaxy law remains close in structure to MOND/RAR and performs
   worse than both.
5. The cluster comparison tests normalized morphology, not absolute lensing
   mass, time delays, shear catalogs, or a relativistic metric.

Thus the current formula is a useful phenomenological field ansatz, not yet new
field physics. To become one, it needs an action or covariant field equation
whose Green's function produces both the scalar law and the routing kernel
without inserting them independently.

## Next falsifiable stage

Freeze `R1322` and a single directional kernel before touching new data, then:

1. refit the galaxy nuisance parameters independently for the arc law and score
   entirely held-out galaxies;
2. predict absolute convergence, shear, and multiple-image positions for new
   clusters using measured stars and gas, rather than normalized maps;
3. derive gravitational slip instead of assuming that photons and massive
   tracers receive the same modification; and
4. reject the family if the one locked setting cannot pass both the galaxy and
   absolute-lensing gates.

Reproduce the two sweeps with:

```powershell
python scripts/run_arc_apogee_cross_domain.py
python scripts/run_arc_apogee_boundary_refinement.py
```

Machine-readable results are in
`results/arc_apogee_cross_domain/` and
`results/arc_apogee_boundary_refinement/`.
