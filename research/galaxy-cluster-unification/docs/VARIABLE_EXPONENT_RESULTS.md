# Universal baryon-conditioned exponent test

## Question

Can the exponent in the curvature-running law be a universal function of the
measured baryons, rather than a fixed square or a different fitted number for
each galaxy?

## Frozen law

The tested acceleration law was

$$
g_{\rm pred}=g_{\rm bar}
\left[1+\left({T_*\over T}\right)^{p(X)}\right]^\epsilon,
\qquad T={g_{\rm bar}\over r},
$$

with the bounded exponent

$$
p(X)=p_0\exp\left[\beta\tanh\left(\ln{X\over X_*}\right)\right].
$$

This has five universal constants: $T_*$, $p_0$, $\beta$, $X_*$, and
$\epsilon$. No gravity constant was fit separately to an individual galaxy or
cluster. Setting $\beta=0$ exactly recovers the constant-exponent power law.
The exponential and hyperbolic tangent guarantee
$p_0e^{-|\beta|}\le p(X)\le p_0e^{|\beta|}$, so the exponent remains positive
and finite even when the baryonic property approaches zero.

Three definitions of $X$ were frozen before scoring:

1. force-equivalent enclosed baryonic mass,
   $M_{\rm eq}=g_{\rm bar}r^2/G$;
2. reconstructed local baryonic volume density, $\rho_b$;
3. distribution shape, $\rho_b/\bar\rho_b$, with
   $\bar\rho_b=3g_{\rm bar}/(4\pi Gr)$.

The first quantity equals enclosed mass for a spherical source but is only a
force-equivalent proxy in a disk. The third is a dimensionless concentration
or radial-profile proxy. Mean enclosed density by itself was not tested as a
new variable because it is exactly proportional to the existing curvature
input $T=g_{\rm bar}/r$; calling it density would add no new information.

## Test design

All five constants were fit only on the five-fold, system-held-out bridge of 44
BCGs and 20 lensing clusters (116 points). Those constants were then transferred
unchanged to 131 SPARC galaxies. Each SPARC galaxy retained the same ordinary
stellar mass-to-light, distance, and inclination nuisance fit used for every
benchmark: nuisances were fit on 2,066 inner points and predictions were scored
on 968 untouched outer points. The same zero-slip law was also sent through the
spent-holdout RX J2129 image-position diagnostic. The Solar-System proxy was
checked from the solar limb through Saturn.

## Results

| Law | held-out BCG + cluster RMSE (dex) | SPARC outer RMSE (km/s) | RX J2129 held-out RMS (arcsec) | result |
|---|---:|---:|---:|---|
| variable enclosed mass | **0.1158** | 61.54 | 9.51 | cluster improvement does not transfer to galaxies |
| variable local density | 0.1241 | 38.87 | 8.77 | same failure, smaller magnitude |
| variable distribution shape | 0.1396 | **15.26** | not scoreable; 6/7 roots | best variable compromise, but no improvement |
| fixed $p=2$ control | 0.1377 | 14.40 | 1.61 | better joint result than every variable law |
| free constant-$p$ control | 0.1383 | 14.58 | 1.53 | better joint result than every variable law |
| fixed RAR | -- | 10.35 | -- | galaxy reference |
| per-galaxy NFW control | -- | 17.80 | -- | flexible galaxy dark-halo reference |
| compact cluster halo | -- | -- | 2.54 | cluster-lensing reference |

The simple-MOND bridge reference is 0.418 dex, so the variable laws remain far
better than simple MOND on the cluster acceleration targets. That alone is not
a unification: the mass and density versions fail their independent galaxy
transfer, while the shape version fails to improve the constant law and does
not produce all raw image roots.

All full fits satisfy the conservative Solar-System proxies. Their maximum
fractional coupling changes between the solar limb and Saturn are
$6.2\times10^{-10}$, $5.0\times10^{-9}$, and $3.4\times10^{-9}$, all below the
$2.3\times10^{-5}$ proxy gate. This does not replace a relativistic PPN
derivation.

## What the fit learned

The mass and density laws did use their new freedom. On the bridge, the mass
law selected exponents from 0.376 to 0.682 and the density law selected 0.611 to
1.004. This let them distinguish BCG/cluster regimes and lower the bridge error.
The same parameters then made the enhancement grow much too strongly through
ordinary galaxy outskirts. The mass law illustrates the problem most clearly:
its isolated $10^{11}\,M_\odot$ diagnostic grows from an enhancement of 3.3 at
10 kpc to 108 at 100 kpc.

The distribution-shape law found the opposite answer. Across every BCG and
cluster bridge point it chose $p=5.0004$ to four decimal places. Its pivot sat
next to the low edge of the allowed range, saturating the bounded function.
In plain language, the data switched the proposed dependence off and returned
an effectively constant exponent. Its SPARC score is accordingly close to the
fixed-$p$ controls, but slightly worse.

The fold fits are also unstable for the mass and density laws: different held-out
folds select distinct combinations of $p_0$, $T_*$, and $\epsilon$. That is a
warning that the bridge improvement is partly parameter tradeoff rather than a
well-identified physical scaling.

## Decision

Making this exponent a function of baryonic mass, density, or distribution is
mathematically valid and can be written compactly. These three versions do not
advance as the universal law. The useful negative result is sharper than “the
fit got worse”: the two variables that create a real cluster improvement break
independent galaxies, while the variable that preserves galaxies collapses to
the old constant-exponent behavior.

A further exponent law should not be tried merely by changing the sigmoid or
adding another power. It would need a new invariant that contains independent
information and a reason from a covariant field equation for why that invariant
changes propagation. Any such proposal should again be fit on one domain and
transferred unchanged to the other before raw lensing.

Artifacts:

- `configs/unbounded_running_variable_exponent_protocol.json`
- `src/voidscreen/unbounded_running.py`
- `scripts/run_unbounded_running_full_test.py`
- `results/unbounded_running_variable_exponent/report.json`
- `results/unbounded_running_variable_exponent/bridge_predictions.csv`
- `results/unbounded_running_variable_exponent/sparc_predictions.csv`
- `results/unbounded_running_variable_exponent/raw_lensing_predictions.csv`
- `results/unbounded_running_variable_exponent/unbounded_running_variable_exponent.png`
