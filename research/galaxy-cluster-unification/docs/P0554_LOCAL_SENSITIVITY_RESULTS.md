# P0554 local sensitivity and compensated-interaction results

## Outcome

Small, frozen changes around the P0554 baryonic field law reveal three distinct
controls:

- radial accumulation shape (`alpha`) is the strongest local galaxy lever;
- baryonic transition-radius scaling is the strongest derived- and raw-cluster
  radial lever; and
- high-acceleration screening sharpness is overwhelmingly the strongest Solar
  System lever.

A separate exact-refit experiment found one useful interaction. Adding a small
concentration/extent response and moving the acceleration screen 20% lower
recovered the missing MACS1931 held-out image root and improved the four
already-complete raw systems by 3.23%. It remained Solar safe. It is not a
solution: its five-cluster RMS is 16.383 arcsec, the historical validation
subset is still worse than the limited compact-halo comparator, and its SPARC
error is worse than the P0554 parent and fixed RAR.

No formula is promoted from either experiment.

## Law being perturbed

For an observed baryonic profile, define

$$
\Phi_b(r)=\int_r^{r_{\max}}g_b(s)\,ds+g_b(r_{\max})r_{\max},
\qquad
\chi=\frac{\Phi_b}{c^2},
\qquad
\ell_\Phi=\frac{\Phi_b}{g_b}.
$$

The generalized P0554 response can be written compactly as

$$
r_s=\sqrt{\frac{GM_b}{a_0}}
\left(\frac{M_b}{10^{10}M_\odot}\right)^\delta,
\qquad x=\frac{r}{r_s},
$$

$$
C(x)=\frac{x^\alpha}{1+(x/\zeta)^\alpha},
\qquad
S(g_b)=\frac{1}{1+[g_b/(a_0s)]^n},
$$

$$
I(r)=
\left[1+\left(\frac{\chi}{\chi_t}\right)^p\right]
\left(\frac{\ell_\Phi}{r}\right)^\beta,
\qquad
\Delta=q\,E^\epsilon C S I,
$$

$$
g_{\rm dyn}=g_b(1+\Delta),
\qquad
g_{\rm lens}=g_b(1+m_\gamma\Delta).
$$

Here `E` is a bounded measure derived from baryonic concentration. In plain
language, `C` describes how an extra baryonic field contribution accumulates
with distance and eventually saturates; `S` turns that contribution off in
high-acceleration environments; and `I` asks whether the whole baryonic
potential and its effective path length alter the response. The optional
`m_gamma` multiplies only the new lensing channel. It is a diagnostic slip, not
a derived photon theory.

The P0554 parent uses

| Quantity | Value | Meaning |
|---|---:|---|
| $q$ | 1.23007 | universal extra-channel amplitude |
| $\alpha$ | 0.75 | radial accumulation shape |
| $\zeta$ | 100 | saturation/apogee radius in units of $r_s$ |
| $n$ | 1.0 | high-acceleration screen sharpness |
| $s$ | 1.0 | screen location relative to $a_0$ |
| $\delta$ | 0 | extra baryonic-mass dependence of $r_s$ |
| $\epsilon$ | 0 | concentration leakage into scalar amplitude |
| $p$ | 1.2 | potential-depth response power |
| $\chi_t$ | $2.0\times10^{-6}$ | potential-depth transition |
| $\beta$ | 0.25 | potential path-ratio power |
| $m_\gamma$ | 1.75 | new-channel lensing/dynamics ratio |

## Frozen local experiment

The first protocol was frozen before any new score. It evaluated the parent
plus low/high perturbations of 11 parameters: 23 formulas total. The universal
amplitude and all raw-lens geometries stayed fixed except in the explicit
amplitude pair. Coverage was 131 SPARC galaxies and 968 outer points, 20 CLASH
systems and 84 derived acceleration points, five raw clusters and 18 held-out
images, plus the same analytic Solar proxies used for P0554.

The parent scores were 12.571 km/s on SPARC, 0.1991 dex on CLASH, 17/18 raw
image roots, and -1.730 mas/century supplementary Mercury precession.

### Impact ranking

The table reports the low-to-high span divided by the relevant parent error or,
for Mercury, its 3.1 mas/century margin. Raw RMS spans use only systems with all
roots in both variants.

| Domain | Largest local lever | Normalized span | Raw interpretation |
|---|---|---:|---|
| SPARC rotation | $\alpha$ | 0.162 | 2.042 km/s span |
| CLASH derived radial lensing | $\delta$ | 0.314 | 0.0625 dex span |
| RX J2129 raw lensing | $\epsilon$ | 1.367 | 1.702 arcsec span |
| Other four raw clusters | $\delta$ | 0.242 | 5.187 arcsec on only two mutually complete systems |
| Solar System | $n$ | 24.073 | 74.63 mas/century Mercury span |

Three parameter pairs moved materially in a favorable common direction across
at least two domains: extent leakage, screen location, and potential-depth
power. This was only a selection clue for the predeclared interaction stage.

Raw topology itself was highly sensitive: the formulas produced between 13
and 18 of the same 18 held-out image roots. A missing root was never converted
to a finite RMS or counted as an accuracy improvement.

### Solar boundaries

Only three local variants failed the analytic Solar gates:

| Variant | Mercury prediction | Result |
|---|---:|---|
| $\alpha=0.675$ | -3.510 mas/century | fails the 3.1 margin |
| $n=0.8$ | -74.666 mas/century | fails strongly |
| $\delta=+0.05$ | -4.102 mas/century | fails the 3.1 margin |

This makes the screen exponent a boundary condition, not a useful knob to tune
galaxy or cluster accuracy freely.

## Exact geometry-refit experiment

Twelve interactions were frozen after the local screen. No gravity coefficient
was fit. For every formula and cluster, six ordinary lens-geometry nuisances
were refit with eight starts, source positions were reprofiled, and exact
held-out image roots were solved.

Making the radial law shallower ($\alpha=0.675$) failed Mercury. Pairing it with
the predeclared sharper screen ($n=1.2$) reduced the Solar prediction from
-3.510 to -0.080 mas/century, proving that a clean scale separation can restore
Solar safety. It did not restore galaxy accuracy or solve raw lensing.

The Solar-safe formulas that recovered every held-out root were:

| Formula change | SPARC RMSE | CLASH RMSE | Five-cluster raw RMS | Roots |
|---|---:|---:|---:|---:|
| $\epsilon=0.05$, $s=0.8$ | 13.019 km/s | 0.1936 dex | 16.383 arcsec | 18/18 |
| $p=1.08$ | 12.773 km/s | 0.1874 dex | 17.036 arcsec | 18/18 |
| $s=0.8$ | 12.327 km/s | 0.1953 dex | 17.565 arcsec | 18/18 |
| $\epsilon=0.05$, $\zeta=80$ | 13.198 km/s | 0.1954 dex | 17.593 arcsec | 18/18 |

For the leading `extent_screen_scale` interaction:

- RX J2129 improved from 1.256 to 1.182 arcsec;
- the three non-MACS1931 systems complete for both parent and candidate
  improved from 21.391 to 20.701 arcsec;
- MACS1931 recovered its fourth root and then scored 7.412 arcsec;
- on all four parent-complete clusters plus RX J2129, the matched improvement
  was 3.23%; and
- its 16.383-arcsec all-five score is descriptive, not a direct parent ratio,
  because the parent has no finite all-five score.

On the historical MACS1115+1931 validation pair, only MACS1115 is directly
matched to the incomplete parent and it worsens by 0.026%. Including the
recovered MACS1931 root gives 18.188 arcsec, still well above the limited
compact-halo comparator's 9.989 arcsec. It also raises SPARC error by 3.57%
relative to the P0554 parent and remains 25.8% above fixed RAR.

## What the data support

1. **The extra channel needs a sharp high-acceleration shutoff.** This is the
   only tested way to vary the low-acceleration radial law substantially while
   retaining the analytic Solar limits.
2. **Potential and transition-radius terms control radial cluster amplitude.**
   They are much more consequential there than extending the nominal apogee.
3. **A weak environmental/extent term controls image topology.** A change of
   only 0.05 in its exponent can create or remove strong-lensing roots while
   barely moving the one-dimensional population scores.
4. **Galaxy and cluster objectives still pull apart.** The leading raw-lensing
   interaction improves cluster behavior by reducing predicted acceleration;
   its SPARC residuals become more negative, especially in bulge- and
   early-type bins.
5. **A pure lensing multiplier is not a free rescue.** It leaves galaxy
   dynamics unchanged but destabilizes roots and did not produce a universal
   raw-lensing improvement.
6. **Scalar radial accuracy is not spatial lens accuracy.** The low-potential-
   power variant has the best CLASH score in the exact-refit set but is not the
   best raw-image formula.

## What remains unknown

The CLASH radial target is inferred through conventional GR/NFW models, the
five raw clusters and all interaction choices are spent exploratory data, and
the Solar checks are weak-field proxies rather than a full ephemeris
likelihood. The RX J2129 component inventory also cannot yet support a strict
component-resolved field reconstruction: BCG decomposition systematics are
unquantified, ICL has not passed a PSF-aware reconstruction gate, gas has only
one accepted cumulative-mass anchor, and satellite membership/mass/3D profile
uncertainties are incomplete. Failed component gates were not treated as
measurements.

The next discriminating experiment should therefore freeze the three observed
roles—Solar screen, scalar radial response, and two-dimensional extent/tidal
response—in one field operator, then test new raw clusters with accepted
stellar, gas, ICL, and member maps. It should not add another scalar amplitude
coefficient to these five spent systems.

## Reproduction

```powershell
python scripts/run_p0554_local_cross_domain_sensitivity.py
python scripts/run_p0554_compensated_interactions.py
python -m pytest tests/test_p0554_local_cross_domain_sensitivity.py tests/test_p0554_compensated_interactions_results.py -q
```

Machine-readable outputs are in
`results/p0554_local_cross_domain_sensitivity/` and
`results/p0554_compensated_interactions/`.

## Multi-scale continuation

The one-pair rankings have now been checked with central perturbations at four
scales. Galaxy and derived CLASH sensitivities remain smooth, while raw-lens
topology changes under several 1--2% perturbations. The mass-radius exponent
also prefers opposite signs in RX J2129 and the other four raw clusters. See
[`P0554_MULTISCALE_ELASTICITY_RESULTS.md`](P0554_MULTISCALE_ELASTICITY_RESULTS.md)
for the 89-formula experiment and its stricter stable-sensitivity ranking.
