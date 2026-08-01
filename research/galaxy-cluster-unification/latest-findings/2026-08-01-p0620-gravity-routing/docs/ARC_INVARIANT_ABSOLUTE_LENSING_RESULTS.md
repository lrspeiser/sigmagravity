# Arc-invariant absolute-lensing results

## Main result

Absolute lensing changes the conclusion drawn from normalized cluster maps.
The original arc-apogee law reproduces some spatial morphology but is too weak:
on 84 CLASH points it underpredicts the inferred acceleration by a median factor
of 3.68 and scores 0.599 dex, compared with 0.508 dex for fixed RAR.

The most consequential small change is to let the response depend on the
baryonic potential depth and its effective path length. A universal
potential/path formula reduces CLASH error to 0.163 dex without treating light
differently from matter, although its galaxy error rises to 16.09 km/s. A
photon-channel multiplier produces a better galaxy/cluster compromise, but no
tested form passes the raw RX J2129 0.5-arcsec criterion.

These are exploratory response laws. They do not demonstrate a new field or
supplant dark matter, MOND, or the project's previous hybrid candidate.

## The field quantities

For each observed baryonic profile, define

$$
\Phi_b(r)=\int_r^{r_{\max}}g_b(s)\,ds+g_b(r_{\max})r_{\max},
$$

where the second term is the declared point-mass tail. Then define

$$
\chi=\frac{\Phi_b}{c^2},
\qquad
\ell_\Phi=\frac{\Phi_b}{g_b}.
$$

$\chi$ measures potential depth. $\ell_\Phi$ is the distance over which the
local field would have to remain constant to accumulate that potential. The
ratio $\ell_\Phi/r$ equals one for an isolated point mass but differs from one
inside an extended mass distribution. That makes it a direct test of the idea
that field-line residence depends on the matter the route passes through, not
only on distance from a center.

The refined dynamical law is

$$
\frac{g_{\rm dyn}}{g_b}
=1+q\,
\frac{x^{0.75}}{1+(x/100)^{0.75}}
\frac{1}{1+g_b/a_0}
\left(\frac{\ell_\Phi}{r}\right)^\beta
\left[1+\left(\frac{\chi}{\chi_t}\right)^p\right],
\qquad
x=\frac{r}{\sqrt{GM_b/a_0}}.
$$

One optional gravitational-slip diagnostic was also tested:

$$
\frac{g_{\rm lens}}{g_b}
=1+m_\gamma\left(\frac{g_{\rm dyn}}{g_b}-1\right).
$$

$m_\gamma=1$ is zero slip. Values above one mean that photons respond more
strongly to only the new channel; ordinary baryonic GR remains unchanged. This
is a phenomenological test, not a derived photon coupling.

## Data and validation structure

| Stage | Coverage | Protection against object tuning |
|---|---:|---|
| Coarse micro-sweep | 58 one-at-a-time laws | One $q$ fitted only to galaxy inner radii |
| Fine potential/path sweep | 576 combinations | Five held-out-galaxy folds; no cluster amplitude fit |
| SPARC | 131 galaxies, 968 outer points | Same universal law and fixed RAR-derived nuisances |
| CLASH | 20 clusters, 84 radial points | Direct forward score, no per-cluster parameters |
| Solar System | Every law | Earth, Mercury, and photon-channel Cassini proxies |
| Raw RX J2129 | Four candidates selected before their raw score | 15 training images, seven held out, 16 geometry starts each |

CLASH's equivalent accelerations are deprojected from conventional NFW lens
profiles. They are valuable for population-wide parameter response but are not
theory-neutral. RX J2129 uses observed multiple-image positions and
spectroscopic source redshifts, although its baryonic radial profile still has
a literature-based spherical closure.

## Coarse parameter impacts

The initial 58-law sweep ranked the controls by their span in absolute CLASH
RMSE:

| Change | CLASH span (dex) | Galaxy span (km/s) | Interpretation |
|---|---:|---:|---|
| Potential depth | 0.270 | 20.20 | Largest scalar bridge and largest tradeoff |
| Photon multiplier | 0.215 | 0.00 | Pure lensing-amplitude lever |
| Potential length | 0.206 | 12.49 | Extended profiles matter |
| Mass-radius exponent | 0.157 | 2.93 | Mass scaling moves clusters but not enough |
| Potential path ratio | 0.141 | 6.27 | Improves both domains at moderate power |
| Enclosed-mass growth | 0.068 | 8.79 | Smaller and less clean effect |
| Accumulation exponent $\alpha$ | 0.0165 | 1.40 | Minor near the selected boundary |
| Screen acceleration scale | 0.0133 | 2.98 | Mostly a galaxy/Solar control |
| Concentration leakage into amplitude | 0.0115 | 0.31 | Nearly irrelevant at small leakage |
| Screen exponent | 0.0052 | 0.87 | Minor after enforcing the Solar-safe region |
| Apogee ratio | 0.0036 | 0.26 | Effectively saturated; not the missing physics |

This is the clearest universal result from this stage: once the basic
low-acceleration and Solar limits are in place, changing the old transition
parameters barely affects absolute cluster lensing. Quantities derived from the
whole baryonic potential affect it by an order of magnitude more.

## Fine-grid outcomes

| Candidate | $p$ | $\chi_t$ | $\beta$ | $m_\gamma$ | $q$ | Galaxy RMSE | CLASH RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0070, best CLASH under 1.5x-RAR galaxy limit | 0.5 | $1.5\times10^{-6}$ | 0.60 | 2.25 | 0.863 | 15.501 km/s | 0.111 dex |
| P0420, best zero-slip under that limit | 1.0 | $1.25\times10^{-6}$ | 0.50 | 1.00 | 1.008 | 15.062 km/s | 0.167 dex |
| P0554, best galaxy score below 0.2-dex CLASH | 1.2 | $2.0\times10^{-6}$ | 0.25 | 1.75 | 1.230 | 12.592 km/s | 0.199 dex |
| P0396, best zero-slip CLASH overall | 1.0 | $1.0\times10^{-6}$ | 0.50 | 1.00 | 0.955 | 16.094 km/s | 0.163 dex |

Reference values are 10.348 km/s for fixed RAR on galaxies and 0.508 dex for
fixed RAR on CLASH. Thus:

- P0070 lowers the CLASH error by 78.1%, but its galaxy error is 49.8% above
  RAR.
- P0554 lowers the CLASH error by 60.8% while remaining 21.7% above RAR on
  galaxies.
- P0396 proves that the potential/path response can supply most of the cluster
  change without a special photon coupling, but with a larger galaxy cost.

All selected candidates pass the specified Solar proxies. Their five galaxy
fold values of $q$ also remain tight; for P0554 they span 1.2244--1.2427. The
tradeoff therefore is not caused by unstable per-fold amplitudes.

Within the fine grid, potential exponent $p$ remains the largest cluster
control, followed by $m_\gamma$, path power $\beta$, and potential scale
$\chi_t$. The best median values differ by observable: galaxies favor larger
$p$ and smaller $\beta$, whereas clusters favor $p\simeq0.6$ and
$\beta\simeq0.5$. That tension is a real target for the next formula, not a
coefficient to average away.

## Raw RX J2129 result

The four formulas were selected using SPARC and CLASH before their raw image
positions were scored. Each used the same six structural lens nuisances and 16
optimization starts.

| Candidate | Held-out roots | Held-out RMS | Geometry at bound? |
|---|---:|---:|---|
| P0554 | 7/7 | **1.245 arcsec** | no |
| P0396 | 7/7 | 1.306 arcsec | no |
| P0070 | 7/7 | 1.324 arcsec | no |
| P0420 | 6/7 | no finite score | no |

For context, the previous locked project candidate scored 1.064 arcsec, the
compact one-halo control scored 2.536 arcsec, and the frozen absolute gate is
0.5 arcsec. P0554 is about 51% lower than the compact-halo control but 17%
higher than the previous candidate. None advances.

The key lesson is that matching a radial CLASH acceleration profile is not
enough to reproduce a strong-lensing image configuration. P0070 has an
excellent 0.111-dex CLASH score with almost zero mean bias, yet its raw score is
worse than P0554's. The missing quantity is therefore increasingly likely to
be spatial tensor structure, substructure, or a derived lens geometry rather
than another scalar amplitude correction.

## Galaxy morphology of the best compromise

P0554's full-$q$ outer errors remain above RAR in every tested category:

| Subsample | P0554 RMSE | RAR RMSE | Ratio |
|---|---:|---:|---:|
| Late type | 9.467 | 8.132 | 1.164 |
| Disk dominated | 11.621 | 9.942 | 1.169 |
| Giant mass | 15.254 | 12.808 | 1.191 |
| Intermediate mass | 11.907 | 9.828 | 1.212 |
| Gas poor | 12.140 | 9.628 | 1.261 |
| Bulge dominated | 15.745 | 12.306 | 1.279 |
| Early type | 13.791 | 10.726 | 1.286 |
| Gas rich | 11.222 | 8.625 | 1.301 |
| Dwarf mass | 9.350 | 6.315 | 1.481 |

Potential dependence improves the earlier giant/gas-poor overprediction, but
now produces its largest relative deficit in dwarfs. This suggests that the
next useful invariant should distinguish potential generated locally from
potential supplied by an extended external environment.

## What should be retained

1. **Retain the potential/path variables.** They are much more influential for
   cluster amplitude than further tuning of $\alpha$, apogee, or Solar screen.
2. **Do not promote the photon multiplier.** It helps population radial scores
   but provides no raw-lensing advantage over the zero-slip P0396 candidate.
3. **Keep concentration out of scalar strength.** Small leakage is nearly
   irrelevant here and the earlier broad sweep showed that large leakage harms
   galaxies.
4. **Stop optimizing scalar cluster amplitude on these spent data.** The raw
   test says the next discriminating change must affect two-dimensional/tensor
   structure.
5. **Require a field equation next.** A credible continuation should derive
   $\Phi$, $\ell_\Phi$, dynamics, and gravitational slip from one operator,
   rather than multiplying four empirical response terms indefinitely.

Reproduce the stages with:

```powershell
python scripts/run_arc_invariant_absolute_lensing.py
python scripts/run_arc_invariant_pareto_refinement.py
```

Machine-readable outputs are in
`results/arc_invariant_absolute_lensing/` and
`results/arc_invariant_pareto_refinement/`.
