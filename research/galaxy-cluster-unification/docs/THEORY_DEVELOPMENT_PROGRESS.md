# Theory-development progress against the stage gates

## 2026-08-03 Sigma v1 pure-geometry cycle

The renewed action-first goal has now tested the smallest one-metric,
baryon-only symmetric-teleparallel action.  The nonlinear nonmetricity action
passes its invariant, deep-field, high-field, parameter-count, and external
dwarf-galaxy gates.  Its regular isolated weak-field equations prove
`Phi=Psi` and reduce exactly to standard-mu AQUAL.  It therefore inherits the
frozen AQUAL raw-lensing result: `0.333` root convergence in both ready
clusters and incorrect held-out topology.  The action is retired without a
fit.  See [`SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md`](SIGMA_V1_NONMETRICITY_ACTION_RESULTS.md).

This closes the pure one-invariant geometric route.  Any next action must add
a baryon-predictable vector/tensor or causal nonlocal state that supplies
anisotropic stress; another scalar interpolation of the same invariant is not
a materially new cycle.

Status: active, updated 2026-07-29. The thresholds were recorded before H7a or
H7s was scored. Neither candidate is advanced by changing a bound after the
result.

The later unbounded curvature-running cycle is documented separately in
`docs/UNBOUNDED_CURVATURE_RUNNING_RESULTS.md`. Its best balanced setting passes
the Solar-System screening proxy and the broad BCG/cluster gates, and improves
on the fitted NFW galaxy reference, but it remains 39% worse than fixed RAR on
untouched SPARC outskirts and has unacceptable raw-lensing chi-square. It is
retained as a phenomenological control, not advanced as a theory survivor.
The subsequent locked multi-cluster raw-image transfer also fails: the two
controls score 18.2--18.6 arcsec equal-system held-out RMS on four raw coordinate
likelihoods, compared with 9.05 arcsec for an inadequate compact halo. A
post-failure per-cluster amplitude grid is non-universal and does not pass the
rescue gate. The first spatial-vector target is now complete as well. A frozen
mass-conserving redistribution of the same baryonic monopole into 63--120
observed member-light directions per cluster gives 18.2--18.7 arcsec and makes
every predictive score slightly worse. A post-failure all-root-converged oracle
over all 148 universal grid settings improves the best parent by only 1.6% and
still has more than twice the compact-halo error. Member light alone is not the
missing source variable. A common-200-kpc aperture correction also fails, with
18.210 arcsec for its best all-root setting versus 18.165 arcsec for the parent.
A further cluster test requires complete gas, BCG, ICL, and satellite
surface-density maps rather than a covariant completion of either failed scalar
closure.

## Current decision

The frozen measured/profile-constrained Stage 4 cycle supports the baryonic-
potential environment variable, but none of the attempted local closures survives.
H7s remains on its Stage 3 hard bound. The subsequent five-parameter EA-Q0
action passes its conservation, unit-vector, mode-speed, and quasistatic checks,
but fails before fitting: the reciprocal Aether source changes the dynamical
environment field by orders of magnitude more than the allowed 5%. EA-Q0 is
retired. The five-parameter EMOG-Q0 control then passes its local field-health,
conservation, short-distance, large-distance, and one-metric lensing derivations,
but its monotone chameleon response and one universal Yukawa range fail the
joint 5% structural target. EMOG-Q0 is also retired. The declared next stage is
a premise-level rethink, not another interpolation term.

## Gate scoreboard

| Stage/outcome | Concrete threshold | Result | Decision |
|---|---:|---:|---|
| SPARC inverse usable | at least 70% | 93.84% | pass |
| CLASH inverse usable | at least 70% | 100% | pass |
| Analytic inverse round trip | max relative error $\le10^{-10}$ | $4.45\times10^{-16}$ | pass |
| Central galaxy/cluster $\log_{10}\chi$ overlap | diagnostic | 0 dex; 0.462-dex central gap | warning |
| H7a SPARC $\chi^2/N$ | $\le9.41784$ | 9.138 | pass |
| H7a raw CLASH $\chi^2/N$ | $\le5.00$ | 4.809 | pass |
| H7a macro $\chi^2/N$ | $\le7.20$ | 6.974 | pass |
| H7a parameters off bounds | all five folds | width at lower bound in 4 folds | **fail** |
| H7s SPARC $\chi^2/N$ | $\le9.41784$ | 9.220 | pass |
| H7s raw CLASH $\chi^2/N$ | $\le5.00$ | 4.114 | pass |
| H7s CLASH with scatter | $\le2.50$ | 2.286 | pass |
| H7s CLASH RMS | $\le0.160$ dex | 0.156 dex | pass |
| H7s macro $\chi^2/N$ | $\le7.20$ | 6.667 | pass |
| H7s parameters off bounds | all five folds | $F=100$ in 3 folds | **fail** |
| Original 50 BCG eRASS gas-scale coverage | at least 30 systems and 80% | 1 system; 9.1% of public-footprint subset | **fail** |
| SPIDERS-MaNGA bridge systems | at least 30 | 34 unique hosts | pass |
| SPIDERS host-scale coverage | at least 80% | 100% | pass |
| Disjoint JAM proxy calibration | diagnostic RMS | 0.093 dex $g_{\rm obs}$; 0.098 dex $g_{\rm bar}$ | usable |
| Local-only H7s BCG $\chi^2/N$ | $\le5.0$ | 5.218 | **fail** |
| Local-only H7s BCG mean residual | $|\bar\Delta|\le0.15$ dex | -0.227 dex | **fail** |
| eRASS-median gas host $\chi^2/N$ | $\le5.0$ | 3.982 | pass |
| eRASS-median gas host mean residual | $|\bar\Delta|\le0.15$ dex | -0.184 dex | **fail** |
| Cosmic-baryon host $\chi^2/N$ | $\le5.0$ | 2.814 | pass |
| Cosmic-baryon host RMS | $\le0.17$ dex | 0.168 dex | pass |
| Cosmic-baryon host mean residual | science: $|\bar\Delta|\le0.10$ dex | -0.135 dex | **fail** |
| Frozen profile-constrained systems | at least 30 | 34/34 | pass |
| Independent satellite-catalog union | at least 30 | 30/34 | pass |
| Measured/profile host $\chi^2/N$ | science: $\le3.0$ | 1.658 | pass |
| Measured/profile host RMS | science: $\le0.17$ dex | 0.132 dex | pass |
| Measured/profile host mean residual | science: $|\bar\Delta|\le0.10$ dex | -0.083 dex | pass |
| Stage 4 uncertainty pass probability | continue $\ge0.80$; science $\ge0.50$ | 1.000; 1.000 | pass |
| EA-Q0 global parameters | at most 5 | 5 | pass |
| EA-Q0 tensor speed | $|c_T/c-1|\le10^{-15}$ | $c_{13}=0$, $c_T=c$ | pass |
| EA-Q0 deep/high-field limits | 5%; $10^{-5}$ | $2.50\times10^{-4}$; $5.00\times10^{-11}$ | pass |
| EA-Q0 BCG $Q$ reproduction | fractional change $\le0.05$ for all 34 | minimum lower bound 216 | **fail** |
| EA-Q0 allowed/required response | diagnostic | required $\eta$ is 84,469 times allowed | **fail** |
| EMOG-Q0 global parameters | at most 5 | 5 | pass |
| EMOG-Q0 local mode health | positive kinetic/gradient; $c_T=c$ | all principal speeds $c$ | pass |
| EMOG-Q0 universal-vector Solar bound | $|\gamma_{\rm eff}-1|\le2.3\times10^{-5}$ | $\alpha\le1.15\times10^{-5}$; favorable target envelope needs $\sim0.98$ | **fail** |
| EMOG-Q0 $1/r$ radial support | broad observed support | 0.055 dex for slope $-1\pm0.05$ | **fail** |
| EMOG-Q0 CLASH environment ordering | pointwise error $\le0.05$ | analytic lower bound 0.590 | **fail** |
| EMOG-Q0 joint force target | all SPARC/CLASH/BCG points within 5% | 0 points pass in favorable envelope | **fail** |

H7s numerically clears the stronger science-score targets, but the bound failure
has priority. Widening $F$ now would turn the test into post-hoc parameter
search. Its paired macro improvement over U0 also does not exclude zero: the
95% interval is -1.280 to +0.293 per point.

## What Stage 1 established

The pointwise reconstruction retained every row and marked
$g_{\rm obs}\le g_{\rm bar}$ points as unavailable for analytic inversion.
Across valid points, the median RAR-equivalent scales are

$$
\log_{10}a_{\rm eff}=-9.922\quad\text{(SPARC)},
$$

$$
\log_{10}a_{\rm eff}=-8.722\quad\text{(CLASH)}.
$$

The broad frozen U0 transition contains 31 SPARC systems and all 20 CLASH
systems, so the declared count gate passes. However, the central 10--90%
potential supports do not overlap. The CLASH 10th percentile lies 0.462 dex
above the SPARC 90th percentile. This makes the intermediate BCG regime a
required identification test rather than an optional external check.

Artifacts:

- `results/constitutive_target/report.json`
- `results/constitutive_target/constitutive_target.png`
- `scripts/reconstruct_constitutive_target.py`

## What the two action-derived cycles established

H7a and H7s use the same three global parameters, no per-object force term, and
no lensing-only multiplier. They differ only in the constitutive derivative of
the weak-field action:

$$
\mu_a(x)=\frac{x}{1+x},
\qquad
\mu_s(x)=\frac{x}{\sqrt{1+x^2}}.
$$

Both preserve SPARC while improving CLASH over U0. This is evidence that an
action-derived nonlinear Poisson limit can reproduce the useful part of U0; it
is not evidence that the potential transition or a void origin has been
established. The repeated boundary behavior and separated sample supports are
the structural warning.

Artifacts:

- `docs/H7A_WEAK_FIELD_DERIVATION.md`
- `results/h7a_cv/report.json`
- `results/h7s_cv/report.json`
- `scripts/cross_validate_h7a.py`

## Independent BCG bridge and host-scale result

The direct eRASS1 optical-BCG cross-match yields only 25 unique MaNGA BCGs, so
it cannot satisfy the frozen count gate. The declared sample rethink combines
three official products without using an acceleration value to select a system:

- GEMA-VAC identifies the brightest galaxy in each MaNGA group.
- SPIDERS supplies spectroscopically confirmed X-ray hosts, luminosities, and
  $R_{200}$ estimates.
- MaNGA DynPop supplies quality-assessed mass-follows-light JAM dynamics.

The frozen 200-kpc and $|\Delta z|\le0.01(1+z)$ match gives 34 unique hosts.
Eleven have direct Tian et al. accelerations. For the other 23, the DynPop/NSA
acceleration proxy is calibrated on 33 disjoint Tian systems; no test system
calibrates itself. The calibration RMS is 0.093 dex for $g_{\rm obs}$ and 0.098
dex for $g_{\rm bar}$, and those scatters enter the proxy uncertainties.

The frozen full-development H7s fit remains at $F=100$, so this cannot repair
its Stage 3 identifiability failure. It nevertheless supplies a useful external
diagnostic. With only BCG baryonic potential it gives $\chi^2/N=5.218$, RMS
0.253 dex, and mean residual -0.227 dex. The direct and proxy subsets have
nearly identical mean residuals, arguing against the proxy calibration being
the source of the missing acceleration.

The first host completion contains no fitted BCG parameter:

$$
M_{200}=\frac{4\pi}{3}200\rho_c(z)R_{200}^3,
\qquad
\chi_{\rm host}=\frac{Gf_bM_{200}}{R_{200}c^2},
$$

with $f_b=\Omega_b/\Omega_m$ from Planck18. This optimistic retained-cosmic-
baryon scale improves the score to 2.814, the RMS to 0.168 dex, and the mean
residual to -0.135 dex. It passes the continue gate but misses the 0.10-dex
scientific bias limit.

That upper-bound result is not enough. Across 10,440 eRASS1 systems the catalog
$f_{\rm gas,500}$ median is 0.064 and the 90th percentile is 0.098, versus a
larger cosmic baryon fraction. Applying those independent gas fractions to the
same SPIDERS potential scale gives:

| Host input | $\chi^2/N$ | RMS (dex) | Mean residual (dex) | Continue? |
|---|---:|---:|---:|---|
| none | 5.218 | 0.253 | -0.227 | no |
| eRASS median gas fraction | 3.982 | 0.210 | -0.184 | no |
| eRASS 90th-percentile gas fraction | 3.493 | 0.193 | -0.165 | no |
| retained cosmic baryon fraction | 2.814 | 0.168 | -0.135 | yes, not science success |

Thus measured hot gas at the catalog scale helps but is insufficient by itself.
The remaining physically allowed terms are the host's stellar/satellite baryons
and the central weighting of the gas profile. Neither may be normalized using
the BCG residual.

Artifacts:

- `configs/bcg_bridge_sample.json`
- `scripts/build_and_test_bcg_bridge.py`
- `data/derived/bcg_bridge_sample.csv`
- `results/bcg_bridge_sample/report.json`
- `results/bcg_bridge_sample/predictions.csv`

All source hashes and selection rules are recorded in the report. Large FITS
files remain local and reproducible through
`scripts/download_bcg_environment_catalogs.ps1`.

## Completed measured/profile-constrained host cycle

The archival audit found direct eRASS or pointed Chandra/XMM coverage for only
10 of the 34 frozen hosts. It would therefore be inaccurate to call this a
30-host directly measured X-ray-profile test. The preregistered alternative is
the profile-constrained route:

- each host retains its measured SPIDERS $R_{200}$ and derived halo scale;
- a 10,439-system eRASS calibration fixes the gas-mass relation and its scatter;
- 46 published Chandra density profiles fix the gas radial-shape population;
- each BCG's NSA Sersic profile supplies the stellar exterior-shell correction;
- a published 21-cluster satellite-mass relation and redMaPPer radial scale fix
  the satellite population; and
- SPIDERS/redMaPPer provide an independent member-catalog coverage check for
  30 of 34 systems, although those members are not normalized with BCG data.

For a spherical component truncated at $R$, the potential is integrated as

$$
\chi_b(r)=\frac{G}{c^2}\left[
\frac{M_b(<r)}{r}+\int_r^R\frac{dM_b(s)}{s}
\right].
$$

This corrects the point-scale approximation by including exterior shells and
the central weighting of extended gas and satellite profiles. It introduces no
host normalization. All 34 systems are scored with the unchanged H7s vector.

| Result | Point estimate | 5--95% Monte Carlo interval | Gate |
|---|---:|---:|---|
| $\chi^2/N$ | 1.658 | 1.493--1.818 | $\le3.0$ |
| RMS | 0.132 dex | 0.125--0.139 dex | $\le0.17$ dex |
| Mean residual | -0.083 dex | -0.087 to -0.070 dex | absolute value $\le0.10$ dex |

The direct Tian subset has $\chi^2/N=2.764$, RMS $=0.140$ dex, and mean
residual $=-0.100$ dex; the 23 calibrated proxy systems give 1.129, 0.128 dex,
and -0.074 dex. The agreement is not driven solely by the proxy subset. Every
one of the 5,000 uncertainty realizations passes both declared Stage 4 gates.

Artifacts:

- `configs/measured_host_profile_validation.json`
- `scripts/download_host_profile_catalog.py`
- `scripts/inventory_host_profile_coverage.py`
- `scripts/validate_measured_host_profiles.py`
- `data/derived/host_profile_coverage.csv`
- `data/derived/measured_host_profile_sample.csv`
- `results/measured_host_profiles/coverage_report.json`
- `results/measured_host_profiles/report.json`

## EA-Q0 derivation result

The selected local EA-Q0 action has one minimally coupled physical metric, a
unit timelike Aether, and a scalar-curvature environment field. Its five global
parameters are $\{\beta,L_Q,\eta,c_1,c_{14}\}$; $c_{13}=0$ exactly and no
per-object or lensing-only parameter is present. The scalar, Aether, constraint,
and metric equations were varied from the same action, and their diffeomorphism
Noether identity verifies on-shell stress-energy conservation.

The action produces the desired standard-$\mu$ static response and healthy
declared high-/low-field limits. It also necessarily produces the reciprocal
source

$$
(\Box-L_Q^{-2})Q=-\frac{R}{2}
-\frac{\eta a_Q^2}{2\beta}
\left(\mathcal H_s-Y\mathcal H_{s,Y}\right).
$$

The second term cannot be removed without abandoning the action. The frozen
spherical check maximizes $\beta$ under the PPN $\gamma$ gate, uses the shortest
field range consistent with 5% potential accuracy, omits all exterior baryons,
and continues only already enclosed mass. Even this lower bound changes $Q$ by
factors of 216--5,211 across the 34 BCGs. None passes the 5% gate. The required
environment response is 84,469 times stronger than the largest response that
keeps every BCG within the gate.

Artifacts:

- `configs/eaq0_derivation.json`
- `docs/EAQ0_DERIVATION.md`
- `src/voidscreen/eaq.py`
- `scripts/check_eaq0_derivation.py`
- `results/eaq0_derivation/report.json`
- `results/eaq0_derivation/feedback_points.csv`

## EMOG-Q0 result and premise-level rethink

The [environmental MOG control](ENVIRONMENTAL_MOG0_DERIVATION.md) freezes one
physical metric, a canonical chameleon scalar, a positive-energy Proca vector,
and one conserved composition-independent charge. Its five parameters are
$\{\beta,\Lambda_s,n,\mu,\alpha\}$. The full metric, scalar, vector, and matter
equations follow from the same action, and the diffeomorphism Noether identity
verifies their on-shell conservation. The regular $F>0$ field domain has
positive scalar and Proca kinetic terms, luminal principal characteristics, and
$c_T=c$. The same metric potential supplies the light-deflection prediction.

At a constant scalar background the massive-particle law is

$$
{g_{\rm dyn}\over g_{\rm bar}}={1\over F_0}
-\alpha(1+\mu r)e^{-\mu r}.
$$

The matched condition $F_0^{-1}=1+\alpha$ recovers Newtonian gravity at small
$r$ and enhanced attraction at large $r$. It does not solve the radial-shape
problem: the extra point-mass acceleration has slope $-1\pm0.05$ for only
$1.682<\mu r<1.908$, or 0.055 dex. Nor can the matching persist when the scalar
responds to environment, because $F(s)$ changes while the conserved vector
charge $\alpha$ is universal.

The action's adiabatic scalar response predicts $F^{-1}$ increasing as mean
baryonic density decreases. CLASH alone contains a lower-density point
requiring enhancement 4.217 and a higher-density point requiring 16.368. Any
response with the action's ordering must miss at least one by 59.0%, independent
of optimizer or power-law details. A deliberately favorable global envelope,
which lets the scalar follow its density minimum instantly, finds no point in
SPARC, CLASH, or the 34 BCGs within the 5% gate.

Artifacts:

- `configs/environmental_mog0_derivation.json`
- `docs/ENVIRONMENTAL_MOG0_DERIVATION.md`
- `src/voidscreen/mog.py`
- `scripts/check_environmental_mog0.py`
- `results/environmental_mog0/report.json`
- `results/environmental_mog0/feasibility_points.csv`

EMOG-Q0 is retired before fitting; no Stage 3 or Stage 4 configuration is
frozen. All three declared relativistic completion routes have now failed a
pre-fit structural or identifiability gate. Under the stopping rule, the next
cycle must revisit the one-field environmental-unification premise and the
meaning of the 5% pointwise target. Adding an interpolation term, making the
range object-dependent, or introducing a lensing-only amplitude is forbidden.
The concrete R0--R3 checkpoints are frozen in
[`PREMISE_LEVEL_RETHINK.md`](PREMISE_LEVEL_RETHINK.md); R0 begins with the raw
observable and covariance provenance behind the CLASH and BCG targets.

## Universal variable-exponent result

The curvature-running exponent was promoted from a constant to the bounded
universal function

$$
p(X)=p_0\exp[\beta\tanh(\ln(X/X_*))],
$$

using force-equivalent enclosed baryonic mass, local baryonic density, or the
local-to-mean density ratio as $X$. Five gravity constants were fit on the
system-held-out BCG+cluster bridge and transferred unchanged to 131 SPARC
galaxies. The mass version improved the bridge to 0.1158 dex but failed SPARC at
61.54 km/s; density scored 0.1241 dex and 38.87 km/s. The best transfer was the
distribution-shape version at 0.1396 dex and 15.26 km/s, but it saturated at
$p=5.0004$ throughout the bridge, effectively returning to a constant exponent.
The fixed $p=2$ control remains better at 0.1377 dex and 14.40 km/s and also
retains all RX J2129 image roots. No variable-exponent candidate advances. See
[`VARIABLE_EXPONENT_RESULTS.md`](VARIABLE_EXPONENT_RESULTS.md).

## Galaxy-locked metric-slip result

The photon/matter distinction was isolated without changing the galaxy force
law. Four smooth curvature matter laws were calibrated only on inner SPARC
radii and failed untouched outer radii at 82.81--88.38 km/s, versus 10.35 km/s
for fixed RAR. Fixed RAR was therefore locked as the matter potential before
any lensing data were used.

The weak-field split

$$
\Phi=\Phi_N+\phi,\qquad \Psi=\Phi_N+(1+s)\phi
$$

was then tested on raw cluster image positions. One shared, complete-root value
$s=5$ was selected on MACS0329 and MACS0429 and transferred unchanged to
MACS1115 and MACS1931. It lowers the unseen equal-system radial RMS from 25.67
to 18.43 arcsec, but the compact-halo control reaches 9.99 arcsec. The far-tail
control changes the result by only 0.51%, so the failure is not caused by the
RAR integration cutoff.

The same slip improves the secondary 20-system radial lensing score from 0.509
to 0.161 dex, showing that a light/matter amplitude difference is useful.
However, it cannot alter the fixed-RAR BCG dynamics, which remain at 0.299 dex
RMSE and only 55.3% of observed acceleration at the median. Raw-image failures
and inconsistent cluster preferences show that one scalar amplitude lacks the
required spatial structure. The candidate is retired; the next justified test
is a universal tidal-tensor slip evaluated with explicit member and gas maps.
See [`METRIC_SLIP_RESULTS.md`](METRIC_SLIP_RESULTS.md).

## Spherical spacetime and hard-cavity result

The proposed spherical-medium picture was separated into an exact closed
three-space Gauss law and an exact impermeable-sphere potential-flow analogy.
For a closed three-space the force enhancement is

$$
{g\over g_{\rm bar}}=\left[{r/L\over\sin(r/L)}\right]^2.
$$

Keeping the same geometry valid through the 3,000-kpc raw-lensing integral
forces $L$ above 1,005 kpc. The fit reaches its lower bound at 1,096 kpc and is
then too weak on galaxies: 72.39 km/s outer SPARC RMSE versus 10.35 km/s for
fixed RAR. A galaxy-only curvature radius fails at 88.74 km/s and reaches its
antipodal singularity before cluster scales. A screened local-curvature variant
improves BCG dynamics to 0.248 dex but catastrophically extrapolates to 177.97
km/s and becomes invalid on clusters.

For a hard spherical cavity the linear directional correction scales as
$(a/r)^3$, cancels in the isotropic average, and leaves an RMS correction of
order $(a/r)^6$. Even treating an entire disk scale as a perfectly hard cavity
gives only a 1.0027 median favorable-axis factor against a required 3.8476;
none of 960 outer points is reached. Real stellar covering fractions are below
$2.2\times10^{-11}$ in the generous upper-bound calculation.

The frozen post-failure raw transfer scores 25.15 arcsec on unseen cluster
images, indistinguishable from baryons at 25.20 and much worse than the compact
halo at 9.99. The result changes only 0.75% across 600--3,000-kpc cutoffs. The
literal global-sphere and hard-cavity candidates are retired. The remaining
direction is a sourced, conservative tensor constitutive law, not another
spherical amplitude. See
[`SPHERICAL_SPACETIME_CAVITY_RESULTS.md`](SPHERICAL_SPACETIME_CAVITY_RESULTS.md).
