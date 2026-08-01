# P0556: cross-domain parameter-impact atlas

## P0557 registered-baryon tensor follow-up

The missing-data audit found adequate registered HST and Chandra morphology
for a proxy test, but no spectroscopy-derived projected gas-mass map. The
prospectively frozen P0557 factorial tested 108 morphology/operator/coupling
combinations. Its selected 75%-starlight + 25%-square-root-X-ray contrast
tensor (`t=+0.3`) improved different-cluster held-out exact RMS by only 0.215%
and remained 1.841 times the compact-halo error. The small gain came from one
of two validation clusters. This preserves a weak non-circular baryon-shape
clue but does not advance the formula; see
`docs/P0557_BARYON_PROXY_TIDAL_RESULTS.md`.

P0558 then replaced the selected universal 75/25 morphology mixture with the
published central BCG/gas mass ratio in each cluster, without adding or fitting
a gravity parameter. The result worsened four-cluster held-out exact RMS by
1.52%. Half gas was the least-bad nonzero sensitivity and double gas lost an
image root. The weak P0557 signal is therefore not explained by the measured
global central baryon ratio; see `docs/P0558_MASS_ANCHOR_PROXY_TIDAL_RESULTS.md`.

## Purpose

This atlas answers a narrower and more useful question than “which formula
wins?”: which small formula changes actually move galaxy rotation, derived
cluster lensing, raw multiple-image positions, and Solar constraints, and do
those changes move the domains in compatible directions?

The entries combine frozen or explicitly post-result sensitivity studies.  The
data are exploratory and repeatedly used; the atlas ranks leverage and
conflict, not statistical discovery.

## Highest-leverage coordinates

| Physical coordinate | Galaxy effect | Derived-cluster effect | Raw-lens effect | Solar effect | Main lesson |
|---|---:|---:|---:|---:|---|
| high-acceleration screen exponent | small | moderate | moderate and topology-sensitive | **dominant** | Necessary safety control, not the missing cluster mechanism |
| mass dependence of transition radius | **large** | **largest scalar effect** | **large, often destructive** | can cross Mercury limit | Strongest cross-scale lever, but direction is not universal |
| radial residence exponent | **largest local galaxy span** | large | moderate | large | Galaxy and cluster/raw directions conflict |
| potential-depth transition scale | modest | **large** | moderate | negligible | Useful cluster lever; raw direction changes by system |
| concentration/extent leakage | small-to-moderate | modest | **largest RXJ2129/topology span** | negligible | Can recover a missing root but worsens galaxies and other lenses |
| photon-channel addition softness | none for matter | moderate | high local raw sensitivity | none | Strong lens-only lever; does not by itself explain galaxy motion |
| coherent-to-branched fraction | none by construction | none radially | monotonic degradation | none | Persistent source-to-target gravity tubes are disfavored |
| constant radial displacement | sub-percent | sub-percent | system-dependent, up to 10.7% | safe | Direction changes between systems; not universal |

## New 64-setting compensation test

The strongest unresolved local coordinates were crossed in a factorial grid:

\[
r_M(M)=r_{M,0}\left(\frac{M_b}{10^{10}M_\odot}\right)^{\delta_M},
\]

\[
E_{\rm extent}=E_{\rm concentration}^{\epsilon},
\]

\[
S(g_b)=\left[1+\left(\frac{g_b}{a_0}\right)^n\right]^{-1}.
\]

The grid used

- \(\delta_M=0,0.01,0.02,0.03\),
- \(\epsilon=0,0.02,0.04,0.06\),
- \(n=1,1.05,1.10,1.15\).

All 64 formulas used the same universal values for every object.  Selection
used 91 discovery galaxies, 13 derived-cluster systems, three raw discovery
clusters, and Solar gates.  Transfer used 40 galaxies, seven derived clusters,
and RXJ2129, MACS1931, and RXJ1347.

### Outcome

No non-parent formula improved all three discovery domains.  The least-bad
non-parent formula was

\[
(\delta_M,\epsilon,n)=(0,0.02,1),
\]

which changed the metrics as follows:

| Test | Gain versus exact parent |
|---|---:|
| discovery galaxies | -0.441% |
| discovery derived clusters | -0.133% |
| discovery raw lenses | -0.438% |
| formula-holdout galaxies | -1.224% |
| formula-holdout derived clusters | +0.865% |
| formula-holdout raw matched RMS | -4.120% |

It remained Solar-safe and recovered MACS1931's missing held-out root, showing
that concentration leakage genuinely changes lens topology.  It did not
improve positional accuracy or transfer.

## What the factorial effects say

### Mass scaling is the most consequential non-Solar coordinate

Across the grid, mass-radius scaling produced:

- a 10.0 percentage-point span in raw-discovery gain;
- a 5.05 percentage-point span in derived-cluster discovery gain;
- an enormous response in raw holdouts, where small nonzero values can worsen
  the matched RXJ2129+RXJ1347 score by several hundred percent.

At the otherwise unchanged parent, increasing \(\delta_M\) from 0 to 0.03:

- improves discovery galaxy RMSE by 2.94%;
- worsens discovery derived-cluster RMSE by 4.93%;
- worsens discovery raw-lens RMS by 10.58%;
- improves the separate derived-cluster holdout by 8.55%;
- worsens the matched raw holdout by more than a factor of ten in fractional
  gain units.

The sign reversal between the two CLASH partitions is especially important:
mass scaling is high leverage but not a stable universal correction.

### Concentration leakage is a topology lever

Positive extent leakage can recover MACS1931's missing root.  It simultaneously
worsens galaxy accuracy and has different positional effects among clusters.
This makes it a candidate descriptor of caustic topology, not a universal
gravity-amplitude term.

### The Solar screen does its intended job

The small grid remained Solar-safe.  The screen exponent dominated Mercury's
response, with a 2.13 mas/century main-effect span, and its interaction with
mass scaling added 0.70 mas/century.  Screening can protect the Solar System,
but it cannot reconcile the galaxy/raw-lens sign conflict.

### Interactions cannot hide the main conflict

For raw discovery, the mass-scaling main-effect span was 10.0 percentage
points, while the largest pair interaction was 1.40 points.  For derived
clusters the corresponding values were 5.05 and 1.16 points.  Compensation is
real but too small to overturn the main-effect disagreement.

## Universal truths supported by the combined data

1. **Scale-dependent transition placement is the strongest amplitude lever.**
   Mass and potential coordinates consistently move results more than route
   width, distance exponent, or radial-displacement bookkeeping.
2. **The most impactful parameters are also the least transferable.** The
   strong mass-scale response flips sign across cluster partitions and damages
   raw image roots.
3. **Raw lens topology is not summarized by a radial acceleration curve.** A
   parameter can improve a derived CLASH profile while losing a root or moving
   RXJ2129/RXJ1347 by several times their parent residual.
4. **Solar screening is comparatively easy.** Many weak-field changes can be
   hidden locally without solving the galaxy/cluster problem.
5. **Coherent local summation is more robust than explicit gravity tubes.**
   Branch preservation worsens monotonically on both discovery and formula
   holdout lens tests.
6. **Member-only anisotropy has already been exhausted.** A future tidal test
   must add registered gas and diffuse light rather than another coupling to
   the same member catalog.

## Data that would unlock the next distinct test

The most discriminating missing dataset is a common-frame baryonic surface
density map for a strong-lensing cluster containing:

- X-ray-derived hot-gas surface density and covariance;
- BCG and diffuse intracluster-light mass maps;
- member-galaxy stellar masses;
- spectroscopic multiple-image positions and redshifts.

That map would test whether the failed member-only tensor was missing the
dominant baryonic component rather than the wrong equation.  Until then, the
best use of existing data is to treat mass scaling and concentration leakage as
high-impact falsification coordinates, not candidate universal constants.

## Artifacts

- `configs/p0554_mass_extent_screen_factorial_protocol.json`
- `scripts/run_p0554_mass_extent_screen_factorial.py`
- `results/p0554_mass_extent_screen_factorial/`
- `results/p0554_local_cross_domain_sensitivity/`
- `results/p0554_route_softness_interaction/`
- `results/p0554_baryonic_network_screen/`
- `results/p0554_route_coherence_transition/`

## P0557-P0559 physical baryon-tensor follow-up

The proposed next distinct test was completed in three increasingly physical
steps. A prospectively selected star/X-ray morphology tensor (P0557) improved
its two-cluster transfer pair by only **0.215%**. Reweighting that morphology
with published central stellar/gas anchors (P0558) worsened the four-cluster
score by **1.52%**. Projecting measured ACCEPT electron-density shells into a
physical gas surface-density map (P0559) still worsened it by **1.41%**.

The P0559 primary changed individual scores by -3.46%, -2.39%, +0.02%, and
+0.73% for MACS0329, MACS0429, MACS1115, and MACS1931 respectively. Rescaling
the gas by factors of 4-14 to match an incompatible independent catalog changed
the aggregate by only 0.013 arcsec. This isolates a new robust lesson:
**baryonic tensor direction is more consequential than absolute gas
normalization, but the preferred direction is not coherent across clusters.**

The next useful sweep is diagnostic rather than a candidate-formula search:
map each system's preferred positive or negative tensor coupling. If the signs
differ, this geometry would require per-cluster field reversal and should be
retired. If the signs agree, freeze a common amplitude on a genuinely unspent
lens sample before making a universality claim.

Additional artifacts:

- `configs/p0557_baryon_proxy_tidal_protocol.json`
- `configs/p0558_mass_anchor_proxy_tidal_protocol.json`
- `configs/p0559_accept_projected_gas_tidal_protocol.json`
- `results/p0557_baryon_proxy_tidal/`
- `results/p0558_mass_anchor_proxy_tidal/`
- `results/p0559_accept_projected_gas_tidal/`

## P0560-P0561 coupling sign and range

A two-sided exact response scan initially appeared to improve the four-cluster
score by 2.74% at its `t=-1` boundary. The prospectively frozen extension to
`|t|<=6`, with twice as many optimizer starts, rejected that interpretation:
zero is best and the best common nonzero point is 2.83% worse. Only `t=0` and
`t=-1` keep every image root in all four systems, despite the operator
remaining positive-elliptic throughout the wider grid.

Individual clusters remain highly sensitive: their spent-grid optima range
from `t=-2` to `t=+4`, with a 40.2% single-system gain for MACS1115. The
near-zero sign divides 2-2, and leave-one-cluster-out choice never yields a
positive transfer to every excluded system. This sharpens the prior lesson:
**tensor direction is high leverage, but neither its sign nor its topology is
universal in the current construction.**

The exercise also exposed nuisance-fit basin sensitivity. At the same map and
`t=-1`, a lower MACS0429 training cost changed its held-out RMS from 14.29 to
19.24 arcsec. Percent-level raw-lens claims must therefore be compared against
a basin ensemble or deterministic global posterior, not one selected local
minimum.

Additional artifacts:

- `configs/p0560_accept_tensor_coupling_response_protocol.json`
- `configs/p0561_accept_tensor_extended_response_protocol.json`
- `results/p0560_accept_tensor_coupling_response/`
- `results/p0561_accept_tensor_extended_response/`

## P0562-P0563 direct response and critical-curve conditioning

Holding lens geometry fixed appeared to reveal an 86.8-87.8% common tensor
gain. A frozen conditioning audit showed that this was an inverse-Jacobian
artifact: critical-curve gain reaches 3,966x, the local/source residual ratio
reaches 207x, and their log correlation is 0.877.

Removing the Jacobian inversion and scoring unweighted source-plane closure
reduces the common response to **0.169-0.178%** at `t=+2.5` to `+2.75`. Both
geometry ensembles give the same per-cluster signs, but MACS0429 prefers
negative while the other three prefer positive; MACS0429 and MACS1115 also
select opposite grid boundaries individually.

This adds two robust methodological truths: **never screen near-critical raw
lenses with an unqualified inverse-Jacobian residual**, and require a proposed
effect to exceed both optimizer-basin variability and conditioning
amplification before interpreting it physically.

Additional artifacts:

- `configs/p0562_accept_tensor_direct_response_protocol.json`
- `configs/p0563_accept_tensor_source_plane_response_protocol.json`
- `results/p0562_accept_tensor_direct_response/`
- `results/p0563_accept_tensor_source_plane_response/`

## P0564-P0565 multiscale morphology gate

The lone negative-sign cluster, MACS0429, has unusually high inner star-gas
correlation (0.505 versus 0.080-0.151) but a nearly opposed outer star/gas
quadrupole (59.7 degrees misalignment versus 4.6-25.1 degrees). A two-threshold
rule based on those gaps was frozen before examining RX J2129's corresponding
descriptors or physical-tensor response.

RX J2129 triggered both negative conditions and both independent source-plane
geometry ensembles preferred negative coupling near zero. This is one
successful observable-to-direction transfer. Exact `t=-0.3` roots all remain,
but RMS is 1.20% worse in one ensemble and 0.48% better in the other, so the
gate is not validated.

The new candidate coordinate is therefore **radial change in star-gas
coherence**, not gas amplitude. Its leverage appears directional, while its
exact predictive magnitude remains below optimizer-basin variability.

Additional artifacts:

- `configs/p0564_baryon_morphology_sign_audit_protocol.json`
- `configs/p0565_rxj2129_morphology_gate_transfer_protocol.json`
- `results/p0564_baryon_morphology_sign_audit/`
- `results/p0565_rxj2129_morphology_gate_transfer/`

## P0566 morphology-gate replication

The frozen two-condition sign rule was transferred to A383 and MS2137. Both
were assigned positive coupling, but for different reasons. A383's unbiased
source-plane slope preferred negative coupling and its exact `t=+0.3` score
worsened by 0.46--0.63%. MS2137 could not solve the baryon-only exact roots at
either zero or `+0.3`; a compact fitted halo did solve them. The original AND
rule therefore failed its first expanded replication. A post-hoc outer
alignment-only sign coordinate labels all seven inspected systems correctly,
but has not earned prospective status.

This reinforces the distinction between direction and structural closure:
even a correct tensor sign cannot repair a missing radial field or image root.

## P0567-P0568 flux backtracking and forward compression

P0567 recast apparent dark-matter locations as a response-divergence map in

\[
\mathbf J_b=\mathbf K(\mathbf x)\mathbf g_{\rm lens},
\qquad \nabla\!\cdot\mathbf J_b\propto-\Sigma_b.
\]

Across ten fresh RELICS systems, 95.8% of convergence-weighted area admits a
local symmetric positive tensor. The minimum anisotropy is only 1.56:1 in the
median and 5.59:1 at the 90th percentile. Twenty-four of 25 apparent-dark
residual peaks backtrack to catalogued baryons, with median projected path
58.7 kpc. This proves geometric availability, not prediction.

P0568 then attempted the required forward compression with nine baryon-only
operators, 468 tensor candidates, 1,000 Lenstool uncertainty scores, GLAFIC
controls, 968 SPARC outer points, and Solar proxies. Only a tidal tensor
weakened in high-density regions selected nonzero coupling. The primary
`w=100 kpc, t=-0.15` setting improved its three heldout normalized maps by
8.01% versus the development-selected local-light null and 30.91% versus a
central Gaussian, but missed the 10% gate. Width/coupling refinement selected
`w=125 kpc, t=-0.30` on development and worsened the prior transfer score.

The parameter-impact result is much stronger than the candidate formula:

- changing local baryon smoothing from 20 to 100 kpc improved development JS
  by 67.5%; the primary tensor added only 2.08%;
- pure field, gradient, full-tidal, blended, and noncircular families all
  selected zero;
- large directional couplings have large JS spans but harmful minima;
- the refined tensor's system-score pattern is closest to an ordinary 125 kpc
  smoothed baryon map;
- SPARC remains the decisive failure: the selected and refined tensor proxies
  score 65.6 and 57.0 km/s, respectively, versus 10.35 km/s for fixed RAR;
- Solar screening suppresses both below (7\times10^{-13}), so Solar safety
  does not discriminate among these weak-field operators.

The current universal truth is therefore: **measured or phenomenological
baryonic extent matters far more than member-only tensor orientation, while
the angular cluster correction does not generate the radial galaxy force.**
The next distinct experiment should replace 75--125 kpc smoothing with
registered gas, BCG/ICL, and stellar-mass maps rather than add another
orientation parameter.

Additional artifacts:

- `docs/P0567_BARYON_FLUX_TENSOR_BACKTRACK_RESULTS.md`
- `docs/P0568_BARYON_ONLY_TENSOR_FORWARD_RESULTS.md`
- `results/p0567_baryon_flux_tensor_backtrack/`
- `results/p0568_baryon_only_tensor_forward/`
- `results/p0568b_tensor_width_refinement/`
- `results/p0568c_width_coupling_interaction/`

## P0569-P0570 measured extent and conservative map residual

The four registered CLASH stars-plus-gas maps have a median equivalent
Gaussian width of 153.3 kpc from RMS radius and 161.5 kpc from R80. All four
fall outside the phenomenological P0568 75--125 kpc band. Measured broad
baryonic extent therefore explains why a narrow member-light proxy was poor,
but not the exact width selected by the earlier map comparison.

P0570 used the measured stellar and ACCEPT gas maps directly. It subtracted
their exact circular potential and added only the conservative noncircular
deflection with one universal response `q`. Across 45 candidates, development
selected sqrt-morphology gas at 0.75 times measured extent and `q=2`.

That correction improves two held-out raw lenses by only 0.140%, versus a 5%
gate, and remains 1.844 times the compact-halo RMS. It also loses an exact
image root in MACS0329. Component, response, and extent change the development
screen by only 0.65%, 0.34%, and 0.15%, respectively. The potential solver
passes curl and circular-point-source null checks.

This isolates a useful negative result: **ordinary noncircular baryon-map
deflection, even doubled, is not the cluster closure. The high-impact
coordinate must alter radial/nonlocal organization rather than merely amplify
the measured angular residual.** The term remains galaxy- and Solar-null only
as an angular addition to a locked scalar parent; it is not an independent
no-MOND galaxy solution.

Additional artifacts:

- `docs/P0569_P0570_MEASURED_BARYON_EXTENT_AND_LENSING_RESULTS.md`
- `results/p0569_measured_baryon_extent_audit/`
- `results/p0570_physical_baryon_residual_lensing/`

## P0571-P0572 tidal-cancellation location and forward test

An eight-feature, 480-setting same-radius audit found that apparent-dark
residual peaks prefer high baryonic tidal balance. That individual feature
transferred to all three held-out systems and GLAFIC but failed the best-of-480
search control (`p=0.249`) and lacked an isolated-source null.

The predeclared null-safe refinement

\[
A=\sqrt{1-C}\,B_T
\]

combines member-field vector cancellation with tidal balance. It achieved a
0.275 development centered-rank effect, 0.224 on the earlier validation
systems, 0.410 on three separate pilot systems, and 0.171 under GLAFIC. The
nine-formula search-aware probability was 0.00389. It is exactly zero for a
centered point source and net axisymmetric component fields.

The first forward construction nevertheless failed. Development chose a fully
routed, field-weighted, 100-kpc arrival map; it worsened every held-out cluster,
all 300 held-out posterior realizations, and the GLAFIC aggregate. Arrival
smoothing was the largest coordinate (10.0% main-effect span), followed by
carrier (4.1%) and routed fraction (2.9%).

A post-hoc 50-kpc tidal-weighted setting with `f=0.8` was then locked and
transferred to three pilot maps. It improved mean JS by 21.8%, all three
systems, and all 300 realizations, while raising mean Pearson from 0.669 to
0.720. Because peak locations from those pilot maps already entered the
activation analysis, this is replication stress evidence rather than fresh
validation.

The current high-impact observation is: **apparent residual locations favor a
null-safe product of vector cancellation and tidal balance, but the predictive
translation depends strongly on carrier and a roughly 50-kpc arrival scale.**
This angular layer is Solar- and axisymmetric-null and therefore cannot explain
galaxy rotation on its own.

Additional artifacts:

- `docs/P0571_P0572_TIDAL_CANCELLATION_RESULTS.md`
- `results/p0571_apparent_peak_baryon_invariant/`
- `results/p0571b_null_safe_tidal_cancellation/`
- `results/p0572_tidal_cancellation_arrival_forward/`
- `results/p0572b_pilot_arrival_transfer/`

## P0573-P0574 fresh replication and symmetry separation

The P0572B setting was frozen and tested on three RELICS systems unused by
every prior local score: RXC J2211.7-0350, SMACS J0723.3-7327, and SPT-CL
J0615-5746. Each has an HST member catalog, 100 Lenstool maps, and an
independent GLAFIC map. The input set was frozen from archive filenames and
coverage before any convergence pixels were opened.

The no-gate 50-kpc, `f=0.8` law improved equal-system Lenstool JS by 18.37%,
all three systems, and 88.7% of 300 uncertainty maps. It also improved the
independent GLAFIC aggregate by 28.32% on all three. That is the first genuinely
fresh replication of the cluster morphology signal.

The stronger cross-domain audit then falsified an earlier claim: a resolved
circular exponential disk gives activation RMS 0.439, not zero. The activation
is exactly null for one source but not for an extended axisymmetric source.

P0574 added one dimensionless environment factor derived only from baryons:

\[
Q_{90}={\sum |B_{50}-R_{90}B_{50}|\over 2\sum B_{50}},\qquad
H={Q_{90}^{n}\over Q_{90}^{n}+Q_0^n},\qquad f_{\rm eff}=fH.
\]

Fourteen frozen one-at-a-time variants were selected on 13 older systems. The
winner retained the original powers and fraction, used `Q0=0.05`, `n=4`, and
broadened the arrival width from 50 to 60 kpc. On the three P0573 systems it
improved JS by 15.77%, all three systems, and 84.3% of realizations. GLAFIC
improved by 24.97% on all three. It retained 85.8% of the no-gate cluster gain
while giving exactly zero effective angular route fraction for a centered
circular disk, the Solar point source, and all 175 deprojected axisymmetric
SPARC profiles.

The one-at-a-time impact ranking is unusually clear:

| Coordinate | Historical relative JS span |
|---|---:|
| Arrival width, 40--60 kpc | 5.209% |
| Routed fraction, 0.7--0.9 | 1.112% |
| Tidal-balance power, 0.8--1.2 | 0.613% |
| Cancellation power, 0.4--0.6 | 0.205% |
| Symmetry sharpness, 2--6 | 0.0777% |
| Symmetry threshold, 0.03--0.08 | 0.0172% |

The current universal truth is: **the repeatable cluster signal is controlled
primarily by a roughly 50--60 kpc spatial reach, while a very insensitive
quarter-turn baryon-symmetry factor can turn the angular layer off in circular
galaxy and Solar environments.** This is a clean domain separator, not yet a
unified force law: it changes no SPARC speed and must be paired with a radial
acceleration equation.

Additional artifacts:

- `docs/P0573_P0574_FRESH_REPLICATION_AND_SYMMETRY_RESULTS.md`
- `results/p0573_tidal_arrival_fresh_replication/`
- `results/p0574_symmetry_gated_arrival_microvariation/`

## P0575-P0576 raw positions and the routed propagator

Twelve pre-JWST SMACS J0723 image positions from four spectroscopic source
families provided the first raw lens-equation check. One positive deflection
amplitude was calibrated on families 1 and 2 and held fixed on families 5 and
19. Under an ordinary Poisson potential, the P0574 arrival map worsened held-
out source-plane scatter by 6.63% versus local member light. It failed in all
six two-family partitions, with a median change of 7.84% worse. Two- through
four-times FFT padding changed the primary result by less than 0.04 percentage
point. The processed Lenstool convergence reference was best in all six splits.

P0576 held the 60-kpc destination map fixed and changed only how its modes
enter deflection:

\[
\alpha_{D,p}(k)=\alpha_{D,1}(k)
\left({k\over k_0}\right)^{2(1-p)},\qquad k_0={2\pi\over60\ {\rm kpc}}.
\]

The symmetry gate mixes this routed field with the ordinary local field, so
the modification remains exactly absent for Solar and deprojected
axisymmetric-SPARC environments. Calibration selected the grid boundary
`p=1.5, f_alpha=1`. On the two held-out families it reduced source-plane RMS
from 1.285 to 0.709 arcsec, a 44.83% improvement, and improved both families.
The processed Lenstool reference scored 0.646 arcsec in the same statistic.

This isolates a new high-impact coordinate: **the conversion from apparent
arrival density to deflection is more important than small activation or
symmetry-gate changes. The raw images favor a long-wavelength-enhanced routed
potential rather than the ordinary Poisson link.** Because both coordinates
hit their scan boundaries and only one cluster is tested, this is a lead for
an extended scan and another raw cluster, not a measured universal exponent.

Additional artifacts:

- `docs/P0575_P0576_RAW_POSITION_AND_PROPAGATOR_RESULTS.md`
- `results/p0575_smacs0723_raw_position/`
- `results/p0575b_raw_position_robustness/`
- `results/p0576_fractional_routed_propagator/`

## P0576B-P0578 degeneracy correction and two-cluster raw response

Extending the fractional scan from `p=1.5` to `p=2.6` made source-plane RMS
fall monotonically to 0.087 arcsec and improved all six family splits. This was
not a physical optimum. At `p=2.6`, 99.9925% of the field variation was an
affine mapping of image position and inferred sources collapsed to 0.91% of
their no-lens radius. The earlier P0576 gain was a mass-sheet/source-plane
metric failure.

A fixed Jacobian-aware linearized image-plane metric selected an interior
`p=1.75` on SMACS, but it improved only one of two held-out families and still
had mass-sheet `R^2=0.9976`. Applied without exponent reselection to the
independent SPT0615 raw table, it worsened held-out RMS by 61.0% and helped only
one of three subfamilies. SPT's own internal selection again ran to the power
and fraction boundaries and improved by only 6.15%. Fractional power is not a
cross-cluster coordinate in the present construction.

P0578 returned to ordinary Poisson propagation and varied only member-light
broadening. Equal-cluster calibration selected a fully broadened 125-kpc map.
It improved SPT by 26.24% but worsened SMACS by 4.98%; only 40% of held-out
subfamilies improved. The fully broad fraction was much more influential than
width: its calibration main-effect span was 2.254 arcsec, versus 0.777 arcsec
for width. Width itself had a broad calibration minimum around 60--100 kpc.

The corrected universal truth is: **radial/spatial organization is the largest
raw-lensing coordinate, but neither a fractional propagator nor one Gaussian
baryon width transfers between the two clusters. Source-plane scatter must not
be used for propagator selection without a mass-sheet-resistant check.** The
next useful formula should respond to measured baryonic concentration or
multi-scale structure rather than use one fixed width or exponent.

Additional artifacts:

- `docs/P0576B_P0578_DEGENERACY_AND_TWO_CLUSTER_RESULTS.md`
- `results/p0576b_fractional_boundary_extension/`
- `results/p0576c_source_plane_degeneracy_audit/`
- `results/p0576d_linearized_image_plane/`
- `results/p0577_spt0615_raw_response/`
- `results/p0578_two_cluster_baryon_broadening/`

## P0579-P0580 inverse-derived return geometry and conservative amplitude

P0579 froze 432 extent-scaled route kernels before scoring held-out raw image
families in SMACS0723 and SPT0615. The externally locked inverse-derived
endpoint rule used a `0.36 R80` return distance and `0.23 R80` endpoint width.
It improved both clusters and reduced equal-cluster held-out RMS from 6.977 to
6.126 arcsec, a 12.20% gain. Three of five held-out subfamilies improved. The
result failed the strict mass-sheet gate only: maximum `R2=0.960` versus the
frozen 0.95 cutoff.

The 432-setting calibration winner did not transfer. It selected a fully
routed symmetric transverse arc, reached 2.000 arcsec on calibration, then
worsened held-out RMS to 14.724 arcsec with `R2=0.9963`. Route flexibility on
four calibration families is therefore not a reliable selector.

Route-residence mode was the largest raw-lensing coordinate by a wide margin:
its post-open held-out span was 8.527 arcsec, versus 1.796 for width, 1.342 for
return length, 0.631 for the extent gate, and 0.387 for routed fraction.
Endpoint-only return was best; symmetric transverse residence was worst.

P0580 then applied the same kernels to 131 SPARC galaxies as strictly
conservative force-equivalent radial redistribution. The locked rule improved
110 galaxies but changed outer RMS only from 72.399 to 70.926 km/s, far from
fixed RAR at 10.348 km/s. Even the post-hoc best of 432 reached only 69.321
km/s. Route mode again dominated, with a 1.823 km/s median main-effect span;
the concentration gate and routed fraction changed the median by less than
0.05 km/s.

The new universal truth is: **arrival geometry can improve where cluster
lensing appears, and inward return has the right sign in most galaxies, but a
fixed conserved baryonic budget cannot generate the galaxy-scale amplitude.
The field should keep ordinary baryonic gravity local, create one universal
low-acceleration excess, and route only that excess through an endpoint
kernel.**

Additional artifacts:

- `docs/P0579_P0580_RETURN_FIELD_RESULTS.md`
- `results/p0579_extent_gated_return_raw/`
- `results/p0580_conservative_return_sparc/`

## P0581 exact-root transfer and topology sensitivity

P0581 translated the locked K0338 endpoint geometry into a conservative
potential field on four clusters independent of its selection. Ordinary
baryonic gravity remained local; only the angular distribution of the P0554
scalar excess was modified. The correction was explicitly curl-free and had
zero annular monopole.

The locked formula did not validate. It recovered MACS1931's missing held-out
root and very slightly improved MACS1115, but it lost one MACS0329 root and
worsened MACS0429. Only MACS0429 and MACS1115 were complete under both formulas;
on that matched pair the endpoint rule worsened equal-system RMS by 0.70%.
Its validation RMS was 1.91 times the compact-halo comparator, and every frozen
performance gate failed.

The one-at-a-time replay exposed a new distinction between residual and
topology impact. Removing the concentration gate reduced total held-out roots
from 10 to 7. Moving return length away from the inverse-derived `0.36 R80`
reduced roots from 10 to 8 in either direction, despite only a 0.021-arcsec
common-system RMS span. Most importantly, lowering the pre-normalization endpoint contrast cap
from 20 to 5 or 10 restored all 11 roots and all four complete systems while
changing matched RMS by only 0.034 arcsec.

The updated universal truth is: **sharp redirected fields must be judged by
caustic/root stability, not residual RMS alone. Environment gating is required,
the `0.36 R80` reach remains an interior topology optimum, and smoothly bounded
endpoint contrast is the next high-value formula coordinate.** The lower caps
are post-hoc diagnostics and require a new cluster for validation.

Additional artifacts:

- `docs/P0581_LOCKED_ENDPOINT_EXACT_ROOT_RESULTS.md`
- `results/p0581_locked_endpoint_exact_root/`

## P0582 smooth saturation and the caustic response window

P0582 varied only the pointwise contrast transform while holding P0581's lens
geometry, source positions, baryons, scalar parent, route gate, `0.36 R80`
return length, and `0.23 R80` width fixed. Four transforms (hard, tanh,
exponential, and rational) were crossed with six nominal saturation scales,
creating 24 variants and 96 cluster fields.

Five variants found all 11 held-out roots: hard caps 5, 7.5, 10, and 15, plus
`20 tanh(x/20)`. The diagnostic RMS winner was hard cap 5 at 19.040 arcsec;
the smooth complete form reached 19.159 arcsec. At nominal scale 20 the modes
were sharply different: tanh found 11 roots, hard 10, exponential 9, and
rational 8. Therefore neither the maximum nominal scale nor smoothness alone
controls the result; the full compression curve matters.

The cluster split explains the pattern. MACS0429 and MACS1115 retained every
root under all 24 variants. MACS0329 failed only for hard cap 20, while MACS1931
completed only six variants and generally required a stronger correction. The
smooth tanh-20 curve is the only smooth candidate that remained below the
MACS0329 upper caustic boundary while remaining strong enough for MACS1931.

An implementation detail was also corrected: the nominal cap is applied before
carrier-weighted annular renormalization. It is not a global bound on the final
field weight; post-normalization weights reached roughly 730 in sparsely
populated annuli while the physical convergence and curl audits remained
finite and conservative. Future formulas should parameterize a directly
auditable field-response norm rather than call this number a literal contrast
ceiling.

The updated universal truth is: **cluster lens roots occupy a response window.
Too much angular correction destroys a MACS0329 root, too little fails to
recover MACS1931 roots, and a gently bending tanh response can thread both.
Image topology is controlled by the full nonlinear response curve, not one cap
or residual score.** All systems were already opened, so tanh-20 is a candidate
to freeze, not validation evidence.

Additional artifacts:

- `docs/P0582_SMOOTH_ENDPOINT_SATURATION_RESULTS.md`
- `results/p0582_smooth_endpoint_saturation/`

## P0583-P0584 RX J2129 transfer, sign, and overshoot

P0583 froze K0338 with tanh-20 and transferred it to RX J2129, a system absent
from the ten inverse-driver clusters, both P0579 systems, and all four
P0581-P0582 clusters. The formula retained all seven exact held-out roots, but
worsened RMS from 1.256 arcsec for scalar P0554 to 14.130 arcsec. Hard-5 was
nearly identical at 14.135 arcsec. All endpoint fits pushed the center and both
shear components to bounds. Smooth contrast saturation therefore does not fix
angular placement.

At fixed scalar geometry, a signed amplitude screen found zero best on the
original grid. `epsilon=+0.025` lost image 2c, while `epsilon=-0.025` retained
roots but worsened RMS to 1.610 arcsec. Of six images with roots on both sides,
only two moved in the improving direction to first order. The endpoint map is
mostly directionally misaligned, not simply too strong.

P0584 isolated one concrete geometric defect. Constant `0.36 R80` travel sent
15 of 51 member sources, carrying 47.8% of catalog weight, through the light
centroid. Three no-cross laws were tested. The smooth
`ell_i=L*tanh(d_i/L)` rule removed all crossings and was best at a refined
`epsilon=0.005`, reducing RMS to 1.243 arcsec (1.05%). The usable interval was
extremely narrow: at 0.01 the tanh route lost a root, and no complete no-cross
alternative improved at that scale.

The updated universal truth is: **a return path should never overshoot its
destination, and `L*tanh(d/L)` is the cleanest tested travel law. But destination
direction is far more consequential than this repair: one global
light-centroid route does not transfer to RX J2129, and per-cluster amplitude
suppression would not be a universal theory.** The next directional formula
must derive multiple local destinations from baryonic structure or a baryonic
field tensor.

Additional artifacts:

- `docs/P0583_P0584_RXJ2129_FAILURE_FORENSICS.md`
- `results/p0583_tanh_endpoint_rxj2129/`
- `results/p0583b_signed_endpoint_amplitude/`
- `results/p0584_no_overshoot_endpoint/`

## P0585 baryon-derived local attractors

P0585 replaced the single global destination with 32 local-attractor maps. For
source `i`, neighboring baryons define a kernel-weighted destination; a mix
parameter blends that target with the global centroid, and
`ell_i=L*tanh(d_i/L)` prevents crossing. Thirty-three maps including the global
control were crossed with seven small positive amplitudes at frozen RX J2129
scalar geometry.

The best local candidate used 25% local destination, `0.1 R80` softening, and
inverse-distance weighting at `epsilon=0.005`. It reached 1.24235 arcsec versus
1.24301 for the global no-cross control, only a 0.053% improvement. The median
RMS spans were 0.000665 arcsec for local mix, 0.000484 for softening, and
0.000124 for distance power.

The updated universal truth is: **once the endpoint amplitude is reduced to
the narrow root-safe RX J2129 regime, destination geometry has essentially no
leverage. Multiple local attractors are physically cleaner but cannot rescue
the zero-monopole endpoint channel.** That channel is now retired as the main
cluster-lensing mechanism; inverse flow maps remain descriptive, and the next
branch must change a continuous baryonic metric/tidal response or add missing
measured baryons.

Additional artifacts:

- `docs/P0585_LOCAL_ATTRACTOR_RESULTS.md`
- `results/p0585_local_attractor_screen/`

## P0586-P0586D continuous baryonic field metric

P0586 replaced discrete arrival endpoints with a positive constitutive metric
built from measured baryons,

\[
\partial_i(K_b^{ij}\partial_j\Phi)=4\pi G\rho_b,
\qquad
K_b=\epsilon(S)\exp[\tau S H Q_b].
\]

The five-coordinate physical-map screen ranked the selected-neighborhood
cluster impacts as: signed anisotropy `tau` 0.1241 arcsec, minimum
permittivity 0.0550, reach `eta R80` 0.0482, transition power 0.00381, and
acceleration scale 0.00047. The projected cluster gate was already 0.999--1,
so the last two coordinates were effectively saturated rather than measured.

A positive-only boundary extension improved at most three clusters and lost a
required exact root. Restoring both signs revealed four common fixed-geometry
candidates, all with no scalar boost, broad `0.8 R80` reach, and negative
anisotropy. The locked `tau=-1.2` candidate then preserved every exact root and
improved four-cluster RMS from 17.984 to 17.624 arcsec, a monotonic 2.01% gain
across the tested negative strengths. It improved MACS0329 by 4.44% and
MACS1931 by 14.83%, but worsened MACS0429 by 0.66% and MACS1115 by 0.04%.

The result failed two decisive checks. It remained 1.764 times the compact-halo
RMS, and the MACS1115 correction was 99.24% affine over its observed images.
The latter drove the maximum mass-sheet audit above its 0.95 limit. The
selected `epsilon0=1` also turns the scalar branch off, leaving spherical SPARC
at the Newtonian 72.399 km/s. Conversely, the best scalar metric reached 17.100
km/s and passed Solar controls but remained behind fixed RAR at 10.348 km/s.

The updated universal truth is: **a broad, signed continuous tidal metric is a
real and root-safe cluster coordinate, but its small aggregate gain mixes two
helpful clusters, two unfavorable clusters, and one nearly affine response.
Cluster anisotropy and galaxy radial support currently live in different parts
of the formula. The next clean test must remove long-wavelength affine
potential modes using a baryon-defined aperture, never the observed image
positions, and ask whether the non-affine exact gain survives.**

Additional artifacts:

- `docs/P0586_P0586D_CONTINUOUS_BARYONIC_METRIC_RESULTS.md`
- `results/p0586_continuous_baryonic_metric/`
- `results/p0586b_metric_boundary_response/`
- `results/p0586c_signed_metric_response/`
- `results/p0586d_signed_metric_exact/`

## P0587 baryon-defined affine high-pass

P0587 asked whether P0586D's 2% exact gain survives after long-wavelength
affine potential modes are structurally forbidden. It fitted a constant plus
trace-only or symmetric affine field on a circular baryon-centered aperture,
converted that fit to a scalar quadratic potential, and subtracted its gradient
through a cosine window. Image positions and lens targets never entered the
field construction.

The declared symmetric `1.0 R80` full-removal primary reduced its baryon-grid
affine `R2` to roughly `1e-9`, preserved every exact root, and remained 1.12%
better than zero. It was 0.23% worse than the raw metric, still worsened
MACS0429 and MACS1115, and remained 1.769 times the compact-halo comparator.
Most importantly, MACS1115's correction remained 99.19% affine at the sparse
observed image locations. The field is globally non-affine; those images happen
to sample a locally linear patch.

The high-pass parameter spans were small: aperture radius 0.0186 arcsec,
trace-versus-symmetric mode 0.0134, and removal fraction 0.00425. No projection
choice changes the branch's interpretation.

The updated universal truth is: **the affine warning is a property of sparse
strong-lens sampling, not a removable global mode of the baryonic field. A
baryon-defined high-pass does not improve the raw metric, while an image-defined
subtraction would encode the target and is inadmissible. Further progress now
requires missing measured baryons or an independent lens observable, not more
affine-removal parameters.**

Additional artifacts:

- `docs/P0587_BARYONIC_HIGHPASS_RESULTS.md`
- `results/p0587_baryonic_highpass_metric/`
