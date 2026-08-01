# Sigma Gravity versus Refracted Gravity

## Bottom line

Sigma Gravity and Refracted Gravity (RG) are not independent versions of the
same equation, but they overlap strongly in what they try to produce. Sigma
prescribes an acceleration-dependent response and has a fixed-
\(B\) QUMOND auxiliary action. RG changes the Poisson operator through a
density-dependent scalar permittivity and thereby changes both field strength
and, in nonspherical systems, field direction.

Multiplying their two enhancements is not a good combination. It counts the
same missing-gravity phenomenology twice and can create extremely large forces
where both the acceleration and density are low. The most defensible candidate
identified at the formula-comparison stage was a **coherence-partitioned
response** in which measured kinematic order chose continuously between the
Sigma and RG operators. The subsequent measured-data program is now complete:
the declared coherence mapping did not improve held-out BCG dynamics, and
measured ACCEPT density profiles did not produce a universal transfer to CLASH
lensing. CPR0 is therefore retained as a rejected baseline, not the active
candidate.

The later exploratory MOND/dark-matter sweep does not revive CPR0. It replaces
the high-coherence Sigma endpoint with fixed empirical RAR and sharpens the
low-coherence RG gate. That distinct follow-on is documented in
[`MOND_DARK_MATTER_FORMULA_SWEEP_RESULTS.md`](MOND_DARK_MATTER_FORMULA_SWEEP_RESULTS.md).

## Exact formula comparison

### Sigma Gravity

The local manuscript defines

\[
\nabla^2\Phi_N=4\pi G\rho_b,
\qquad g_N=|\nabla\Phi_N|,
\]

\[
\Sigma(g_N,B)=1+B h(g_N),
\qquad
h(g_N)=\sqrt{\frac{g^\dagger}{g_N}}
\frac{g^\dagger}{g^\dagger+g_N},
\]

with \(g^\dagger=9.60\times10^{-11}\ {\rm m\,s^{-2}}\).  Its spherical or
algebraic prediction is

\[
g_\Sigma=\left[1+B h(g_N)\right]g_N.
\]

The low-acceleration term scales as \(\sqrt{g^\dagger g_N}\), while the
fractional response falls as \(g_N^{-3/2}\) at high acceleration.  In the deep
point-mass limit it gives

\[
V^4\longrightarrow B^2 G M_b g^\dagger.
\]

Thus the baryonic Tully--Fisher normalization measures \(B^2g^\dagger\), not
\(B\) and \(g^\dagger\) independently.

For spatially constant, independently prescribed \(B\), the manuscript gives
the QUMOND system

\[
\nabla^2\Phi_N=4\pi G\rho_b,
\qquad
\nabla^2\Phi=\nabla\!\cdot\!\left[(1+B h)\nabla\Phi_N\right].
\]

The locked galaxy implementation is not identical to that action. It uses

\[
B_{\rm gal}=A_0\frac{V_\Sigma^2}{V_\Sigma^2+\sigma^2},
\quad A_0=e^{1/(2\pi)},\quad \sigma=20\ {\rm km\,s^{-1}},
\]

where \(V_\Sigma\) is itself predicted by the model. That factor is an
endogenous regularizer, not an independently measured coherence observable.

### Refracted Gravity

Phenomenological RG instead starts with

\[
\nabla\!\cdot[\epsilon(\rho_b)\nabla\Phi]=4\pi G\rho_b,
\]

and the published smooth transition

\[
\epsilon_{\rm RG}(\rho)=\epsilon_0+
\frac{1-\epsilon_0}{2}
\left\{\tanh\!\left[Q\ln(\rho/\rho_c)\right]+1\right\}.
\]

High density gives \(\epsilon\rightarrow1\); low density gives
\(\epsilon\rightarrow\epsilon_0<1\). In spherical symmetry,

\[
g_{\rm RG}(r)=\frac{G M_b(<r)}{\epsilon[\rho(r)]r^2}.
\]

The direction remains radial in that limit. In a disk,
\(\nabla\epsilon\) need not be parallel to \(\nabla\Phi\), and the expanded
equation

\[
\epsilon\nabla^2\Phi+
\frac{\partial\epsilon}{\partial\rho}
\nabla\rho\!\cdot\!\nabla\Phi=4\pi G\rho
\]

shows the directional term. This is the source of RG's proposed field-line
refraction and disk-plane focusing. The original paper demonstrated disk
rotation curves and cluster X-ray temperature profiles; later DiskMass work fit
radial and vertical galaxy kinematics but found unresolved RAR residual
correlations. A 2025 two-cluster kinematic test found different preferred RG
parameters for the two clusters, with a 13-sigma difference in transition
density and a 4-sigma difference in sharpness
([Matsakos & Diaferio 2016](https://arxiv.org/abs/1603.04943),
[Cesare et al. 2020](https://arxiv.org/abs/2003.07377),
[Pizzuti et al. 2025](https://arxiv.org/abs/2410.19698)).

Covariant RG embeds the permittivity in a scalar--tensor action,

\[
S=\frac{1}{16\pi G}\int d^4x\sqrt{-g}
\left[\varphi R-\frac{1}{\varphi}(\nabla\varphi)^2-2\Xi\varphi\right]+S_m,
\]

and identifies \(\varphi=2\epsilon\) in its weak-field limit. That is a real
advantage over the current Sigma construction, although a covariant action by
itself is not an observational validation
([Sanna, Matsakos & Diaferio 2023](https://doi.org/10.1051/0004-6361/202243553)).

## The overlap with our already-tested scalar basin model

The NBP0 permittivity used

\[
\epsilon_{\rm NBP0}(X)=\epsilon_0+(1-\epsilon_0)
\frac{(X/\rho_c)^q}{1+(X/\rho_c)^q},
\qquad (1-L_X^2\nabla^2)X=\rho_b.
\]

For \(L_X=0\), the published RG formula is exactly the same logistic family
under

\[
X=\rho_b,\qquad q=2Q_{\rm RG}.
\]

The NBP0 sweep included \(L_X=0\), \(\epsilon_0=0.03\) to 1, and
\(q=0.5\) to 8, so it included the published galaxy-scale reference near
\(\epsilon_0=0.089\) and \(Q_{\rm RG}=0.47\). Consequently, simply renaming
NBP0 as RG or changing the transition exponent does not open a genuinely new
branch.

NBP0's failure was specific but important. Across 128 matched disk/bulge
environments, only 28.9% gave the predicted disk-greater-than-bulge enhancement
at 4, 6, and 8 disk scales simultaneously, versus the frozen 80% gate. Adding
morphology to the SPARC held-out model worsened RMSE for every tested stellar
mass-to-light combination. This rejects a stable *population-level morphology
signature* for that scalar implementation. It does not reproduce every
published RG analysis, because our SPARC snapshot lacks measured local gas
density and thickness maps and the empirical morphology test was not a full RG
field fit.

## Candidate combinations

### 1. Product response -- reject

\[
\frac{g}{g_N}=\frac{1+B h(g_N)}{\epsilon_{\rm RG}(\rho)}.
\]

This has simple limits but multiplies two mechanisms designed to explain the
same discrepancy. For a low-acceleration, low-density point, each factor can
already be several to ten, making their product tens of times Newtonian. It also
changes the deep BTFR normalization with environment and aggravates the cluster
overprediction already seen with \(B=8.446\).

### 2. Additive susceptibilities -- diagnostic only

\[
\frac{g}{g_N}=1+B h(g_N)+\left(\epsilon_{\rm RG}^{-1}-1\right).
\]

This is less explosive than the product but still assumes two simultaneous
missing-gravity sources. It has no compelling rule for why both should be active
in the same place and does not retain either parent's nonspherical field
equation.

### 3. Let coherence move \(\rho_c\) or \(\epsilon_0\) -- defer

For example,

\[
\rho_c(C)=\rho_{c0}\exp[\lambda(C-C_0)].
\]

This is easy to fit but creates another continuous degeneracy among
\(C,\lambda,\rho_c,Q\), while our scalar morphology result already shows that
moving the transition surface can reverse the disk/bulge sign with radius.

### 4. Coherence-partitioned response (CPR0) -- tested, did not advance

Define an independently measured, bounded phase-space order statistic such as

\[
C_{\rm kin}=\frac{|\mathbf v_{\rm stream,barycentric}|^2}
{|\mathbf v_{\rm stream,barycentric}|^2+
{\rm tr}(\boldsymbol\sigma^2)},
\qquad 0\le C\le1,
\]

and use the parameter-free smooth weight

\[
w(C)=3C^2-2C^3.
\]

Then define

\[
\epsilon_{\rm mix}=w+(1-w)\epsilon_{\rm RG},
\qquad
\nu_{\rm src}=1+w B_0h(g_N),
\]

\[
\boxed{
\nabla\!\cdot[\epsilon_{\rm mix}\nabla\Phi]
=\nabla\!\cdot[\nu_{\rm src}\nabla\Phi_N]
}.
\]

The reference \(B_0\) is global, initially \(A_0=e^{1/(2\pi)}\), rather than a
separate fitted value for each system.

This construction has exact and useful endpoints:

- \(C=1\): \(\epsilon_{\rm mix}=1\), recovering the fixed-\(B\) Sigma/QUMOND
  equation;
- \(C=0\): \(\nu_{\rm src}=1\), recovering phenomenological RG; and
- high density and high acceleration: both coefficients approach one.

In spherical symmetry,

\[
\frac{g}{g_N}=\frac{1+w B_0h(g_N)}
{w+(1-w)\epsilon_{\rm RG}(\rho)}.
\]

The interpretation changes in an important way. Coherence is not assumed to
make gravity universally stronger. It selects the Sigma-like response in cold,
ordered systems, while disordered, low-density systems use the RG channel. This
can accommodate a disk and a cluster without fitting a cluster-specific Sigma
amplitude, but it is also perilously close to a smooth domain selector. It only
becomes a scientific unification if measured \(C\), frozen before fitting,
predicts intermediate systems such as counterrotators, S0s, ellipticals, BCGs,
groups, and merging clusters.

## First CLASH endpoint screen

The frozen exploratory protocol used the 84 Tian et al. CLASH points in 20
clusters. These total accelerations derive from strong/weak-lensing mass
reconstructions. Because the table provides \(g_{\rm bar}\) but not a local 3D
baryon-density profile, the screen used

\[
\bar\rho_b(<r)=\frac{3g_{\rm bar}}{4\pi G r}
\]

as an explicitly imperfect density proxy. Complete clusters, not radii, were
assigned to five folds. One universal RG parameter vector was fitted to the
other four folds. This is a zero-slip effective-acceleration comparison to the
lensing-derived target, not a photon-geodesic prediction.

| Model | Equal-cluster RMSE (dex) | Median predicted/observed | Radial residual slope (dex/dex) |
|---|---:|---:|---:|
| Sigma, fixed \(B=8.446\) | 0.1884 | 1.382 | +0.131 |
| Sigma, universal \(B\), cluster-CV | **0.1229** | 0.978 | +0.116 |
| Constant \(\epsilon\), cluster-CV | 0.1948 | 0.942 | -0.165 |
| Published galaxy RG parameters, no refit | 0.2513 | 0.813 | +0.342 |
| CPR0 RG endpoint, density-proxy cluster-CV | 0.1432 | **1.003** | **+0.037** |

Relative to the old fixed cluster Sigma amplitude, the CPR0 endpoint improved
equal-cluster RMSE by 24.0%, passed the declared median-ratio and radial-slope
gates, and had stable fits across the five folds:

- \(\epsilon_0=0.112\) to 0.123;
- \(\log_{10}[\rho_c/({\rm g\,cm^{-3}})]=-23.75\) to -23.23; and
- \(Q=0.40\) to 1.00.

None was within 1% of a search bound. However, the freely recalibrated cluster
Sigma law was more accurate and required \(B=5.34\) to 5.92, still roughly five
times the galaxy amplitude. The published galaxy RG parameters did not transfer
under the mean-density proxy. The result therefore identifies a useful shape
mechanism and a next experiment; it does not establish universal RG parameters
or show CPR0 is better than all controls.

## Measured-data follow-up and final disposition

The proxy screen above was followed by measured local densities, an independent
BCG coherence observable, grouped galaxy+cluster fits, radial weak-lensing
profiles, and a final ACCEPT x CLASH match. The last primary sample contains 52
no-extrapolation points in 18 clusters at 100--600 kpc and reduces the
galaxy--cluster density gap to 0.421 dex.

Density-only RG reaches 0.1300 dex held-out cluster RMSE, compared with 0.1369
dex for a constant enhancement. The 0.0069-dex improvement misses the frozen
0.02-dex gate; parameters locked from the earlier galaxy+cluster fit score
0.1459 dex and underpredict lensing by 0.082 dex on average. Adding observed
HST BCG masses and sizes supplies 73.4% of the median central local density, but
the 20 central points still score 0.2277 dex and the full 72-point cluster
sample scores 0.1776 dex.

In the final shared test, RG scores 0.0934 dex for BCG dynamics and 0.1556 dex
for cluster lensing. CPR0 scores 0.0932 and 0.1546 dex, respectively: its
0.00070-dex equal-domain improvement is far below the required 0.01 dex. No
measured-data protocol passed all gates. Full inputs, calibration brackets,
fold fits, and claim boundaries are in
[`CPR0_MEASURED_DENSITY_AND_COHERENCE_RESULTS.md`](CPR0_MEASURED_DENSITY_AND_COHERENCE_RESULTS.md).

## What must happen next

1. Retain the completed local-density and `Lambda_Re` tests as negative
   controls. Do not reopen them by changing parameter bounds or transition
   exponents.
2. Acquire a same-system radial lensing likelihood with covariance plus
   overlapping BCG/member kinematics. Use it to measure both metric potentials,
   rather than assuming zero gravitational slip.
3. If a new coherence, shape, or anisotropy invariant is proposed, measure and
   freeze it independently of acceleration residuals before defining its field
   coupling.
4. Solve any surviving full axisymmetric equation and require simultaneous
   radial, vertical, and off-plane predictions. The prior scalar morphology
   failure remains an active negative control.
5. Only after an empirical pass, derive a closed action in which the added state
   is a dynamical scalar or tensor sourced by covariant matter-flow invariants.
   Derive both weak-field metric potentials, gravitational slip, photon
   propagation, Solar-System behavior, stability, and gravitational-wave
   propagation before calling it a gravity theory.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_sigma_refracted.py
python scripts/run_cpr0_cluster_proxy_screen.py
python scripts/run_cpr0_accept_clash_bridge.py
python scripts/run_cpr0_accept_clash_bcg_stellar.py
```

The proxy protocol is `configs/cpr0_sigma_refracted_protocol.json`. The final
measured-density protocols are `configs/cpr0_accept_clash_bridge_protocol.json`
and `configs/cpr0_accept_clash_bcg_stellar_protocol.json`; numerical outputs
are under their corresponding `results/cpr0_*` directories.
