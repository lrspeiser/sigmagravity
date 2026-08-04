# Sigma v17O halo-scale driver audit

## Purpose

The next field action should not be chosen by taste. Existing modified-gravity
theories already imply different baryonic scalings for a characteristic halo
radius:

\[
r_M=\sqrt{GM_b/a_0}\propto M_b^{1/2}
\]

for a MOND acceleration transition,

\[
r_\rho\propto M_b^{1/3}
\]

for a universal mean-density transition, and

\[
r_C=(r_M\mu^{-2})^{1/3}\propto M_b^{1/6}
\]

for the published AeST quasistatic cutoff. V17O asks which, if any, resembles
the scale radii inferred by conventional halo analyses across both galaxies and
clusters.

The published antecedents are [AeST](https://arxiv.org/abs/2007.00082),
[covariant refracted gravity](https://arxiv.org/abs/2109.11217), and
[TeVeS](https://arxiv.org/abs/astro-ph/0403694). None of those formulas is a
Sigma invention.

## Diagnostic targets

For 131 SPARC galaxies, the audit uses the existing NFW control in which
\(V_{200}\) and concentration were independently fitted on inner rotation-curve
radii. It converts them through

\[
r_{200}={V_{200}\over10H_0},\qquad r_s={r_{200}\over c_{200}}.
\]

For 20 CLASH clusters, it refits the published NFW-deprojected total
acceleration profile with

\[
M(<r)=M_{200}{f(r/r_s)\over f(c_{200})},
\quad
f(x)=\ln(1+x)-{x\over1+x}.
\]

Galaxy baryonic mass and extent come from the catalog disk, bulge, and gas
components. Cluster baryonic mass and half-mass radius are reconstructed from
\(g_{\rm bar}(r)r^2/G\).

These targets are deliberately labeled **diagnostic**. Galaxy NFW radii are
prior-sensitive, and the cluster profiles were already produced under an NFW
interpretation. Matching them would select a scale mechanism; it would not
prove a physical halo or pass the project’s raw-observation gates.

## Frozen comparison

Five one-parameter relations are tested with one shared normalization:

\[
r_s=A r_M,
\quad
r_s=A M_b^{1/3},
\quad
r_s=A r_M^{1/3},
\quad
r_s=A R_{b,50},
\quad
r_s=A\sqrt{r_MR_{b,50}}.
\]

Two diagnostics are also allowed:

\[
r_s=A(M_b/10^{10}M_\odot)^a
\]

and the dimensionally consistent bridge

\[
r_s=A r_M^aR_{b,50}^{1-a}.
\]

Every relation is common to galaxies and clusters. Five-fold predictions use
hash-assigned systems and equal total weight for each domain. A domain-specific
normalization is reported only as a transfer failure and can never become the
final law.

## Decision gates

A fixed relation is considered informative only if its equal-domain
out-of-fold error is at most 0.25 dex and each domain’s median bias is at most
0.15 dex. A continuous geometry invariant is required only if the mass–extent
bridge improves on every mass-only relation by at least 0.05 dex and removes
the domain bias.

If neither happens, these NFW products do not identify a universal halo-radius
law well enough to design the next action. The project must then derive and
score field scales directly from raw rotation and lensing observations.

## Result

The audit retained 129 strict-quality galaxy NFW comparators and all 20 cluster
profiles. The cluster profiles were reproduced by an NFW curve to a maximum
error of 0.00542 dex, so failure of the scaling relations is not caused by a
poor numerical reconstruction of the cluster target.

| Relation | Equal-domain CV RMSE (dex) | Galaxy RMSE / bias (dex) | Cluster RMSE / bias (dex) | Gate |
|---|---:|---:|---:|---|
| (r_s=A R_{b,50}) | 0.3542 | 0.4135 / -0.0923 | 0.2828 / +0.1541 | Fail |
| (r_s=A M_b^{1/3}) | 0.3658 | 0.4552 / +0.2253 | 0.2457 / -0.1473 | Fail |
| (r_s=A\sqrt{r_MR_{b,50}}) | 0.3781 | 0.4624 / -0.0641 | 0.2684 / +0.1629 | Fail |
| (r_s=A r_M) | 0.4301 | 0.5501 / -0.1133 | 0.2597 / +0.1811 | Fail |
| AeST-like (r_s=A r_M^{1/3}) | 0.5293 | 0.5733 / +0.5165 | 0.4813 / -0.4178 | Fail |

The best fixed relation is therefore baryonic extent, but 0.3542 dex means a
typical factor-2.26 scale error and misses the preregistered 0.25-dex gate. The
two-parameter mass--extent bridge reaches only 0.3555 dex. Its improvement over
the best mass-only law is 0.0103 dex, far short of the required 0.05 dex.

The pooled free mass law gives

\[
r_s\mathrel{\propto}M_b^{0.3734},
\]

with a domain-stratified bootstrap interval of 0.3465--0.4028. That number is
not evidence for a universal near-(1/3) density law. Fitted within each domain,
the exponent is 0.0891 for galaxies and 0.1835 for clusters. A galaxy-only fit
misses clusters by 1.159 dex RMSE, while a cluster-only fit misses galaxies by
0.897 dex. The pooled slope is largely the line connecting two populations.

Retaining the two finite galaxy fits that landed on prior boundaries does not
change the decision. Baryonic extent remains best at 0.3561 dex; changes across
all relations are 0.0006--0.0122 dex.

## Decision and implication for the root equation

The frozen outcome is
`halo_scale_not_identifiable_from_current_diagnostic_products`. No holdout was
opened, and no physical halo claim follows.

This rules out a specific shortcut: the root equation should not be built by
assigning every baryonic system a single NFW-like radius from total mass or
half-mass radius. Those scalars capture only coarse trends and do not transfer
with one normalization between galaxies and clusters. The result does not rule
out a baryon-only field law. It says the law must predict the spatial response
directly, using continuous field information such as local acceleration,
density gradients, and tidal or multi-source geometry, rather than first
inventing an effective halo radius.

The next derivation will therefore return to raw radial acceleration and raw
lensing observables. It must place any environment response in a dynamically
healthy field sector, because v17N already ruled out the broad class of
decreasing acceleration screens inserted directly into the matter metric. A
galaxy/cluster label or a domain-specific normalization remains forbidden.

Machine-readable outputs are in
`results/sigma_v17o_halo_scale_driver_audit/report.json`, with object-level
cross-validated predictions in `predictions.csv` and the reconstructed cluster
targets in `cluster_nfw_fits.csv`.
