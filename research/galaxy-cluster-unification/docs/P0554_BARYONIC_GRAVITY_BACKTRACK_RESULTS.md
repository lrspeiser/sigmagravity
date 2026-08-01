# P0554 baryonic-gravity backtracking

## Question

Suppose the locations normally represented by cluster-scale dark-matter halos are not new matter. Instead, they are places where a response launched by baryonic matter is focused or deposited after propagating through a structured gravitational field. Can the assumed halo locations be backtracked to plausible baryonic origins, and can that inverse map be turned into a baryon-only forward prediction?

This study answers the first, descriptive part. It does not yet supply a successful forward law.

## A minimal field representation

A conservative version of the idea can be written as

\[
\rho_{\rm route}(\mathbf y)
=(1-q)\rho_b(\mathbf y)
+q\int K_\theta(\mathbf y\mid\mathbf x,E_b)\rho_b(\mathbf x)\,d^3x,
\]

\[
\nabla^2\Phi(\mathbf y)=4\pi G\rho_{\rm route}(\mathbf y),
\qquad
\int K_\theta(\mathbf y\mid\mathbf x,E_b)\,d^3y=1.
\]

- `rho_b` is the measured baryonic source distribution.
- `q` is the fraction of the response that is relocated. A Solar-System screening rule must make `q` effectively zero locally.
- `K_theta(y|x,E_b)` is the still-unknown propagation kernel: given a baryonic launch point `x` and its baryonic environment `E_b`, it predicts where the response appears at `y`.
- The subtraction implicit in `(1-q)rho_b` prevents the model from silently creating extra total source strength.

A finite-distance, arc-producing candidate family could use

\[
K_\theta(\mathbf y\mid\mathbf x,E_b)
\propto
\exp\!\left[-\frac{(\|\mathbf y-\mathbf x\|-\ell_\theta(E_b))^2}{2w_\theta(E_b)^2}\right]
\exp\!\left[\beta_\theta(E_b)\,\widehat{(\mathbf y-\mathbf x)}\cdot\mathbf e_1(E_b)\right],
\]

where `ell` sets a preferred return distance, `w` sets its spread, and `e_1` is a direction determined from the baryonic tidal or density field. The normalization is essential: the destination must be predicted from baryons, not inserted from a dark-halo fit.

For visualization only, a source-to-arrival pair was lifted into an arc,

\[
\Gamma_{ij}(t)=\left((1-t)\mathbf x_i+t\mathbf y_j,\;4h_{ij}t(1-t)\right).
\]

The two-dimensional observations constrain the endpoints but not the off-plane height `h`, travel time, or actual trajectory.

In a relativistic theory the same idea would require an additional conserved response tensor,

\[
G_{\mu\nu}=8\pi G T^b_{\mu\nu}+\Sigma_{\mu\nu}[g,T_b],
\qquad \nabla^\mu\Sigma_{\mu\nu}=0.
\]

The conservation condition is not optional. An arbitrary collection of curved force lines is not automatically compatible with a metric theory or even with a curl-free Newtonian potential.

## What was actually backtracked

The destination maps were built from the public Lenstool posterior chains for five clusters: RXJ2129, MACS0329, MACS0429, MACS1115, and MACS1931. Seven cluster-scale halo components were found. These are model-dependent halo locations inferred from lensing, not direct observations of dark matter.

Each posterior center was convolved with a universal 50 kpc Gaussian and weighted by its posterior mean velocity-dispersion squared. Baryonic launch maps included member-galaxy light, continuous HST F160W light, and X-ray morphology. X-ray brightness was treated only as an emissivity proxy.

The inverse coupling used balanced entropic optimal transport,

\[
\min_{P\ge0}\sum_{ij}P_{ij}\|\mathbf y_j-\mathbf x_i\|^2
+\epsilon\sum_{ij}P_{ij}(\log P_{ij}-1),
\]

with exact source and destination marginals. A capacity-relaxed version replaced the forced source marginal with

\[
\sum_iP_{ij}=e_j,
\qquad
\sum_jP_{ij}\le q_{\rm cap}b_i.
\]

`q_cap` is only an attribution capacity. It is not a fitted gravity strength.

## Five-cluster result

The balanced reconstruction produced 50 source/target/entropy variants and 160 radial-angle controls. Its median system RMS path was 106.48 kpc. No cluster's real member angles beat the radial-shuffle null at `p <= 0.05`; the inverse therefore did not demonstrate a special angular routing pattern beyond the clusters' radial structure.

Capacity relaxation stabilized at `q_cap` about 2–4:

| `q_cap` | Median system RMS route | Median effective origins | Median top-origin flow |
|---:|---:|---:|---:|
| 1 | 106.48 kpc | 18.03 | 16.7% |
| 2 | 55.32 kpc | 8.01 | 30.8% |
| 4 | 53.40 kpc | 6.00 | 37.2% |
| 8 | 53.14 kpc | 5.54 | 38.6% |

The same dominant origin was recovered at `q_cap = 2, 4, 8` for all seven halos.

| Cluster / halo | Dominant luminous-origin distance from halo median |
|---|---:|
| MACS0329 / 1 | 0.84 kpc |
| MACS0329 / 2 | 23.61 kpc |
| MACS0429 / 1 | 3.68 kpc |
| MACS1115 / 1 | 12.53 kpc |
| MACS1931 / 1 | 6.17 kpc |
| RXJ2129 / 1 | 9.45 kpc |
| MACS1931 / 2 | 498.90 kpc with the original 300 kpc source aperture |

Thus six of seven standard halo components are already nearly co-located with luminous galaxies or subclusters. They cannot distinguish ordinary local mass association from redirected gravity. MACS1931's second, broad southern component is the only strongly nonlocal case in this sample.

Restoring all 120 published MACS1931 members out to 435 kpc reduced the `q_cap=4` RMS route from 321.14 to 269.21 kpc, a 16.2% improvement, but its dominant weighted origin was still more than 400 kpc from the halo median.

## Wide-field MACS1931 audit

The first DESI Legacy Surveys extraction proved unusable for membership: the patch contained only g-band measurements and no usable photo-z values. It was retained as a provenance and star-classification input, not used to claim a galaxy density.

The public CLASH/Subaru Suprime-Cam catalog then supplied 108,658 objects over a roughly 40 by 40 arcmin BVRIz field and BPZ redshifts. The frozen primary selection used a tight cluster-redshift window, at least four detected/observed filters, BPZ odds at least 0.5, and `17 <= IC <= 25.5`.

The southern halo median is at `(-212.0, -649.2)` kpc, 682.9 kpc from the cluster center. All 256 posterior samples lie within the catalog bounds.

### Frozen endpoint result

| Statistic | Result |
|---|---:|
| Tight photo-z candidates in full catalog | 9,415 |
| Nearest selected candidate | 96.95 kpc |
| Candidates within 200 kpc | 14 |
| Count-density ratio to median same-radius rotation | 1.429 |
| Count-density rotation p-value | 0.0278 |
| IC-luminosity density ratio | 1.070 |
| IC-luminosity rotation p-value | 0.3472 |

The frozen counterpart gate required `p <= 0.05`, a density ratio of at least 1.5, and at least five candidates within 200 kpc. The p-value and count requirement passed; the ratio did not. The endpoint therefore remains **no frozen significant baryonic counterpart**.

The spent robustness follow-up removed Legacy `PSF`/Gaia matches and required consistency with a color locus fitted to published members. The combined selection produced ratio 1.623 but `p = 0.0694`, again failing its predeclared joint gate. The strongest magnitude slice was `21 <= IC < 23`, with ratio 1.968 and `p = 0.0556`. This is suggestive of a modest intermediate/faint population, not a secure massive subgroup.

Using the wide photo-z source map shortened the descriptive inverse RMS route from 269.21 kpc to 182.76 kpc for count weights and 186.99 kpc for capped IC-luminosity weights. This shows that source-catalog completeness matters. It is not forward evidence because the halo destination was supplied to the inverse optimization.

## What has been learned

1. The backtracking problem can be made mathematically explicit and reproducible. We now have source-to-arrival couplings, uncertainty-aware arrival maps, capacity sensitivities, same-radius controls, and candidate three-dimensional arc visualizations.
2. Most fitted halo centers in this five-cluster set do not require a long gravity route: six of seven are within 24 kpc of a luminous structure.
3. The sole long-route case is not explained by a convincing luminous group in the current wide-field photometry. It has a mild count excess but no luminosity-weighted or robustness-gated detection.
4. Wider baryonic data reduce the inferred path length substantially. Any routing claim based on a truncated galaxy catalog would have been premature.
5. The inverse map does not identify the absolute response strength, an off-plane arc, or a forward destination rule.

## A decisive observation for a purely conservative model

If the effect only redirects existing baryonic gravity, it cannot create net gravitational flux. A positive apparent-halo region must be balanced by a deficit elsewhere when the complete system is enclosed:

\[
\int \left[\rho_{\rm route}(\mathbf y)-\rho_b(\mathbf y)\right]d^3y=0.
\]

This gives a sharp observational target: reconstruct a sufficiently wide convergence field and ask whether positive lensing residuals around the standard halo locations are accompanied by the predicted negative or under-strength regions. Weak-lensing mass-sheet degeneracy makes shear alone insufficient; magnification, strong lensing, and/or an external absolute calibration are needed. If the wide-field excess is positive everywhere with no compensation, then a strictly conservative redirection model cannot replace the missing mass. It would need a larger fundamental coupling, amplification, or a non-Poisson metric response.

## Next forward test

The next stage must no longer use a halo map to choose destinations on its test clusters.

1. Fit a small, normalized `K_theta(y|x,E_b)` family on a discovery subset using only baryonic features such as local surface density, tidal eigenvectors, concentration, gas morphology, and source luminosity.
2. Freeze all kernel parameters and the screening rule.
3. From baryons alone, predict a convergence/shear map for untouched clusters.
4. Score raw strong-lensing image positions, image multiplicity, weak shear/magnification, and cluster radial/tangential structure.
5. With the same constants, score SPARC galaxy rotation curves and Solar-System constraints.
6. Compare with baryons-only GR, fixed universal MOND/RAR, and standard dark-matter baselines. The fair advantage target is one universal setting, not a per-cluster endpoint fit.

Until this succeeds, the work is a disciplined map of where a new field would have to act—not evidence that such a field exists.

## Public data references

- [MAST CLASH archive](https://archive.stsci.edu/prepds/clash/)
- [MACS1931 Subaru catalog directory](https://archive.stsci.edu/missions/hlsp/clash/macs1931/catalogs/subaru/)
- [NOIRLab Legacy Surveys data access](https://beta.datalab.noirlab.edu/data/legacy-surveys)
- [Ehlert et al. MACS J1931 multiwavelength study](https://academic.oup.com/mnras/article/411/3/1641/972295)
