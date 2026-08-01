# RX J2129 raw strong-lensing pilot

## Bottom line

The locked universal candidate produces all seven held-out RX J2129 image
roots, with an exact image-plane radial RMS of **1.064 arcsec**. It is decisively
better in this pilot than the fixed simple-MOND closure, which recovers only
three of seven held-out roots, and better than the deliberately compact
GR-plus-one-halo control at **2.536 arcsec**. It nevertheless **fails** the
preregistered 0.5-arcsec absolute gate. This is a useful survival result, not a
successful or publication-grade lens model.

The published conventional RX J2129 analysis reaches **0.29 arcsec** after an
all-image fit with 71 halos, including a cluster-scale dark halo and individually
optimized galaxy halos. That number is not a held-out score on our catalog, so
it is a descriptive standard-of-practice reference rather than a like-for-like
likelihood ratio. The candidate's separately labeled all-image RMS is 0.618
arcsec, about 2.13 times the published reference.

## What “raw lensing” means here

The scored lens-side observations are 22 sky positions in seven independently
identified, MUSE-spectroscopic source families from the Caminha et al. catalog.
The likelihood does not read an NFW mass, convergence profile, or a published
lensing residual. One image from every family was frozen as held out; the other
15 images determine seven source positions and the permitted lens geometry.

“Raw” applies only to the lensing observables. The baryonic acceleration profile
is still a literature reconstruction from Chandra gas, HST BCG light, and an
average cluster-galaxy relation. It is independent of these image residuals, but
it is not a native joint likelihood for the same photons and lacks a complete
component covariance.

The word “held out” also applies specifically to image coordinates. The locked
candidate triplet was previously calibrated on the 20-cluster derived CLASH
acceleration bridge, which included RX J2129. These 22 Caminha image residuals
were not used in that calibration, but RX J2129 is not an untouched cluster.
This pilot therefore tests whether a previously calibrated radial field
translates into raw image positions; it is not an independent object-level test
of the universal constants.

## How conventional strong-lensing analyses work

The standard workflow is:

1. identify repeated images in deep imaging and secure source redshifts with
   spectroscopy;
2. specify projected mass components for the cluster halo, BCG, member
   galaxies, gas, and line-of-sight structure;
3. solve the lens equation, `beta = theta - alpha(theta, z_source)`;
4. vary mass, geometry, and nuisance parameters in an image-plane likelihood;
5. report image-position RMS/chi-square and derive mass, magnification, and
   critical-curve maps, with alternative component choices used as systematic
   checks.

Caminha et al. deliberately used spectroscopic families to reduce
misidentification and redshift degeneracies and made the models public. Modern
packages such as Lenstronomy implement this as forward modeling with analytic
lens and light components. The later RX J2129 analysis illustrates the usual
flexibility: eight systems, 71 halos, and 0.29-arcsec image-plane RMS.

Primary references:

- [Caminha et al. (2019), eight spectroscopic CLASH lens models](https://arxiv.org/abs/1903.05103)
- [Jauzac et al. (2021), RX J2129 model and 0.29-arcsec result](https://arxiv.org/abs/2006.10700)
- [Birrer & Amara (2018), Lenstronomy forward-modeling framework](https://arxiv.org/abs/1803.09746)

## Frozen theory-to-photon calculation

For every radial acceleration law, the physical deflection was calculated with
the zero-slip, same-effective-potential closure

$$
\hat\alpha(b)=\frac{4b}{c^2}\int_0^\infty
\frac{g(\sqrt{b^2+z^2})}{\sqrt{b^2+z^2}}\,dz
=\frac{4b}{c^2}\int_0^\infty g(b\cosh t)\,dt,
$$

then reduced by $D_{ds}/D_s$. The candidate had no fitted cluster acceleration
scale, lensing multiplier, gravitational slip, or lensing amplitude. Its five
gravity settings were copied unchanged from the galaxy/cluster bridge. All
fixed radial models received the same six object-specific structural nuisances:
axis ratio, position angle, two center coordinates, and two external-shear
components. Seven two-coordinate source positions were profiled from training
images. Exact nonlinear image roots, not source-plane residuals, were used for
the final score.

The compact dark-halo control added one non-singular isothermal ellipsoid to
the fixed spherical baryons. It has six object-specific halo parameters plus
two shear components. It is intentionally much less expressive than the
published 71-halo model and is not a stand-in for the best dark-matter analysis.

## Predictive results

| Frozen model | Held-out roots | Held-out radial RMS | Held-out reduced chi-square | All-image descriptive RMS |
|---|---:|---:|---:|---:|
| Baryons in GR | 7/7 | 17.925 arcsec | 4170.3 | 10.233 arcsec |
| Fixed simple MOND, $a_0=1.2\times10^{-10}$ m/s² | 3/7 | undefined | undefined | undefined (18/22 roots) |
| **Locked universal candidate** | **7/7** | **1.064 arcsec** | **14.70** | **0.618 arcsec** |
| Cluster-retuned RAR diagnostic | 7/7 | 2.816 arcsec | 102.95 | 1.113 arcsec |
| GR plus compact cluster halo | 7/7 | 2.536 arcsec | 83.48 | 0.624 arcsec |

The candidate's held-out RMS is 58.0% below the compact one-halo control and
62.2% below the non-universal cluster-retuned RAR diagnostic. A numerical RMS
ratio to fixed MOND is intentionally not reported: an RMS is undefined when
four predicted image roots do not exist or cannot be recovered. The correct
comparison is 7/7 roots versus 3/7.

The candidate residual is not uniformly good. Images 1c through 5c are within
0.405--0.842 arcsec, while 6c and 7c miss by 1.698 and 1.586 arcsec. This
localization is consistent with missing angular or member-galaxy structure, but
the pilot cannot distinguish that possibility from an incorrect radial field.

## Frozen sensitivity results

| Input change | Candidate held-out RMS | Roots |
|---|---:|---:|
| Baseline | 1.064 arcsec | 7/7 |
| Baryonic acceleration -0.1 dex | 1.668 arcsec | 7/7 |
| Baryonic acceleration +0.1 dex | 11.248 arcsec | 7/7 |
| Environmental density -0.3 dex | 1.910 arcsec | 7/7 |
| Environmental density +0.3 dex | 1.126 arcsec | 7/7 |

Both MOND baryonic-profile shifts still fail exact held-out root recovery. The
candidate's strong asymmetric response to the baryonic normalization is a major
warning: a native baryonic likelihood and covariance are required before its
apparent advantage can be treated as robust.

## Preregistered gate audit

| Gate | Outcome |
|---|---|
| All seven candidate held-out roots | pass |
| Candidate held-out RMS no more than 0.5 arcsec | **fail: 1.064 arcsec** |
| Better than fixed simple MOND | pass by root recovery |
| No worse than 1.25 times compact-halo RMS | pass: ratio 0.420 |
| No fitted lens amplitude or slip | pass |
| No candidate geometry parameter at a bound | pass |
| Overall advance | **fail** |

This means the candidate has not yet “beaten MOND” under the project's full
works-on-galaxies-and-lensing criterion: it wins the relative raw-lensing pilot,
but it does not fit the lens accurately enough. It also has not beaten dark
matter. It outpredicts one compact held-out halo control, while a substantially
more flexible published conventional model fits all images much more closely.

## What this test establishes—and what it cannot

It establishes that the unchanged candidate amplitude is in the right broad
strong-lensing regime for one cluster, predicts withheld image positions much
better than the declared fixed-MOND closure, and does so without a cluster-only
force scale. It also falsifies the claim that this particular implementation is
already adequate at 0.5 arcsec.

It cannot establish a void origin, a relativistic theory, gravitational slip,
population-level universality, or superiority to state-of-the-art dark-matter
models. It does not include a covariant action, native baryon-image joint
likelihood, correlated astrometric systematics, member-by-member baryonic
perturbers, missing-image selection, weak lensing, magnifications, or time
delays. Because RX J2129 contributed to the earlier derived-field bridge, only a
new cluster can provide a strictly independent test of the exact locked
triplet.

## Required next stage

Formula tuning on these seven held-out images is prohibited. The next honest
test is structural and then external:

1. construct the same-system BCG/ICL, hot-gas, and member-galaxy baryon map with
   covariance, independent of lens residuals;
2. give every gravity law the same photometry-tied angular perturbers and use an
   exact image-plane likelihood during fitting;
3. freeze the candidate settings and photon closure again before looking at a
   new cluster;
4. repeat on several spectroscopic CLASH clusters and compare held-out families
   against both a compact halo and a conventional member-halo model;
5. derive the photon law from a covariant action before making a theory claim.

## Reproduction

```powershell
python scripts/run_rxj2129_raw_theory_lensing.py
pytest -q tests/test_raw_lensing.py tests/test_rxj2129_raw_theory_lensing_results.py
```

The frozen protocol is
`configs/rxj2129_raw_theory_lensing_protocol.json`. Machine-readable outputs are
under `results/rxj2129_raw_theory_lensing/`.
