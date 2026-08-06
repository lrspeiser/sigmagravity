# Sigma V19DA Chandra ICM velocity preregistration

## Outcome

The source-only preregistration and archive preflight pass. V19DA supplies a
second direct-velocity route after the public XRISM/Resolve-v1 regional NXB
likelihood failed. It uses Chandra Fe-K line centroids to measure signed
line-of-sight ICM motion, but it does not yet open a source spectrum or make a
gravity claim.

The complete response archive remains usable:

| Audit | Result |
|---|---:|
| Validated response cells | 5,082 |
| Source/background/ARF/RMF products checked by name and frozen byte length | 20,328 |
| Missing products | 0 |
| Wrong-size products | 0 |
| PHA, ARF or RMF scientific arrays opened by this preflight | 0 |

The report is
`results/sigma_v19da_chandra_icm_velocity_preflight/report.json`; its SHA-256
is `7d0417b4af630d00ab0bbee1573b707fcafc6bc1882c6630c1050e39a3ba3c0a`.
The exact merged-bin membership is in `frozen_region_groups.csv` with SHA-256
`335c650e58658745212c5340c1f74c8c72356286cff37171fb16b497b2b47ff2`.

## One source-only region rule

The existing V19M bins are too small for robust Fe-K work individually. Their
median broad-band net count is about 1,700. V19DA therefore merges adjacent
admitted bins with one rule for both clusters:

1. select the connected component with the smallest net count below target;
2. merge it with the neighbor sharing the longest boundary;
3. break ties by closest mean broad-band surface brightness and then root ID;
4. repeat until every component reaches the target.

Only the frozen 0.5--7 keV bin map, broad counts and pixel adjacency enter.
No hardness ratio, Fe-line count, centroid, temperature, redshift, shock map,
lensing map, halo result or gravity residual is allowed.

| Cluster | Admitted V19M bins | 8,000-count primary regions | 10,000-count robustness regions |
|---|---:|---:|---:|
| Bullet | 366 | 43 | 35 |
| Abell 2146 | 128 | 16 | 12 |

Every admitted bin appears exactly once in each branch, and every merged
region is connected. The 8,000 and 10,000 thresholds were selected during
disclosed source-only method development. They are detector-statistics
settings, not gravity parameters.

## Frozen spectral method

The method follows the central safeguards in Liu et al.'s Bullet development
paper and later cluster application:

- regions are constructed from surface brightness, never spectral outcome;
- the primary redshift likelihood uses ungrouped 2--10 keV spectra;
- two thermal components share one redshift, while both temperatures,
  abundances and normalizations remain free;
- a global redshift profile is searched rather than trusting one local fit;
- unresolved thermal mixtures and gain uncertainty are propagated separately.

The primary modern model is `tbabs*(apec+apec)` using CIAO 4.18 / XSPEC
12.14.0k. The published `tbabs*(mekal+mekal)` choice is a mandatory robustness
branch. Temperatures span 3.5--27 keV and abundances 0.1--1.6 solar. A frozen
4,096-point Sobol design samples the four-dimensional thermal mixture; accepted
full-band mixtures are projected back through the hard-band redshift fit. The
statistical, thermal-mixture and gain intervals are combined in quadrature.

The redshift search covers the optical cluster value plus or minus 0.05. It
uses a 0.001 coarse scan and a 0.0001 fine scan. Regions with a competing
minimum inside `Delta statistic = 6.63`, a boundary-truncated interval, failed
posterior-predictive goodness, or excessive uncertainty are reported and
excluded by those rules only.

The gain audit fits the Ni K-alpha and Au L-alpha background lines per ObsID.
The primary analysis propagates the fitted gain covariance without shifting
source data; a gain-corrected rerun tests sign-topology stability.

Primary sources:

- [Bullet Chandra bulk-motion method](https://arxiv.org/abs/1508.04879)
- [Later multi-cluster Chandra application](https://arxiv.org/abs/1602.07704)
- [CIAO 4.18 APEC model documentation](https://cxc.cfa.harvard.edu/sherpa/ahelp/xsapec.html)

## Evidence split and claim boundary

The Bullet cluster is development-only. Its qualitative published outcome is
known, so reproducing it validates our implementation rather than new physics.
The primary observation set contains the same nine VFAINT ObsIDs used by the
paper; adding FAINT ObsID 554 is a frozen robustness branch.

Abell 2146 remains internally target-sealed until the complete Bullet report,
code and configuration are hashed and the development gates pass. However, a
2016 paper already applied this broad method to Abell 2146. It is therefore an
internally sealed transfer check, **not a pristine literature-blind holdout**.
Its primary branch uses the eight published VFAINT ObsIDs; the two additional
already-processed observations form a robustness branch.

A Bullet pass requires a reproducible nonconstant field and stable velocity
signs across APEC/MEKAL, both region scales, observation sets and gain
treatments. Abell 2146 access remains closed until then. A later Abell pass
would admit signed gas current as a candidate source for covariant action
placement; it would not validate Sigma Gravity. A failed transfer forbids
using Chandra redshift noise as a source and sends the project to a prospective
calorimeter, resolved-kSZ or newly released region-aware likelihood.

## What this changes about the formula search

V19DA directly tests whether the missing cluster variable can be a continuous
time-odd part of the baryonic stress-energy tensor, rather than a label such as
"merger" or "cluster." If a stable signed field exists, the next action can
couple covariantly to baryonic current or anisotropic stress and predict an
oriented metric Hessian. If it does not, the time-odd/current mechanism loses
its observational basis and should not be added merely because it can move a
lensing caustic.

No lensing coordinate, halo map, action, gravity formula or gravity constant
was opened or changed in V19DA.
