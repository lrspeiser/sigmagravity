# One-parameter multi-cluster lens result

## Bottom line

The strongest member of the frozen eight-family search is a baryon-normalized
isothermal tail,

\[
g(r)=g_{\rm bar}(r)+\lambda\,g_{\rm bar}(200\,{\rm kpc})
\frac{200\,{\rm kpc}}{r},\qquad \lambda=9.
\]

The same value of `lambda` was used for every cluster. No cluster-specific
force scale, transition radius, or lensing multiplier was allowed. The law was
chosen with MACS0329 and MACS0429 and then locked before scoring six withheld
images in MACS1115 and MACS1931.

On that two-cluster replay holdout it obtains **9.423 arcsec** equal-cluster RMS,
versus 25.199 for baryons-only GR, 25.636 for simple MOND, 25.673 for fixed RAR,
and 9.989 for the compact-halo comparator. Thus it reduces the baryons-only
error by 62.6% and is 5.7% lower than the deliberately limited compact-halo
aggregate.

This is a useful phenomenological lead, but it does **not** pass the frozen
advance gates. Its error is far above the predeclared 2-arcsec target, its
pooled chi-square is worse than the compact halo, and the post-lock stress test
on RXJ2129 is poor. It is not evidence that dark matter has been beaten.

## What the one parameter means

- `g_bar(r)` is the acceleration calculated from the measured baryonic radial
  profile.
- `g_bar(200 kpc)` lets the strength scale with each cluster's measured baryons;
  it is data-derived, not a fitted cluster parameter.
- `200 kpc/r` supplies a long-range `1/r` acceleration tail. In Newtonian
  language this is equivalent to an effective enclosed mass that grows roughly
  linearly with radius, or an isothermal-like effective density proportional to
  `1/r^2`.
- `lambda=9` is the only fitted gravity parameter. At 200 kpc the extra term is
  nine times the baryonic acceleration at that reference point, so the total is
  ten times the baryonic value there.

The implementation also holds `G`, `c`, `a0`, the 200-kpc reference scale, the
3-Mpc integration cutoff, and zero gravitational slip fixed. Cluster geometry
(center, ellipticity, angle, shear) and source positions are nuisance variables
fit only from each cluster's training images. Calling this a one-parameter law
means one fitted *gravity* parameter, not that the calculation contains no
fixed physical or numerical constants.

## Frozen design

| Role | Clusters | Use |
|---|---|---|
| Development | MACS0329, MACS0429 | Choose one of 8 formula families and one shared value using within-family held-out images |
| Validation | MACS1115, MACS1931 | Fit geometry on training images; score the already locked law on withheld images |
| Stress only | RXJ1347, RXJ2129 | Apply the locked law after validation; RXJ1347 has no eligible within-family holdout |

All of these catalogs appeared in earlier project analyses. The split was frozen
before this one-parameter search, but it is a **replay holdout**, not untouched
external validation.

## Selection and validation

| Stage or model | Shared value | Held-out images | Equal-cluster RMS | All held-out roots? |
|---|---:|---:|---:|---:|
| Development winner | 9 | 5 | 13.120 arcsec | Yes |
| Locked validation | 9 | 6 | 9.423 arcsec | Yes |
| Baryons-only GR | fixed | 6 | 25.199 arcsec | Yes |
| Simple MOND | fixed | 6 | 25.636 arcsec | Yes |
| Fixed RAR, zero slip | fixed | 6 | 25.673 arcsec | Yes |
| GR + compact cluster halo | object-specific halo | 6 | 9.989 arcsec | Yes |

The apparent aggregate win over the compact halo needs care:

| Validation cluster | New law | Baryons GR | Simple MOND | Compact halo |
|---|---:|---:|---:|---:|
| MACS1115 | 9.805 | 29.931 | 29.449 | 14.057 |
| MACS1931 | 9.024 | 19.343 | 21.148 | 1.401 |

The new law is more balanced, so it wins the predeclared equal-cluster RMS. The
compact halo is dramatically better on MACS1931. When residuals are pooled by
image rather than giving each cluster equal weight, the new law has chi-square
2202 versus 1615 for the compact halo (36% worse). This is why the result cannot
be summarized honestly as “better than dark matter.” A modern many-component
cluster model is also much more flexible than this compact-halo control.

## Frozen gate audit

| Gate | Requirement | Result |
|---|---|---|
| Withheld roots | Every root converges | Pass |
| Absolute accuracy | RMS <= 2 arcsec | **Fail: 9.423** |
| Halo proximity | RMS <= 1.25 times compact halo | Pass: 0.943 times |
| Per-cluster transfer | Beats baryons in both validation clusters | Pass |
| Search stability | Selected value is not a grid boundary | Pass |

Overall verdict: **does not advance**, solely because it misses the absolute
accuracy gate. Note separately that a MACS1931 training root did not converge;
the formal root gate was defined on withheld images, but this is another warning
against a strong claim.

## Post-lock stress result

| Cluster | New law | Baryons GR | Compact halo | Interpretation |
|---|---:|---:|---:|---|
| RXJ1347 training | 5.028 | 14.502 | 0.805 | Descriptive only; no eligible holdout |
| RXJ2129 heldout | 13.979 | 17.908 | 2.521 | Improves baryons but is 5.5 times the halo error; one training root is lost |

The RXJ2129 stress result shows that the formula does not generalize at
dark-halo quality across the currently available raw-lens clusters.

## Physical and statistical limitations

1. The `1/r` term is isothermal-halo phenomenology written as a modification of
   acceleration. A covariant field equation, conservation law, and photon
   propagation rule have not been derived.
2. The formula becomes very large at small radius unless a physically derived
   inner completion suppresses it. The present cluster-only test should not be
   extrapolated to galaxies or the Solar System.
3. The baryonic input is a sparse spherical profile. Pseudo-ellipticity and
   external shear do not replace resolved gas, BCG, intracluster light, and
   member-galaxy mass maps.
4. Six validation images in two previously seen clusters are too few for an
   external claim. The next credible test needs new clusters, frozen beforehand,
   with resolved baryonic maps and no formula revision after disclosure.

## Reproduce

```powershell
python -m pytest tests/test_one_parameter_lens.py -q
python scripts/run_one_parameter_multicluster_lens.py
python scripts/run_one_parameter_multicluster_lens_stress.py
python -m pytest tests/test_one_parameter_multicluster_lens_results.py -q
```

Machine-readable evidence is in
`results/one_parameter_multicluster_lens/report.json`; the full coarse and
refined grids, predictions, fitted geometry, radial profiles, and diagnostic
figure are in the same directory.
