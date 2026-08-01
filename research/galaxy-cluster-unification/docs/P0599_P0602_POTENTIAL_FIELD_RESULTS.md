# P0599–P0602: bounded potential field results

## Outcome

The bounded baryonic-potential law is the strongest scalar bridge found in this
stage, but it is not yet a successful raw-lensing law.

The formula frozen by five-fold whole-object cross-validation was

\[
g=\operatorname{RAR}(g_b,a_0)
\left[1+3S_4H(C)
\frac{\chi^4}{\chi^4+(10^{-6})^4}\right],
\qquad
\chi=\frac{\Phi_b}{c^2},\quad C=\frac{R_{50}}{R_{80}}.
\]

Here `RAR` is the fixed galaxy acceleration relation, `S4` turns the added
channel off in high-acceleration stellar environments, `H(C)` is a bounded
radial-shape gate, and `chi` is the dimensionless depth of the potential made
by the observed baryons. The added response saturates at a factor of
`3*S4*H(C)`; it cannot grow without bound.

The same candidate was selected independently in all five folds. It keeps
galaxy rotation close to fixed RAR and supplies the missing radial cluster
amplitude, but its apparent success on deprojected CLASH profiles does not
carry through to raw multiple-image positions.

| Test | Fixed RAR or reference | Frozen P0599 | Result |
|---|---:|---:|---|
| 131 SPARC galaxies, equal-object outer RMSE | 10.716 km/s | 10.883 km/s | 1.55% worse than RAR |
| 20 CLASH clusters, equal-object radial RMSE | 0.509 dex | 0.120 dex | 76.5% lower error; 20/20 clusters improve |
| 34 intermediate-potential BCGs | 0.285 dex fixed RAR | 0.203 dex | improves RAR but misses bias/direct-subset gates |
| RX J2129, seven held-out raw images | fixed RAR recovers 3/7 roots | 1.809 arcsec, 7/7 roots | fails the predeclared 0.5-arcsec gate |

For context, the earlier locked project candidate scores 1.064 arcsec on the
same spent RX J2129 split, the earlier P0554 scalar law scores 1.245 arcsec, a
deliberately compact one-halo control scores 2.536 arcsec, and the published
multi-halo reference is about 0.29 arcsec. The compact one-halo comparison is
not representative of the full flexibility of modern dark-matter lens models.

## What each stage established

### P0599: potential depth controls absolute amplitude

The test covered 480 combinations of spatial base, carrier, maximum amplitude,
potential threshold, and transition power. Five folds held out complete
galaxies and complete clusters. Every fold selected the same local-profile,
potential-times-shape formula shown above.

The one-at-a-time response spans identify the variables that matter most for
the CLASH radial score:

| Ingredient | Cluster RMSE response span |
|---|---:|
| Carrier choice | 0.161 dex |
| Maximum amplitude | 0.135 dex |
| Potential threshold | 0.094 dex |
| Potential power | 0.091 dex |
| Conservative spatial diffusion | 0.024 dex |

Thus potential depth is a useful class-spanning amplitude coordinate, while
the earlier one-dimensional spatial diffusion is not what fixes absolute
cluster amplitude. The Solar source gate is about `4.9e-50`, so the added term
is numerically absent in the Solar proxy; this is screening evidence, not a
full PPN or Cassini derivation.

### P0600: the galaxy–cluster gap remains unresolved

The SPARC potential range ends at `6.74e-7`; the CLASH range begins at
`1.80e-6`. The selected `1e-6` transition lies inside that observational gap,
so it could be an object-class separator rather than new physics.

Thirty-four BCGs partly bridge the gap. The frozen primary bracket reaches
0.203 dex RMS and is better than fixed RAR in 99.4% of bootstrap resamples, but
it narrowly misses the 0.20-dex primary gate, retains a +0.161-dex mean bias,
and scores 0.231 dex on the 11 direct objects. Radial host shape is the largest
uncertainty, followed by host potential and source screening. Resolved gas,
stars, and satellite mass profiles for the same BCG systems are the required
observation.

### P0601: raw images expose the missing structure

P0599 was frozen before replay on 22 RX J2129 image positions. Fifteen images
fit six common lens-geometry nuisances; seven were held out. No gravity,
amplitude, transition, photon, or slip parameter was fitted to this cluster.

The candidate recovers every exact held-out root with no geometry parameter at
a bound, but scores 1.809 arcsec. Across the 15.8–76.5 kpc image annulus, its
added amplitude fraction changes only from 2.21870 to 2.21828. It is therefore
almost a constant rescaling where the raw data needs a structured deflection
field. This explains how a formula can fit a spherically deprojected
acceleration profile yet miss the two-dimensional image configuration.

### P0602: small radial changes do not repair the miss

Seventeen one-at-a-time variants changed amplitude, potential threshold,
transition power, bounded potential-path and mass-growth carriers, or a bounded
radial power. The geometry was locally refitted from the P0601 solution using
training images only. The seven comparison images were already spent and are
reported only as a post-hoc diagnostic.

Lower transition powers reduce training RMS from 0.495 to about 0.443 arcsec,
but worsen the spent held-out RMS to 2.07–2.52 arcsec. Lowering the potential
threshold gives the best post-hoc held-out value, 1.647 arcsec, while worsening
training. The path, potential-gradient, and several radial-power variants lose
exact roots. No tested radial carrier improves both sides of the split.

This is evidence against continuing to tune a spherical multiplier on RX
J2129. It does not rule out field routing. It says that a routing theory must
predict two-dimensional direction and structure.

## Current physical interpretation

The most economical surviving picture is:

1. Baryonic matter sources the ordinary field.
2. A high-acceleration source gate suppresses any nonlocal response in stars
   and the Solar System.
3. Baryonic potential depth controls how much response can reside outside the
   immediately local Newtonian pattern.
4. A scalar radial law is insufficient to say where that response reappears.
5. The next operator must map the baryonic source distribution into a vector
   or tensor deflection field with curl-free/conservative constraints and must
   be tested on raw image positions from a different cluster.

The inverse gravity-flow work already provides the appropriate observational
question: given baryonic source pixels and a conventional lensing-excess map,
find conservative transport paths from the former to the latter, then ask
whether one source-blind kernel predicts an untouched cluster. What remains
missing is a local field equation whose Green function generates those paths,
rather than an optimal-transport backtracking algorithm imposed after the
fact.

## Claim boundary

- CLASH radial totals are NFW-deprojected lens reconstructions, not raw
  theory-neutral observables.
- RX J2129 is a single, now-spent cluster pilot with a literature baryon
  profile and approximate pseudo-elliptical geometry.
- P0599 contains empirical RAR and phenomenological gates; it is not derived
  from a covariant action.
- Beating the compact one-halo diagnostic does not beat dark matter.
- The results identify useful observables and failed formula classes. They do
  not establish negative gravity, field-line return, or a replacement for
  dark matter or MOND.

Reproduce with:

```powershell
python scripts/run_p0599_bounded_potential_amplitude.py
python scripts/run_p0600_bcg_potential_gap.py
python scripts/run_p0601_frozen_potential_raw_lensing.py
python scripts/run_p0602_raw_radial_structure_diagnostic.py
python -m pytest tests/test_p0599_bounded_potential_amplitude_results.py tests/test_p0600_bcg_potential_gap_results.py tests/test_p0601_frozen_potential_raw_lensing_results.py tests/test_p0602_raw_radial_structure_diagnostic_results.py -q
```
