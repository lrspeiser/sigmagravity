# P0554 caustic-margin and image-multiplicity results

## Outcome

The earlier **17/18 versus 18/18** raw-lensing result was too easy to read as
“one formula cannot make the observed image and another can.” A frozen global
root search gives a more precise result:

- all 18 formulas produce at least the three roots required by the three
  observed MACS1931 family-2 images;
- the 11 formulas previously called successful produce **five** roots;
- the seven formulas previously called failed produce **three** roots; and
- the extra two roots in every five-root formula lie within 15 arcseconds of
  observed image 2c.

Thus the old one-seed status was not merely a numerical accident, but neither
was it demonstrated recovery of an otherwise absent required image. It marks a
real **3-to-5-root caustic bifurcation** near 2c. No formula is promoted.

## What was calculated

For a lens mapping

$$
\boldsymbol\beta(\boldsymbol\theta)
=\boldsymbol\theta-\boldsymbol\alpha(\boldsymbol\theta),
$$

the local lens Jacobian is

$$
A(\boldsymbol\theta)
=\frac{\partial\boldsymbol\beta}{\partial\boldsymbol\theta}
=I-\frac{\partial\boldsymbol\alpha}{\partial\boldsymbol\theta}.
$$

A critical curve occurs where $\det A=0$. Mapping that curve through the lens
equation produces a caustic in the source plane. When the fitted source crosses
a caustic, image roots are born or destroyed in pairs. We therefore measured:

- the minimum singular value of $A$ at the observed coordinate and at a solved
  image root;
- distance from that root to the nearest critical curve;
- distance from the fitted source to the corresponding caustic; and
- the number and locations of all unique roots found by a frozen global seed
  grid inside a 120-arcsecond aperture.

No gravity parameter or lens geometry parameter was refitted. The calculation
reused the archived eight-start geometries and training-profiled source
positions from the 18-formula interaction experiment.

## Coverage

| Item | Coverage |
|---|---:|
| Formula variants | 18 |
| Raw clusters reconstructed | 5 |
| Held-out formula-image diagnostics | 324 |
| Prior one-seed failures | 7 |
| Global MACS1931 family-2 searches | 18 |
| Unique global roots | 76 |
| Observed family-2 images | 3 |
| Reused SPARC control | 131 galaxies |
| Reused CLASH control | 20 systems |

## The sharp result

| Old image-2c status | Formulas | Global roots | Roots within 15 arcsec of 2c | Nearest root to 2c | Source-to-caustic margin |
|---|---:|---:|---:|---:|---:|
| One-seed success | 11 | 5 in every formula | 2 in every formula | 1.951--5.883 arcsec | 0.129--0.804 arcsec |
| One-seed failure | 7 | 3 in every formula | 0 in every formula | 24.939--26.042 arcsec | 4.853--5.374 arcsec |

All seven old failures can converge to some valid root under a local multistart
search, but that alternative branch is about 25 arcseconds from image 2c. The
five-root formulas instead have two roots near 2c. Their two nearest roots are
separated by 1.782--10.859 arcseconds.

For the combined parent, the nearby roots are approximately
$(-23.105,22.491)$ and $(-21.795,23.698)$ arcseconds, separated by 1.782
arcseconds. This is not automatically a success: the additional root is a
potential companion image that the observations must contain or render
undetectable.

## Which quantity predicts the transition?

Within the 18 variants of this one spent image, source-to-caustic margin,
image-to-critical-curve distance, solved-root singular value, nearest-root
distance, and globally assigned distance all have directed AUC 1.000 for the
three-root versus five-root regime. The source-to-caustic margin has Spearman
$\rho=-0.846$ with the old binary status.

The local Jacobian at the catalogued image coordinate is much weaker:

| Diagnostic at observed coordinate | Directed AUC |
|---|---:|
| Absolute determinant | 0.545 |
| Minimum singular value | 0.377 |
| Linearized Newton distance | 0.883 |

The useful signal is therefore not simply “gravity is strong at this point.”
It is where the complete lens mapping places an image branch and its caustic.
That is consistent with the proposed idea that baryonic gravity can be
redirected through a spatial structure: topology and accumulated geometry
matter more than a local scalar multiplier.

## What this proves—and what it does not

This result proves for the frozen simplified MACS1931 model that small route and
photon-response changes can move the source across a caustic and create an
extra image pair. It also corrects the earlier interpretation that one model
had no globally sufficient multiplicity: all tested formulas retain the three
required roots.

It does **not** prove that the extra pair is physically present, that the route
is a universal law, or that this model rivals a dark-matter reconstruction.
The clusters and formulas are spent exploratory evidence; the global grid is
finite; 72 of 90 archived geometry fits touch a nuisance boundary; and the
lens omits registered gas and diffuse intracluster-light maps.

That decisive follow-up is now complete. All pairs have opposite parity and a
predicted companion/anchor brightness ratio of 0.917--1.251. No published
family-2 companion is present. After registering each anchor onto observed
image 2c, ten formulas point to clean F160W blank sky and the eleventh is
contaminated/inconclusive rather than confirmed. The extra-pair realization is
therefore disfavored. See
[`P0554_MACS1931_COMPANION_AUDIT_RESULTS.md`](P0554_MACS1931_COMPANION_AUDIT_RESULTS.md).
The next fair test is to repeat frozen multiplicity audits on other clusters
and source families rather than tuning MACS1931 again.

## Reproduction

```powershell
python scripts/run_p0554_caustic_margin.py
python scripts/run_p0554_caustic_margin.py --postprocess-only
python -m pytest tests/test_p0554_caustic_margin.py -q
```

The second command regenerates summaries and the figure from the frozen primary
tables without repeating the expensive lens reconstruction. Machine-readable
outputs are in `results/p0554_caustic_margin/`.
