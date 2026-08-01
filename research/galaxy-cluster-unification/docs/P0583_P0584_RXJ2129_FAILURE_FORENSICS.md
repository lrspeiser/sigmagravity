# P0583-P0584 RX J2129 transfer and failure forensics

## Result in plain language

The smooth K0338 endpoint formula failed decisively on RX J2129. It retained
all seven held-out lens-image roots after a full nuisance refit, but moved them
far from their observed locations: RMS rose from 1.256 arcsec for the scalar
parent to 14.130 arcsec. The fit tried to compensate by pushing center and both
external-shear components to their allowed bounds.

Turning the endpoint field down does not turn this into a substantial result.
At fixed scalar geometry, the best refined amplitude was only 0.005, where a
smooth no-overshoot route reduced RMS to 1.243 arcsec, a 1.05% descriptive
gain. Doubling that amplitude either erased the gain or lost an image root.

The useful lesson is structural: fixed-distance endpoint return can carry
inner baryonic sources through the center, which is physically unnatural and
measurably harmful. Preventing overshoot is the better rule, but it does not
solve the more important problem that one global centroid-return pattern does
not align with lens residuals across clusters.

## Why RX J2129 was informative

RX J2129 was absent from:

- the ten-cluster inverse driver that set K0338's extent relation;
- the two P0579 raw clusters; and
- the four P0581-P0582 exact-root clusters.

Its raw positions have been used by other project branches, so P0583 is new to
this formula rather than globally fresh astronomy. The test contains 51 hard
photometric member sources, 15 training images, and seven held-out images.
No gravity coefficient was selected on RX J2129. Each main variant refitted
only the same six ordinary lens-geometry nuisances and training-family source
positions.

## P0583 frozen exact-root transfer

The tested directional field was

\[
\boldsymbol\alpha=\boldsymbol\alpha_{P0554}
+\boldsymbol\alpha_{K0338},
\]

where the K0338 carrier is the P0554 excess above local baryonic gravity, the
return length is `0.36 R80`, width is `0.23 R80`, the concentration gate is
fixed, and the smooth candidate uses

\[
T(x)=20\tanh(x/20).
\]

| Variant | Held-out roots | Held-out RMS | Change from scalar | Geometry at boundary |
|---|---:|---:|---:|---:|
| Scalar P0554 | 7/7 | 1.256 arcsec | reference | no |
| K0338 hard-20 | 7/7 | 15.232 arcsec | 1113% worse | yes |
| K0338 hard-5 | 7/7 | 14.135 arcsec | 1025% worse | yes |
| K0338 tanh-20 | 7/7 | 14.130 arcsec | 1025% worse | yes |

Tanh-20 and hard-5 are nearly indistinguishable here. Smooth saturation fixed
P0582 root stability but did not fix angular placement. The tanh unit field had
RMS correction 6.246 arcsec and maximum correction 53.08 arcsec. Its
baryon-derived route fraction was 0.289, return distance 66.35 kpc, and width
42.39 kpc.

Five of the seven held-out images had endpoint residuals above 5 arcsec; images
3d and 7c exceeded 20 arcsec. This is not one outlier or one missing-root
penalty.

## P0583B signed-amplitude forensics

The P0583 scalar geometry and source positions were frozen. The tanh-20 field
was then multiplied by signed `epsilon` from -0.2 to 1.0.

| `epsilon` | Roots | RMS | Interpretation |
|---:|---:|---:|---|
| -0.050 | 7/7 | 1.884 arcsec | reversed field is worse |
| -0.025 | 7/7 | 1.610 arcsec | reversed field is worse |
| 0 | 7/7 | 1.256 arcsec | best on the original coarse grid |
| +0.025 | 6/7 | undefined | image 2c root lost |
| +0.10 | 5/7 | undefined | two roots lost |
| +1.00 | 3/7 | undefined | four roots lost |

A central signed perturbation was available for six images. Positive endpoint
response improved only two to first order and worsened four; image 2c failed to
converge already at `epsilon=0.025`. Neither the original sign nor its reversal
provides a broadly aligned correction vector.

## P0584 no-overshoot route law

The original rule gives every source the same travel length `L`:

\[
\ell_i=L.
\]

If a source is closer than `L` to the centroid, it crosses the centroid and
returns on the opposite side. In RX J2129, 15 of 51 sources representing 47.8%
of the catalog weight did this.

Three non-crossing alternatives were tested:

\[
\ell_i=\min(L,d_i),
\qquad
\ell_i=L\tanh(d_i/L),
\qquad
\ell_i={Ld_i\over L+d_i}.
\]

The tanh and rational laws are smooth, remain below `d_i`, approach the center
for inner sources, and saturate at a finite travel scale for distant sources.

| Travel law | Best complete `epsilon` | Held-out RMS | Gain from scalar |
|---|---:|---:|---:|
| Constant | 0.005 | 1.2463 arcsec | 0.79% |
| Hard no-cross | 0.005 | 1.2438 arcsec | 0.99% |
| Rational no-cross | 0.005 | 1.2447 arcsec | 0.91% |
| Tanh no-cross | 0.005 | **1.2430 arcsec** | **1.05%** |

The smoother tanh travel law is the best small change, but the gain is tiny and
post-hoc. At `epsilon=0.01`, tanh already loses a root; hard and rational
no-cross retain the roots but are worse than the scalar baseline. Thus the
usable positive interval is extremely narrow.

## What this establishes

1. P0582's tanh contrast result was a root-stability result on four spent
   clusters, not a transferable angular-lensing solution.
2. RX J2129 rejects unit-strength K0338 endpoint return even with full nuisance
   refitting.
3. Saturation shape and strength are not the main failure: hard-5 and tanh-20
   fail almost identically.
4. Constant-distance centroid crossing is a real formula defect. The physically
   cleaner travel law is `L tanh(d/L)`.
5. Removing overshoot improves the local response by only about one percentage
   point; global-centroid direction remains the dominant problem.
6. A successful next formula should derive multiple local destinations from
   baryonic structure or a baryonic field tensor. It should not rescue this
   branch with a per-cluster amplitude.

The angular construction remained curl-free and zero-monopole throughout, so
Solar and axisymmetric galaxy controls remain null. That consistency does not
offset the failed raw lens transfer.

## Reproduction

```powershell
python scripts/run_p0583_tanh_endpoint_rxj2129.py
python scripts/run_p0583b_signed_endpoint_amplitude.py
python scripts/run_p0584_no_overshoot_endpoint.py
pytest -q tests/test_p0583_p0584_rxj2129_results.py
```

Machine-readable outputs are in:

- `results/p0583_tanh_endpoint_rxj2129/`;
- `results/p0583b_signed_endpoint_amplitude/`; and
- `results/p0584_no_overshoot_endpoint/`.

## Subsequent local-attractor result

P0585 replaced the global centroid with 32 baryon-derived local-attractor
maps. The best local map reached 1.24235 arcsec at `epsilon=0.005`, versus
1.24301 for the global no-cross map: only a 0.053% difference. Local/global
mix, softening, and falloff each had spans below 0.001 arcsec. This confirms
that destination refinement cannot rescue a channel whose root-safe amplitude
is already nearly zero.
