# P0554 multifamily image-multiplicity results

## Outcome

The global root test now covers all 27 source families and 77 published images
in the five raw clusters. It compares five representative formulas using the
same 120-arcsecond aperture, two interlaced seed grids, archived geometry and
source positions, and no gravity refit.

No formula predicts the published multiplicity in every family. The present
angular route improves position residuals but produces more potentially visible
surplus roots than scalar P0554. Its only root-count changes occur in two
MACS1931 families, where it creates a pair in each.

## Frozen primary result

An unassigned root is screened as “potentially observable” when its absolute
magnification is at least 0.25 times that of the faintest assigned root in the
same formula and source family. This is a diagnostic threshold, not a survey
completeness model.

| Formula | Missing families | Exact families | Demagnified-only surplus | Potentially observable surplus families | Potentially observable surplus roots | Equal-family position RMS |
|---|---:|---:|---:|---:|---:|---:|
| Scalar P0554 | 7 | 12 | 1 | 7 | 8 | 10.531 arcsec |
| Photon softness 0.98 | 8 | 12 | 1 | 6 | 7 | 9.986 |
| Route only | 7 | 11 | 0 | 9 | 12 | 9.228 |
| Combined parent | 7 | 11 | 0 | 9 | 12 | 9.178 |
| Combined route power 2.4 | 7 | 11 | 1 | 8 | 10 | 9.060 |

The route-only RMS is 12.4% lower than baseline and the power-2.4 combined RMS
is 14.0% lower. That is a useful continuous effect, but it is not a clean win:
route-only adds four potentially observable surplus roots relative to baseline.

## Where topology actually changes

The route, combined parent, and power-2.4 formula have exactly the same root
count as scalar P0554 in 25 of 27 families. The only changes are:

| System and family | Published images | Scalar roots | Route roots | Change |
|---|---:|---:|---:|---:|
| MACS1931 family 2 | 3 | 3 | 5 | +2 |
| MACS1931 family 3 | 2 | 5 | 7 | +2 |

This makes the earlier MACS1931 result less likely to be a universal beneficial
feature. The route is locally pushing an already fragile MACS1931 mapping
through caustics. Family 2's new pair is unsupported by the direct F160W audit;
family 3 now becomes another concrete counterimage target.

Photon softness has the opposite topology effect in one place only: it changes
MACS1931 family 1 from five roots to three for four published images, turning a
surplus into missing multiplicity. Continuous residual improvement can therefore
also cross the wrong caustic boundary.

## Threshold sensitivity

Because root detectability is not a fundamental 0.25 cutoff, the saved roots
were rescored descriptively at four relative-magnification thresholds:

| Threshold vs faintest assigned image | Baseline surplus families / roots | Route surplus families / roots | Combined parent | Power 2.4 |
|---:|---:|---:|---:|---:|
| 0.10 | 8 / 10 | 9 / 12 | 9 / 14 | 9 / 13 |
| 0.25 | 7 / 8 | 9 / 12 | 9 / 12 | 8 / 10 |
| 0.50 | 7 / 8 | 9 / 11 | 9 / 11 | 8 / 9 |
| 1.00 | 6 / 7 | 8 / 10 | 8 / 10 | 6 / 7 |

Route-only has more surplus roots than baseline at every checked threshold.
The magnitude changes, but the direction does not.

One assignment caveat is instructive. In MACS1931 family 2, the power-2.4
formula assigns the second bright near-2c root to observed image 2a despite a
19.1-arcsecond error. A remaining unassigned root sits at 0.2496 times the
faintest assigned magnification—just below the primary cutoff. This is why
binary surplus labels cannot replace inspection of positions, parities, and
all individual roots.

## Universal lesson and next formula target

The data favor a constrained objective, not simply “more routed gravity”:

$$
\text{improve existing-image positions}
\quad\text{subject to}\quad
N_{\rm observable\ roots}=N_{\rm observed\ images}.
$$

The route contains a smooth positional benefit, but its MACS1931 amplitude and
shape cross two unwanted caustics. That continuation has now been run. Eta =
0.30 is the best descriptive subcritical setting, but improves the MACS1931
equal-family assigned RMS by only 0.468%. Family 3 first crosses a caustic at
eta = 0.60, family 2 at eta = 1.00, and the full-strength 24.855% improvement
carries six potentially observable surplus roots instead of two. See
[`P0554_SUBCRITICAL_ROUTE_SCAN_RESULTS.md`](P0554_SUBCRITICAL_ROUTE_SCAN_RESULTS.md).

## Limits

Two-image catalogs commonly omit demagnified central images, so raw surplus
count alone is not a rejection. Published catalogs may be incomplete; the
relative-magnification screen ignores foreground masking and actual source
flux; the root search is finite; and all systems are spent evidence. The direct
MACS1931 F160W result is stronger than this cross-family screening statistic.

## Reproduction

```powershell
python scripts/run_p0554_multifamily_multiplicity.py
python scripts/run_p0554_multifamily_multiplicity.py --postprocess-only
python -m pytest tests/test_p0554_multifamily_multiplicity.py -q
```

Machine-readable outputs are in `results/p0554_multifamily_multiplicity/`.
