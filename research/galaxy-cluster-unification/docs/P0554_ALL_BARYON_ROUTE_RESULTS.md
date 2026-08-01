# P0554 registered baryon-proxy route results

## Outcome

Registered continuous starlight and X-ray morphology do **not** improve the
member-galaxy route direction in this test. All eight map-based directions
remain better than turning the angular route off, but all are worse than the
discrete member-only direction on the four-cluster primary aggregate.

That is useful evidence against the simplest version of the idea. If apparent
halo gravity is baryonic gravity being redirected, the direction is not just
the gradient toward one smooth map of everything visible. The result instead
favors separated baryonic structures—or an unmeasured property correlated
with their locations—as the current directional clue.

No variant passes the frozen exact-refit follow-up rule and no formula is
promoted.

## Data added

The acquisition adds the public CLASH RX J2129 F160W science and weight mosaics
and selected public Chandra primary event/image products for the other four
clusters. Together with already archived inputs, the test has:

- ten registered F160W science/weight files covering every one of 84 cataloged
  image coordinates;
- eleven Chandra observations totaling 307.3 ks;
- 4,840 to 47,504 soft-band events per system inside 100 arcseconds.

Every input hash, FITS event table, HST coverage gate, and predeclared count
gate passes. X-ray brightness is an emissivity/location proxy, not a gas-mass
map.

## Backtracking equation

For every route-launching member at position $\mathbf x_i$, a registered proxy
map $\Sigma_P$ supplies the direction

$$
\hat{\mathbf d}_{P,i}=\operatorname{unit}\left[
\int d^2x\,\Sigma_P(\mathbf x)
\frac{\mathbf x-\mathbf x_i}
{(|\mathbf x-\mathbf x_i|^2+s^2)^{3/2}}
\right],
$$

with the previously selected $s=200$ kpc. This is the continuous-map analogue
of the member-to-member inverse-square direction. Only the unit direction is
retained; neither HST flux nor X-ray counts set the route strength.

The endpoint remains

$$
\mathbf y_i=\mathbf x_i+L_i\hat{\mathbf d}_{P,i},
$$

and the inherited conservative map moves the same amount away from each launch
point and toward its endpoint. The circular monopole is removed before the
deflection is added, so this screen cannot silently add mass.

## Frozen variants and score

The fixed-geometry screen tested no route, member-only, continuous F160W,
unmasked and approximately point-source-masked X-ray maps, three star/gas
direction blends, and two member/map blends.

| Direction | Primary RMS | Change vs no route | Change vs member route | Primary clusters better than member |
|---|---:|---:|---:|---:|
| Discrete members | **14.198 arcsec** | **+0.824%** | baseline | — |
| 50% member + 50% continuous stars | 14.204 | +0.781% | -0.043% | 0 of 4 |
| Continuous F160W | 14.225 | +0.630% | -0.195% | 0 of 4 |
| 50% member + 50% star/gas blend | 14.234 | +0.566% | -0.260% | 1 of 4 |
| 25% gas in star/gas direction | 14.247 | +0.479% | -0.347% | 0 of 4 |
| Masked X-ray morphology | 14.250 | +0.460% | -0.367% | 1 of 4 |
| Unmasked X-ray morphology | 14.252 | +0.445% | -0.382% | 1 of 4 |
| 75% gas in star/gas direction | 14.254 | +0.427% | -0.400% | 1 of 4 |
| 50% gas in star/gas direction | 14.256 | +0.416% | -0.411% | 0 of 4 |
| No route | 14.315 | baseline | -0.831% | 0 of 4 |

The point-source mask changes the gas result only slightly, so compact X-ray
sources are not the reason the gas direction loses.

## Structural lesson

In RX J2129 and MACS0329, continuous-starlight directions are closely aligned
with the member direction (weighted cosines 0.983 and 0.961). In MACS1931 the
alignment falls to 0.568. MACS0429's gas/member alignment is only 0.413, yet
its held-out lens score is almost insensitive to every route direction. The
clusters therefore separate two effects:

1. Baryonic components can genuinely point in different projected directions.
2. A direction difference matters only where the available images constrain
   that angular mode.

The data do not support replacing member structure with a smooth all-baryon
center. The next backtracking test should start at the *assumed excess-gravity
arrival map* and ask which member launch points can supply it, rather than
choosing a baryonic destination direction first.

## Cross-domain controls and limits

SPARC outer error (12.571 km/s), radial CLASH error (0.1964 dex), Mercury
precession (-1.730 mas/century), and all Solar proxies remain unchanged by
construction. The route is angular and zero-monopole. This is preservation,
not an improvement in those domains.

This is a fixed-geometry, Jacobian-linearized screen on spent clusters. F160W
has mass-to-light and contamination uncertainty; X-ray surface brightness is
not exposure/temperature/deprojection-corrected gas mass; and the approximate
point-source mask is not a final gas analysis. Exact roots were not run because
no map variant met the predeclared gate.

## Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_p0554_all_baryon_route.ps1
python scripts/audit_p0554_all_baryon_route_inputs.py
python scripts/run_p0554_all_baryon_route_screen.py
python scripts/run_p0554_all_baryon_route_screen.py --postprocess-only
python -m pytest tests/test_baryon_morphology.py tests/test_route_template.py tests/test_p0554_all_baryon_route_inputs.py tests/test_p0554_all_baryon_route_screen.py -q
```

Machine-readable products are in
`results/p0554_all_baryon_route_input_audit/` and
`results/p0554_all_baryon_route_screen/`.
