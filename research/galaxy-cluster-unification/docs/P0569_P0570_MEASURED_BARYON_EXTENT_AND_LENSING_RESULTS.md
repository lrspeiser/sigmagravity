# P0569-P0570: measured baryon extent and raw-lensing results

## Question

P0568 found that spreading member-galaxy light over roughly 75--125 kpc
changed inferred cluster morphology much more than changing the directional
gravity tensor. P0569 asks whether that distance is already present in
measured baryons. P0570 then replaces the smoothed-light proxy with registered
stellar and hot-gas maps and calculates their noncircular weak-field
deflection directly.

## P0569 measured extent

The audit used four CLASH clusters with registered F160W starlight, Chandra
morphology, and physical ACCEPT gas-density shells. For a circular 2D Gaussian,
the equivalent width inferred from RMS radius is

\[
\sigma_{\rm RMS}=\frac{\sqrt{\langle R^2\rangle}}{\sqrt{2}},
\]

and the independent 80%-enclosed width is

\[
\sigma_{80}=\frac{R_{80}}{\sqrt{2\ln 5}}.
\]

For the primary stars-plus-gas map, the median widths were 153.3 and 161.5
kpc, respectively. Individual RMS widths ranged from 143.2 to 174.2 kpc. Zero
of four clusters lay in the P0568 75--125 kpc band by either definition.

| Component | Median sigma from RMS | Median sigma from R80 |
|---|---:|---:|
| ACCEPT gas, spherical | 153.1 kpc | 160.1 kpc |
| ACCEPT gas, sqrt morphology | 153.0 kpc | 160.1 kpc |
| Registered starlight | 160.9 kpc | 170.9 kpc |
| Member catalogue | 161.8 kpc | 162.1 kpc |
| Stars + gas, sqrt morphology | 153.3 kpc | 161.5 kpc |

The maps are broad enough to explain why broadening helped P0568, but their
measured width does not reproduce the fitted interval. Hot gas supplies
94.8--99.1% of the assigned projected map mass in this construction, so the
combined extent is effectively a gas-extent measurement. The 120-arcsec map
window also truncates still more extended material.

## P0570 conservative residual formula

The locked scalar lens already supplies a circular radial profile. To avoid
counting that profile twice, P0570 adds only the measured map's angular
residual:

\[
\boldsymbol\alpha=
\boldsymbol\alpha_{\rm locked}
+q\left(\boldsymbol\alpha_b-\boldsymbol\alpha_{b,m=0}\right)
+\boldsymbol\alpha_{\rm shear}.
\]

Here, \(\boldsymbol\alpha_b\) is the standard weak-field point-mass sum from
the registered baryon map, and \(\boldsymbol\alpha_{b,m=0}\) is its exact
azimuthally averaged potential differentiated on the same grid. This
potential-level subtraction makes the correction conservative. The sole
gravity-response number is universal: \(q=1\) means ordinary GR-strength
angular baryon deflection; \(q=0\) is the control.

Five measured components, three fixed extent rescalings, and three nonzero
responses produced 45 candidates. Two clusters selected the candidate using
unweighted source-plane closure. Two different clusters were then refitted as
the held-out validation set.

## Raw-lensing result

Selection chose ACCEPT gas with sqrt X-ray morphology, 0.75 times its measured
extent, and \(q=2\). That means the best development setting asked for twice
the ordinary angular residual rather than discovering a parameter-free
GR-strength effect.

| Validation metric | Result |
|---|---:|
| Control held-out RMS | 18.444 arcsec |
| Selected held-out RMS | 18.418 arcsec |
| Held-out improvement | 0.140% |
| Frozen required improvement | 5.0% |
| Compact-halo benchmark | 9.989 arcsec |
| Selected / compact-halo RMS | 1.844 |
| Allowed ratio | 1.25 |

The selected correction loses at least one required exact image root in
MACS0329 on the development set. Both validation clusters preserve their
roots, but their combined gain is negligible. Thus the formula fails the
development-root, held-out-improvement, and compact-halo gates.

The screen itself is insensitive to all three tested coordinates: component
choice changes mean source-plane RMS by only 0.65%, response by 0.34%, and
extent by 0.15%. This is much smaller than the 67.5% map-shape effect caused by
the phenomenological smoothing sweep in P0568.

## Numerical and cross-domain audit

- Maximum normalized curl over all 60 constructed fields: 0.
- Centered circular point-source residual: \(7.23\times10^{-15}\) of its full
  field.
- Centered circular point-source normalized curl: 0.
- The angular residual is exactly zero for an isolated axisymmetric galaxy
  and centered Solar point mass.

The last item is a limit, not a galaxy success. P0570 inherits the locked
fixed-RAR galaxy score of 10.35 km/s and zero Solar change only because it is
an angular closure. It does not replace the scalar law or independently
explain galaxy rotation without MOND/RAR-like behavior.

## What changed our understanding

1. The useful P0568 distance was not simply the physical RMS width of stars
   plus gas; the measured maps are broader.
2. Using real gas geometry directly does not recover the compact-halo lensing
   result. Its held-out effect is only 0.14%.
3. Merely multiplying the measured angular residual is not the missing law.
   Even \(q=2\) is too small in predictive impact and can destroy image roots.
4. A successful gravity-flow model must change how baryonic influence is
   organized radially or nonlocally, not just add the ordinary noncircular
   weak-field vectors already present in the map.
5. Solar safety remains easy for an explicitly angular/circular-null term;
   the hard problem is jointly producing galaxy radial support and cluster
   image topology.

## Next distinct test

The next formula should not add another free orientation or simple amplitude.
It should test a conservative nonlocal routing kernel whose characteristic
length is calculated from a measured baryonic observable, then normalize the
redistributed field so total baryon-sourced flux is conserved. Candidate
observables are gas entropy/pressure scale length, star-gas coherence change
with radius, and the radius where baryonic acceleration crosses one universal
threshold. The length rule must be frozen on galaxies or development clusters
and transferred unchanged to held-out clusters, SPARC, and the Solar null.

## Reproduce

```powershell
python scripts/run_p0569_measured_baryon_extent_audit.py
python scripts/run_p0570_physical_baryon_residual_lensing.py
python -m pytest -q tests/test_p0569_p0570_measured_baryon_results.py
```
