# P0695 radial path potential mathematical audit

Frozen before synthetic metrics: 2026-08-02

Verdict: radial/path identities pass; first-order interpolation **fails** the
spherical tangential-leakage and angular-scatter gates

## Audited operator

\[
\Phi_{\rm path}(\mathbf{x})=\Phi_N(\mathbf{x}_c)+
\int_0^1 -\nu\!\left({|\mathbf g_N(\mathbf{x}_c+t\mathbf d)|\over a_0}\right)
\mathbf g_N(\mathbf{x}_c+t\mathbf d)\cdot\mathbf d\,dt,
\]

\[
\Phi_{\rm joint}=\Phi_{\rm path}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

The primary uses 24-point Gauss-Legendre quadrature and first-order Cartesian
field interpolation. A 48-point run is the convergence control. No galaxy or
cluster outcome is read.

## Frozen result

| Metric | Result | Gate | Verdict |
|---|---:|---:|---|
| spherical radial relative RMS | `0.01165` | `<=0.05` | pass |
| spherical median absolute relative error | `0.00841` | `<=0.03` | pass |
| spherical tangential/radial RMS | `0.09148` | `<=0.03` | fail |
| maximum spherical angular scatter | `0.05794` | `<=0.05` | fail |
| 24-to-48 acceleration difference | `0.00441` | `<=0.02` | pass |
| 90-degree rotation potential RMS | `1.03e-15` | `<=1e-10` | pass |
| 90-degree rotation acceleration RMS | `6.11e-15` | `<=1e-10` | pass |
| maximum normalized curl | `1.06e-16` | `<=1e-10` | pass |
| hybrid potential identity error | `0` | `<=1e-14` | pass |
| routing-correction boundary mismatch | `0` | `<=1e-14` | pass |

The radial concept works: over the frozen spherical annulus, the path field is
within about one percent of its algebraic simple-MOND radial derivative. It is
also a genuine scalar field with negligible curl, exact 90-degree covariance,
stable quadrature, and exact boundary cancellation of the routing correction.

The failure comes from linear interpolation of the Cartesian Newtonian vector
samples along oblique rays. It transfers the underlying grid anisotropy into
angular variations of the completed potential. The radial shell means remain
accurate, but the derived acceleration contains too much tangential power for
the preregistered spherical limit.

## Next numerical audit

The only admissible follow-up changes the already-supported numerical
interpolation order from one to three. It must retain the same physical
equation, grids, density fields, 24/48 quadrature orders, masks, and thresholds
in a separately frozen protocol. No observational score may be calculated.

If cubic interpolation still fails, the straight-ray Cartesian implementation
is retired. If it passes, that exact implementation may advance to one
separately frozen spent DDO154/RX J2129 joint screen.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0695_radial_path_potential_math_audit.py
python -m pytest tests/test_radial_path_potential.py tests/test_source_routing_qumond.py tests/test_field_solvers.py -q
```

Artifacts are in
`results/p0695_radial_path_potential_math_audit/`.

## Claim boundary

P0695 is a numerical audit, not evidence for a gravity theory. Straight rays
from a global baryonic centroid are nonlocal and lack a covariant action or
causal evolution law. P0633 and P0640 remain sealed.
