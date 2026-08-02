# P0695B cubic radial path audit

Frozen before synthetic metrics: 2026-08-02

Verdict: cubic interpolation improves accuracy but **fails** the unchanged
tangential and angular gates; straight-ray Cartesian path implementation
retired

## Single allowed change

P0695B repeats P0695 with only the field-interpolation order changed from one
to three. The physical equation, synthetic densities, 33³ grid, 24/48
Gauss-Legendre orders, comparison annulus, and every rejection threshold are
unchanged. No observational outcome is read.

| Metric | Linear P0695 | Cubic P0695B | Gate | Verdict |
|---|---:|---:|---:|---|
| spherical radial relative RMS | `0.01165` | `0.00652` | `<=0.05` | pass |
| spherical median absolute error | `0.00841` | `0.00398` | `<=0.03` | pass |
| tangential/radial RMS | `0.09148` | `0.06608` | `<=0.03` | fail |
| maximum angular scatter | `0.05794` | `0.06011` | `<=0.05` | fail |
| 24/48 acceleration difference | `0.00441` | `0.0000427` | `<=0.02` | pass |
| rotation potential / acceleration RMS | machine scale | `1.14e-15 / 6.94e-15` | `<=1e-10` | pass |
| maximum normalized curl | `1.06e-16` | `1.18e-16` | `<=1e-10` | pass |
| correction boundary mismatch | `0` | `0` | `<=1e-14` | pass |

Cubic interpolation removes nearly all quadrature sensitivity and improves the
mean radial response, but it does not remove the angular imprint of sampling a
Cartesian Newtonian vector field along oblique rays. The straight-ray
implementation therefore does not advance to DDO154 or RX J2129.

## Next multipole operator

Avoid ray interpolation entirely. About the baryonic centroid, decompose the
Newtonian potential into a spherical monopole plus measured multipoles:

\[
\Phi_N(\mathbf{x})=\Phi_{N,0}(r)+\delta\Phi_N(\mathbf{x}).
\]

Boost only the coherent monopole by integrating its spherical simple-MOND
acceleration, while retaining the observed Newtonian multipoles:

\[
\Phi_{\rm coh}(\mathbf{x})=Phi_{M,0}(r)+\delta\Phi_N(\mathbf{x}),
\qquad
{d\Phi_{M,0}\over dr}=g_{M,0}(r).
\]

Then retain the successful cluster topology correction:

\[
\Phi_{\rm joint}=\Phi_{\rm coh}
+e_{2D}(\Phi_{\rm route}-\Phi_{\rm local}).
\]

This is a genuine multipole/coherence hypothesis: only the component shared
by every direction receives the low-acceleration enhancement; bars, clumps,
and asymmetries remain present at their Newtonian multipole strength. It has no
new fitted physical constant, exact spherical and high-acceleration limits,
and no interpolated path. It requires another synthetic, no-observation audit
before any empirical score.

## Reproduction

```powershell
$env:PYTHONPATH='src'
python scripts/run_p0695_radial_path_potential_math_audit.py --config configs/p0695b_cubic_radial_path_potential_math_audit.json
```

Artifacts are in
`results/p0695b_cubic_radial_path_potential_math_audit/`.

## Claim boundary

P0695B rejects one numerical realization, not all path-dependent physics. No
galaxy, cluster, P0633, or P0640 outcome was opened.
