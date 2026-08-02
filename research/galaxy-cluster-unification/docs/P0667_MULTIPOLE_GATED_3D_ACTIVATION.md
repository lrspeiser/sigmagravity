# P0667 multipole-gated 3D activation

## New geometric term

P0667 multiplies the local physical-length tensor coefficient by a dimensionless
baryonic multipole invariant:

\[
D^2={|c_\star-c_g|^2\over {\rm tr}Q_\star+{\rm tr}Q_g},
\qquad
Q^2=\left\|{Q_\star\over{\rm tr}Q_\star}
-{Q_g\over{\rm tr}Q_g}\right\|_F^2,
\]

\[
M_{\rm multipole}=1-e^{-(D^2+Q^2)},
\qquad
\sigma_{\rm final}=M_{\rm multipole}\,\sigma_{\rm local}.
\]

The centroids and covariance tensors are measured directly from the baryonic
components. The gate is dimensionless and adds no length, amplitude, or
per-object gravity parameter.

## Frozen result

All 18 gates pass:

- co-centered radial multipole gate: `1.68e-32`;
- co-centered radial mass-weighted `sigma`: `3.65e-37`;
- displaced multipole gate: `0.340739`;
- displaced mass-weighted `sigma`: `0.0233408`;
- displaced signal retained from P0666: `34.074%`;
- rotation error: `2.22e-16`;
- component-exchange error: `0.0`;
- translation error: `0.133%`;
- scale-covariance error within the frozen threshold; and
- positive constitutive eigenvalues with bounded `sigma`.

No spent or sealed outcome was opened.

## Interpretation

Unlike a low-acceleration or surface-density switch, this term measures a
specific multipole relationship between baryonic components. A spherical
stellar core and spherical gas halo at the same center produce no tensor
response regardless of their different sizes. A displaced centroid or
different normalized quadrupole activates it.

This is closer to the proposed gravity-routing idea: the field responds when
different baryonic components create a persistent, spatially organized set of
directions rather than merely when gravity becomes weak.

## Claim boundary

The gate is global. Locally opposed substructures could cancel in its moments,
so registered maps and raw lens topology remain essential. It is not yet
derived from a covariant action or microscopic vector-transport theory.

## Reproduction

```powershell
python scripts/run_p0667_multipole_gated_3d_activation.py
python -m pytest tests/test_multipole_activation_3d.py tests/test_p0667_multipole_gated_3d_activation.py -q
```
