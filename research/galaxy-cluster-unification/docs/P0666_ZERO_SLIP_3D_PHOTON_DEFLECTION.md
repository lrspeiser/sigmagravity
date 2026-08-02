# P0666 zero-slip 3D photon deflection

## Frozen test

P0666 combined two new pieces without opening an image catalog:

1. a three-dimensional baryonic activation constructed from separately solved
   stellar and gas Newtonian fields; and
2. the fixed zero-slip weak-field photon closure

\[
\boldsymbol\alpha=-{2\over c^2}{D_{ds}\over D_s}
\int \mathbf a_\perp\,dz.
\]

## Result

P0666 fails one of 18 frozen gates:

- point-mass GR deflection median error: `1.528%`;
- point-mass p95 error: `2.910%`;
- linear mass-scaling error: below the frozen threshold;
- rotation covariance error: `4.19e-15`;
- normalized deflection curl: `1.53e-16`;
- surface-to-volume mass error: below the frozen threshold;
- offset 3D mass-weighted `sigma`: `0.06850`; and
- co-centered radial 3D mass-weighted `sigma`: `2.168e-5`, above the
  required `1e-8` null.

The candidate does not advance to an RX J2129 map build. No spent or sealed
lensing outcome was opened.

## Diagnosis

The zero-slip photon normalization is sound. The failure lies in the 3D
activation: separately discretized spherical stellar and gas fields are not
perfectly parallel on a finite Cartesian grid. Squaring their small angular
difference produces a false positive even though the continuum fields are
radial.

The offset signal is more than three thousand times larger than the radial
artifact, but the frozen null is absolute and cannot be relaxed. A legitimate
next test must remove only the measured discretization-odd component—for
example by cubic reflection/rotation symmetrization—without reducing a real
offset signal or adding a physical threshold.

## Claim boundary

The weak-field zero-slip metric remains a closure rather than a derived
covariant action. Point-mass recovery validates normalization only. P0666 is a
failed activation test, not lensing evidence.

## Reproduction

```powershell
python scripts/run_p0666_zero_slip_3d_photon_deflection.py
python -m pytest tests/test_metric_lensing_3d.py tests/test_p0666_zero_slip_3d_photon_deflection.py -q
```
