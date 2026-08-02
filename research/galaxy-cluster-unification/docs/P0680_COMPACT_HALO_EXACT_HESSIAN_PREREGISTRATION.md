# P0680 compact-halo exact-Hessian preregistration

P0679 shows second-order curl convergence but fails its frozen count because
the `0.1 arcsec` direct step misses `1e-5` by four percent. Its `0.01 arcsec`
curl is already `1.04e-7`, and convergence/Jacobian derivatives are stable
below one part per million.

P0680 follows the prescribed replacement path. At the identical 92 P0678
strong-lens points it requests `LensModel.hessian(..., diff=None)`, which uses
exact lens-model derivatives when supplied. It freezes three decisive gates:

- Hessian symmetry, expressed as normalized curl, below `1e-12`;
- convergence agreement with direct `0.01 arcsec` differences below `1e-5`;
  and
- lens-Jacobian determinant agreement below `1e-5`.

Every point must be finite and all P0678/P0679 provenance failures must match
exactly. A pass qualifies P0678 as a numerically reliable *spent target
specification* while preserving both failed historical reports. It fits no
formula and scores no image root or sealed target.
