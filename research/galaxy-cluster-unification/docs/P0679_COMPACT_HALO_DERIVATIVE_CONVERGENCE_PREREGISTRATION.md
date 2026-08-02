# P0679 compact-halo derivative convergence preregistration

P0678 fails only because the analytic NIE comparator has normalized curl
`0.01254` when its sampled 33-cell image is differentiated. Since an NIE
deflection derives from a lens potential, P0679 tests whether this is the
expected coarse finite-difference error without lowering the `1e-5` gate.

Two independent checks are frozen:

1. sample the identical NIE on 33, 65, 129, and 257 cells across the same
   physical domain and require normalized curl to decrease at every step and
   improve by at least tenfold; and
2. at the original strong-lens points, evaluate the analytic deflection at
   `x+-h` and `y+-h` for six steps from `0.5` to `0.01 arcsec`. At least four
   steps and the smallest step must beat `1e-5`; the two smallest steps must
   agree in convergence and Jacobian determinant to relative RMS `1e-5`.

Every derivative must be finite. No formula is fitted, image root scored, or
sealed target opened. A pass qualifies P0678 as a numerically reliable spent
target specification; it does not validate the compact halo or a new theory.
